# region imports
from AlgorithmImports import *
import numpy as np
import random
from collections import deque
from datetime import timedelta

from infra   import add_universe, KRAKEN_PAIRS, OrderHelper, PartialFillTracker, PersistenceManager
from models  import build_features, compute_atr, VoxEnsemble, build_training_data, walk_forward_train
from risk    import RegimeFilter, RiskManager, compute_qty
# endregion

# ── Strategy constants (all overridable via the QC parameter panel) ───────────
TAKE_PROFIT          = 0.020   # +2.0 %  close long on gain
STOP_LOSS            = 0.012   # −1.2 %  close long on loss
TIMEOUT_HOURS        = 3.0     # close after this many hours regardless
ATR_TP_MULT          = 2.0     # TP = entry + ATR_TP_MULT × ATR
ATR_SL_MULT          = 1.2     # SL = entry − ATR_SL_MULT × ATR
SCORE_MIN            = 0.60    # minimum mean_proba to open a position
SCORE_GAP            = 0.05    # required lead of top coin over runner-up
MAX_DISPERSION       = 0.15    # max std_proba across models
MIN_AGREE            = 4       # min models with proba >= 0.5
ALLOCATION           = 0.50    # fallback fraction of portfolio if Kelly disabled
KELLY_FRAC           = 0.25    # fractional-Kelly multiplier
MAX_ALLOC            = 0.80    # hard ceiling on any single trade allocation
USE_KELLY            = True    # set False to use flat ALLOCATION
MAX_DAILY_SL         = 2       # halt new entries after this many SL hits per day
COOLDOWN_MINS        = 15      # minutes to wait after any exit before re-entering
SL_COOLDOWN_MINS     = 60      # per-coin cooldown specifically after an SL exit
MAX_DD_PCT           = 0.08    # drawdown circuit-breaker: halt if equity drops > 8 %
CASH_BUFFER          = 0.99    # keep 1 % cash headroom for fees/rounding
RESOLUTION_MINUTES   = 5       # subscribe at 5-min bars, consolidate internally
DECISION_INTERVAL_MIN = 15     # only evaluate entries at 15-min boundaries
WARMUP_DAYS          = 90      # bars of history needed before trading

# ─────────────────────────────────────────────────────────────────────────────
# Vox — ML Ensemble Kraken Rotation Strategy
#
# Architecture (see README.md for full diagram):
#   Data → 5-min consolidator → feature deques
#        → 15-min decision tick
#        → VoxEnsemble (5-model soft vote)
#        → confidence gate (score_min / score_gap / dispersion / n_agree)
#        → regime gate (4h BTC SMA + slope)
#        → Kelly sizer
#        → pre-trade validation (price, cash, lot-size, min-order)
#        → market_order
#        → fill-driven state machine in on_order_event
#
# Key design decisions (inherited from main.py):
#   1. Position state updated ONLY via on_order_event confirmed fills.
#   2. Exits use actual portfolio quantity (never assumed target).
#   3. Pre-trade validation guards against Price=0 / insufficient cash.
#   4. _reconcile() corrects state drift on every bar.
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
random.seed(42)


class VoxAlgorithm(QCAlgorithm):
    """
    Execution-first ML ensemble strategy for Kraken spot crypto.

    Inherits the fill-driven state machine from KrakenTopCoinAlgorithm and
    layers on:
      • A 5-model heterogeneous ensemble (LogReg / RF / LGBM / ET / GNB).
      • Triple-barrier labeling for supervised training.
      • Fractional-Kelly position sizing.
      • 4h BTC regime filter.
      • Full risk management (drawdown CB, per-coin SL cooldown, daily cap).
      • ObjectStore persistence for model and trade log.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(10_000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.set_time_zone(TimeZones.UTC)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # ── Parameters — overridable via QC parameter panel ───────────────────
        self._tp       = float(self.get_parameter("take_profit")      or TAKE_PROFIT)
        self._sl       = float(self.get_parameter("stop_loss")        or STOP_LOSS)
        self._toh      = float(self.get_parameter("timeout_hours")    or TIMEOUT_HOURS)
        self._atr_tp   = float(self.get_parameter("atr_tp_mult")      or ATR_TP_MULT)
        self._atr_sl   = float(self.get_parameter("atr_sl_mult")      or ATR_SL_MULT)
        self._s_min    = float(self.get_parameter("score_min")        or SCORE_MIN)
        self._s_gap    = float(self.get_parameter("score_gap")        or SCORE_GAP)
        self._max_disp = float(self.get_parameter("max_dispersion")   or MAX_DISPERSION)
        self._min_agr  = int(self.get_parameter("min_agree")          or MIN_AGREE)
        self._alloc    = float(self.get_parameter("allocation")       or ALLOCATION)
        self._kf       = float(self.get_parameter("kelly_frac")       or KELLY_FRAC)
        self._max_alloc= float(self.get_parameter("max_alloc")        or MAX_ALLOC)
        _uk_raw        = self.get_parameter("use_kelly")
        self._use_kelly= (str(_uk_raw).lower() in ("true","1","yes")) if _uk_raw else USE_KELLY
        self._max_sl   = int(self.get_parameter("max_daily_sl")       or MAX_DAILY_SL)
        self._cd_mins  = int(self.get_parameter("cooldown_mins")      or COOLDOWN_MINS)
        self._sl_cd    = int(self.get_parameter("sl_cooldown_mins")   or SL_COOLDOWN_MINS)
        self._max_dd   = float(self.get_parameter("max_dd_pct")       or MAX_DD_PCT)
        self._cb       = float(self.get_parameter("cash_buffer")      or CASH_BUFFER)

        # ── Universe ──────────────────────────────────────────────────────────
        self._symbols  = add_universe(self)
        self._btc_sym  = next(
            (s for s in self._symbols if s.value.upper().startswith("BTC")), None
        )

        # ── Slippage model ────────────────────────────────────────────────────
        for sym in self._symbols:
            self.securities[sym].set_slippage_model(ConstantSlippageModel(0.001))

        # ── Per-symbol state deques ───────────────────────────────────────────
        _dq = lambda n: deque(maxlen=n)
        self._state = {}
        for sym in self._symbols:
            self._state[sym] = {
                "closes":  _dq(200),
                "highs":   _dq(200),
                "lows":    _dq(200),
                "volumes": _dq(200),
            }

        # ── 5-minute consolidators ────────────────────────────────────────────
        for sym in self._symbols:
            self.consolidate(
                sym,
                timedelta(minutes=RESOLUTION_MINUTES),
                lambda bar, s=sym: self._on_5m(s, bar),
            )

        # ── Regime filter (4h BTC consolidator wired inside) ──────────────────
        self._regime = RegimeFilter()
        if self._btc_sym:
            self._regime.update_btc(self, self._btc_sym)

        # ── Risk manager ──────────────────────────────────────────────────────
        self._risk = RiskManager(
            max_daily_sl     = self._max_sl,
            cooldown_mins    = self._cd_mins,
            sl_cooldown_mins = self._sl_cd,
            max_dd_pct       = self._max_dd,
            cash_buffer      = self._cb,
        )

        # ── Ensemble + persistence ────────────────────────────────────────────
        self._ensemble    = VoxEnsemble(logger=self.log)
        self._persistence = PersistenceManager(self)
        self._fill_tracker= PartialFillTracker()
        self._model_ready = False

        # Try to load a previously trained model
        saved = self._persistence.load_model()
        if saved is not None:
            self._ensemble.load_state(saved)
            self._model_ready = self._ensemble.is_fitted
            self.log("[vox] Loaded pre-trained model from ObjectStore.")

        # ── Position state — updated ONLY via on_order_event ──────────────────
        self._pos_sym     = None    # confirmed open position symbol
        self._entry_px    = 0.0     # confirmed entry fill price
        self._entry_time  = None    # confirmed entry fill time
        self._tp_dyn      = 0.0     # ATR-derived TP fraction for current trade
        self._sl_dyn      = 0.0     # ATR-derived SL fraction for current trade
        self._pending_sym = None    # symbol of in-flight entry order
        self._pending_oid = None    # order ID of in-flight entry order
        self._exiting     = False   # True while an exit order is in flight
        self._exit_time   = None    # time of most recent completed exit

        # ── Warm-up ───────────────────────────────────────────────────────────
        self.set_warm_up(timedelta(days=WARMUP_DAYS))

        # ── Scheduled events ──────────────────────────────────────────────────
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(0, 0),
            self._reset_daily,
        )
        # Weekly retrain on Monday at 00:05 UTC
        self.train(
            self.date_rules.week_start(),
            self.time_rules.at(0, 5),
            self._retrain,
        )

    # ── 5-minute bar handler ──────────────────────────────────────────────────

    def _on_5m(self, sym, bar):
        """Append OHLCV from a freshly closed 5-min bar to the state deques."""
        st = self._state[sym]
        st["closes"].append(float(bar.close))
        st["highs"].append(float(bar.high))
        st["lows"].append(float(bar.low))
        st["volumes"].append(float(bar.volume))

    # ── 4h BTC bar handler (wired via RegimeFilter.update_btc) ───────────────
    # (RegimeFilter registers its own consolidator callback; the method below
    #  is kept for documentation / direct wiring if needed)

    def _on_4h_btc(self, bar):
        """Relay a 4h BTC bar to the regime filter (not used — wired inside RegimeFilter)."""
        pass

    # ── Main data handler ─────────────────────────────────────────────────────

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Update rolling high for drawdown circuit-breaker on every bar
        self._risk.update_rolling_high(self.portfolio.total_portfolio_value)

        # Reconcile internal state against actual holdings
        self._reconcile()

        # ── Exit check — minute-level precision ───────────────────────────────
        if self._pos_sym is not None and not self._exiting:
            if self._pos_sym in data.bars:
                self._check_exit(float(data.bars[self._pos_sym].close))

        # ── Entry logic — fire only at DECISION_INTERVAL_MIN boundaries ───────
        if self.time.minute % DECISION_INTERVAL_MIN != 0:
            return
        if self._pos_sym is not None or self._pending_sym is not None:
            return

        # Kill switch
        if self._persistence.is_kill_switch_active():
            self.log("[vox] Kill switch active — skipping entry.")
            return

        # Model must be trained
        if not self._model_ready:
            return

        # Risk guards
        pv = self.portfolio.total_portfolio_value
        can, reason = self._risk.can_enter(
            sym=None, current_time=self.time, portfolio_value=pv
        )
        if not can:
            self.debug(f"[vox] Entry blocked: {reason}")
            return

        self._try_enter()

    # ── Order-event state machine ──────────────────────────────────────────────

    def on_order_event(self, order_event):
        oid   = order_event.order_id
        order = self.transactions.get_order_by_id(oid)
        if order is None:
            return
        sym = order.symbol
        tag = order.tag or ""

        if order_event.status == OrderStatus.FILLED:
            if tag == "ENTRY" and sym == self._pending_sym:
                # Entry fill confirmed — mark position active
                self._pos_sym    = sym
                self._entry_px   = float(order_event.fill_price)
                self._entry_time = self.time
                self._fill_tracker.on_fill(oid, abs(float(order_event.fill_quantity)))
                self._pending_sym = None
                self._pending_oid = None
                self.debug(
                    f"FILL ENTRY {sym.value}  px={self._entry_px:.4f}"
                    f"  qty={order_event.fill_quantity:.6f}"
                )

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                # Exit fill confirmed — clear position state
                ret = (
                    (order_event.fill_price - self._entry_px) / self._entry_px
                    if self._entry_px else 0.0
                )
                self.debug(
                    f"FILL EXIT {sym.value}  px={order_event.fill_price:.4f}"
                    f"  ret={ret:.3%}  tag={tag}"
                )
                is_sl = tag == "EXIT_SL"
                self._risk.record_exit(sym, is_sl=is_sl, exit_time=self.time)
                self._fill_tracker.clear(oid)
                self._pos_sym    = None
                self._entry_px   = 0.0
                self._entry_time = None
                self._tp_dyn     = 0.0
                self._sl_dyn     = 0.0
                self._exiting    = False
                self._exit_time  = self.time

        elif order_event.status == OrderStatus.PARTIALLY_FILLED:
            self._fill_tracker.on_fill(oid, abs(float(order_event.fill_quantity)))
            # For ENTRY: mark position active once we have any fill so we can
            # track the position; full completion handled by is_complete check
            if tag == "ENTRY" and sym == self._pending_sym and self._pos_sym is None:
                self._pos_sym    = sym
                self._entry_px   = float(order_event.fill_price)
                self._entry_time = self.time
                self.debug(
                    f"PARTIAL FILL ENTRY {sym.value}  px={self._entry_px:.4f}"
                    f"  qty_so_far={self._fill_tracker.get_filled(oid):.6f}"
                )
            if self._fill_tracker.is_complete(oid):
                self._pending_sym = None
                self._pending_oid = None

        elif order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            if tag == "ENTRY" and sym == self._pending_sym:
                self.debug(
                    f"ENTRY order {oid} for {sym.value} — status={order_event.status},"
                    f" clearing pending"
                )
                self._fill_tracker.clear(oid)
                self._pending_sym = None
                self._pending_oid = None
                self._pos_sym    = None
                self._entry_px   = 0.0
                self._entry_time = None

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                self.debug(
                    f"EXIT order {oid} for {sym.value} — status={order_event.status},"
                    f" will retry"
                )
                self._exiting = False

    # ── Reconciliation ────────────────────────────────────────────────────────

    def _reconcile(self):
        """
        Compare internal tracking state against actual portfolio holdings.
        Clears stale state when they diverge, so the next scoring pass starts
        from a clean slate.
        """
        if self._pos_sym is not None:
            qty = self.portfolio[self._pos_sym].quantity
            if qty <= 0:
                self.debug(
                    f"RECONCILE: tracking {self._pos_sym.value} but qty={qty:.6f}"
                    f" — clearing stale state"
                )
                self._pos_sym    = None
                self._entry_px   = 0.0
                self._entry_time = None
                self._tp_dyn     = 0.0
                self._sl_dyn     = 0.0
                self._exiting    = False

        # Safety net for stale pending orders
        if self._pending_oid is not None:
            order = self.transactions.get_order_by_id(self._pending_oid)
            if order is not None and order.status in (
                OrderStatus.FILLED, OrderStatus.INVALID, OrderStatus.CANCELED
            ):
                if order.status != OrderStatus.FILLED:
                    self._fill_tracker.clear(self._pending_oid)
                    self._pending_sym = None
                    self._pending_oid = None

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, price):
        """Evaluate ATR-based (or fixed) TP / SL / timeout; submit market sell."""
        ret     = (price - self._entry_px) / self._entry_px
        elapsed = (self.time - self._entry_time).total_seconds() / 3600.0

        tp_use = self._tp_dyn if self._tp_dyn > 0 else self._tp
        sl_use = self._sl_dyn if self._sl_dyn > 0 else self._sl

        reason = None
        if ret >= tp_use:
            reason = "EXIT_TP"
        elif ret <= -sl_use:
            reason = "EXIT_SL"
        elif elapsed >= self._toh:
            reason = "EXIT_TIMEOUT"

        if not reason:
            return

        qty = self.portfolio[self._pos_sym].quantity
        if qty > 0:
            self._exiting = True
            self.market_order(self._pos_sym, -qty, tag=reason)
            self.debug(
                f"EXIT order {self._pos_sym.value}  reason={reason}"
                f"  qty={qty:.6f}  ret={ret:.3%}"
            )
        else:
            # Portfolio already flat — clear stale state
            self.debug(
                f"EXIT {self._pos_sym.value}: portfolio qty=0,"
                f" clearing stale state"
            )
            self._pos_sym    = None
            self._entry_px   = 0.0
            self._entry_time = None
            self._tp_dyn     = 0.0
            self._sl_dyn     = 0.0
            self._exit_time  = self.time

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_enter(self):
        """
        Score all symbols; if a clear winner passes all gates, place a buy order.

        Gate pipeline (in order):
          1. Sufficient feature history
          2. Ensemble confidence: mean_proba >= SCORE_MIN
          3. Confidence gap to runner-up >= SCORE_GAP
          4. Ensemble dispersion: std_proba <= MAX_DISPERSION
          5. Model agreement: n_agree >= MIN_AGREE
          6. Regime filter: 4h BTC SMA + slope
          7. Risk manager: cooldown, daily SL cap, drawdown CB
          8. Pre-trade validation: price > 0, cash, lot-size, min-order
        """
        scores    = {}
        conf_data = {}

        for sym in self._symbols:
            st = self._state.get(sym)
            if st is None:
                continue

            closes  = list(st["closes"])
            volumes = list(st["volumes"])
            btc_closes = (
                list(self._state[self._btc_sym]["closes"])
                if self._btc_sym else []
            )

            feat = build_features(
                closes     = closes,
                volumes    = volumes,
                btc_closes = btc_closes,
                hour       = self.time.hour,
            )
            if feat is None:
                continue

            try:
                conf = self._ensemble.predict_with_confidence(feat.reshape(1, -1))
            except Exception as exc:
                self.debug(f"[vox] predict_with_confidence failed for {sym.value}: {exc}")
                continue

            mean_p = conf["mean_proba"]
            std_p  = conf["std_proba"]
            n_agr  = conf["n_agree"]

            # Dispersion and agreement gates (per symbol, not just top)
            if std_p > self._max_disp:
                continue
            if n_agr < self._min_agr:
                continue
            if mean_p < self._s_min:
                continue

            scores[sym]    = mean_p
            conf_data[sym] = conf

        if not scores:
            return

        ranked   = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sym, top_sc = ranked[0]
        second_sc       = ranked[1][1] if len(ranked) > 1 else 0.0

        if (top_sc - second_sc) < self._s_gap:
            self.debug(
                f"[vox] Gap too small: top={top_sc:.3f}"
                f"  second={second_sc:.3f}  gap={top_sc-second_sc:.3f}"
            )
            return

        # ── Regime gate ───────────────────────────────────────────────────────
        if not self._regime.is_risk_on(self._btc_sym, sym=top_sym):
            self.debug(f"[vox] Regime block for {top_sym.value}")
            return

        # ── Risk manager gate ──────────────────────────────────────────────────
        pv = self.portfolio.total_portfolio_value
        can, reason = self._risk.can_enter(
            sym=top_sym, current_time=self.time, portfolio_value=pv
        )
        if not can:
            self.debug(f"[vox] Risk block for {top_sym.value}: {reason}")
            return

        # ── Pre-trade validation ───────────────────────────────────────────────
        price = float(self.securities[top_sym].price)
        if price <= 0:
            self.debug(f"[vox] ENTRY skip {top_sym.value}: price={price} invalid")
            return

        # ATR-based dynamic TP/SL (fall back to fixed if insufficient data)
        st    = self._state[top_sym]
        atr   = compute_atr(
            highs  = list(st["highs"]),
            lows   = list(st["lows"]),
            closes = list(st["closes"]),
        )
        if atr > 0 and price > 0:
            tp_use = (atr * self._atr_tp) / price
            sl_use = (atr * self._atr_sl) / price
        else:
            tp_use = self._tp
            sl_use = self._sl

        # Kelly / flat sizing
        qty, alloc = compute_qty(
            mean_proba      = top_sc,
            tp              = tp_use,
            sl              = sl_use,
            price           = price,
            portfolio_value = pv,
            kelly_frac      = self._kf,
            max_alloc       = self._max_alloc,
            cash_buffer     = self._cb,
            use_kelly       = self._use_kelly,
            allocation      = self._alloc,
        )

        # Cash check
        cash = self.portfolio.cash
        if qty * price > cash * self._cb:
            self.debug(
                f"[vox] ENTRY skip {top_sym.value}: insufficient cash"
                f" (need {qty*price:.2f}, have {cash:.2f})"
            )
            return

        # Lot-size rounding and minimum order validation
        sec      = self.securities[top_sym]
        lot_size = OrderHelper.get_lot_size(sec)
        min_ord  = OrderHelper.get_min_order_size(sec)
        qty      = OrderHelper.round_qty(qty, lot_size)

        if not OrderHelper.validate_qty(qty, min_ord):
            self.debug(
                f"[vox] ENTRY skip {top_sym.value}: qty={qty:.8f}"
                f" < min_order={min_ord}"
            )
            return
        if qty <= 0:
            self.debug(
                f"[vox] ENTRY skip {top_sym.value}: computed qty={qty:.8f}"
            )
            return

        # Place entry order
        order = self.market_order(top_sym, qty, tag="ENTRY")
        self._pending_sym = top_sym
        self._pending_oid = order.order_id
        self._tp_dyn      = tp_use
        self._sl_dyn      = sl_use
        self._fill_tracker.start_order(order.order_id, qty)

        self.debug(
            f"ENTRY order {top_sym.value}  score={top_sc:.3f}"
            f"  price={price:.4f}  qty={qty:.6f}"
            f"  alloc={alloc:.3f}  tp={tp_use:.4f}  sl={sl_use:.4f}"
        )

        # Log trade attempt to persistence
        self._persistence.log_trade({
            "event":      "entry_attempt",
            "time":       str(self.time),
            "symbol":     top_sym.value,
            "price":      price,
            "qty":        qty,
            "alloc":      alloc,
            "mean_proba": top_sc,
            "n_agree":    conf_data[top_sym]["n_agree"],
            "std_proba":  conf_data[top_sym]["std_proba"],
            "tp":         tp_use,
            "sl":         sl_use,
        })

    # ── Retrain ───────────────────────────────────────────────────────────────

    def _retrain(self):
        """Weekly retrain on all accumulated history.  Saves model to ObjectStore."""
        self.log("[vox] Starting scheduled retrain …")
        timeout_bars = int(self._toh * 60 / DECISION_INTERVAL_MIN)

        X, y = build_training_data(
            algorithm             = self,
            symbols               = self._symbols,
            state_dict            = self._state,
            tp                    = self._tp,
            sl                    = self._sl,
            timeout_bars          = timeout_bars,
            decision_interval_min = DECISION_INTERVAL_MIN,
        )
        if X is None or len(X) < 50:
            self.log("[vox] Retrain skipped: insufficient training data.")
            return

        try:
            self._ensemble = walk_forward_train(self._ensemble, X, y)
            self._model_ready = self._ensemble.is_fitted
            self._persistence.save_model(self._ensemble)
            self.log(
                f"[vox] Retrain complete. Samples={len(X)}, fitted={self._model_ready}"
            )
        except Exception as exc:
            self.log(f"[vox] Retrain failed: {exc}")

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily(self):
        """Reset daily counters at midnight UTC."""
        self._risk.reset_daily()
