# region imports
from AlgorithmImports import *
import numpy as np
import random
from collections import deque
from datetime import timedelta

from infra   import add_universe, KRAKEN_PAIRS, OrderHelper, PartialFillTracker, PersistenceManager
from models  import (
    build_features, compute_atr, VoxEnsemble, build_training_data, walk_forward_train,
    LABEL_TP, LABEL_SL, LABEL_HORIZON_BARS, LABEL_COST_BPS,
)
from risk    import RegimeFilter, RiskManager, compute_qty
# endregion

from config   import *                       # noqa: F401,F403  strategy constants
from config   import setup_risk_profile
from momentum import check_momentum_override_conditions, compute_momentum_score

np.random.seed(42)
random.seed(42)


class VoxAlgorithm(QCAlgorithm):
    """Execution-first ML ensemble strategy for Kraken spot crypto.

    See README.md for architecture details and risk-profile documentation.
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
        self._s_min_floor = SCORE_MIN_FLOOR
        self._s_gap    = float(self.get_parameter("score_gap")        or SCORE_GAP)  # probability-gap threshold (mean_proba units)
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
        self._exit_qty_buffer = int(
            self.get_parameter("exit_qty_buffer_lots") or EXIT_QTY_BUFFER_LOTS
        )
        self._cost_bps = float(self.get_parameter("cost_bps")         or COST_BPS)
        self._min_ev   = float(self.get_parameter("min_ev")           or MIN_EV)
        self._ev_gap   = float(self.get_parameter("ev_gap")           or EV_GAP)   # EV-gap threshold (return-fraction units, NOT probability units)
        self._pred_return_min = float(self.get_parameter("pred_return_min") or PRED_RETURN_MIN)
        self._min_hold_minutes = int(self.get_parameter("min_hold_minutes") or MIN_HOLD_MINUTES)
        self._emergency_sl = float(self.get_parameter("emergency_sl") or EMERGENCY_SL)
        self._penalty_losses = int(self.get_parameter("penalty_cooldown_losses") or PENALTY_COOLDOWN_LOSSES)
        self._penalty_hours  = float(self.get_parameter("penalty_cooldown_hours") or PENALTY_COOLDOWN_HOURS)
        _uc_raw        = self.get_parameter("use_calibration")
        self._use_calibration = (str(_uc_raw).lower() in ("true", "1", "yes")) if _uc_raw else True
        # Minimum allocation floor — prevents Kelly from shrinking positions below
        # this fraction.  Default 0.0 preserves existing balanced-mode behaviour.
        self._min_alloc = float(self.get_parameter("min_alloc") or 0.0)
        # Runner / trailing-profit mode — ruthless only by default.
        # setup_risk_profile() will override these for the ruthless profile.
        self._runner_mode    = False
        self._trail_after_tp = 0.04    # activate trailing once return ≥ this
        self._trail_pct      = 0.025   # trail stop: price drops this % from high
        self._label_tp      = float(self.get_parameter("label_tp")           or LABEL_TP)
        self._label_sl      = float(self.get_parameter("label_sl")           or LABEL_SL)
        self._label_horizon = int(self.get_parameter("label_horizon_bars")   or LABEL_HORIZON_BARS)
        self._label_cost_bps = float(self.get_parameter("label_cost_bps")    or LABEL_COST_BPS)

        # ── Risk profile + momentum override setup ────────────────────────────
        # Resolves effective profile (ruthless/aggressive/conservative/balanced)
        # from QC parameters and applies gate/sizing overrides.  See config.py.
        setup_risk_profile(self)

        # ── Universe ──────────────────────────────────────────────────────────
        self._symbols  = add_universe(self)
        self._btc_sym  = next(
            (s for s in self._symbols if s.value.upper().startswith("BTC")), None
        )

        # ── Slippage model ────────────────────────────────────────────────────
        for sym in self._symbols:
            self.securities[sym].set_slippage_model(ConstantSlippageModel(0.001))

        # ── Per-symbol state deques ───────────────────────────────────────────
        # Bars per day at the chosen resolution × WARMUP_DAYS × 1.2 safety margin.
        _bars_per_day = int(24 * 60 / RESOLUTION_MINUTES)
        _state_max    = int(WARMUP_DAYS * _bars_per_day * 1.2)   # ~31k for 90d @ 5m
        _dq = lambda n=_state_max: deque(maxlen=n)
        self._state = {}
        for sym in self._symbols:
            self._state[sym] = {
                "closes":  _dq(),
                "highs":   _dq(),
                "lows":    _dq(),
                "volumes": _dq(),
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
        self._ensemble    = VoxEnsemble(logger=self.log, use_calibration=self._use_calibration)
        self._persistence = PersistenceManager(self)
        self._fill_tracker= PartialFillTracker()
        self._model_ready = False

        # Try to load a previously trained model
        saved = self._persistence.load_model()
        if saved is not None:
            self._ensemble.load_state(saved)
            if hasattr(self._ensemble, "set_logger"):
                self._ensemble.set_logger(self.log)
            self._model_ready = self._ensemble.is_fitted
            self.log("[vox] Loaded pre-trained model from ObjectStore.")

        # ── Position state — updated ONLY via on_order_event ──────────────────
        self._pos_sym          = None    # confirmed open position symbol
        self._entry_px         = 0.0     # confirmed entry fill price
        self._entry_time       = None    # confirmed entry fill time
        self._tp_dyn           = 0.0     # ATR-derived TP fraction for current trade
        self._sl_dyn           = 0.0     # ATR-derived SL fraction for current trade
        self._pending_sym      = None    # symbol of in-flight entry order
        self._pending_oid      = None    # order ID of in-flight entry order
        self._exiting          = False   # True while an exit order is in flight
        self._exit_time        = None    # time of most recent completed exit
        self._exit_retry_count = 0       # consecutive INVALID exit retry counter
        # Runner / trailing-profit state — reset on every entry and exit
        self._trail_active     = False   # True once trailing stop has been activated
        self._trail_high_px    = 0.0     # highest price seen since trail activation

        # ── Entry prediction store (for realized-EV logging at exit) ──────────
        # sym -> {class_proba, pred_return, ev, final_score, tp, sl, time}
        self._entry_predictions = {}

        # ── Per-symbol penalty cooldown ────────────────────────────────────────
        # Tracks recent trade outcomes per symbol to apply extended cooldowns
        # after repeated losses (independent of the per-coin SL cooldown in RiskManager).
        from collections import deque as _deque
        self._sym_outcomes     = {sym: _deque(maxlen=10) for sym in self._symbols}
        self._sym_penalty_until = {}  # sym -> datetime when penalty ends

        # ── Diagnostic throttling ──────────────────────────────────────────────
        # Routine skip messages (EV gap too small, regime block, risk block) are
        # suppressed unless at least SKIP_DIAG_INTERVAL_SECS have passed since the
        # last one, to prevent QuantConnect log rate-limiting.
        self._last_skip_diag_time = None
        # No-candidate summary throttle (logged at most every DIAG_INTERVAL_HOURS).
        self._last_nocandidate_diag_time = None
        # Last completed retrain time (used to suppress duplicate retrains shortly
        # after the initial train or a recent scheduled retrain).
        self._last_retrain_time = None

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

    def on_warmup_finished(self):
        """Schedule initial hydration + training off the time loop via self.train().
        Synchronous heavy work in on_warmup_finished can exceed QC's 10-minute
        Isolator watchdog and crash the algorithm."""
        self.log("[vox] Warmup finished — scheduling initial train off-loop.")
        self.train(self._initial_train)

    def _initial_train(self):
        """Off-loop initial training: hydrate history, then train."""
        import time
        t0 = time.time()
        try:
            self._hydrate_state_from_history(WARMUP_DAYS)
            self.log(f"[vox] hydrate took {time.time()-t0:.1f}s")
        except Exception as exc:
            self.log(f"[vox] history hydrate failed: {exc}")
        t1 = time.time()
        try:
            self._retrain()
            self.log(f"[vox] initial _retrain took {time.time()-t1:.1f}s")
        except Exception as exc:
            self.log(f"[vox] Initial train failed: {exc}")

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
            elif self._entry_time is not None:
                # No bar for this symbol this tick (illiquid / low-volume pair).
                # Normal TP/SL checks are skipped, but still check the timeout
                # so the hold is bounded even when bars are absent.
                elapsed = (self.time - self._entry_time).total_seconds() / 3600.0
                if elapsed >= self._toh:
                    fallback_px = float(self.securities[self._pos_sym].price)
                    if fallback_px > 0:
                        self._check_exit(fallback_px)

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
            self._throttled_skip_debug(f"[vox] Entry blocked: {reason}")
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

                # ── Realized EV logging ────────────────────────────────────────
                # Log predicted vs realized for each completed trade so the user
                # can evaluate calibration and model quality out-of-sample.
                # Retrieve the entry predictions stored at entry time.
                entry_pred = self._entry_predictions.pop(sym, None)
                self._persistence.log_trade({
                    "event":              "exit",
                    "time":               str(self.time),
                    "symbol":             sym.value,
                    "exit_reason":        tag,
                    "realized_return":    round(ret, 6),
                    "predicted_class_proba": round(entry_pred["class_proba"], 4) if entry_pred else None,
                    "predicted_return":   round(entry_pred["pred_return"], 6)    if entry_pred else None,
                    "predicted_ev":       round(entry_pred["ev"], 6)             if entry_pred else None,
                    "final_score":        round(entry_pred["final_score"], 6)    if entry_pred else None,
                    "tp":                 round(entry_pred["tp"], 4)             if entry_pred else None,
                    "sl":                 round(entry_pred["sl"], 4)             if entry_pred else None,
                    "entry_path":         entry_pred.get("entry_path", "ml")     if entry_pred else None,
                })

                # ── Per-symbol outcome tracking (penalty cooldown) ─────────────
                if sym not in self._sym_outcomes:
                    from collections import deque as _dq
                    self._sym_outcomes[sym] = _dq(maxlen=10)
                self._sym_outcomes[sym].append(ret)
                self._update_penalty_cooldown(sym, is_sl)

                self._pos_sym       = None
                self._entry_px      = 0.0
                self._entry_time    = None
                self._tp_dyn        = 0.0
                self._sl_dyn        = 0.0
                self._exiting       = False
                self._exit_time     = self.time
                self._trail_active  = False
                self._trail_high_px = 0.0

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
                self._pending_sym   = None
                self._pending_oid   = None
                self._pos_sym       = None
                self._entry_px      = 0.0
                self._entry_time    = None
                self._trail_active  = False
                self._trail_high_px = 0.0

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                self._exit_retry_count += 1
                # Before allowing a retry, check if we can actually submit a
                # valid sell.  In Kraken cash mode the portfolio holding can
                # exceed the actual CashBook balance after fees; if the safe
                # qty is zero/dust, clear state immediately instead of looping.
                _sec      = self.securities[sym]
                _lot_size = OrderHelper.get_lot_size(_sec)
                _min_ord  = OrderHelper.get_min_order_size(_sec)
                _safe_qty = OrderHelper.safe_crypto_sell_qty(
                    self, sym, _lot_size, _min_ord,
                    exit_qty_buffer_lots=self._exit_qty_buffer,
                )
                if _safe_qty <= 0 or self._exit_retry_count >= MAX_EXIT_RETRY_COUNT:
                    # Dust position or retry cap hit — treat as non-actionable
                    self.debug(
                        f"EXIT order {oid} for {sym.value} —"
                        f" status={order_event.status},"
                        f" safe_qty={_safe_qty:.8f}"
                        f" retry={self._exit_retry_count}"
                        f" — clearing as dust"
                    )
                    is_sl = (tag == "EXIT_SL")
                    self._risk.record_exit(sym, is_sl=is_sl, exit_time=self.time)
                    self._pos_sym          = None
                    self._entry_px         = 0.0
                    self._entry_time       = None
                    self._tp_dyn           = 0.0
                    self._sl_dyn           = 0.0
                    self._exiting          = False
                    self._exit_time        = self.time
                    self._exit_retry_count = 0
                    self._trail_active     = False
                    self._trail_high_px    = 0.0
                else:
                    self.debug(
                        f"EXIT order {oid} for {sym.value} —"
                        f" status={order_event.status},"
                        f" safe_qty={_safe_qty:.8f}"
                        f" retry={self._exit_retry_count} — will retry"
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
                sym = self._pos_sym   # capture before clearing
                self.debug(
                    f"RECONCILE: tracking {sym.value} but qty={qty:.6f}"
                    f" — clearing stale state"
                )
                self._pos_sym       = None
                self._entry_px      = 0.0
                self._entry_time    = None
                self._tp_dyn        = 0.0
                self._sl_dyn        = 0.0
                self._exiting       = False
                self._exit_time     = self.time
                self._trail_active  = False
                self._trail_high_px = 0.0
                # Inform the risk manager so cooldown accounting is not bypassed.
                self._risk.record_exit(sym, is_sl=False, exit_time=self.time)

        # Safety net for stale pending orders.
        # If the entry order already settled but on_order_event missed the fill
        # (synchronous fill race: market_order filled before _pending_sym was set),
        # reconstruct position state here so the state machine can continue.
        if self._pending_oid is not None:
            order = self.transactions.get_order_by_id(self._pending_oid)
            if order is not None and order.status in (
                OrderStatus.FILLED, OrderStatus.INVALID, OrderStatus.CANCELED
            ):
                if order.status == OrderStatus.FILLED:
                    # Recover from a missed synchronous fill.
                    if self._pos_sym is None and self._pending_sym is not None:
                        held_qty = self.portfolio[self._pending_sym].quantity
                        if held_qty > 0:
                            self._pos_sym    = self._pending_sym
                            self._entry_time = self.time
                            # Fill price unavailable here; approximate with current price.
                            self._entry_px   = float(
                                self.securities[self._pending_sym].price
                            )
                            self.debug(
                                f"RECONCILE: recovered missed ENTRY fill for"
                                f" {self._pending_sym.value}"
                                f"  approx_px={self._entry_px:.4f}"
                            )
                    self._fill_tracker.clear(self._pending_oid)
                    self._pending_sym = None
                    self._pending_oid = None
                else:   # INVALID or CANCELED
                    self._fill_tracker.clear(self._pending_oid)
                    self._pending_sym = None
                    self._pending_oid = None

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, price):
        """Evaluate ATR-based (or fixed) TP / SL / timeout; submit market sell.

        Minimum-hold protection:
          If elapsed time < min_hold_minutes, ordinary TP/SL/timeout exits are
          suppressed.  Only an emergency stop is allowed (loss > emergency_sl).
          This prevents the algo from entering and immediately being shaken out
          by microstructure noise within a few bars.

        Runner mode (ruthless only):
          When runner_mode=True the strategy does not immediately exit at the TP
          trigger.  Instead it activates a trailing stop once the return crosses
          trail_after_tp (or tp_use, whichever is lower).  The trailing stop
          follows the highest price seen since activation and exits with tag
          EXIT_TRAIL when price pulls back by trail_pct.  Hard SL, emergency SL,
          and timeout remain active throughout.
        """
        # Capture immutable local references BEFORE placing any order.
        # market_order() can fill synchronously in QuantConnect/LEAN, meaning
        # on_order_event may clear self._pos_sym (and related fields) before
        # this function returns.  Using locals throughout avoids NoneType errors.
        sym        = self._pos_sym
        entry_px   = self._entry_px
        entry_time = self._entry_time

        if sym is None or entry_time is None or entry_px <= 0:
            return

        ret             = (price - entry_px) / entry_px
        elapsed_sec     = (self.time - entry_time).total_seconds()
        elapsed_hours   = elapsed_sec / 3600.0
        elapsed_minutes = elapsed_sec / 60.0

        tp_use = self._tp_dyn if self._tp_dyn > 0 else self._tp
        sl_use = self._sl_dyn if self._sl_dyn > 0 else self._sl

        # ── Minimum hold period ────────────────────────────────────────────────
        # During the minimum hold window, suppress ordinary exits.  Only the
        # emergency SL is allowed to exit early to protect against large gaps.
        if elapsed_minutes < self._min_hold_minutes:
            if ret <= -self._emergency_sl:
                reason = "EXIT_SL"   # emergency stop — override min-hold
            else:
                return               # hold — suppress normal TP/SL/timeout
        elif self._runner_mode:
            # ── Runner / trailing-profit mode ─────────────────────────────────
            # Hard SL and timeout always take priority over trailing logic.
            if ret <= -sl_use:
                reason = "EXIT_SL"
            elif elapsed_hours >= self._toh:
                reason = "EXIT_TIMEOUT"
            elif self._trail_active:
                # Update the high-water mark in price space
                if price > self._trail_high_px:
                    self._trail_high_px = price
                # Trail stop: exit if price falls trail_pct from the high
                if price <= self._trail_high_px * (1.0 - self._trail_pct):
                    reason = "EXIT_TRAIL"
                else:
                    return   # still running — keep holding
            else:
                # Check if we should activate the trailing stop.
                # Activate at whichever threshold is reached first: the
                # configured trail_after_tp (e.g. +4%) or the ATR-derived
                # tp_use (in case ATR produces a target below trail_after_tp).
                trail_trigger = min(tp_use, self._trail_after_tp)
                if ret >= trail_trigger:
                    self._trail_active  = True
                    self._trail_high_px = price
                    return   # don't exit yet — trail is now live
                else:
                    return   # below trigger — keep holding
        else:
            # ── Normal (balanced / conservative / aggressive) exit logic ───────
            reason = None
            if ret >= tp_use:
                reason = "EXIT_TP"
            elif ret <= -sl_use:
                reason = "EXIT_SL"
            elif elapsed_hours >= self._toh:
                reason = "EXIT_TIMEOUT"

            if not reason:
                return

        # ── Safe sell quantity — guards against CashBook / portfolio mismatch ──
        # In Kraken cash mode, portfolio[sym].quantity can exceed the actual
        # base-currency CashBook balance after fees/rounding.  Submitting a
        # sell for the raw quantity causes an INVALID order and an unbounded
        # retry loop.  Use the min(portfolio, CashBook) with a lot-sized buffer.
        sec      = self.securities[sym]
        lot_size = OrderHelper.get_lot_size(sec)
        min_ord  = OrderHelper.get_min_order_size(sec)
        qty = OrderHelper.safe_crypto_sell_qty(
            self, sym, lot_size, min_ord,
            exit_qty_buffer_lots=self._exit_qty_buffer,
        )

        if qty > 0:
            self._exiting          = True
            self._exit_retry_count = 0
            # Log BEFORE market_order() so we never dereference self._pos_sym
            # after a potential synchronous fill clears it.
            self.debug(
                f"EXIT order {sym.value}  reason={reason}"
                f"  qty={qty:.6f}  ret={ret:.3%}"
                f"  elapsed_min={elapsed_minutes:.1f}"
                + (f"  trail_high={self._trail_high_px:.4f}" if self._trail_active else "")
            )
            self.market_order(sym, -qty, tag=reason)
        else:
            # Portfolio is flat or remaining position is non-actionable dust.
            portfolio_qty = float(self.portfolio[sym].quantity)
            self.debug(
                f"EXIT {sym.value}: safe sell qty=0 (dust/flat),"
                f" portfolio_qty={portfolio_qty:.8f}  reason={reason}"
                f" — clearing state"
            )
            is_sl = (reason == "EXIT_SL")
            self._risk.record_exit(sym, is_sl=is_sl, exit_time=self.time)
            self._pos_sym          = None
            self._entry_px         = 0.0
            self._entry_time       = None
            self._tp_dyn           = 0.0
            self._sl_dyn           = 0.0
            self._exiting          = False
            self._exit_time        = self.time
            self._exit_retry_count = 0
            self._trail_active     = False
            self._trail_high_px    = 0.0

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_enter(self):
        """Score all symbols; if a clear winner passes all gates, place a buy order.

        Gate pipeline: feature history → penalty cooldown → ensemble confidence
        → dispersion → model agreement → EV gate → predicted-return gate
        → final-score gap → regime filter → risk manager → pre-trade validation.
        See README.md for full gate documentation.
        """
        scores         = {}   # sym -> final_score
        conf_data      = {}   # sym -> confidence dict from ensemble
        tp_sl_data     = {}   # sym -> (tp_use, sl_use, atr, price) for re-use
        ev_data        = {}   # sym -> ev_after_costs
        entry_path_data = {}  # sym -> "ml" | "momentum_override"

        cost_fraction = self._cost_bps * 1e-4
        reg_fitted    = getattr(self._ensemble, "_reg_fitted", False)

        btc_closes = (
            list(self._state[self._btc_sym]["closes"])
            if self._btc_sym else []
        )

        candidates = []   # list of (sym, feat)
        for sym in self._symbols:
            st = self._state.get(sym)
            if st is None:
                continue

            # ── Penalty cooldown gate ─────────────────────────────────────────
            # Skip symbols that are in their post-repeated-loss cooldown window.
            if self._is_in_penalty_cooldown(sym):
                continue

            closes  = list(st["closes"])
            volumes = list(st["volumes"])

            feat = build_features(
                closes     = closes,
                volumes    = volumes,
                btc_closes = btc_closes,
                hour       = self.time.hour,
            )
            if feat is None:
                continue

            candidates.append((sym, feat))

        if not candidates:
            # Only log occasionally to avoid spamming the 100KB log cap
            if self.time.minute == 0 and self.time.hour % 6 == 0:
                self.log(
                    f"[diag] no_features symbols={len(self._symbols)} "
                    f"(build_features returned None for all)"
                )
            return

        X_all = np.vstack([c[1] for c in candidates])
        try:
            confs = self._ensemble.predict_with_confidence_batch(X_all)
        except Exception as exc:
            self.debug(f"[vox] batch predict failed: {exc}")
            return

        # Base-rate-aware effective thresholds (computed once per tick)
        pr            = self._ensemble.base_rate
        score_min_eff = float(np.clip(
            max(self._s_min_floor, SCORE_MIN_MULT * pr), self._s_min_floor, self._s_min
        ))
        agree_thr     = self._ensemble._agree_threshold()

        # Per-gate pass counters and best-candidate values for diagnostics
        n_pass_disp     = 0
        n_pass_agree    = 0
        n_pass_score    = 0
        n_pass_ev       = 0
        n_pass_pred_ret = 0
        n_momentum_override = 0
        _best_ev_diag   = float("-inf")   # best ev_after_costs seen (for diagnostics)

        for (sym, feat), conf in zip(candidates, confs):
            class_proba   = conf["class_proba"]    # weighted VotingClassifier probability
            std_proba     = conf["std_proba"]
            n_agree       = conf["n_agree"]
            pred_return   = conf["pred_return"]     # 0.0 if regressors not trained

            passed_disp   = std_proba   <= self._max_disp
            passed_agree  = n_agree     >= self._min_agr
            passed_score  = class_proba >= score_min_eff
            if passed_disp:   n_pass_disp  += 1
            if passed_agree:  n_pass_agree += 1
            if passed_score:  n_pass_score += 1

            ml_gates_pass = passed_disp and passed_agree and passed_score
            entry_path    = "ml"

            if not ml_gates_pass:
                # ── Momentum breakout override (aggressive/ruthless only) ─────
                if not self._momentum_override:
                    continue
                if not check_momentum_override_conditions(
                    feat,
                    self._momentum_ret4_min,
                    self._momentum_ret16_min,
                    self._momentum_volume_min,
                    self._momentum_btc_rel_min,
                ):
                    continue
                entry_path = "momentum_override"

            # ── Per-candidate ATR-based TP/SL for EV computation ─────────────
            price = float(self.securities[sym].price)
            if price <= 0:
                continue
            st  = self._state[sym]
            atr = compute_atr(
                highs  = list(st["highs"]),
                lows   = list(st["lows"]),
                closes = list(st["closes"]),
            )
            if atr > 0:
                tp_use = (atr * self._atr_tp) / price
                sl_use = (atr * self._atr_sl) / price
            else:
                tp_use = self._tp
                sl_use = self._sl

            # ── Vox v2 decision score ─────────────────────────────────────────
            # ev = classifier probability × TP − (1−prob) × SL − costs
            ev_after_costs = (
                class_proba * tp_use
                - (1.0 - class_proba) * sl_use
                - cost_fraction
            )
            # Track best EV seen (for no-candidate diagnostics)
            if ev_after_costs > _best_ev_diag:
                _best_ev_diag = ev_after_costs

            if entry_path == "momentum_override":
                # Momentum override: EV must not be catastrophically negative
                if ev_after_costs < self._momentum_override_min_ev:
                    continue
                self.log(
                    f"[vox] MOMENTUM OVERRIDE candidate={sym.value}"
                    f" ret4={feat[1]:.4f} ret16={feat[3]:.4f}"
                    f" vol_r={feat[6]:.2f} btc_rel={feat[7]:.4f}"
                    f" ev={ev_after_costs:.5f} pred_ret={pred_return:.5f}"
                    f" proba={class_proba:.3f}"
                )
                n_momentum_override += 1
                n_pass_ev += 1
                n_pass_pred_ret += 1
            else:
                # Normal ML path
                if ev_after_costs <= self._min_ev:
                    continue   # negative edge after costs — skip
                n_pass_ev += 1

                # Predicted-return gate (only applied when regressors are trained).
                if reg_fitted and pred_return < self._pred_return_min:
                    continue
                n_pass_pred_ret += 1

            # ── Final score ───────────────────────────────────────────────────
            if self._use_momentum_score:
                # Aggressive/ruthless: momentum-boosted scoring formula.
                momentum_score = compute_momentum_score(feat)
                if reg_fitted and pred_return != 0.0:
                    final_score = (
                        0.50 * ev_after_costs
                        + 0.25 * pred_return
                        + 0.25 * momentum_score
                    )
                else:
                    confidence_adj = max(0.0, 1.0 - std_proba)
                    final_score    = (
                        0.75 * ev_after_costs * confidence_adj
                        + 0.25 * momentum_score
                    )
            else:
                # Balanced/conservative: original Vox v2 scoring formula
                if reg_fitted and pred_return != 0.0:
                    final_score = 0.6 * ev_after_costs + 0.4 * pred_return
                else:
                    # Fallback when regressors not yet trained: EV × confidence adj
                    confidence_adj = max(0.0, 1.0 - std_proba)
                    final_score    = ev_after_costs * confidence_adj

            scores[sym]          = final_score
            conf_data[sym]       = conf
            tp_sl_data[sym]      = (tp_use, sl_use, atr, price)
            ev_data[sym]         = ev_after_costs
            entry_path_data[sym] = entry_path

        if scores:
            ranked      = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_sym_d   = ranked[0][0]
            top_sc_d    = ranked[0][1]
            second_sc_d = ranked[1][1] if len(ranked) > 1 else 0.0
            top_cd      = conf_data[top_sym_d]
            top_tp, top_sl, top_atr, top_px = tp_sl_data[top_sym_d]
            self.log(
                f"[diag] candidates={len(scores)}"
                f" top={top_sym_d.value}"
                f" path={entry_path_data.get(top_sym_d, 'ml')}"
                f" final_score={top_sc_d:.5f}"
                f" ev_score={ev_data[top_sym_d]:.5f}"
                f" pred_ret={top_cd['pred_return']:.5f}"
                f" gap={top_sc_d-second_sc_d:.5f}"
                f" class_proba={top_cd['class_proba']:.3f}"
                f" std_proba={top_cd['std_proba']:.3f}"
                f" n_agree={top_cd['n_agree']}"
                f" tp={top_tp:.4f} sl={top_sl:.4f}"
                + (f" mo_overrides={n_momentum_override}" if n_momentum_override else "")
            )
        else:
            # Throttle no-candidate summary to at most once per DIAG_INTERVAL_HOURS
            # to protect QuantConnect's 100KB log cap during multi-year backtests.
            _emit_diag = (
                self._last_nocandidate_diag_time is None
                or (self.time - self._last_nocandidate_diag_time).total_seconds()
                   >= DIAG_INTERVAL_HOURS * 3600
            )
            if _emit_diag:
                self._last_nocandidate_diag_time = self.time
                best_proba  = max(c["class_proba"] for c in confs)
                best_nagree = max(c["n_agree"]     for c in confs)
                best_std    = min(c["std_proba"]   for c in confs)
                best_pred   = max(c["pred_return"] for c in confs)
                best_ev_str = f"{_best_ev_diag:.5f}" if _best_ev_diag > float("-inf") else "n/a"
                self.log(
                    f"[diag] eval={len(candidates)} "
                    f"pass_disp={n_pass_disp} pass_agree={n_pass_agree} "
                    f"pass_score={n_pass_score} pass_ev={n_pass_ev} "
                    f"pass_pred_ret={n_pass_pred_ret} "
                    f"best_proba={best_proba:.3f} best_agree={best_nagree} "
                    f"best_disp={best_std:.3f} best_pred_ret={best_pred:.5f} "
                    f"best_ev={best_ev_str} "
                    f"(thresh: score>={score_min_eff:.3f} "
                    f"agree>={self._min_agr} disp<={self._max_disp} "
                    f"ev>{self._min_ev:.5f} pred_ret>={self._pred_return_min:.5f} "
                    f"cost={cost_fraction:.4f})"
                )

        if not scores:
            return

        ranked   = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sym, top_sc = ranked[0]
        second_sc       = ranked[1][1] if len(ranked) > 1 else 0.0

        # Gap check on final score — uses self._ev_gap (return-fraction units),
        # NOT self._s_gap (probability units).  SCORE_GAP=0.02 was designed for
        # probability differences (0–1 scale) and is far too large for EV scores
        # which are typically in the 0.001–0.02 range.
        # Skip the gap check when there is only one candidate.
        ev_gap_actual = top_sc - second_sc
        if len(ranked) > 1 and ev_gap_actual < self._ev_gap:
            self._throttled_skip_debug(
                f"[vox] Score gap too small: top={top_sc:.5f}"
                f"  second={second_sc:.5f}"
                f"  gap={ev_gap_actual:.5f}"
                f"  required={self._ev_gap:.5f}"
            )
            return

        # ── Regime gate ───────────────────────────────────────────────────────
        if not self._regime.is_risk_on(self._btc_sym, sym=top_sym):
            self._throttled_skip_debug(f"[vox] Regime block for {top_sym.value}")
            return

        # ── Risk manager gate ──────────────────────────────────────────────────
        pv = self.portfolio.total_portfolio_value
        can, reason = self._risk.can_enter(
            sym=top_sym, current_time=self.time, portfolio_value=pv
        )
        if not can:
            self._throttled_skip_debug(f"[vox] Risk block for {top_sym.value}: {reason}")
            return

        # ── Pre-trade validation ───────────────────────────────────────────────
        tp_use, sl_use, atr, price = tp_sl_data[top_sym]
        class_proba_top  = conf_data[top_sym]["class_proba"]
        pred_return_top  = conf_data[top_sym]["pred_return"]
        ev_top           = ev_data[top_sym]

        # Kelly / flat sizing (uses class_proba and ATR TP/SL for Kelly edge)
        qty, alloc = compute_qty(
            mean_proba      = class_proba_top,
            tp              = tp_use,
            sl              = sl_use,
            price           = price,
            portfolio_value = pv,
            kelly_frac      = self._kf,
            max_alloc       = self._max_alloc,
            cash_buffer     = self._cb,
            use_kelly       = self._use_kelly,
            allocation      = self._alloc,
            min_alloc       = self._min_alloc,
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

        # Place entry order.
        # IMPORTANT: set _pending_sym (and _tp_dyn/_sl_dyn) BEFORE calling
        # market_order().  In QuantConnect/LEAN, market orders with the default
        # ImmediateFillModel fill synchronously — on_order_event fires inside
        # market_order() before it returns.  If _pending_sym is still None when
        # that happens the ENTRY fill is silently ignored and the state machine
        # gets permanently stuck (_pending_sym stays set, _pos_sym never set).
        self._pending_sym   = top_sym
        self._tp_dyn        = tp_use
        self._sl_dyn        = sl_use
        self._trail_active  = False   # reset trail state on every new entry
        self._trail_high_px = 0.0
        order = self.market_order(top_sym, qty, tag="ENTRY")
        # order_id is only available on the returned ticket; assign after the call.
        self._pending_oid = order.order_id
        self._fill_tracker.start_order(order.order_id, qty)

        # Store entry predictions for realized-EV logging at exit.
        self._entry_predictions[top_sym] = {
            "class_proba": class_proba_top,
            "pred_return": pred_return_top,
            "ev":          ev_top,
            "final_score": top_sc,
            "tp":          tp_use,
            "sl":          sl_use,
            "time":        self.time,
            "entry_path":  entry_path_data.get(top_sym, "ml"),
        }

        _top_entry_path = entry_path_data.get(top_sym, "ml")
        self.debug(
            f"ENTRY order {top_sym.value}"
            f"  path={_top_entry_path}"
            f"  final_score={top_sc:.5f}"
            f"  ev={ev_top:.5f}"
            f"  pred_ret={pred_return_top:.5f}"
            f"  class_proba={class_proba_top:.3f}"
            f"  std_proba={conf_data[top_sym]['std_proba']:.3f}"
            f"  n_agree={conf_data[top_sym]['n_agree']}"
            f"  price={price:.4f}  qty={qty:.6f}"
            f"  alloc={alloc:.3f}  tp={tp_use:.4f}  sl={sl_use:.4f}"
        )

        # Log trade attempt to persistence
        self._persistence.log_trade({
            "event":        "entry_attempt",
            "time":         str(self.time),
            "symbol":       top_sym.value,
            "price":        price,
            "qty":          qty,
            "alloc":        alloc,
            "class_proba":  class_proba_top,
            "pred_return":  pred_return_top,
            "n_agree":      conf_data[top_sym]["n_agree"],
            "std_proba":    conf_data[top_sym]["std_proba"],
            "tp":           tp_use,
            "sl":           sl_use,
            "ev_score":     ev_top,
            "final_score":  top_sc,
            "cost_bps":     self._cost_bps,
            "entry_path":   _top_entry_path,
        })

    # ── Retrain ───────────────────────────────────────────────────────────────

    def _retrain(self):
        """Weekly retrain on all accumulated history.  Saves model to ObjectStore."""
        # Skip during warmup — _initial_train() handles the first fit after warmup.
        if self.is_warming_up:
            return

        # Avoid duplicate retrain shortly after the initial train or a recent
        # scheduled retrain.  Skip if the last retrain completed within
        # MIN_RETRAIN_INTERVAL_HOURS hours.
        if self._last_retrain_time is not None:
            elapsed_h = (self.time - self._last_retrain_time).total_seconds() / 3600.0
            if elapsed_h < MIN_RETRAIN_INTERVAL_HOURS:
                self.debug(
                    f"[vox] Scheduled retrain skipped: last retrain was"
                    f" {elapsed_h:.1f}h ago (min={MIN_RETRAIN_INTERVAL_HOURS}h)"
                )
                return

        self.log("[vox] Starting scheduled retrain …")
        timeout_bars = int(self._toh * 60 / DECISION_INTERVAL_MIN)

        X, y_class, y_return = build_training_data(
            algorithm             = self,
            symbols               = self._symbols,
            state_dict            = self._state,
            tp                    = self._tp,
            sl                    = self._sl,
            timeout_bars          = timeout_bars,
            decision_interval_min = DECISION_INTERVAL_MIN,
            label_tp              = self._label_tp,
            label_sl              = self._label_sl,
            label_horizon_bars    = self._label_horizon,
            cost_bps              = self._label_cost_bps,
        )
        if X is None or len(X) < 50:
            self.log("[vox] Retrain skipped: insufficient training data.")
            return

        try:
            self._ensemble = walk_forward_train(self._ensemble, X, y_class, y_return)
            self._model_ready = self._ensemble.is_fitted
            self._persistence.save_model(self._ensemble)
            reg_trained = getattr(self._ensemble, "_reg_fitted", False)
            self._last_retrain_time = self.time
            self.log(
                f"[vox] Retrain complete. Samples={len(X)}, fitted={self._model_ready}"
                f", reg_fitted={reg_trained}"
            )
        except Exception as exc:
            self.log(f"[vox] Retrain failed: {exc}")

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily(self):
        """Reset daily counters at midnight UTC."""
        self._risk.reset_daily()

    # ── Per-symbol penalty cooldown ───────────────────────────────────────────

    def _is_in_penalty_cooldown(self, sym):
        """
        Return True if *sym* is currently blocked by the penalty cooldown.

        The penalty cooldown is applied when a symbol has had
        ``_penalty_losses`` consecutive SL exits.  It is independent of the
        per-coin SL cooldown in RiskManager (which is much shorter).
        """
        penalty_until = self._sym_penalty_until.get(sym)
        if penalty_until is None:
            return False
        if self.time >= penalty_until:
            # Cooldown expired — clean up
            del self._sym_penalty_until[sym]
            return False
        return True

    def _update_penalty_cooldown(self, sym, is_sl):
        """Apply penalty cooldown if sym has had _penalty_losses consecutive SL exits."""
        if not is_sl:
            return   # Only SL exits contribute to penalty counting

        outcomes = list(self._sym_outcomes.get(sym, []))
        if len(outcomes) < self._penalty_losses:
            return   # Not enough history yet

        # Check that the last N outcomes are all losses (SL returns are negative)
        recent = outcomes[-self._penalty_losses:]
        all_losses = all(r < 0 for r in recent)
        if all_losses:
            penalty_end = self.time + timedelta(hours=self._penalty_hours)
            self._sym_penalty_until[sym] = penalty_end
            self.log(
                f"[vox] PENALTY COOLDOWN: {sym.value} has had {self._penalty_losses}"
                f" consecutive losses — blocked until {penalty_end} "
                f"({self._penalty_hours}h)"
            )

    # ── Throttled diagnostic helper ───────────────────────────────────────────

    def _throttled_skip_debug(self, message: str) -> None:
        """
        Emit a debug-level diagnostic message for routine trade skips, but only
        if at least SKIP_DIAG_INTERVAL_SECS seconds have elapsed since the last
        such message.  This prevents QuantConnect from rate-limiting the browser
        log during normal backtests where skip conditions fire every 15 minutes.

        Important events (fills, exits, errors) must NOT use this helper — call
        self.debug() or self.log() directly for those.
        """
        now = self.time
        if (
            self._last_skip_diag_time is None
            or (now - self._last_skip_diag_time).total_seconds() >= SKIP_DIAG_INTERVAL_SECS
        ):
            self._last_skip_diag_time = now
            self.debug(message)

    # ── History hydration ─────────────────────────────────────────────────────

    def _hydrate_state_from_history(self, days):
        """Fetch `days` of consolidated bars per symbol via self.history() and
        populate the state deques. Used by _retrain to guarantee a full
        training window regardless of consolidator timing."""
        bars_needed = int(days * 24 * 60 / RESOLUTION_MINUTES)
        bars_needed = min(bars_needed, MAX_HISTORY_BARS)   # safety cap (~104 days @ 5m)
        for sym in self._symbols:
            try:
                hist = self.history(
                    sym,
                    bars_needed,
                    Resolution.MINUTE,
                )
                if hist is None or hist.empty:
                    continue
                # If MultiIndex (symbol, time), select this symbol.
                if hasattr(hist.index, "levels") and len(hist.index.levels) > 1:
                    df = hist.loc[sym] if sym in hist.index.get_level_values(0) else hist
                else:
                    df = hist
                # Resample 1-min bars to RESOLUTION_MINUTES (open not stored in state).
                df = df.resample(f"{RESOLUTION_MINUTES}min").agg({
                    "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
                st = self._state[sym]
                st["closes"].clear(); st["highs"].clear()
                st["lows"].clear();   st["volumes"].clear()
                for _, row in df.iterrows():
                    st["closes"].append(float(row["close"]))
                    st["highs"].append(float(row["high"]))
                    st["lows"].append(float(row["low"]))
                    st["volumes"].append(float(row["volume"]))
            except Exception as exc:
                self.log(f"[vox] history hydrate failed for {sym.value}: {exc}")

    # ── End of algorithm ──────────────────────────────────────────────────────

    def on_end_of_algorithm(self):
        try:
            self._persistence.flush_trade_log()
        except Exception as exc:
            self.log(f"[vox] Failed to flush trade log on shutdown: {exc}")
