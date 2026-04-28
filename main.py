# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

# ─────────────────────────────────────────────────────────────────────────────
# Drag the Waters — Execution-First Kraken Top-Coin Rotation
#
# This version prioritises correct order handling and fill-driven state
# management over model sophistication.  All position state is updated
# ONLY in on_order_event after confirmed fills, eliminating the
# "Price=0 / Invalid order / state-drift" bugs seen in earlier backtests.
#
# Key fixes vs prior version:
#   1. Position is NOT marked active until the ENTRY fill is confirmed.
#   2. Exits use actual portfolio quantity, not an assumed target size.
#   3. Pre-trade validation rejects orders when price or buying power is bad.
#   4. on_order_event drives all state transitions.
#   5. _reconcile() clears stale internal state when holdings diverge.
#
# Universe    : 10 fixed Kraken USD pairs
# Cadence     : score all coins every 15 minutes
# Selection   : enter the top coin when score >= SCORE_MIN and it leads
#               the runner-up by at least SCORE_GAP
# Sizing      : explicit qty = (ALLOCATION × portfolio_value) / price
# Exits       : take-profit (+1.2 %), stop-loss (−0.7 %), 2-hour timeout
# Risk guards : post-exit cooldown; daily stop-loss cap
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE = [
    "BTCUSD", "ETHUSD", "AVAXUSD", "XRPUSD", "ADAUSD",
    "LTCUSD", "LINKUSD", "DOTUSD", "TRXUSD", "SOLUSD",
]

# Default strategy constants — all overridable via the QC parameter panel
TAKE_PROFIT   = 0.012   # +1.2 %  close long when price rises this much
STOP_LOSS     = 0.007   # −0.7 %  close long when price falls this much
TIMEOUT_HOURS = 2.0     # close if neither TP nor SL triggered within 2 h
SCORE_MIN     = 0.60    # minimum composite score required to open a position
SCORE_GAP     = 0.04    # required score lead of #1 coin over #2 coin
ALLOCATION    = 0.80    # fraction of portfolio per trade
MAX_DAILY_SL  = 2       # halt new entries for the day after this many SL hits
COOLDOWN_MINS = 15      # minutes to wait after any exit before re-entering
CASH_BUFFER   = 0.99   # keep 1 % cash headroom for fees/rounding at entry
QTY_PRECISION = 6      # decimal places for lot-size flooring (Kraken min lots)
# Safety buffer lots subtracted from the sell quantity to absorb fee/rounding
# precision mismatch between portfolio.quantity and the CashBook balance.
EXIT_QTY_BUFFER_LOTS = 1
# Quote suffixes for base-currency resolution (longest first).
# QC SymbolProperties does not expose base_currency; we derive it by stripping
# one of these suffixes from the symbol value.  Matches OrderHelper in Vox/infra.py.
_CRYPTO_QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH")


class KrakenTopCoinAlgorithm(QCAlgorithm):
    """
    Execution-first single-file QuantConnect strategy for Kraken spot.

    State transitions are driven exclusively by on_order_event fills.
    This prevents invalid orders (Price=0) and internal state drift from
    assuming fills that have not yet occurred.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(5_000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # Parameters — each can be overridden via the QC parameter panel
        self._tp      = float(self.get_parameter("take_profit")   or TAKE_PROFIT)
        self._sl      = float(self.get_parameter("stop_loss")     or STOP_LOSS)
        self._toh     = float(self.get_parameter("timeout_hours") or TIMEOUT_HOURS)
        self._s_min   = float(self.get_parameter("score_min")     or SCORE_MIN)
        self._s_gap   = float(self.get_parameter("score_gap")     or SCORE_GAP)
        self._alloc   = float(self.get_parameter("allocation")    or ALLOCATION)
        self._max_sl  = int(self.get_parameter("max_daily_sl")    or MAX_DAILY_SL)
        self._cd_mins = int(self.get_parameter("cooldown_mins")   or COOLDOWN_MINS)
        # Register securities and per-symbol 15-minute history
        self._symbols = []
        self._state   = {}   # symbol -> {"closes": deque, "volumes": deque}
        for ticker in UNIVERSE:
            sym = self.add_crypto(ticker, Resolution.MINUTE, Market.KRAKEN).symbol
            self._symbols.append(sym)
            self._state[sym] = {
                "closes":  deque(maxlen=30),   # 30 × 15-min bars ≈ 7.5 h
                "volumes": deque(maxlen=30),
            }
            # 15-min consolidator to keep close/volume history fresh
            self.consolidate(
                sym,
                timedelta(minutes=15),
                lambda bar, s=sym: self._on_15m(s, bar),
            )

        # ── Position state — updated ONLY via on_order_event ──────────────────
        self._pos_sym     = None   # confirmed open position symbol
        self._entry_px    = 0.0    # confirmed entry fill price
        self._entry_time  = None   # confirmed entry fill time
        self._pending_sym = None   # symbol of in-flight entry order
        self._pending_oid = None   # order ID of in-flight entry order
        self._exiting     = False  # True while an exit order is in flight
        self._exit_time   = None   # time of most recent completed exit
        self._daily_sl    = 0      # stop-loss hits today

        self.set_warm_up(timedelta(days=5))

        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(0, 0),
            self._reset_daily,
        )

    # ── 15-minute bar update ──────────────────────────────────────────────────

    def _on_15m(self, sym, bar):
        """Append close and volume from a freshly closed 15-min bar."""
        st = self._state[sym]
        st["closes"].append(float(bar.close))
        st["volumes"].append(float(bar.volume))

    # ── Main data handler ─────────────────────────────────────────────────────

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Reconcile internal state against actual holdings on every tick
        self._reconcile()

        # Exit check — minute-level precision; skip while an exit is in flight
        if self._pos_sym is not None and not self._exiting:
            if self._pos_sym in data.bars:
                self._check_exit(float(data.bars[self._pos_sym].close))
            elif self._entry_time is not None:
                # No bar this tick — still check the timeout so the position
                # is bounded even for illiquid symbols with sparse bars.
                elapsed = (self.time - self._entry_time).total_seconds() / 3600.0
                if elapsed >= self._toh:
                    fallback_px = float(self.securities[self._pos_sym].price)
                    if fallback_px > 0:
                        self._check_exit(fallback_px)

        # Entry logic — fire only at 15-minute bar boundaries
        if self.time.minute % 15 != 0:
            return
        if self._pos_sym is not None or self._pending_sym is not None:
            return
        if self._exit_time is not None and (
            self.time - self._exit_time < timedelta(minutes=self._cd_mins)
        ):
            return
        if self._daily_sl >= self._max_sl:
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
                # Entry fill confirmed — now mark the position active
                self._pos_sym     = sym
                self._entry_px    = float(order_event.fill_price)
                self._entry_time  = self.time
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
                self._pos_sym    = None
                self._entry_px   = 0.0
                self._entry_time = None
                self._exiting    = False
                self._exit_time  = self.time

        elif order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            if tag == "ENTRY" and sym == self._pending_sym:
                self.debug(
                    f"ENTRY order {oid} for {sym.value} — status={order_event.status},"
                    f" clearing pending"
                )
                self._pending_sym = None
                self._pending_oid = None

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                # Exit order failed; allow _check_exit to retry next bar
                self.debug(
                    f"EXIT order {oid} for {sym.value} — status={order_event.status},"
                    f" will retry"
                )
                self._exiting = False

    # ── Reconciliation ────────────────────────────────────────────────────────

    def _reconcile(self):
        """
        Compare internal tracking state against actual portfolio holdings.
        If they diverge, clear the stale internal state so the next scoring
        pass starts from a clean slate.
        """
        if self._pos_sym is not None:
            qty = self.portfolio[self._pos_sym].quantity
            if qty <= 0:
                sym = self._pos_sym   # capture before clearing
                self.debug(
                    f"RECONCILE: tracking {sym.value} but qty={qty:.6f}"
                    f" — clearing stale state"
                )
                self._pos_sym    = None
                self._entry_px   = 0.0
                self._entry_time = None
                self._exiting    = False
                self._exit_time  = self.time

        # Safety net: if a pending entry order has already settled (should have
        # been caught by on_order_event), clean it up here.
        # Handles the synchronous fill race where on_order_event fires inside
        # market_order() before _pending_sym was set, causing the fill to be
        # ignored.  Reconstruct minimal position state when FILLED.
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
                            self._entry_px   = float(
                                self.securities[self._pending_sym].price
                            )
                            self._entry_time = self.time
                            self.debug(
                                f"RECONCILE: recovered missed ENTRY fill for"
                                f" {self._pending_sym.value}"
                                f"  approx_px={self._entry_px:.4f}"
                            )
                    self._pending_sym = None
                    self._pending_oid = None
                else:   # INVALID or CANCELED
                    self._pending_sym = None
                    self._pending_oid = None

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _safe_sell_qty(self, sym):
        """
        Return a safe sell quantity that will not exceed the actual CashBook
        balance for the base currency.

        In Kraken cash mode, ``portfolio[sym].quantity`` can be slightly higher
        than the exchangeable base-currency balance after fees/rounding.
        Selling the raw portfolio quantity therefore causes an INVALID order.

        This helper takes the minimum of the portfolio holding and the CashBook
        amount, floors to ``QTY_PRECISION`` decimal places (as an approximation
        of the exchange lot size), then subtracts one unit at that precision as
        a safety buffer.

        Returns 0.0 when the position is dust / non-actionable.
        """
        portfolio_qty = float(self.portfolio[sym].quantity)
        if portfolio_qty <= 0:
            return 0.0

        # Determine base currency.
        # QC SymbolProperties does not expose base_currency; derive it from
        # quote_currency (which QC does expose) or by stripping known suffixes.
        base_ccy = None
        sym_val  = sym.value.upper()
        try:
            quote = str(
                self.securities[sym].symbol_properties.quote_currency
            ).upper()
            if quote and sym_val.endswith(quote):
                cand = sym_val[: -len(quote)]
                if cand:
                    base_ccy = cand
        except Exception:
            pass

        if not base_ccy:
            for suffix in _CRYPTO_QUOTE_SUFFIXES:
                if sym_val.endswith(suffix):
                    cand = sym_val[: -len(suffix)]
                    if cand:
                        base_ccy = cand
                        break

        cash_qty = portfolio_qty   # default: trust portfolio
        if base_ccy:
            try:
                cash_qty = float(
                    self.portfolio.cash_book[base_ccy].amount
                )
            except Exception:
                pass

        # Sellable = min(portfolio, cash balance)
        sellable = min(portfolio_qty, cash_qty)

        # Floor to QTY_PRECISION decimal places and subtract one unit buffer
        precision_factor = 10 ** QTY_PRECISION
        sellable         = int(sellable * precision_factor) / precision_factor
        safety_buffer_qty = 1.0 / precision_factor
        sellable = sellable - safety_buffer_qty

        if sellable <= 0:
            return 0.0
        return float(sellable)

    def _check_exit(self, price):
        """Evaluate TP / SL / timeout; submit a market sell for safe qty."""
        # Capture local references BEFORE placing any order.
        # market_order() can fill synchronously in QuantConnect/LEAN and
        # on_order_event may clear self._pos_sym before this function returns,
        # causing a NoneType error if we read self._pos_sym after the call.
        sym        = self._pos_sym
        entry_px   = self._entry_px
        entry_time = self._entry_time

        if sym is None or entry_time is None or entry_px <= 0:
            return

        ret     = (price - entry_px) / entry_px
        elapsed = (self.time - entry_time).total_seconds() / 3600.0

        reason = None
        if ret >= self._tp:
            reason = "EXIT_TP"
        elif ret <= -self._sl:
            reason = "EXIT_SL"
        elif elapsed >= self._toh:
            reason = "EXIT_TIMEOUT"

        if not reason:
            return

        # Safe sell quantity — guards against CashBook / portfolio precision
        # mismatch in Kraken cash mode.  Submitting portfolio.quantity verbatim
        # can cause an INVALID order when the CashBook balance is slightly lower
        # after fees/rounding.
        qty = self._safe_sell_qty(sym)
        if qty > 0:
            self._exiting = True   # suppress duplicate exit attempts
            # Log BEFORE market_order — fill may be synchronous.
            self.debug(
                f"EXIT order {sym.value}  reason={reason}"
                f"  qty={qty:.6f}  ret={ret:.3%}"
            )
            self.market_order(sym, -qty, tag=reason)
            if reason == "EXIT_SL":
                self._daily_sl += 1
        else:
            # Dust position or portfolio already flat — clear stale state
            portfolio_qty = float(self.portfolio[sym].quantity)
            self.debug(
                f"EXIT {sym.value}: safe sell qty=0 (dust/flat),"
                f" portfolio_qty={portfolio_qty:.8f}  reason={reason}"
                f" — clearing state"
            )
            self._pos_sym    = None
            self._entry_px   = 0.0
            self._entry_time = None
            self._exit_time  = self.time

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_enter(self):
        """Score all coins; if a clear winner emerges, place a buy order."""
        scores = {}
        for sym in self._symbols:
            sc = self._score(sym)
            if sc is not None:
                scores[sym] = sc

        if not scores:
            return

        ranked      = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sym, top_sc = ranked[0]
        second_sc       = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_sc < self._s_min or (top_sc - second_sc) < self._s_gap:
            return

        # ── Pre-trade validation ───────────────────────────────────────────────
        price = float(self.securities[top_sym].price)
        if price <= 0:
            self.debug(f"ENTRY skip {top_sym.value}: price={price} invalid")
            return

        portfolio_value = self.portfolio.total_portfolio_value
        target_value    = portfolio_value * self._alloc
        cash            = self.portfolio.cash
        if target_value > cash * CASH_BUFFER:
            self.debug(
                f"ENTRY skip {top_sym.value}: insufficient cash"
                f" (need {target_value:.2f}, have {cash:.2f})"
            )
            return

        # Explicit quantity — floor to QTY_PRECISION d.p. to avoid fractional-lot errors
        _factor = 10 ** QTY_PRECISION
        qty = float(int(target_value / price * _factor) / _factor)
        if qty <= 0:
            self.debug(f"ENTRY skip {top_sym.value}: computed qty={qty:.8f}")
            return

        # IMPORTANT: set _pending_sym BEFORE calling market_order().
        # In QuantConnect/LEAN, market orders with ImmediateFillModel fill
        # synchronously — on_order_event fires inside market_order() before it
        # returns.  If _pending_sym is still None at that point the ENTRY fill
        # check (sym == self._pending_sym) fails and the state machine gets
        # permanently stuck.
        self._pending_sym = top_sym
        order = self.market_order(top_sym, qty, tag="ENTRY")
        self._pending_oid = order.order_id
        self.debug(
            f"ENTRY order {top_sym.value}  score={top_sc:.3f}"
            f"  price={price:.4f}  qty={qty:.6f}"
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, sym):
        """
        Composite score in [0, 1] using three sub-signals:

        * 50% momentum  — return over last 4 intervals (4 × 15 min ≈ 1 h);
                          1 % move normalised to ±1.
        * 30% RSI(14)   — reward 55–74 (uptrend); penalise ≥ 75 (overbought)
                          or < 40 (oversold); simple non-smoothed calculation.
        * 20% vol spike — current-bar volume vs prior 15-bar mean; capped at 1.

        Returns None when there is insufficient history (< 20 bars).
        """
        st      = self._state[sym]
        closes  = st["closes"]
        volumes = st["volumes"]

        if len(closes) < 20 or len(volumes) < 2:
            return None

        c = list(closes)   # index 0 = oldest, index -1 = most recent

        # 1) Momentum: return over the last 4 intervals (4 × 15 min ≈ 1 h)
        mom   = (c[-1] - c[-5]) / c[-5]
        mom_n = max(min(mom / 0.01, 1.0), -1.0)

        # 2) RSI(14) — simple (non-smoothed Wilder) computation
        deltas = [c[i] - c[i - 1] for i in range(-14, 0)]
        gains  = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        avg_g  = gains / 14
        avg_l  = losses / 14
        rsi    = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)

        if rsi >= 75:
            rsi_s = -0.5    # overbought — penalise entry
        elif rsi >= 55:
            rsi_s =  1.0    # healthy uptrend — reward
        elif rsi >= 40:
            rsi_s =  0.3    # neutral
        else:
            rsi_s = -0.3    # oversold — slight penalty

        # 3) Volume spike vs prior 15-bar mean (exclude current bar)
        vols      = list(volumes)
        prior_avg = float(np.mean(vols[-16:-1]))   # exactly 15 prior bars
        vol_spike = (vols[-1] / prior_avg - 1.0) if prior_avg > 0 else 0.0
        vol_s     = min(max(vol_spike, 0.0), 1.0)   # clamp to [0, 1]

        # Weighted sum in [−1, 1] mapped to [0, 1]
        raw = 0.50 * mom_n + 0.30 * rsi_s + 0.20 * vol_s
        return (raw + 1.0) / 2.0

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily(self):
        """Clear the daily stop-loss counter at midnight."""
        self._daily_sl = 0
