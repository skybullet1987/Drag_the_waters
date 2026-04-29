# ── Vox Risk ──────────────────────────────────────────────────────────────────
#
# Consolidated module for all pre-trade guards:
#   • RegimeFilter  — 4h BTC SMA(20) + slope gate
#   • kelly_fraction / compute_qty — fractional-Kelly position sizing
#   • RiskManager   — per-coin cooldown, daily SL cap, drawdown circuit-breaker
# Previously split across regime.py, sizing.py, and risk.py.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from datetime import timedelta
from AlgorithmImports import *


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeFilter:
    """
    Bitcoin-based macro regime filter.

    Logic
    -----
    The filter collects 4-hour BTC bars via a QC consolidator.  On each
    decision call it evaluates two conditions:

    1. **SMA gate** — the latest 4h close must be above the 20-bar SMA.
    2. **Slope gate** — a linear slope over the last 5 bars must be positive.

    If either condition fails the filter returns False (block alt longs).
    BTC itself is always permitted regardless of regime.

    Insufficient history (< 20 bars) is treated as "risk on".

    Usage
    -----
    In ``initialize``::

        self._regime = RegimeFilter()
        self._regime.update_btc(self, self._btc_sym)

    In entry logic::

        if not self._regime.is_risk_on(self._btc_sym, sym=top_sym):
            return
    """

    _SMA_PERIOD   = 20
    _SLOPE_PERIOD = 5

    def __init__(self):
        self._closes = deque(maxlen=self._SMA_PERIOD + self._SLOPE_PERIOD)

    def update_btc(self, algorithm, btc_sym):
        """
        Register a 4-hour consolidator on *btc_sym* that feeds this filter.
        Call once from ``algorithm.initialize()``.
        """
        algorithm.consolidate(
            btc_sym,
            timedelta(hours=4),
            self._on_4h_bar,
        )

    def _on_4h_bar(self, bar):
        """Receive a closed 4-hour bar and append the close price."""
        self._closes.append(float(bar.close))

    def is_risk_on(self, btc_sym, sym=None):
        """
        Return True if the macro regime permits a long entry.

        Parameters
        ----------
        btc_sym : Symbol — BTC symbol (exempt from the regime gate).
        sym     : Symbol or None — symbol being evaluated.

        Returns
        -------
        bool
        """
        if sym is not None and sym == btc_sym:
            return True

        closes = list(self._closes)
        if len(closes) < self._SMA_PERIOD:
            return True   # insufficient history → allow trading

        sma20  = float(np.mean(closes[-self._SMA_PERIOD:]))
        latest = closes[-1]
        if latest < sma20:
            return False

        if len(closes) >= self._SLOPE_PERIOD and self._SLOPE_PERIOD > 1:
            window = closes[-self._SLOPE_PERIOD:]
            slope  = (window[-1] - window[0]) / (self._SLOPE_PERIOD - 1)
            if slope < 0.0:
                return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING  (fractional Kelly)
# ═══════════════════════════════════════════════════════════════════════════════

def kelly_fraction(p, tp, sl, kelly_frac=0.25, max_alloc=0.80):
    """
    Compute the fractional-Kelly allocation for a long trade.

    Full-Kelly formula::

        b       = tp / sl          (payoff ratio)
        f_full  = (p × (b + 1) − 1) / b

    The result is scaled by *kelly_frac* (quarter-Kelly) and clamped to
    [0, max_alloc].

    Parameters
    ----------
    p          : float — model probability P(win) in (0, 1).
    tp         : float — take-profit fraction (e.g. 0.020).
    sl         : float — stop-loss fraction   (e.g. 0.012).
    kelly_frac : float — fractional-Kelly multiplier (default 0.25).
    max_alloc  : float — hard ceiling on allocation  (default 0.80).

    Returns
    -------
    float — allocation fraction in [0, max_alloc].
    """
    if sl <= 0:
        return 0.0
    b      = tp / sl
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, min(f_full * kelly_frac, max_alloc))


def compute_qty(
    mean_proba,
    tp,
    sl,
    price,
    portfolio_value,
    kelly_frac,
    max_alloc,
    cash_buffer,
    use_kelly,
    allocation,
    min_alloc=0.0,
):
    """
    Compute the quantity (in coin units) to purchase.

    Sizing logic
    ────────────
    1. If *use_kelly* is True, compute the Kelly allocation.
       - If Kelly <= 0 (negative edge), fall back to flat *allocation*.
       - If Kelly > 0 and *min_alloc* > 0, apply the floor:
         ``alloc = max(kelly_alloc, min_alloc)``
    2. If *use_kelly* is False, use flat *allocation* directly.
    3. Apply *max_alloc* cap (always honoured, even after min_alloc floor).
    4. Apply *cash_buffer* to leave headroom for fees.
    5. Convert dollar value to coin units.

    Parameters
    ----------
    min_alloc : float
        Minimum allocation fraction when Kelly is positive (default 0.0).
        Set to e.g. 0.75 for ruthless mode so Kelly cannot shrink trades
        below 75 % of portfolio.  Ignored when use_kelly=False.

    Returns
    -------
    tuple[float, float]
        ``(qty, alloc_fraction)`` where qty is in coin units.
    """
    if use_kelly:
        alloc = kelly_fraction(mean_proba, tp, sl, kelly_frac, max_alloc)
        if alloc <= 0.0:
            alloc = allocation
        elif min_alloc > 0.0:
            alloc = max(alloc, min_alloc)
    else:
        alloc = allocation

    alloc        = min(alloc, max_alloc)   # honour hard ceiling after any floor
    dollar_value = portfolio_value * alloc * cash_buffer
    qty          = dollar_value / price if price > 0 else 0.0
    return qty, alloc


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER


class RiskManager:
    """
    Pre-trade risk guardrails for the Vox strategy.

    Parameters
    ----------
    max_daily_sl    : int   — halt new entries after this many SL hits.
    cooldown_mins   : float — global cooldown (minutes) after any exit.
    sl_cooldown_mins: float — per-coin cooldown (minutes) specifically after
                              a stop-loss exit.
    max_dd_pct      : float — circuit-breaker threshold (e.g. 0.08 = 8 %).
    cash_buffer     : float — reserved; not used internally but stored for
                              consistency with config.
    """

    def __init__(
        self,
        max_daily_sl,
        cooldown_mins,
        sl_cooldown_mins,
        max_dd_pct,
        cash_buffer,
    ):
        self._max_daily_sl     = max_daily_sl
        self._cooldown         = timedelta(minutes=cooldown_mins)
        self._sl_cooldown      = timedelta(minutes=sl_cooldown_mins)
        self._max_dd_pct       = max_dd_pct
        self._cash_buffer      = cash_buffer

        # Mutable state
        self._daily_sl         = 0          # stop-loss hits today
        self._last_exit_time   = None       # time of most recent exit (any)
        self._sl_exit_times    = {}         # sym -> time of last SL exit
        self._rolling_high     = None       # rolling portfolio high-water mark

    # ── State updates ─────────────────────────────────────────────────────────

    def record_exit(self, sym, is_sl, exit_time):
        """
        Record that a position in *sym* was exited at *exit_time*.

        Parameters
        ----------
        sym       : Symbol — the exited symbol.
        is_sl     : bool   — True if the exit was triggered by stop-loss.
        exit_time : datetime — the exit timestamp.
        """
        self._last_exit_time = exit_time
        if is_sl:
            self._daily_sl += 1
            self._sl_exit_times[sym] = exit_time

    def record_sl(self):
        """Increment the daily stop-loss counter (alternative to record_exit)."""
        self._daily_sl += 1

    def reset_daily(self):
        """Reset daily counters.  Call at midnight via scheduled event."""
        self._daily_sl       = 0

    def update_rolling_high(self, portfolio_value):
        """
        Update the 30-day rolling high-water mark.

        Parameters
        ----------
        portfolio_value : float — current total portfolio value.
        """
        if self._rolling_high is None or portfolio_value > self._rolling_high:
            self._rolling_high = portfolio_value

    # ── Checks ────────────────────────────────────────────────────────────────

    def check_drawdown(self, portfolio_value):
        """
        Return True (circuit-breaker tripped) if equity has dropped more than
        *max_dd_pct* from the rolling high-water mark.

        Parameters
        ----------
        portfolio_value : float

        Returns
        -------
        bool
            True  → halt trading.
            False → trading permitted.
        """
        if self._rolling_high is None or self._rolling_high == 0:
            return False
        dd = (self._rolling_high - portfolio_value) / self._rolling_high
        return dd > self._max_dd_pct

    def can_enter(self, sym, current_time, portfolio_value, rolling_high=None):
        """
        Evaluate all pre-trade risk checks for a potential long entry.

        Parameters
        ----------
        sym             : Symbol  — the symbol to be traded.
        current_time    : datetime — current algorithm time.
        portfolio_value : float   — current total portfolio value.
        rolling_high    : float or None — caller-provided rolling high; if None
                          the internally tracked value is used.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)`` where *reason* describes the decision.
        """
        # Update rolling high with caller-provided value if given
        if rolling_high is not None:
            self.update_rolling_high(rolling_high)
        self.update_rolling_high(portfolio_value)

        # ── Daily SL cap ──────────────────────────────────────────────────────
        if self._daily_sl >= self._max_daily_sl:
            return False, f"daily_sl_cap ({self._daily_sl}/{self._max_daily_sl})"

        # ── Drawdown circuit-breaker ───────────────────────────────────────────
        if self.check_drawdown(portfolio_value):
            return False, "drawdown_circuit_breaker"

        # ── Global post-exit cooldown ─────────────────────────────────────────
        if self._last_exit_time is not None:
            elapsed = current_time - self._last_exit_time
            if elapsed < self._cooldown:
                remaining = int((self._cooldown - elapsed).total_seconds() / 60)
                return False, f"global_cooldown ({remaining}m remaining)"

        # ── Per-coin SL cooldown ──────────────────────────────────────────────
        if sym in self._sl_exit_times:
            elapsed = current_time - self._sl_exit_times[sym]
            if elapsed < self._sl_cooldown:
                remaining = int((self._sl_cooldown - elapsed).total_seconds() / 60)
                return False, f"sl_cooldown_{sym.value} ({remaining}m remaining)"

        return True, "ok"
