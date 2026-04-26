# ── Vox Risk Manager ──────────────────────────────────────────────────────────
#
# Centralises all pre-trade risk checks:
#   • Per-coin cooldown after SL exits
#   • Global cooldown after any exit
#   • Daily stop-loss cap
#   • Drawdown circuit-breaker
# ─────────────────────────────────────────────────────────────────────────────

from datetime import timedelta


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
