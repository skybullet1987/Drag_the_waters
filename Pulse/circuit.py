"""circuit — multi-layer risk circuit breakers.

Four independent risk gates:

1. ``DrawdownCircuitBreaker``
   - Trips at MAX_DRAWDOWN_TRIP_PCT (default 20%); halts at MAX_DRAWDOWN_HALT_PCT (25%)
   - Trip = pause new entries; halt = also liquidate
   - Auto-reset only after equity recovers MAX_DD_RECOVERY_PCT above the trip-low
   - Ported from Sweet Water v3-2 with two-tier behavior (trip vs halt)

2. ``RollingMaxDrawdown``
   - Pure tracker: rolling max DD over a window
   - Useful for monitoring without halting

3. ``PerTradeKill``
   - Catastrophic per-trade -8% market liquidation override
   - Backtest's largest intra-trade DD on MG36 was -41%; that's the time bomb
   - This bounds the worst case at -8% per name

4. ``EquityCurveStop`` (PLAN.md §6.E.2)
   - After N calendar days without a new equity high: pause new entries for M days
   - Open positions are managed normally (exits still fire)
   - Resume after pause window OR when equity stamps a new high
   - Catches "strategy stopped working" without waiting for a full drawdown

All pure-Python, no QC dep. Tests work offline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# ─── DrawdownCircuitBreaker ──────────────────────────────────────────────────

@dataclass
class CircuitState:
    """Read-only snapshot of CircuitBreaker state."""
    peak_equity: float
    current_equity: float
    drawdown_pct: float
    tripped: bool
    halted: bool
    trip_time: datetime | None
    trip_low_equity: float
    recovery_target_equity: float | None


class DrawdownCircuitBreaker:
    """Two-tier equity-curve circuit breaker.

    State machine:
        OK → TRIPPED (at -trip%):
            New entries paused.
            trip_low_equity = current equity at trip moment.
            recovery_target_equity = trip_low * (1 + recovery_pct)
        OK → HALTED (at -halt%):
            All positions liquidated (caller must observe this).
        TRIPPED → OK:
            When equity recovers to recovery_target_equity.
        HALTED → OK:
            Manual reset only (call .reset_halt()).

    Args:
        trip_drawdown_pct:   e.g. 0.20 for 20%; pause new entries
        halt_drawdown_pct:   e.g. 0.25 for 25%; also liquidate
        recovery_pct:        e.g. 0.05; equity must recover this much
                             above trip-low for auto-reset
    """

    def __init__(
        self,
        trip_drawdown_pct: float = 0.20,
        halt_drawdown_pct: float = 0.25,
        recovery_pct:      float = 0.05,
    ):
        if not (0 < trip_drawdown_pct < halt_drawdown_pct < 1.0):
            raise ValueError(
                f"trip_drawdown_pct ({trip_drawdown_pct}) must be < "
                f"halt_drawdown_pct ({halt_drawdown_pct}), both in (0, 1)"
            )
        self.trip_pct = trip_drawdown_pct
        self.halt_pct = halt_drawdown_pct
        self.recovery_pct = recovery_pct

        self._peak_equity: float = 0.0
        self._tripped: bool = False
        self._halted: bool = False
        self._trip_time: datetime | None = None
        self._trip_low_equity: float = 0.0
        self._recovery_target: float | None = None

    # ── Mutation: caller updates with current equity each tick ──────────────

    def update(self, current_equity: float, now: datetime) -> dict:
        """Update state with the latest equity value.

        Returns a small action dict describing what happened, e.g.:
            {"action": "noop"}
            {"action": "tripped", "drawdown_pct": 0.21, "peak": 1000.0}
            {"action": "halted",  "drawdown_pct": 0.26, "peak": 1000.0}
            {"action": "reset",   "trip_low": 800.0, "recovered_to": 840.0}
        """
        if current_equity <= 0:
            return {"action": "noop_zero_equity"}

        # First-ever update: seed peak and return early
        if self._peak_equity <= 0:
            self._peak_equity = current_equity
            return {"action": "init", "peak": current_equity}

        # Track new peak (only when not in trip — peak is the high before trip)
        if not self._tripped and current_equity > self._peak_equity:
            self._peak_equity = current_equity

        dd = (self._peak_equity - current_equity) / self._peak_equity

        # ── Halt path (most severe) ──
        if dd >= self.halt_pct and not self._halted:
            self._halted = True
            self._tripped = True
            self._trip_time = now
            self._trip_low_equity = current_equity
            self._recovery_target = current_equity * (1.0 + self.recovery_pct)
            return {
                "action":       "halted",
                "drawdown_pct": dd,
                "peak":         self._peak_equity,
                "current":      current_equity,
            }

        # ── Trip path ──
        if dd >= self.trip_pct and not self._tripped:
            self._tripped = True
            self._trip_time = now
            self._trip_low_equity = current_equity
            self._recovery_target = current_equity * (1.0 + self.recovery_pct)
            return {
                "action":       "tripped",
                "drawdown_pct": dd,
                "peak":         self._peak_equity,
                "current":      current_equity,
            }

        # ── While tripped: track trip-low + maybe recover ──
        if self._tripped:
            if current_equity < self._trip_low_equity:
                self._trip_low_equity = current_equity
                self._recovery_target = current_equity * (1.0 + self.recovery_pct)
                return {"action": "new_trip_low", "trip_low": current_equity}

            # Halted requires manual reset; otherwise check auto-reset
            if not self._halted and self._recovery_target is not None:
                if current_equity >= self._recovery_target:
                    return self._reset(current_equity, "reset")

        return {"action": "noop", "drawdown_pct": dd}

    def _reset(self, current_equity: float, label: str) -> dict:
        old_low    = self._trip_low_equity
        old_target = self._recovery_target
        self._tripped = False
        self._trip_time = None
        self._recovery_target = None
        # Reseed peak to the recovery point so a fresh DD calc starts here
        self._peak_equity = max(self._peak_equity, current_equity)
        return {
            "action":        label,
            "trip_low":      old_low,
            "recovered_to":  current_equity,
            "target_was":    old_target,
        }

    def reset_halt(self, current_equity: float | None = None) -> dict:
        """Manual reset after a HALT. Caller must explicitly invoke."""
        if not self._halted:
            return {"action": "noop_not_halted"}
        eq = current_equity if current_equity else self._peak_equity
        self._halted = False
        return self._reset(eq, "manual_reset")

    # ── Read-only queries ──

    def can_enter_new_positions(self) -> bool:
        """True only when neither tripped nor halted."""
        return not (self._tripped or self._halted)

    def should_liquidate_all(self) -> bool:
        """True only on HALT. Caller is responsible for actually liquidating."""
        return self._halted

    def state(self, current_equity: float = 0.0) -> CircuitState:
        dd = ((self._peak_equity - current_equity) / self._peak_equity
              if self._peak_equity > 0 and current_equity > 0 else 0.0)
        return CircuitState(
            peak_equity=self._peak_equity,
            current_equity=current_equity,
            drawdown_pct=dd,
            tripped=self._tripped,
            halted=self._halted,
            trip_time=self._trip_time,
            trip_low_equity=self._trip_low_equity,
            recovery_target_equity=self._recovery_target,
        )


# ─── RollingMaxDrawdown ──────────────────────────────────────────────────────

class RollingMaxDrawdown:
    """Track maximum drawdown over a fixed-size rolling window of equity values.

    Useful for monitoring strategy health on a sliding basis without
    triggering hard halts.
    """

    def __init__(self, lookback_bars: int = 1440):
        if lookback_bars < 2:
            raise ValueError("lookback_bars must be >= 2")
        self.lookback_bars = lookback_bars
        self._equities: deque[float] = deque(maxlen=lookback_bars)

    def update(self, equity: float) -> None:
        self._equities.append(float(equity))

    def get_max_drawdown(self) -> float:
        """Return rolling max DD as a positive fraction (e.g. 0.15 = 15%)."""
        if len(self._equities) < 2:
            return 0.0
        running_peak = self._equities[0]
        max_dd = 0.0
        for eq in self._equities:
            if eq > running_peak:
                running_peak = eq
            if running_peak > 0:
                dd = (running_peak - eq) / running_peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def get_current_drawdown(self) -> float:
        """Current draw from rolling-window peak."""
        if not self._equities:
            return 0.0
        peak = max(self._equities)
        cur  = self._equities[-1]
        return (peak - cur) / peak if peak > 0 else 0.0

    def __len__(self) -> int:
        return len(self._equities)


# ─── PerTradeKill ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TradeKillDecision:
    """Outcome of evaluating a position against the per-trade kill threshold."""
    symbol: str
    return_pct: float          # current PnL fraction, e.g. -0.082
    should_kill: bool
    reason: str | None         # e.g. 'hard_kill_-8.20%' or None


@dataclass
class PerTradeKill:
    """Hard-kill any single trade that hits the catastrophe threshold.

    MG36 backtest's largest intra-trade DD was -41%. PER_TRADE_HARD_KILL_PCT=0.08
    bounds the worst case at -8% per name regardless of the strategy's other logic.
    """
    threshold_pct: float = 0.08

    def __post_init__(self):
        if self.threshold_pct <= 0:
            raise ValueError("threshold_pct must be > 0 (positive fraction)")

    def evaluate(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
    ) -> TradeKillDecision:
        if entry_price <= 0 or current_price <= 0:
            return TradeKillDecision(symbol, 0.0, False, None)
        ret = (current_price - entry_price) / entry_price
        if ret <= -self.threshold_pct:
            return TradeKillDecision(
                symbol=symbol,
                return_pct=ret,
                should_kill=True,
                reason=f"hard_kill_{ret*100:.2f}%",
            )
        return TradeKillDecision(symbol, ret, False, None)


# ─── EquityCurveStop ─────────────────────────────────────────────────────────

@dataclass
class EquityStopState:
    """Read-only snapshot of EquityCurveStop state."""
    last_high_equity:         float
    last_high_time:           datetime | None
    days_since_high:          int
    paused_until:             datetime | None
    is_paused:                bool


class EquityCurveStop:
    """Pause new entries when the equity curve stalls.

    Triggered when equity has gone N calendar days without making a new high.
    On trigger, new entries are paused for M days. Open positions are not
    affected (exits still fire normally).

    Usage::

        stop = EquityCurveStop(stale_days=14, pause_days=7)
        # Each day:
        action = stop.update(current_equity, now)
        if not stop.can_enter_new_positions(now):
            return   # skip entry; managing-only mode

    Args:
        stale_days: trigger after this many calendar days without a new high
        pause_days: pause new entries for this many days when triggered
    """

    def __init__(self, stale_days: int = 14, pause_days: int = 7):
        if stale_days <= 0 or pause_days <= 0:
            raise ValueError("stale_days and pause_days must be > 0")
        self.stale_days = stale_days
        self.pause_days = pause_days

        self._last_high_equity: float = 0.0
        self._last_high_time: datetime | None = None
        self._paused_until: datetime | None = None

    def update(self, equity: float, now: datetime) -> dict:
        """Update state with the latest equity value.

        Returns an action dict, e.g.:
            {"action": "init"}             — first call
            {"action": "new_high"}         — equity made a new high
            {"action": "stale"}            — N+ days without new high; pause now
            {"action": "released"}         — pause window ended
            {"action": "noop"}             — within pause window or below stale threshold
        """
        if equity <= 0:
            return {"action": "noop_invalid_equity"}

        # First-ever update: seed
        if self._last_high_equity <= 0 or self._last_high_time is None:
            self._last_high_equity = equity
            self._last_high_time = now
            return {"action": "init", "high": equity}

        # New high — reset pause if any
        if equity > self._last_high_equity:
            self._last_high_equity = equity
            self._last_high_time = now
            old_pause = self._paused_until
            self._paused_until = None
            if old_pause is not None and now < old_pause:
                return {
                    "action":      "new_high_released_pause",
                    "high":        equity,
                    "released_at": now.isoformat(),
                }
            return {"action": "new_high", "high": equity}

        # Currently paused — check if window expired
        if self._paused_until is not None:
            if now >= self._paused_until:
                self._paused_until = None
                # Reset the staleness clock so we don't immediately re-trigger
                self._last_high_time = now
                return {"action": "released"}
            return {"action": "noop_paused"}

        # Not paused — check if we've been stale long enough to trigger
        days_stale = (now - self._last_high_time).days
        if days_stale >= self.stale_days:
            self._paused_until = now + timedelta(days=self.pause_days)
            return {
                "action":         "stale",
                "days_stale":     days_stale,
                "paused_until":   self._paused_until.isoformat(),
            }

        return {"action": "noop", "days_stale": days_stale}

    def can_enter_new_positions(self, now: datetime) -> bool:
        """True if we are NOT in a pause window."""
        return not (self._paused_until is not None and now < self._paused_until)

    def state(self, now: datetime) -> EquityStopState:
        if self._last_high_time is None:
            days = 0
        else:
            days = (now - self._last_high_time).days
        return EquityStopState(
            last_high_equity=self._last_high_equity,
            last_high_time=self._last_high_time,
            days_since_high=days,
            paused_until=self._paused_until,
            is_paused=not self.can_enter_new_positions(now),
        )
