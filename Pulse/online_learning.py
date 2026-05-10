"""online_learning — adapt scalp engine thresholds to live PnL.

After every closed trade Pulse can learn whether its entry/conviction
thresholds are too tight (missing winners) or too loose (taking losers).
This module implements a simple, robust gradient-style update on the
score-thresholds without overfitting:

  - Rolling window of last N closed trades (default 30)
  - Compute the "edge" of the threshold: WR × avg_win − (1−WR) × avg_loss
  - If edge < 0 over the rolling window: TIGHTEN threshold by step
  - If edge > 0 AND trades-per-day < target: LOOSEN threshold by step
  - Clamp thresholds to a hard min/max envelope so we can't drift away

Two threshold pairs adapted independently:
  - SCALP_ENTRY_THRESHOLD (default 0.55, range 0.45-0.70)
  - SCALP_HIGH_CONVICTION_THRES (default 0.70, range 0.60-0.85)

Pure-Python, no QC dep. Tests exercise every path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence


# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_LOOKBACK_TRADES         = 30
DEFAULT_MIN_TRADES_BEFORE_TUNE  = 15
DEFAULT_STEP_SIZE               = 0.005   # 0.5pp adjustment per cycle
DEFAULT_TIGHTEN_EDGE_THRESHOLD  = 0.0      # negative edge → tighten
DEFAULT_LOOSEN_EDGE_THRESHOLD   = 0.005   # positive edge above this → consider loosening
DEFAULT_TARGET_TRADES_PER_DAY   = 4.0
DEFAULT_TUNE_INTERVAL_HOURS     = 12

# Hard envelopes
ENTRY_THRESHOLD_MIN  = 0.45
ENTRY_THRESHOLD_MAX  = 0.70
HC_THRESHOLD_MIN     = 0.60
HC_THRESHOLD_MAX     = 0.85


# ─── Data types ────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """One closed trade outcome for learning."""
    timestamp:  datetime
    pnl_pct:    float       # net of fees
    score:      float       # the entry score that triggered this trade
    high_conviction: bool   # was it an HC trade?


@dataclass
class ThresholdState:
    """Mutable state for the online learner."""
    entry_threshold:        float = 0.55
    high_conviction_thres:  float = 0.70
    last_tune_time:         datetime | None = None
    trades:                 deque = field(default_factory=lambda: deque(maxlen=200))


@dataclass(frozen=True)
class TuneAction:
    """Result of one tune cycle (informational)."""
    timestamp:       datetime
    action:          str       # "tightened", "loosened", "noop", "warmup"
    old_entry:       float
    new_entry:       float
    old_hc:          float
    new_hc:          float
    edge_pct:        float
    win_rate:        float
    trades_observed: int
    reason:          str


# ─── Pure helpers ──────────────────────────────────────────────────────────

def compute_edge(trades: Sequence[TradeResult]) -> tuple[float, float, int]:
    """Return (edge_per_trade, win_rate, n) over the supplied trades.

    edge = WR × avg_win + (1-WR) × avg_loss   (avg_loss is negative)
    """
    if not trades:
        return 0.0, 0.0, 0
    n = len(trades)
    wins   = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
    if not wins and not losses:
        return 0.0, 0.0, n
    wr      = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_los = sum(losses) / len(losses) if losses else 0.0
    edge    = wr * avg_win + (1 - wr) * avg_los
    return edge, wr, n


def trades_per_day(trades: Sequence[TradeResult]) -> float:
    """Approximate trades-per-day from the timestamps in the rolling window."""
    if len(trades) < 2:
        return 0.0
    span_seconds = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
    if span_seconds <= 0:
        return float(len(trades))
    days = span_seconds / 86400.0
    return len(trades) / days if days > 0 else float(len(trades))


# ─── OnlineThresholdLearner ────────────────────────────────────────────────

class OnlineThresholdLearner:
    """Adaptive scalp-threshold tuner.

    Usage::

        learner = OnlineThresholdLearner()
        # After every closed trade:
        learner.record(now, pnl_pct, score, high_conviction)
        # Periodically (e.g. once per hour):
        action = learner.tune(now)
        if action.action == 'tightened':
            ...   # log it
        # Each entry tick read the live thresholds:
        thr = learner.entry_threshold
        hc  = learner.high_conviction_thres
    """

    def __init__(
        self,
        initial_entry_threshold:        float = 0.55,
        initial_high_conviction_thres:  float = 0.70,
        lookback_trades:                int   = DEFAULT_LOOKBACK_TRADES,
        min_trades_before_tune:         int   = DEFAULT_MIN_TRADES_BEFORE_TUNE,
        step_size:                      float = DEFAULT_STEP_SIZE,
        tighten_edge_threshold:         float = DEFAULT_TIGHTEN_EDGE_THRESHOLD,
        loosen_edge_threshold:          float = DEFAULT_LOOSEN_EDGE_THRESHOLD,
        target_trades_per_day:          float = DEFAULT_TARGET_TRADES_PER_DAY,
        tune_interval_hours:            float = DEFAULT_TUNE_INTERVAL_HOURS,
        entry_min:                      float = ENTRY_THRESHOLD_MIN,
        entry_max:                      float = ENTRY_THRESHOLD_MAX,
        hc_min:                         float = HC_THRESHOLD_MIN,
        hc_max:                         float = HC_THRESHOLD_MAX,
    ):
        if not (entry_min <= initial_entry_threshold <= entry_max):
            raise ValueError(
                f"initial_entry_threshold {initial_entry_threshold} outside "
                f"[{entry_min}, {entry_max}]"
            )
        if not (hc_min <= initial_high_conviction_thres <= hc_max):
            raise ValueError(
                f"initial_high_conviction_thres {initial_high_conviction_thres} "
                f"outside [{hc_min}, {hc_max}]"
            )
        self.lookback_trades         = lookback_trades
        self.min_trades_before_tune  = min_trades_before_tune
        self.step_size               = step_size
        self.tighten_edge_threshold  = tighten_edge_threshold
        self.loosen_edge_threshold   = loosen_edge_threshold
        self.target_trades_per_day   = target_trades_per_day
        self.tune_interval_hours     = tune_interval_hours
        self.entry_min, self.entry_max = entry_min, entry_max
        self.hc_min,    self.hc_max    = hc_min, hc_max

        self._state = ThresholdState(
            entry_threshold=initial_entry_threshold,
            high_conviction_thres=initial_high_conviction_thres,
            trades=deque(maxlen=max(lookback_trades, 50)),
        )

    # ── Read-only properties ────────────────────────────────────────────────

    @property
    def entry_threshold(self) -> float:
        return self._state.entry_threshold

    @property
    def high_conviction_thres(self) -> float:
        return self._state.high_conviction_thres

    @property
    def trade_count(self) -> int:
        return len(self._state.trades)

    # ── Mutators ────────────────────────────────────────────────────────────

    def record(self, timestamp: datetime, pnl_pct: float,
               score: float, high_conviction: bool) -> None:
        self._state.trades.append(TradeResult(
            timestamp=timestamp, pnl_pct=pnl_pct,
            score=score, high_conviction=high_conviction,
        ))

    def tune(self, now: datetime) -> TuneAction:
        """Run one tuning cycle. Idempotent within tune_interval_hours."""
        s = self._state

        # Throttle: don't re-tune until interval has passed
        if (s.last_tune_time is not None
                and (now - s.last_tune_time).total_seconds() / 3600
                    < self.tune_interval_hours):
            return TuneAction(
                timestamp=now, action="noop",
                old_entry=s.entry_threshold, new_entry=s.entry_threshold,
                old_hc=s.high_conviction_thres, new_hc=s.high_conviction_thres,
                edge_pct=0.0, win_rate=0.0, trades_observed=len(s.trades),
                reason=f"throttled (next tune at +{self.tune_interval_hours}h)",
            )

        # Window: most recent N trades
        recent = list(s.trades)[-self.lookback_trades:]
        if len(recent) < self.min_trades_before_tune:
            return TuneAction(
                timestamp=now, action="warmup",
                old_entry=s.entry_threshold, new_entry=s.entry_threshold,
                old_hc=s.high_conviction_thres, new_hc=s.high_conviction_thres,
                edge_pct=0.0, win_rate=0.0, trades_observed=len(recent),
                reason=f"warmup ({len(recent)}/{self.min_trades_before_tune})",
            )

        edge, wr, n = compute_edge(recent)
        tpd = trades_per_day(recent)
        old_entry, old_hc = s.entry_threshold, s.high_conviction_thres
        action_label, reason = "noop", "edge ok and trade-rate ok"

        # Negative edge: tighten both thresholds
        if edge < self.tighten_edge_threshold:
            s.entry_threshold = min(self.entry_max,
                                     s.entry_threshold + self.step_size)
            s.high_conviction_thres = min(self.hc_max,
                                           s.high_conviction_thres + self.step_size)
            action_label = "tightened"
            reason = f"edge={edge:+.4f} < tighten_threshold={self.tighten_edge_threshold:+.4f}"
        # Positive edge AND under-trading: loosen
        elif (edge > self.loosen_edge_threshold
              and tpd < self.target_trades_per_day):
            s.entry_threshold = max(self.entry_min,
                                     s.entry_threshold - self.step_size)
            s.high_conviction_thres = max(self.hc_min,
                                           s.high_conviction_thres - self.step_size)
            action_label = "loosened"
            reason = (f"edge={edge:+.4f} > loosen_threshold={self.loosen_edge_threshold:+.4f}"
                      f" AND trades/day={tpd:.1f} < target={self.target_trades_per_day}")

        s.last_tune_time = now
        return TuneAction(
            timestamp=now, action=action_label,
            old_entry=old_entry, new_entry=s.entry_threshold,
            old_hc=old_hc,        new_hc=s.high_conviction_thres,
            edge_pct=edge, win_rate=wr, trades_observed=n,
            reason=reason,
        )

    def reset(self) -> None:
        self._state.trades.clear()
        self._state.last_tune_time = None
