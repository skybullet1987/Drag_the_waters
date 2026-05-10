"""portfolio — multi-strategy capital allocator (the big "huge profits" lever).

PLAN.md §6.A.3: run 3 sub-strategies in parallel and dynamically reallocate
capital toward whichever has the best recent live Sharpe. No single strategy
works in every regime — the portfolio compounds across regimes.

Three sub-strategies:
  scalp  — MicroScalpEngine v8 (1-min crypto, alt-season harvester)
  trend  — TrendBasket (HYDRA done right, weekly daily-basket)
  mr     — MR engine (chop-regime capitulation buys)

Allocation rules:
  - Each strategy gets at least MIN_FLOOR (default 20%)
  - Each strategy capped at MAX_CEILING (default 60%)
  - Beyond floor, allocate proportionally to rolling 30-day Sharpe
  - If 2 of 3 strategies are negative-PnL over rolling window, halve
    exposure on the 3rd (cross-strategy risk veto)
  - "Last man standing" rule: never pause all 3 — at least one always
    runs as a probe with the floor allocation

Pure-Python — fully unit-testable. The QC integration (PulseAlgorithm)
plugs in by passing rolling outcome histories per strategy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence


STRATEGIES = ("scalp", "trend", "mr")


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MIN_FLOOR_FRAC          = 0.20    # each strategy gets at least 20%
DEFAULT_MAX_CEILING_FRAC        = 0.60    # capped at 60%
DEFAULT_LOOKBACK_TRADES         = 30      # rolling window for Sharpe estimation
DEFAULT_MIN_TRADES_FOR_ALLOC    = 10      # below this, use equal weights
DEFAULT_DECAY_FACTOR            = 0.95    # exponential decay for older trades
DEFAULT_NEGATIVE_PNL_THRESHOLD  = 0.0     # negative cumulative PnL = "in trouble"
DEFAULT_RISK_VETO_HALF_MULT     = 0.5     # halve exposure on 3rd strategy when 2 lose


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class StrategyOutcome:
    """One closed trade's outcome for the rolling-Sharpe estimate."""
    timestamp: datetime
    pnl_pct:   float    # net PnL fraction, e.g. +0.012 or -0.005


@dataclass
class StrategyState:
    """Per-strategy mutable state."""
    name:        str
    enabled:     bool = True
    paused_until: datetime | None = None
    outcomes:    deque = field(default_factory=lambda: deque(maxlen=200))

    def add_outcome(self, ts: datetime, pnl_pct: float):
        self.outcomes.append(StrategyOutcome(ts, pnl_pct))

    def is_paused(self, now: datetime) -> bool:
        return self.paused_until is not None and now < self.paused_until


@dataclass
class AllocationDecision:
    """Output of StrategyAllocator.compute_allocation()."""
    weights:       dict[str, float] = field(default_factory=dict)
    sharpes:       dict[str, float] = field(default_factory=dict)
    cum_pnls:      dict[str, float] = field(default_factory=dict)
    n_trades:      dict[str, int]   = field(default_factory=dict)
    risk_veto_active: bool = False
    notes:         list[str] = field(default_factory=list)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs):
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def rolling_sharpe(
    outcomes: Sequence[StrategyOutcome],
    lookback: int = DEFAULT_LOOKBACK_TRADES,
    decay:    float = DEFAULT_DECAY_FACTOR,
) -> float:
    """Decay-weighted Sharpe-like ratio over the last `lookback` trades.

    Returns 0.0 if fewer than 2 trades.
    """
    last = list(outcomes)[-lookback:]
    if len(last) < 2:
        return 0.0
    weights = [decay ** (len(last) - 1 - i) for i in range(len(last))]
    pnls    = [o.pnl_pct for o in last]
    wsum    = sum(weights)
    mean    = sum(p * w for p, w in zip(pnls, weights)) / wsum
    var     = sum(w * (p - mean) ** 2 for p, w in zip(pnls, weights)) / wsum
    std     = math.sqrt(var)
    if std <= 0:
        return 0.0
    return mean / std


def cumulative_pnl(
    outcomes: Sequence[StrategyOutcome],
    lookback: int = DEFAULT_LOOKBACK_TRADES,
) -> float:
    return sum(o.pnl_pct for o in list(outcomes)[-lookback:])


# ─── StrategyAllocator ──────────────────────────────────────────────────────

class StrategyAllocator:
    """Multi-strategy capital allocator with rolling-Sharpe weighting.

    Usage:
        alloc = StrategyAllocator()
        # Each time a sub-strategy closes a trade:
        alloc.record_outcome("scalp", now, pnl_pct=0.012)
        # Each decision tick:
        decision = alloc.compute_allocation(now=now, total_equity=1000.0)
        # decision.weights = {"scalp": 0.45, "trend": 0.30, "mr": 0.25}
        # → scalp_capital = 1000 * 0.45 = $450
    """

    def __init__(
        self,
        strategies: Sequence[str] = STRATEGIES,
        min_floor_frac:    float = DEFAULT_MIN_FLOOR_FRAC,
        max_ceiling_frac:  float = DEFAULT_MAX_CEILING_FRAC,
        lookback_trades:   int   = DEFAULT_LOOKBACK_TRADES,
        min_trades_alloc:  int   = DEFAULT_MIN_TRADES_FOR_ALLOC,
        decay:             float = DEFAULT_DECAY_FACTOR,
        risk_veto_threshold: float = DEFAULT_NEGATIVE_PNL_THRESHOLD,
        risk_veto_half_mult: float = DEFAULT_RISK_VETO_HALF_MULT,
    ):
        if min_floor_frac * len(strategies) > 1.0:
            raise ValueError(
                f"min_floor_frac × n_strategies > 1.0 "
                f"({min_floor_frac} × {len(strategies)})"
            )
        if max_ceiling_frac < min_floor_frac:
            raise ValueError("max_ceiling_frac must be >= min_floor_frac")

        self.strategies = list(strategies)
        self.min_floor_frac = min_floor_frac
        self.max_ceiling_frac = max_ceiling_frac
        self.lookback_trades = lookback_trades
        self.min_trades_alloc = min_trades_alloc
        self.decay = decay
        self.risk_veto_threshold = risk_veto_threshold
        self.risk_veto_half_mult = risk_veto_half_mult

        self._states: dict[str, StrategyState] = {
            s: StrategyState(name=s) for s in self.strategies
        }

    # ── Mutators ────────────────────────────────────────────────────────────

    def record_outcome(self, strategy: str, ts: datetime, pnl_pct: float):
        if strategy not in self._states:
            raise KeyError(f"Unknown strategy {strategy!r}")
        self._states[strategy].add_outcome(ts, pnl_pct)

    def pause_strategy(self, strategy: str, until: datetime):
        if strategy in self._states:
            self._states[strategy].paused_until = until

    def disable_strategy(self, strategy: str):
        if strategy in self._states:
            self._states[strategy].enabled = False

    # ── Read-only ───────────────────────────────────────────────────────────

    def state(self, strategy: str) -> StrategyState:
        return self._states[strategy]

    # ── Main ────────────────────────────────────────────────────────────────

    def compute_allocation(self, now: datetime,
                           total_equity: float = 1.0) -> AllocationDecision:
        """Compute per-strategy capital weight given current rolling stats.

        `total_equity` is informational only — weights are returned as fractions.
        """
        d = AllocationDecision()

        # Collect stats per strategy
        active: list[str] = []
        for s in self.strategies:
            st = self._states[s]
            outs = st.outcomes
            d.n_trades[s] = len(outs)
            d.sharpes[s]  = rolling_sharpe(outs, self.lookback_trades, self.decay)
            d.cum_pnls[s] = cumulative_pnl(outs, self.lookback_trades)
            if st.enabled and not st.is_paused(now):
                active.append(s)

        if not active:
            # Last-man-standing rule: revive at least one strategy at floor
            # Pick the one with the best historical Sharpe
            best = max(self.strategies, key=lambda s: d.sharpes[s])
            d.weights[best] = self.min_floor_frac
            d.notes.append(f"all_paused; reviving {best} at floor")
            return d

        # Risk veto: count strategies with cumulative PnL <= threshold (recent loss)
        n_in_trouble = sum(
            1 for s in active
            if d.cum_pnls[s] <= self.risk_veto_threshold
            and d.n_trades[s] >= self.min_trades_alloc
        )
        d.risk_veto_active = (n_in_trouble >= 2)
        if d.risk_veto_active:
            d.notes.append(
                f"risk_veto: {n_in_trouble} strategies negative-PnL → "
                f"halve exposure on the third"
            )

        # Compute base weights
        # If any strategy has < min_trades_alloc, use equal weights for all.
        if any(d.n_trades[s] < self.min_trades_alloc for s in active):
            equal = 1.0 / len(active)
            for s in active:
                d.weights[s] = equal
            d.notes.append("warmup: equal-weight (insufficient trades)")
        else:
            # Sharpe-proportional with floor + ceiling
            # Step 1: shift sharpes to be non-negative for proportional alloc
            min_sharpe = min(d.sharpes[s] for s in active)
            shifted = {s: max(d.sharpes[s] - min_sharpe + 0.01, 0.01)
                       for s in active}
            total = sum(shifted.values())
            for s in active:
                d.weights[s] = shifted[s] / total

        # Apply floor + ceiling — iteratively as in trend_engine
        for _ in range(10):
            changed = False
            for s in active:
                if d.weights[s] < self.min_floor_frac:
                    d.weights[s] = self.min_floor_frac
                    changed = True
                if d.weights[s] > self.max_ceiling_frac:
                    d.weights[s] = self.max_ceiling_frac
                    changed = True
            if changed:
                # Renormalize among strategies whose weights changed
                s = sum(d.weights.values())
                if s > 0 and abs(s - 1.0) > 1e-6:
                    # Try to rebalance: take excess from any strategy that
                    # exceeds floor + epsilon
                    excess = s - 1.0
                    if excess > 0:
                        # Take from non-pinned; can't reduce below floor
                        slack = {st: d.weights[st] - self.min_floor_frac
                                  for st in active}
                        slack = {st: v for st, v in slack.items() if v > 0}
                        slack_total = sum(slack.values())
                        if slack_total > 0:
                            for st, v in slack.items():
                                d.weights[st] -= excess * (v / slack_total)
                    else:
                        # Need to add capacity — give to non-ceiling
                        room = {st: self.max_ceiling_frac - d.weights[st]
                                 for st in active}
                        room = {st: v for st, v in room.items() if v > 0}
                        room_total = sum(room.values())
                        if room_total > 0:
                            for st, v in room.items():
                                d.weights[st] += (-excess) * (v / room_total)
                continue
            break

        # Apply risk veto: halve the third strategy's allocation
        if d.risk_veto_active:
            healthy = [s for s in active
                       if d.cum_pnls[s] > self.risk_veto_threshold]
            for s in healthy:
                d.weights[s] = max(self.min_floor_frac,
                                    d.weights[s] * self.risk_veto_half_mult)

        # Final normalization
        total = sum(d.weights.values())
        if total > 0:
            for s in d.weights:
                d.weights[s] /= total

        return d


# ─── Convenience helpers ────────────────────────────────────────────────────

def equal_weight_allocation(strategies: Sequence[str] = STRATEGIES) -> dict[str, float]:
    """Fallback equal-weight allocation; useful at strategy startup."""
    n = len(strategies)
    return {s: 1.0 / n for s in strategies}


def capital_per_strategy(
    decision: AllocationDecision,
    total_capital_usd: float,
) -> dict[str, float]:
    """Convert weights into dollar allocations."""
    return {s: total_capital_usd * w for s, w in decision.weights.items()}
