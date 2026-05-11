"""Pulse.apex.exits — exit logic for Apex positions.

Three exit gates evaluated per minute tick:

  1. PROBABILITY_FLIP   ML prob (re-scored at next 4h tick) drops below
                        APEX_EXIT_THRESHOLD → market liquidate
  2. ATR_TRAIL          Trailing stop based on entry's ATR; tightens with MFE
  3. TIME_STOP          Held longer than APEX_HOLD_DAYS_MAX → liquidate
  4. PER_TRADE_KILL     Loss exceeds APEX_PER_TRADE_HARD_KILL_PCT → liquidate

The first three are "soft" exits — they let the strategy follow its
plan. The last is a HARD KILL that overrides everything.

Pure-Python — no QC dependency. Caller wires actual order placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from Pulse.apex.config import (
    APEX_ATR_TRAIL_MULT,
    APEX_HOLD_DAYS_MAX,
    APEX_PER_TRADE_HARD_KILL_PCT,
    APEX_EXIT_THRESHOLD,
)


# ─── Position state ─────────────────────────────────────────────────────────


@dataclass
class ApexPosition:
    """Per-symbol position state owned by the Apex engine."""
    symbol:           str
    entry_time:       datetime
    entry_price:      float
    quantity:         float
    entry_atr:        float            # ATR at entry; used for trail width
    high_water:       float            # highest price seen since entry
    last_seen_prob:   float = 0.5      # most-recent ML prob

    def update_high_water(self, price: float) -> None:
        if price > self.high_water:
            self.high_water = price

    def held_days(self, now: datetime) -> float:
        return (now - self.entry_time).total_seconds() / 86_400.0


# ─── Exit decision ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExitDecision:
    should_exit:  bool
    reason:       Optional[str] = None
    detail:       Optional[dict] = None


def should_exit_per_trade_kill(pos: ApexPosition, current_price: float,
                                hard_kill_pct: float = APEX_PER_TRADE_HARD_KILL_PCT
                                ) -> ExitDecision:
    if pos.entry_price <= 0:
        return ExitDecision(False)
    pnl_pct = (current_price - pos.entry_price) / pos.entry_price
    if pnl_pct <= -hard_kill_pct:
        return ExitDecision(True, "PER_TRADE_KILL",
                            {"pnl_pct": pnl_pct,
                             "hard_kill_pct": -hard_kill_pct})
    return ExitDecision(False)


def should_exit_time_stop(pos: ApexPosition, now: datetime,
                           max_days: float = APEX_HOLD_DAYS_MAX
                           ) -> ExitDecision:
    if pos.held_days(now) >= max_days:
        return ExitDecision(True, "TIME_STOP",
                            {"held_days": pos.held_days(now),
                             "max_days": max_days})
    return ExitDecision(False)


def should_exit_atr_trail(pos: ApexPosition, current_price: float,
                           trail_mult: float = APEX_ATR_TRAIL_MULT
                           ) -> ExitDecision:
    """Trailing stop = high_water - trail_mult × entry_atr.

    Returns False when price hasn't moved up enough to define a trail
    above the entry (i.e. high_water == entry_price).
    """
    if pos.entry_atr <= 0:
        return ExitDecision(False)
    trail = pos.high_water - trail_mult * pos.entry_atr
    if current_price <= trail and pos.high_water > pos.entry_price:
        return ExitDecision(True, "ATR_TRAIL",
                            {"price": current_price,
                             "trail": trail,
                             "high_water": pos.high_water})
    return ExitDecision(False)


def should_exit_prob_flip(pos: ApexPosition,
                           latest_prob: float,
                           exit_threshold: float = APEX_EXIT_THRESHOLD
                           ) -> ExitDecision:
    if latest_prob < exit_threshold:
        return ExitDecision(True, "PROB_FLIP",
                            {"latest_prob": latest_prob,
                             "exit_threshold": exit_threshold})
    return ExitDecision(False)


def evaluate_exits(pos: ApexPosition, *,
                    current_price: float,
                    now: datetime,
                    latest_prob: Optional[float] = None,
                    hard_kill_pct: float = APEX_PER_TRADE_HARD_KILL_PCT,
                    max_days: float = APEX_HOLD_DAYS_MAX,
                    trail_mult: float = APEX_ATR_TRAIL_MULT,
                    exit_threshold: float = APEX_EXIT_THRESHOLD,
                    ) -> ExitDecision:
    """Evaluate ALL exit gates in priority order. First hit wins."""
    pos.update_high_water(current_price)

    # Hard kill first — overrides everything
    d = should_exit_per_trade_kill(pos, current_price, hard_kill_pct)
    if d.should_exit:
        return d

    # Time stop
    d = should_exit_time_stop(pos, now, max_days)
    if d.should_exit:
        return d

    # Probability flip (only when we have a fresh probability)
    if latest_prob is not None:
        d = should_exit_prob_flip(pos, latest_prob, exit_threshold)
        if d.should_exit:
            return d

    # Soft trail
    d = should_exit_atr_trail(pos, current_price, trail_mult)
    if d.should_exit:
        return d

    return ExitDecision(False)
