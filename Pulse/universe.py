"""universe — Pulse capacity-aware universe gate + 4-tier symbol classifier.

This module addresses the #1 finding from the MG36 paper-trade audit:
the live universe expanded to 73 symbols including FARTCOIN, PEAQ, MORPHO,
TOSHI, NOS, AKT, ZRO, JUP — micro-cap meme coins with 100-300bp round-trip
slippage that ate every signal alive.

Two responsibilities:

1. ``UniverseGate``  — decides whether a symbol is *eligible* to trade. A coin
   passes only if it meets all of:
      - listed on Kraken cash spot
      - rolling 24h dollar volume ≥ MIN_DOLLAR_VOL_24H_USD
      - rolling 60-bar mean spread ≤ MAX_AVG_SPREAD_BPS
      - price ≥ MIN_PRICE_USD
      - ≥ MIN_DAYS_HISTORY of clean OHLCV

2. ``SymbolTierClassifier`` — assigns each *eligible* symbol to one of
   {major, large, mid, micro}, each tier carrying its own per-trade max
   USD and slippage budget. Tiers can auto-demote / eject when recent
   live slippage exceeds the budget on the last N trades.

All decisions are pure-Python dataclass logic — no QC dependency — so the
whole module is unit-testable offline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from Pulse.config import (
    UNIV_MIN_DOLLAR_VOL_24H_USD,
    UNIV_MAX_AVG_SPREAD_BPS,
    UNIV_MIN_PRICE_USD,
    UNIV_MIN_DAYS_HISTORY,
    TIER_MAJOR_SYMBOLS,
    TIER_LARGE_SYMBOLS,
    TIER_MID_SYMBOLS,
    TIER_LIMITS,
    TIER_SLIP_LOOKBACK_TRADES,
    TIER_DEMOTE_BUDGET_MULT,
    TIER_EJECT_BUDGET_MULT,
    TIER_EJECT_DURATION_HOURS,
)


# ─── Reasons & decisions ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateDecision:
    """Outcome of running a symbol through UniverseGate.is_eligible."""
    symbol: str
    eligible: bool
    reasons: tuple[str, ...]   # empty when eligible

    def __bool__(self) -> bool:
        return self.eligible


@dataclass(frozen=True)
class SymbolStats:
    """Snapshot of recent stats for a candidate symbol.

    Caller is responsible for computing these from QC history / quote bars
    and passing them into UniverseGate.is_eligible. Keeping the gate pure
    (no QC dependency) keeps it unit-testable.
    """
    symbol: str
    rolling_24h_dollar_vol_usd: float
    rolling_60bar_mean_spread_bps: float
    last_price_usd: float
    days_of_history: int
    is_kraken_cash_spot: bool = True
    is_wrapped_or_staked: bool = False


# ─── Universe gate ───────────────────────────────────────────────────────────

class UniverseGate:
    """Decide which symbols may be traded.

    All thresholds are sourced from Pulse.config; constructor accepts overrides
    for unit tests and walk-forward sweeps.
    """

    def __init__(
        self,
        min_dollar_vol_24h_usd: float = UNIV_MIN_DOLLAR_VOL_24H_USD,
        max_avg_spread_bps: float = UNIV_MAX_AVG_SPREAD_BPS,
        min_price_usd: float = UNIV_MIN_PRICE_USD,
        min_days_history: int = UNIV_MIN_DAYS_HISTORY,
    ):
        self.min_dollar_vol_24h_usd = min_dollar_vol_24h_usd
        self.max_avg_spread_bps = max_avg_spread_bps
        self.min_price_usd = min_price_usd
        self.min_days_history = min_days_history

    def is_eligible(self, stats: SymbolStats) -> GateDecision:
        reasons: list[str] = []

        if not stats.is_kraken_cash_spot:
            reasons.append("not_kraken_cash_spot")
        if stats.is_wrapped_or_staked:
            reasons.append("wrapped_or_staked_variant")
        if stats.last_price_usd < self.min_price_usd:
            reasons.append(
                f"price_below_floor({stats.last_price_usd:.4f}<{self.min_price_usd})"
            )
        if stats.rolling_24h_dollar_vol_usd < self.min_dollar_vol_24h_usd:
            reasons.append(
                f"dollar_vol_too_low("
                f"${stats.rolling_24h_dollar_vol_usd:,.0f}"
                f"<${self.min_dollar_vol_24h_usd:,.0f})"
            )
        if stats.rolling_60bar_mean_spread_bps > self.max_avg_spread_bps:
            reasons.append(
                f"spread_too_wide("
                f"{stats.rolling_60bar_mean_spread_bps:.1f}bp"
                f">{self.max_avg_spread_bps:.1f}bp)"
            )
        if stats.days_of_history < self.min_days_history:
            reasons.append(
                f"insufficient_history({stats.days_of_history}d<{self.min_days_history}d)"
            )

        return GateDecision(
            symbol=stats.symbol,
            eligible=not reasons,
            reasons=tuple(reasons),
        )

    def filter(self, candidates: Iterable[SymbolStats]) -> list[GateDecision]:
        """Run the gate on a batch — returns list of decisions in input order."""
        return [self.is_eligible(s) for s in candidates]


# ─── Symbol tier classifier ──────────────────────────────────────────────────

@dataclass
class TierState:
    """Mutable per-symbol tier state with auto-demote + eject memory."""
    symbol: str
    base_tier: str              # "major" | "large" | "mid" | "micro"
    current_tier: str           # may differ if auto-demoted
    recent_slippage_bps: deque  # rolling last N round-trip slippages
    ejected_until: datetime | None = None

    def is_ejected(self, now: datetime) -> bool:
        return self.ejected_until is not None and now < self.ejected_until


class SymbolTierClassifier:
    """Assign each symbol to a tier and track auto-demote / eject state.

    Tier assignment rules:
        - if symbol in TIER_MAJOR_SYMBOLS → "major"
        - elif in TIER_LARGE_SYMBOLS → "large"
        - elif in TIER_MID_SYMBOLS → "mid"
        - else → "micro"

    Each tier defines (max_pos_usd, slip_budget_bps). After a trade closes,
    call ``record_trade_slippage(symbol, round_trip_bps, now)``. If the
    rolling mean of the last N trades exceeds:
        - DEMOTE_BUDGET_MULT × budget → step the tier down one notch
        - EJECT_BUDGET_MULT × budget → eject for EJECT_DURATION_HOURS

    Call ``classify(symbol, now=...)`` each cycle to get the *current*
    effective tier, which may be lower than the base tier.
    """

    TIER_ORDER = ("major", "large", "mid", "micro", "ejected")

    def __init__(self):
        self._state: dict[str, TierState] = {}

    def _base_tier(self, symbol: str) -> str:
        if symbol in TIER_MAJOR_SYMBOLS:
            return "major"
        if symbol in TIER_LARGE_SYMBOLS:
            return "large"
        if symbol in TIER_MID_SYMBOLS:
            return "mid"
        return "micro"

    def _ensure(self, symbol: str) -> TierState:
        s = self._state.get(symbol)
        if s is None:
            base = self._base_tier(symbol)
            s = TierState(
                symbol=symbol,
                base_tier=base,
                current_tier=base,
                recent_slippage_bps=deque(maxlen=TIER_SLIP_LOOKBACK_TRADES),
            )
            self._state[symbol] = s
        return s

    def classify(self, symbol: str, now: datetime | None = None) -> str:
        """Return the *current* effective tier for `symbol`."""
        s = self._ensure(symbol)
        if now and s.is_ejected(now):
            return "ejected"
        return s.current_tier

    def limits_for(self, symbol: str, now: datetime | None = None) -> dict:
        """Return {'max_pos_usd', 'slip_budget_bps'} for current tier.

        Returns 0/0 limits if symbol is currently ejected.
        """
        tier = self.classify(symbol, now=now)
        if tier == "ejected":
            return {"max_pos_usd": 0.0, "slip_budget_bps": 0.0, "tier": "ejected"}
        d = dict(TIER_LIMITS[tier])
        d["tier"] = tier
        return d

    def record_trade_slippage(
        self,
        symbol: str,
        round_trip_bps: float,
        now: datetime,
    ) -> dict:
        """Update rolling slippage for `symbol`. May demote or eject the tier.

        Returns a small action dict describing what happened.
        """
        s = self._ensure(symbol)

        # If currently ejected and still in window, don't update (shouldn't
        # be trading anyway, but defensive).
        if s.is_ejected(now):
            return {"action": "ignored_ejected"}

        s.recent_slippage_bps.append(float(round_trip_bps))

        if len(s.recent_slippage_bps) < TIER_SLIP_LOOKBACK_TRADES:
            return {"action": "noop_warmup",
                    "trades_observed": len(s.recent_slippage_bps)}

        budget = TIER_LIMITS[s.current_tier]["slip_budget_bps"]
        avg = sum(s.recent_slippage_bps) / len(s.recent_slippage_bps)

        # Eject (most severe — checked first)
        if avg > TIER_EJECT_BUDGET_MULT * budget:
            s.ejected_until = now + timedelta(hours=TIER_EJECT_DURATION_HOURS)
            old_tier = s.current_tier
            # Reset to base_tier so post-eject the symbol gets a clean restart;
            # classify() consults ejected_until directly during the window.
            s.current_tier = s.base_tier
            s.recent_slippage_bps.clear()
            return {
                "action": "ejected",
                "from_tier": old_tier,
                "until": s.ejected_until.isoformat(),
                "avg_slip_bps": round(avg, 2),
                "budget_bps": budget,
            }

        # Demote one tier
        if avg > TIER_DEMOTE_BUDGET_MULT * budget:
            cur = s.current_tier
            try:
                idx = self.TIER_ORDER.index(cur)
            except ValueError:
                idx = 0
            new_idx = min(idx + 1, len(self.TIER_ORDER) - 2)  # don't auto-eject here
            new_tier = self.TIER_ORDER[new_idx]
            if new_tier != cur:
                s.current_tier = new_tier
                s.recent_slippage_bps.clear()  # fresh start at new tier
                return {
                    "action": "demoted",
                    "from": cur,
                    "to": new_tier,
                    "avg_slip_bps": round(avg, 2),
                    "budget_bps": budget,
                }

        return {"action": "noop_within_budget",
                "avg_slip_bps": round(avg, 2),
                "budget_bps": budget}

    def reset(self, symbol: str | None = None):
        """Reset state for one symbol (or all). Used in tests."""
        if symbol is None:
            self._state.clear()
        else:
            self._state.pop(symbol, None)


# ─── Convenience: filter eligible candidates and tier them ──────────────────

def screen(
    candidates: Iterable[SymbolStats],
    gate: UniverseGate | None = None,
    classifier: SymbolTierClassifier | None = None,
    now: datetime | None = None,
) -> dict:
    """One-shot screen: returns {eligible: [...], rejected: [...], tiers: {...}}.

    Used both at startup (initial universe build) and at periodic
    re-screening (when a symbol's rolling stats change).
    """
    g = gate or UniverseGate()
    c = classifier or SymbolTierClassifier()
    eligible: list[SymbolStats] = []
    rejected: list[GateDecision] = []
    tiers: dict[str, dict] = {}
    for s in candidates:
        d = g.is_eligible(s)
        if d.eligible:
            eligible.append(s)
            tiers[s.symbol] = c.limits_for(s.symbol, now=now)
        else:
            rejected.append(d)
    return {"eligible": eligible, "rejected": rejected, "tiers": tiers}
