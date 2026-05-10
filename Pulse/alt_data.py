"""alt_data — alternative data subscriptions and signal helpers.

User decision (PLAN.md §0.B): drop WhaleAlert; properly implement Fear & Greed.

Two layers:

1. ``FearGreedData`` — QC custom-data subscription class for the
   alternative.me Fear & Greed Index (free, no API key, daily).
   Identical to the version Macro Cannon used.

2. ``FearGreedSignal`` — the part Macro Cannon was missing: a *pure-Python*
   signal helper that converts the index value into actionable size and
   gating decisions:
       value < 25 (extreme fear) → +1.2× size; bias toward bounce setups
       value > 75 (extreme greed) → -0.5× size; halve max_positions
       else → 1.0× size

   Plus utility helpers:
       fg_size_multiplier(value) → [0.5, 1.2]
       fg_max_positions_multiplier(value) → [0.5, 1.0]
       fg_regime(value) → 'extreme_fear' | 'fear' | 'neutral' | 'greed' | 'extreme_greed'

The two layers are decoupled so the FearGreedSignal logic is unit-testable
without QC (we just feed in float values).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from Pulse.config import (
    FG_GREED_EXTREME_THRESHOLD,
    FG_FEAR_EXTREME_THRESHOLD,
    FG_SIZE_MULT_GREED,
    FG_SIZE_MULT_FEAR,
)

try:
    from AlgorithmImports import (   # type: ignore  # noqa: F401
        PythonData,
        SubscriptionDataSource,
        SubscriptionTransportMedium,
    )
    from datetime import datetime, timedelta
    HAS_QC = True
except Exception:
    HAS_QC = False
    PythonData = object
    SubscriptionDataSource = None  # type: ignore
    SubscriptionTransportMedium = None  # type: ignore


# ─── Pure-Python signal layer ────────────────────────────────────────────────

FG_REGIMES = ("extreme_fear", "fear", "neutral", "greed", "extreme_greed")


def fg_regime(value: float | None) -> str:
    """Map raw F&G index value (0-100) to a named regime."""
    if value is None:
        return "neutral"
    v = float(value)
    if v <= FG_FEAR_EXTREME_THRESHOLD:                   # ≤ 25
        return "extreme_fear"
    if v < 50:
        return "fear"
    if v < FG_GREED_EXTREME_THRESHOLD:                   # < 75
        return "greed" if v > 50 else "neutral"
    return "extreme_greed"                                # ≥ 75


def fg_size_multiplier(value: float | None) -> float:
    """Per-trade size multiplier from F&G value.

    Anchored to the user's Path A defaults (PLAN.md §6.D.1):
      extreme_fear  (≤25) → 1.2× (capitulation buys)
      fear          (<50) → 1.05× (slight upsize)
      neutral             → 1.0×
      greed         (<75) → 0.85× (slight downsize)
      extreme_greed (≥75) → 0.5× (halve sizing — everyone is long)
    """
    r = fg_regime(value)
    return {
        "extreme_fear":  FG_SIZE_MULT_FEAR,    # 1.2
        "fear":          1.05,
        "neutral":       1.0,
        "greed":         0.85,
        "extreme_greed": FG_SIZE_MULT_GREED,   # 0.5
    }.get(r, 1.0)


def fg_max_positions_multiplier(value: float | None) -> float:
    """Multiplier on `max_positions` (caller does int(round(base * mult))).

      extreme_greed → 0.5  (halve concurrent positions; nowhere to dump to)
      greed         → 0.75
      neutral/fear  → 1.0
      extreme_fear  → 1.0  (we don't expand positions in fear, just resize)
    """
    r = fg_regime(value)
    return {
        "extreme_fear":  1.0,
        "fear":          1.0,
        "neutral":       1.0,
        "greed":         0.75,
        "extreme_greed": 0.5,
    }.get(r, 1.0)


def fg_block_new_entries(value: float | None,
                         block_above: float = 90.0) -> bool:
    """Hard gate: above this value, block all new entries.

    Default 90 = essentially "panic-greed only". Off by default in normal
    operation; surfaced as a hook the strategy can use as a kill switch.
    """
    if value is None:
        return False
    return float(value) >= block_above


def fg_bias_toward_bounce(value: float | None,
                          fear_threshold: float = 30.0) -> bool:
    """Should the strategy bias toward mean-reversion / bounce setups?
    True when F&G is in fear territory."""
    if value is None:
        return False
    return float(value) <= fear_threshold


@dataclass(frozen=True)
class FGSignal:
    """One-shot snapshot — what the strategy should do given the F&G value."""
    value: float | None
    regime: str
    size_multiplier: float
    max_positions_multiplier: float
    block_new_entries: bool
    bias_toward_bounce: bool

    @classmethod
    def from_value(cls, value: float | None) -> "FGSignal":
        return cls(
            value=value,
            regime=fg_regime(value),
            size_multiplier=fg_size_multiplier(value),
            max_positions_multiplier=fg_max_positions_multiplier(value),
            block_new_entries=fg_block_new_entries(value),
            bias_toward_bounce=fg_bias_toward_bounce(value),
        )


# ─── QC custom-data class ────────────────────────────────────────────────────

if HAS_QC:

    class FearGreedData(PythonData):
        """alternative.me Fear & Greed daily index — free, no API key needed."""

        def GetSource(self, config, date, isLiveMode):
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            return SubscriptionDataSource(
                url, SubscriptionTransportMedium.RemoteFile,
            )

        def Reader(self, config, line, date, isLiveMode):
            if not line or line.strip() == "":
                return None
            try:
                obj = json.loads(line)
                data_list = obj.get("data", [])
                if not data_list:
                    return None
                entry = data_list[0]
                value = float(entry["value"])
                timestamp = int(entry["timestamp"])
                result = FearGreedData()
                result.Symbol = config.Symbol
                result.Time = datetime.utcfromtimestamp(timestamp)
                result.Value = value
                result.EndTime = result.Time + timedelta(days=1)
                return result
            except Exception:
                return None
else:
    FearGreedData = None  # type: ignore
