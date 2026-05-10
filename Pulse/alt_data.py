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


# ─── Tier C.4: Funding rate proxy ───────────────────────────────────────────

"""
Funding rate proxy (Tier C.4 from PLAN.md §6.C.4).

Even though we trade Kraken cash spot, the funding rate on Bybit / Binance
perpetuals is a free signal we can read. It captures positioning extremes:

  - High positive funding (>+0.05% per 8h)
      Late longs paying premium to shorts.
      Crowded trade → mean-reversion BEARISH risk.
  - High negative funding (<-0.05% per 8h)
      Late shorts paying premium to longs.
      Crowded short squeeze → mean-reversion BULLISH opportunity.
  - Mild funding (-0.01% to +0.01%)
      Balanced; no crowding signal.

The Bybit/Binance public funding endpoints don't need an API key, but we
keep the QC subscription class optional and graceful — when the data isn't
available the strategy reverts to a neutral signal (multiplier = 1.0).

Two layers (mirrors the Fear&Greed pattern):
  1. ``BybitFundingData`` — QC custom-data subscription (Bybit BTCUSD perp)
  2. ``FundingSignal`` — pure-Python signal helper (testable without QC)
"""


FUNDING_REGIMES = (
    "deep_short_squeeze",   # rate <= -0.05%
    "shorts_paying",         # -0.05% < rate <= -0.01%
    "balanced",              # -0.01% < rate < +0.01%
    "longs_paying",          # +0.01% <= rate < +0.05%
    "deep_long_crowd",       # rate >= +0.05%
)

# Default thresholds (per 8h funding period, Bybit/Binance convention)
DEFAULT_FUNDING_DEEP_NEG_THRESHOLD = -0.0005   # -0.05%
DEFAULT_FUNDING_NEG_THRESHOLD      = -0.0001   # -0.01%
DEFAULT_FUNDING_POS_THRESHOLD      = +0.0001   # +0.01%
DEFAULT_FUNDING_DEEP_POS_THRESHOLD = +0.0005   # +0.05%


def funding_regime(rate: float | None) -> str:
    """Map raw funding rate (decimal, NOT percent) to a named regime."""
    if rate is None:
        return "balanced"
    r = float(rate)
    if r <= DEFAULT_FUNDING_DEEP_NEG_THRESHOLD:
        return "deep_short_squeeze"
    if r <= DEFAULT_FUNDING_NEG_THRESHOLD:
        return "shorts_paying"
    if r < DEFAULT_FUNDING_POS_THRESHOLD:
        return "balanced"
    if r < DEFAULT_FUNDING_DEEP_POS_THRESHOLD:
        return "longs_paying"
    return "deep_long_crowd"


def funding_size_modifier(rate: float | None) -> float:
    """Per-trade size multiplier from funding regime.

    Long-bias strategies (Pulse is long-only on Kraken cash):
      deep_short_squeeze (rate < -0.05%) → 1.20  (squeeze opportunity)
      shorts_paying      (rate < -0.01%) → 1.05
      balanced                            → 1.00
      longs_paying       (rate > +0.01%) → 0.85
      deep_long_crowd    (rate > +0.05%) → 0.50  (crowded — danger)
    """
    return {
        "deep_short_squeeze": 1.20,
        "shorts_paying":      1.05,
        "balanced":           1.00,
        "longs_paying":       0.85,
        "deep_long_crowd":    0.50,
    }.get(funding_regime(rate), 1.00)


def funding_block_new_entries(rate: float | None,
                              block_above: float = 0.0010) -> bool:
    """Hard gate: block new long entries when funding is extreme positive.

    Default threshold +0.10% per 8h ≈ +109% annualized — pure mania territory.
    """
    if rate is None:
        return False
    return float(rate) >= block_above


def funding_bias_toward_bounce(rate: float | None,
                               threshold: float = -0.0005) -> bool:
    """Bias mean-reversion / bounce setups when shorts are crowded."""
    if rate is None:
        return False
    return float(rate) <= threshold


@dataclass(frozen=True)
class FundingSignal:
    """One-shot snapshot of the funding-rate signal for a strategy tick."""
    rate:                     float | None
    regime:                   str
    size_modifier:            float
    block_new_entries:        bool
    bias_toward_bounce:       bool

    @classmethod
    def from_rate(cls, rate: float | None) -> "FundingSignal":
        return cls(
            rate=rate,
            regime=funding_regime(rate),
            size_modifier=funding_size_modifier(rate),
            block_new_entries=funding_block_new_entries(rate),
            bias_toward_bounce=funding_bias_toward_bounce(rate),
        )


# QC custom-data class (Bybit BTC perp funding, public, no key)
if HAS_QC:

    class BybitFundingData(PythonData):
        """Bybit BTCUSDT perpetual funding rate — public, no API key needed.

        Bybit returns the most recent funding rate per call. We poll once
        per 4h via QC's hourly resolution to stay well below rate limits.
        """

        def GetSource(self, config, date, isLiveMode):
            url = (
                "https://api.bybit.com/v5/market/funding/history"
                "?category=linear&symbol=BTCUSDT&limit=1"
            )
            return SubscriptionDataSource(
                url, SubscriptionTransportMedium.RemoteFile,
            )

        def Reader(self, config, line, date, isLiveMode):
            if not line or not line.strip():
                return None
            try:
                obj = json.loads(line)
                result_list = obj.get("result", {}).get("list", [])
                if not result_list:
                    return None
                latest = result_list[0]
                rate = float(latest["fundingRate"])
                ts_ms = int(latest.get("fundingRateTimestamp", 0))
                from datetime import datetime as _dt, timedelta as _td
                t = _dt.utcfromtimestamp(ts_ms / 1000) if ts_ms else date
                result = BybitFundingData()
                result.Symbol = config.Symbol
                result.Time = t
                # Store the rate in .Value for downstream access
                result.Value = rate
                # Funding cycle is 8h on Bybit
                result.EndTime = result.Time + _td(hours=8)
                return result
            except Exception:
                return None
else:
    BybitFundingData = None  # type: ignore
