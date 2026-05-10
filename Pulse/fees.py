"""fees — Kraken fee models for Pulse.

Two models, both ported from the user's existing strategies:

1. ``KrakenTieredFeeModel`` (from Sweet Water v3-2)
   - Volume-tiered fee model matching Kraken Pro Canada schedule.
   - Tracks cumulative volume; applies rolling 30-day-equivalent tier.
   - Default 25% taker / 75% maker for limit orders.

2. ``MakerTakerFeeModel`` (from MG36)
   - Flat-rate variant (0.40% taker / 0.25% maker).
   - Default 40% taker for limit orders.

Calibration note: based on MG36 paper-trade evidence (0/2 maker fills
within 30s TTL), the *realistic* taker ratio is much higher than the
defaults above. The Pulse harsh simulator uses 90-100% taker.

Both classes work without QC AlgorithmImports for unit testing — the
fee calculation is a pure function exposed via `compute_fee_pct()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple

try:
    from AlgorithmImports import (   # type: ignore  # noqa: F401
        FeeModel, OrderFee, CashAmount, OrderType,
    )
    HAS_QC = True
except Exception:
    HAS_QC = False
    FeeModel = object
    OrderFee = None    # type: ignore
    CashAmount = None  # type: ignore
    OrderType = None   # type: ignore


# ─── Pure-Python computation core ────────────────────────────────────────────

# Kraken Pro Canada schedule — (min_30d_volume_usd, maker, taker)
KRAKEN_FEE_TIERS: list[Tuple[float, float, float]] = [
    (500_000, 0.0008, 0.0018),   # $500K+
    (250_000, 0.0010, 0.0020),   # $250K+
    (100_000, 0.0012, 0.0022),   # $100K+
    (50_000,  0.0014, 0.0024),   # $50K+
    (25_000,  0.0020, 0.0035),   # $25K+
    (10_000,  0.0022, 0.0038),   # $10K+
    (2_500,   0.0030, 0.0060),   # $2.5K+
    (0,       0.0040, 0.0080),   # $0+ (most retail)
]


def lookup_kraken_tier(monthly_volume_usd: float) -> Tuple[float, float]:
    """Return (maker_rate, taker_rate) for the given 30-day volume.

    Tiers are searched highest-volume-first so the first match wins.
    """
    for min_vol, maker, taker in KRAKEN_FEE_TIERS:
        if monthly_volume_usd >= min_vol:
            return maker, taker
    return KRAKEN_FEE_TIERS[-1][1], KRAKEN_FEE_TIERS[-1][2]


def compute_fee_pct(
    *,
    is_limit_order: bool,
    monthly_volume_usd: float = 0.0,
    limit_taker_ratio: float = 0.25,
) -> float:
    """Pure function: blended fee % for one order.

    Market orders always pay full taker rate.
    Limit orders pay a blended (1-r)*maker + r*taker rate.
    """
    maker, taker = lookup_kraken_tier(monthly_volume_usd)
    if not is_limit_order:
        return taker
    r = limit_taker_ratio
    return (1.0 - r) * maker + r * taker


def compute_flat_fee_pct(
    *,
    is_limit_order: bool,
    maker_pct: float = 0.0025,
    taker_pct: float = 0.0040,
    limit_taker_ratio: float = 0.40,
) -> float:
    """MakerTakerFeeModel flat-rate variant from MG36."""
    if not is_limit_order:
        return taker_pct
    return (1.0 - limit_taker_ratio) * maker_pct + limit_taker_ratio * taker_pct


# ─── Trailing-30-day volume tracker ──────────────────────────────────────────

@dataclass
class VolumeTracker:
    """Cumulative volume tracker that approximates a 30-day rolling estimate.

    Strategy:
        Track total volume + days elapsed since first order; project to
        30-day-equivalent via `monthly = total * 30 / elapsed_days`.
        This is the same approach Sweet Water used.

    For QC backtests starting at zero capital this is a fair approximation
    that tightens as the strategy runs.
    """
    cumulative_volume_usd: float = 0.0
    start_time: datetime | None = None

    def record(self, time: datetime, trade_value_usd: float) -> None:
        if trade_value_usd <= 0:
            return
        self.cumulative_volume_usd += trade_value_usd
        if self.start_time is None:
            self.start_time = time

    def estimated_30d_volume(self, now: datetime) -> float:
        if self.start_time is None or self.cumulative_volume_usd <= 0:
            return 0.0
        elapsed_days = max((now - self.start_time).days, 1)
        return self.cumulative_volume_usd * 30.0 / elapsed_days


# ─── QC adapter classes ──────────────────────────────────────────────────────

if HAS_QC:

    class KrakenTieredFeeModel(FeeModel):
        """LEAN-compatible volume-tiered Kraken fee model.

        Wraps a `VolumeTracker` to compute the right tier per order.
        """

        LIMIT_TAKER_RATIO = 0.25

        def __init__(self):
            self._tracker = VolumeTracker()

        def GetOrderFee(self, parameters):
            order = parameters.Order
            price = parameters.Security.Price
            trade_value = float(order.AbsoluteQuantity) * float(price)

            self._tracker.record(order.Time, trade_value)
            monthly_vol = self._tracker.estimated_30d_volume(order.Time)
            fee_pct = compute_fee_pct(
                is_limit_order=(order.Type == OrderType.Limit),
                monthly_volume_usd=monthly_vol,
                limit_taker_ratio=self.LIMIT_TAKER_RATIO,
            )
            return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


    class MakerTakerFeeModel(FeeModel):
        """LEAN-compatible flat-rate Kraken fee model.

        Simpler than the tiered model; used when the strategy doesn't
        plan to compound to volume tiers.
        """

        LIMIT_TAKER_RATIO = 0.40
        MAKER_PCT = 0.0025
        TAKER_PCT = 0.0040

        def GetOrderFee(self, parameters):
            order = parameters.Order
            price = parameters.Security.Price
            trade_value = float(order.AbsoluteQuantity) * float(price)
            fee_pct = compute_flat_fee_pct(
                is_limit_order=(order.Type == OrderType.Limit),
                maker_pct=self.MAKER_PCT,
                taker_pct=self.TAKER_PCT,
                limit_taker_ratio=self.LIMIT_TAKER_RATIO,
            )
            return OrderFee(CashAmount(trade_value * fee_pct, "USD"))

else:
    KrakenTieredFeeModel = None  # type: ignore
    MakerTakerFeeModel = None    # type: ignore
