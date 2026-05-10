"""slippage — RealisticCryptoSlippage (calibrated against MG36 paper trade).

Ported from Sweet Water v3-2's `realistic_slippage.py` — the most evolved
slippage model in the user's portfolio — with calibration upgrades from
the live MG36 evidence (mean per-fill 111bp, max 197bp).

Key design points:
- Volume-aware: order size relative to recent bar volume drives impact
- Spread-aware: uses bid/ask when available, synthetic spread floor otherwise
- Price-tier multipliers: low-price alts have wider proportional spreads
- 2.5% cap (was 2% in Sweet Water; raised based on live KASUSD 197bp single fill)

The model is implemented in pure Python (no QC dep) so it's unit-testable.
A `QCRealisticSlippage` adapter is exposed when AlgorithmImports is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from AlgorithmImports import *  # type: ignore  # noqa: F401,F403
    HAS_QC = True
except Exception:
    HAS_QC = False


# ─── Calibration defaults ────────────────────────────────────────────────────

@dataclass
class SlippageParams:
    """Parameters for RealisticCryptoSlippage.

    Defaults calibrated against the MG36 paper-trade evidence:
    - base 30bp + spread + volume impact + tier multiplier
    - Mean simulated round-trip ~100bp on majors, 250+ on micros
    """
    base_slippage_pct:         float = 0.0030   # 30bp baseline
    volume_impact_factor:      float = 0.40     # market impact at 1% participation
    volume_impact_exponent:    float = 1.5      # convex (worse at higher participation)
    max_slippage_pct:          float = 0.025    # 2.5% cap (was 2.0% in Sweet Water)

    # Synthetic spread floors (used when bid/ask unavailable in backtest)
    spread_floor_dust:    float = 0.020         # price < $0.01
    spread_floor_micro:   float = 0.010         # price < $0.10
    spread_floor_small:   float = 0.005         # price < $1
    spread_floor_mid:     float = 0.003         # price < $10
    spread_floor_large:   float = 0.0016        # price < $100
    spread_floor_major:   float = 0.0010        # price >= $100

    # Price tier multipliers (low price = wider proportional spread)
    tier_mult_dust:  float = 4.0
    tier_mult_micro: float = 2.5
    tier_mult_small: float = 1.8
    tier_mult_mid:   float = 1.2


def _spread_floor(price: float, p: SlippageParams) -> float:
    if price < 0.01:  return p.spread_floor_dust
    if price < 0.10:  return p.spread_floor_micro
    if price < 1.0:   return p.spread_floor_small
    if price < 10.0:  return p.spread_floor_mid
    if price < 100.0: return p.spread_floor_large
    return p.spread_floor_major


def _tier_multiplier(price: float, p: SlippageParams) -> float:
    if price < 0.01: return p.tier_mult_dust
    if price < 0.10: return p.tier_mult_micro
    if price < 1.0:  return p.tier_mult_small
    if price < 10.0: return p.tier_mult_mid
    return 1.0


def estimate_slippage_pct(
    *,
    price: float,
    order_quantity: float,
    bar_volume: float,
    bid: float = 0.0,
    ask: float = 0.0,
    params: SlippageParams | None = None,
) -> float:
    """Compute slippage as a fraction of price (e.g. 0.005 = 50bp).

    Pure-Python: returns the percent slippage; caller multiplies by price
    to get the absolute dollar amount.
    """
    p = params or SlippageParams()
    if price <= 0:
        return 0.0

    slip = p.base_slippage_pct

    # Spread component: real bid/ask preferred; synthetic floor otherwise
    if bid > 0 and ask > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
        if mid > 0:
            slip += (ask - bid) / (2.0 * mid)
    else:
        slip += _spread_floor(price, p)

    # Volume impact: convex in participation rate
    if bar_volume > 0:
        order_value  = abs(order_quantity) * price
        volume_value = bar_volume * price
        if volume_value > 0:
            participation = order_value / volume_value
            slip += p.volume_impact_factor * (participation ** p.volume_impact_exponent)

    # Price-tier multiplier
    slip *= _tier_multiplier(price, p)

    # Cap
    return min(slip, p.max_slippage_pct)


# ─── QC adapter ──────────────────────────────────────────────────────────────

if HAS_QC:

    class RealisticCryptoSlippage:
        """LEAN-compatible slippage model.

        Uses duck typing — QuantConnect Python LEAN matches by method
        name `GetSlippageApproximation(asset, order)` and does not
        require inheriting from any base class.
        """

        def __init__(self, params: SlippageParams | None = None):
            self.params = params or SlippageParams()

        def GetSlippageApproximation(self, asset, order):
            try:
                price = float(asset.Price)
                if price <= 0:
                    return 0
                bid = float(getattr(asset, "BidPrice", 0) or 0)
                ask = float(getattr(asset, "AskPrice", 0) or 0)
                vol = float(getattr(asset, "Volume", 0) or 0)
                qty = float(order.Quantity)
                pct = estimate_slippage_pct(
                    price=price, order_quantity=qty, bar_volume=vol,
                    bid=bid, ask=ask, params=self.params,
                )
                return price * pct
            except Exception:
                return 0
else:
    RealisticCryptoSlippage = None  # type: ignore
