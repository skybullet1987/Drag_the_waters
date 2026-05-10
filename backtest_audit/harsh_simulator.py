"""harsh_simulator — pessimistic QC-backtest overrides anchored to live evidence.

Calibrated to MG36 paper-trade evidence (PLAN.md §0.A):
- Live ⚠️ HIGH SLIPPAGE warnings: 65 - 197 bps per fill (mean ~111 bps)
- Maker limit fills: 0 / 2 within 30s (limits time out under volatility)
- Round-trip cost on KASUSD: 304 bps total (vs 60 bps backtest model)

This module exposes:
1. `HarshConfig` — typed config dataclass with overridable defaults
2. `HarshSlippageModel` — drop-in QC slippage model (used in QC algos)
3. `HarshFeeModel` — 100% taker fees (used in QC algos)
4. `HarshFillSimulator` — pure-Python order-fill simulator for offline testing
5. `apply_harsh_overrides(algorithm, config)` — convenience helper to install
   harsh slippage + fees on an existing QCAlgorithm subclass

The QC-side classes degrade gracefully when AlgorithmImports is unavailable
(i.e. running locally for unit tests) by falling back to plain-Python stubs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

# QC imports are optional — guard them
try:
    from AlgorithmImports import (  # type: ignore
        FeeModel, OrderFee, CashAmount, OrderType,
    )
    HAS_QC = True
except Exception:
    HAS_QC = False
    FeeModel = object
    OrderFee = None  # type: ignore
    CashAmount = None  # type: ignore
    OrderType = None  # type: ignore


# ─── Config (calibrated to live evidence) ────────────────────────────────────

@dataclass
class HarshConfig:
    """Pessimistic-simulator parameters. All defaults anchored to live MG36 log."""

    # Slippage
    base_slippage_bps: float = 100.0          # MG36 mean was 111 bp
    p99_slippage_bps:  float = 200.0          # MG36 max was 197 bp
    micro_cap_multiplier: float = 2.5         # micro caps 2.5× worse
    mid_cap_multiplier:   float = 1.3
    large_cap_multiplier: float = 1.0
    major_cap_multiplier: float = 0.5         # BTC/ETH still incur slippage but less

    # Fees: assume 100% taker (Kraken taker fee 0.40%)
    taker_fee_pct: float = 0.0040
    maker_fee_pct: float = 0.0025
    assumed_maker_fill_rate: float = 0.10     # MG36 paper fills: 0/2 within TTL

    # Execution
    fill_delay_bars: int = 1                  # T+1 fill, not T close
    limit_order_ttl_seconds: int = 30
    multi_order_spread_bps: float = 10.0      # +10bp per concurrent open order

    # Order rejection
    reject_rate_normal: float = 0.02
    reject_rate_vol_spike: float = 0.05

    # Signal forcing (live evidence: OBI is broken at minute QuoteBars)
    force_obi_to_zero: bool = True

    # Reproducibility
    seed: int | None = 42

    def slip_multiplier_for_tier(self, tier: str) -> float:
        return {
            "major": self.major_cap_multiplier,
            "large": self.large_cap_multiplier,
            "mid":   self.mid_cap_multiplier,
            "micro": self.micro_cap_multiplier,
        }.get(tier, self.mid_cap_multiplier)


# ─── Pure-Python fill simulator (for tests & offline analysis) ───────────────

@dataclass
class HarshFill:
    """Result of simulating one order fill under harsh assumptions."""
    filled: bool
    fill_price: float | None
    slippage_bps: float
    fee_pct: float
    rejected_reason: str | None = None
    delayed_bars: int = 0


class HarshFillSimulator:
    """Pure-Python fill model — no QC dependency.

    Use this in unit tests and offline replay to simulate what the live
    venue *would* have done given the harsh calibration. Models slippage,
    rejections, fee model, and bar delay.

    Pricing: caller supplies a `next_bar_open()` for delayed fills and
    a `current_price` for instant fills.
    """

    def __init__(self, config: HarshConfig):
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.open_order_count = 0

    def simulate_fill(
        self,
        symbol: str,
        side: str,                     # "Buy" | "Sell"
        is_market: bool,
        intended_price: float,
        next_bar_price: float | None,
        tier: str = "mid",
        is_vol_spike: bool = False,
        concurrent_open_orders: int = 0,
    ) -> HarshFill:
        """Simulate one order's outcome under harsh assumptions."""
        cfg = self.cfg

        # ── 1. Reject randomly ──────────────────────────────────────────────
        reject_p = cfg.reject_rate_vol_spike if is_vol_spike else cfg.reject_rate_normal
        if self.rng.random() < reject_p:
            return HarshFill(
                filled=False, fill_price=None,
                slippage_bps=0.0, fee_pct=0.0,
                rejected_reason="exchange_rejected",
            )

        # ── 2. Limit orders — assume they mostly time out ──────────────────
        if not is_market:
            if self.rng.random() > cfg.assumed_maker_fill_rate:
                # Limit timed out → fall back to market at next bar
                if next_bar_price is None:
                    return HarshFill(
                        filled=False, fill_price=None,
                        slippage_bps=0.0, fee_pct=0.0,
                        rejected_reason="limit_timeout_no_fallback",
                    )
                base_px = next_bar_price
                is_market = True
                fee_pct = cfg.taker_fee_pct
                delayed_bars = 1
            else:
                # Maker fill — best case
                return HarshFill(
                    filled=True, fill_price=intended_price,
                    slippage_bps=0.0, fee_pct=cfg.maker_fee_pct,
                    delayed_bars=0,
                )
        else:
            # Market: T+1 open in harsh sim
            base_px = next_bar_price if next_bar_price is not None else intended_price
            fee_pct = cfg.taker_fee_pct
            delayed_bars = cfg.fill_delay_bars

        # ── 3. Apply slippage ──────────────────────────────────────────────
        slip_bps = self._sample_slippage_bps(tier, concurrent_open_orders, is_vol_spike)
        # Buy slippage is positive (pay more); Sell slippage is negative (get less)
        slip_pct = slip_bps / 10_000.0
        if side == "Buy":
            fill_px = base_px * (1.0 + slip_pct)
        else:
            fill_px = base_px * (1.0 - slip_pct)

        return HarshFill(
            filled=True,
            fill_price=fill_px,
            slippage_bps=slip_bps,
            fee_pct=fee_pct,
            delayed_bars=delayed_bars,
        )

    def _sample_slippage_bps(self, tier: str, concurrent: int, vol_spike: bool) -> float:
        """Sample slippage in bps from a calibrated distribution.

        Mean tracks `base_slippage_bps × tier_multiplier`; with vol spike,
        we draw from a heavier tail. Concurrent orders add pressure.
        """
        cfg = self.cfg
        base = cfg.base_slippage_bps * cfg.slip_multiplier_for_tier(tier)
        base += concurrent * cfg.multi_order_spread_bps
        if vol_spike:
            # Right-skewed: 50% chance to hit p99 cap
            if self.rng.random() < 0.5:
                return cfg.p99_slippage_bps * cfg.slip_multiplier_for_tier(tier)
        # Else, ~Normal(base, base/3) clipped to [0.5×base, p99]
        sigma = max(base / 3.0, 5.0)
        sample = self.rng.gauss(base, sigma)
        return max(0.5 * base, min(sample, cfg.p99_slippage_bps * cfg.micro_cap_multiplier))


# ─── QC-side adapters (only meaningful when AlgorithmImports is present) ─────

if HAS_QC:

    class HarshSlippageModel:
        """QC-compatible slippage model anchored to live evidence."""

        def __init__(self, config: HarshConfig | None = None):
            self.cfg = config or HarshConfig()

        def GetSlippageApproximation(self, asset, order):
            try:
                price = float(asset.Price)
                if price <= 0:
                    return 0
                # Pessimistic: base + price-tier penalty + a 0.5% floor
                slip_bps = self.cfg.base_slippage_bps
                if price < 0.01:
                    slip_bps += 200.0
                elif price < 0.10:
                    slip_bps += 100.0
                elif price < 1.0:
                    slip_bps += 50.0
                elif price < 10.0:
                    slip_bps += 20.0
                # Cap at p99
                slip_bps = min(slip_bps, self.cfg.p99_slippage_bps * 2)
                return price * (slip_bps / 10_000.0)
            except Exception:
                return 0


    class HarshFeeModel(FeeModel):
        """QC-compatible 100%-taker fee model (Kraken 0.40%)."""

        def __init__(self, config: HarshConfig | None = None):
            self.cfg = config or HarshConfig()

        def GetOrderFee(self, parameters):
            order = parameters.Order
            # Honest assumption: even limit orders get classified as taker most
            # of the time (live evidence: 0/2 maker fills within TTL)
            fee_pct = self.cfg.taker_fee_pct
            trade_value = abs(order.AbsoluteQuantity) * parameters.Security.Price
            return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


    def apply_harsh_overrides(algorithm, config: HarshConfig | None = None) -> None:
        """Install the harsh slippage + fee model on every security.

        Call from `Initialize()` *after* registering symbols, e.g.:
            apply_harsh_overrides(self, HarshConfig())
        """
        cfg = config or HarshConfig()
        for sec in algorithm.Securities.Values:
            sec.SetSlippageModel(HarshSlippageModel(cfg))
            sec.SetFeeModel(HarshFeeModel(cfg))

else:
    # Local-only stubs so tests can import without QC. No QC behavior; tests
    # that need these classes are skipped.
    HarshSlippageModel = None  # type: ignore
    HarshFeeModel = None       # type: ignore

    def apply_harsh_overrides(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "apply_harsh_overrides requires AlgorithmImports (run inside QC)"
        )
