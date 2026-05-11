"""Pulse.apex.data.stablecoin_supply — USDT/USDC mint/burn signal.

Source: pre-computed via Etherscan ERC-20 events (free, 5 req/sec).
The offline tool `apex_data/fetch_stablecoin_supply.py` writes a CSV of
daily total supply, columns: Date, USDT_supply_usd, USDC_supply_usd.

Signal logic
------------
For each stablecoin compute the 3-day net Δ supply.
Score = z-score of total Δ against the 30-day distribution.

  z > +1.5 → strong minting → fresh capital entering crypto → bullish
  z < -1.5 → strong burning → capital leaving crypto       → bearish

This is a GLOBAL signal — same score for every symbol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


DEFAULT_DELTA_LOOKBACK = 3
DEFAULT_BASELINE_LOOKBACK = 30
DEFAULT_BULLISH_Z = 1.5


def _delta(series: Sequence[float], lookback: int) -> float | None:
    if len(series) < lookback + 1:
        return None
    return float(series[-1]) - float(series[-1 - lookback])


def _all_deltas(series: Sequence[float], lookback: int,
                window: int) -> list[float]:
    n = len(series)
    if n < lookback + 1:
        return []
    start = max(lookback, n - window)
    return [series[i] - series[i - lookback] for i in range(start, n)]


def compute_stablecoin_score(
    usdt_supply: Sequence[float],
    usdc_supply: Sequence[float],
    *,
    delta_lookback:    int = DEFAULT_DELTA_LOOKBACK,
    baseline_lookback: int = DEFAULT_BASELINE_LOOKBACK,
    bullish_z:         float = DEFAULT_BULLISH_Z,
) -> tuple[float, dict]:
    cur_total = (_delta(usdt_supply, delta_lookback) or 0.0) + \
                (_delta(usdc_supply, delta_lookback) or 0.0)
    baseline = []
    for s in (usdt_supply, usdc_supply):
        baseline.extend(_all_deltas(s, delta_lookback, baseline_lookback))
    if len(baseline) < 5:
        return 0.0, {"error": "insufficient_history",
                     "have": len(baseline)}
    m = sum(baseline) / len(baseline)
    var = sum((x - m) ** 2 for x in baseline) / len(baseline)
    if var <= 0:
        return 0.0, {"error": "no_variance"}
    sd = math.sqrt(var)
    if abs(m) > 0 and (sd / abs(m)) < 1e-4:
        return 0.0, {"error": "near_constant"}
    z = (cur_total - m) / sd
    score = max(-1.0, min(1.0, z / bullish_z))
    return score, {"z": z, "cur_total_delta": cur_total,
                   "baseline_mean_delta": m}


def make_stablecoin_signal_fn(supply_provider, **kwargs):
    """`supply_provider(context)` → (usdt_series, usdc_series)."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            ut, uc = supply_provider(context)
        except Exception as exc:   # noqa: BLE001
            return SignalScore("stablecoin_mint", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_stablecoin_score(ut or [], uc or [], **kwargs)
        valid = "error" not in meta
        return SignalScore("stablecoin_mint", symbol, score,
                           valid=valid, meta=meta)
    return _fn


def register_stablecoin(registry: SignalRegistry, supply_provider, **kwargs):
    registry.register("stablecoin_mint",
                      make_stablecoin_signal_fn(supply_provider, **kwargs))


@dataclass
class StablecoinSupplyStore:
    keep_days: int = 60
    usdt: list[float] = field(default_factory=list)
    usdc: list[float] = field(default_factory=list)

    def record(self, usdt_supply_usd: float | None,
               usdc_supply_usd: float | None) -> None:
        if usdt_supply_usd is not None:
            self.usdt.append(float(usdt_supply_usd))
            if len(self.usdt) > self.keep_days:
                del self.usdt[0]
        if usdc_supply_usd is not None:
            self.usdc.append(float(usdc_supply_usd))
            if len(self.usdc) > self.keep_days:
                del self.usdc[0]

    def load_csv(self, content: str) -> None:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("Date") or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                ut = float(parts[1]) if parts[1] else None
                uc = float(parts[2]) if parts[2] else None
                self.record(ut, uc)
            except ValueError:
                continue

    def get(self, context: dict | None = None) -> tuple[list[float], list[float]]:
        return list(self.usdt), list(self.usdc)
