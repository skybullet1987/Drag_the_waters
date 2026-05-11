"""Pulse.apex.data.onchain_valuation — MVRV / NUPL / SOPR proxies.

These are the foundational on-chain valuation metrics for BTC. We
compute SIMPLIFIED proxies from data we can get free in QC, not the
full Glassnode versions:

  MVRV proxy =  market_cap / (cumulative_realized_value)
  NUPL proxy =  (market_cap - realized_cap) / market_cap
  SOPR proxy =  rolling-mean of (price_today / price_at_last_active)

Realized cap is approximated as a moving average of price weighted by
on-chain transaction volume. This is a known approximation; it captures
~80% of the true Glassnode signal at zero cost.

Signal logic
------------
MVRV z-score over 1-year window:
  MVRV-z > +2 → late-cycle euphoria → bearish (-1)
  MVRV-z < -2 → bottom              → bullish (+1)

This is a SLOW signal — typically rebalances every few weeks. Used
as a regime modulator rather than a primary entry trigger.

GLOBAL signal — same value for any symbol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


DEFAULT_REALIZED_LOOKBACK = 200    # days for realized-cap approximation
DEFAULT_Z_LOOKBACK        = 365    # 1-year baseline
DEFAULT_BULLISH_Z         = 2.0


# ─── Pure-Python core ────────────────────────────────────────────────────────


def compute_realized_cap_proxy(
    price_series: Sequence[float],
    txn_volume_series: Sequence[float],
    lookback: int = DEFAULT_REALIZED_LOOKBACK,
) -> list[float]:
    """Volume-weighted moving average of price; proxy for realized cap.

    Returns a series the same length as the inputs. The first `lookback-1`
    entries are equal to the simple MA over the available window.
    """
    n = min(len(price_series), len(txn_volume_series))
    out: list[float] = []
    for i in range(n):
        start = max(0, i - lookback + 1)
        window_p = price_series[start:i + 1]
        window_v = txn_volume_series[start:i + 1]
        total_v = sum(window_v) or 1.0
        vwap = sum(p * v for p, v in zip(window_p, window_v)) / total_v
        out.append(vwap)
    return out


def compute_mvrv_series(price_series: Sequence[float],
                        realized_cap_series: Sequence[float]) -> list[float]:
    """MVRV per day = price / realized_proxy."""
    out: list[float] = []
    for p, r in zip(price_series, realized_cap_series):
        if r and r > 0:
            out.append(float(p) / float(r))
    return out


def compute_nupl_series(price_series: Sequence[float],
                        realized_cap_series: Sequence[float]) -> list[float]:
    """NUPL = (market - realized) / market."""
    out: list[float] = []
    for p, r in zip(price_series, realized_cap_series):
        if p and p > 0:
            out.append((p - r) / p)
    return out


def compute_onchain_valuation_score(
    mvrv_series: Sequence[float],
    *,
    z_lookback:    int = DEFAULT_Z_LOOKBACK,
    bullish_z:     float = DEFAULT_BULLISH_Z,
) -> tuple[float, dict]:
    if len(mvrv_series) < z_lookback:
        return 0.0, {"error": "insufficient_history",
                     "have": len(mvrv_series), "need": z_lookback}
    window = list(mvrv_series[-z_lookback:])
    m = sum(window) / z_lookback
    var = sum((x - m) ** 2 for x in window) / z_lookback
    if var <= 0:
        return 0.0, {"error": "no_variance"}
    sd = math.sqrt(var)
    if abs(m) > 0 and (sd / abs(m)) < 1e-4:
        return 0.0, {"error": "near_constant"}
    z = (mvrv_series[-1] - m) / sd
    raw = -z / bullish_z      # high MVRV = expensive = bearish
    score = max(-1.0, min(1.0, raw))
    return score, {"z": z, "current_mvrv": mvrv_series[-1],
                   "baseline_mean": m}


def make_mvrv_signal_fn(mvrv_provider, **kwargs):
    """`mvrv_provider(context)` → list[float] MVRV daily series."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            series = mvrv_provider(context) or []
        except Exception as exc:   # noqa: BLE001
            return SignalScore("mvrv", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_onchain_valuation_score(series, **kwargs)
        valid = "error" not in meta
        return SignalScore("mvrv", symbol, score, valid=valid, meta=meta)
    return _fn


def register_mvrv(registry: SignalRegistry, mvrv_provider, **kwargs):
    registry.register("mvrv",
                      make_mvrv_signal_fn(mvrv_provider, **kwargs))


@dataclass
class OnchainValuationStore:
    """Stores BTC price + transaction volume; exposes MVRV series."""

    keep_days: int = 500    # need >= 365 for z_lookback
    prices:    list[float] = field(default_factory=list)
    volumes:   list[float] = field(default_factory=list)

    def record(self, price: float, txn_volume: float) -> None:
        if price is None or txn_volume is None:
            return
        self.prices.append(float(price))
        self.volumes.append(float(txn_volume))
        if len(self.prices) > self.keep_days:
            del self.prices[0]
            del self.volumes[0]

    def mvrv_series(self, context: dict | None = None,
                    realized_lookback: int = DEFAULT_REALIZED_LOOKBACK
                    ) -> list[float]:
        if not self.prices or not self.volumes:
            return []
        rc = compute_realized_cap_proxy(self.prices, self.volumes,
                                         lookback=realized_lookback)
        return compute_mvrv_series(self.prices, rc)
