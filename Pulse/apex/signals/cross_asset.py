"""Pulse.apex.signals.cross_asset — macro risk-on/risk-off signal.

Sources (all native to QC equity/forex feeds):
  - DXY (US Dollar Index)   strong dollar = risk-off for crypto
  - GLD (gold ETF)           gold rallying = inflation/uncertainty regime
  - SPY (S&P 500 ETF)        equities up = risk-on, mostly bullish for crypto

Signal logic
------------
Compute 5-day return for each macro asset. Then:

    risk_on_score  =  +0.5 * spy_5d_return_z
                      -0.5 * dxy_5d_return_z
                      -0.2 * gld_5d_return_z

Where _z means z-scored against a 60-day rolling window.

Score is GLOBAL (returns same value regardless of symbol). For BTC,
score is multiplied by +1; for alts, by +1.2 (alts are higher beta to
risk-on regimes than BTC).

Returns 0/invalid if any of the three series is missing or too short.
"""

from __future__ import annotations

import math
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


# ─── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_RETURN_LOOKBACK = 5
DEFAULT_BASELINE_LOOKBACK = 60
DEFAULT_BULLISH_Z = 1.5
DEFAULT_ALT_BETA  = 1.2

DEFAULT_WEIGHTS = {
    "spy": +0.5,
    "dxy": -0.5,
    "gld": -0.2,
}


# ─── Pure-Python core ────────────────────────────────────────────────────────


def _pct_return(series: Sequence[float], lookback: int) -> float | None:
    if len(series) < lookback + 1:
        return None
    old = series[-lookback - 1]
    new = series[-1]
    if old <= 0:
        return None
    return (new - old) / old


def _rolling_returns(series: Sequence[float], lookback: int,
                     window: int) -> list[float]:
    """All `lookback`-day returns within the trailing `window` bars."""
    out: list[float] = []
    n = len(series)
    if n < lookback + 1:
        return out
    start = max(lookback, n - window)
    for i in range(start, n):
        old = series[i - lookback]
        new = series[i]
        if old <= 0:
            continue
        out.append((new - old) / old)
    return out


def _z_score(value: float, sample: Sequence[float]) -> float | None:
    """Z-score with a *meaningful* variance floor.

    A series of returns that's effectively constant (e.g. exponential
    growth at a fixed rate) shows tiny float-noise variance ~1e-32.
    Treat any sample with sd < 1e-6 as having no usable spread, since
    that means we're amplifying float artifacts rather than measuring
    a real distribution.
    """
    if len(sample) < 5:
        return None
    m = sum(sample) / len(sample)
    var = sum((x - m) ** 2 for x in sample) / len(sample)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    if sd < 1e-6:    # functionally constant series
        return None
    return (value - m) / sd


def compute_cross_asset_score(
    series: dict[str, Sequence[float]],
    *,
    is_alt: bool = True,
    return_lookback:   int = DEFAULT_RETURN_LOOKBACK,
    baseline_lookback: int = DEFAULT_BASELINE_LOOKBACK,
    bullish_z: float = DEFAULT_BULLISH_Z,
    alt_beta:  float = DEFAULT_ALT_BETA,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """Returns (score ∈ [-1, +1], meta).

    `series` is dict mapping {"spy", "dxy", "gld"} → daily price series
    (oldest → newest). Missing series → that asset's contribution = 0.
    """
    weights = weights or DEFAULT_WEIGHTS
    contributions: dict[str, float] = {}
    raw_z: dict[str, float | None] = {}
    total = 0.0
    used_weight = 0.0

    for asset, w in weights.items():
        s = series.get(asset)
        if not s:
            raw_z[asset] = None
            contributions[asset] = 0.0
            continue
        ret = _pct_return(s, return_lookback)
        if ret is None:
            raw_z[asset] = None
            contributions[asset] = 0.0
            continue
        sample = _rolling_returns(s, return_lookback, baseline_lookback)
        z = _z_score(ret, sample)
        if z is None:
            raw_z[asset] = None
            contributions[asset] = 0.0
            continue
        raw_z[asset] = z
        unit = max(-1.0, min(1.0, z / bullish_z))
        contributions[asset] = unit * w
        total += unit * w
        used_weight += abs(w)

    if used_weight <= 0:
        return 0.0, {"error": "all_assets_missing", "raw_z": raw_z,
                     "contributions": contributions}

    score = total / used_weight   # bounded to [-1, +1]
    if is_alt:
        score *= alt_beta
        score = max(-1.0, min(1.0, score))
    return score, {
        "raw_z":         raw_z,
        "contributions": contributions,
        "is_alt":        is_alt,
        "alt_beta":      alt_beta if is_alt else 1.0,
    }


# ─── Registry callable factory ──────────────────────────────────────────────

def _is_alt(symbol: str) -> bool:
    s = symbol.upper()
    return not (s.startswith("BTC") or s in {"XBTUSD", "WBTCUSD"})


def make_cross_asset_signal_fn(
    series_provider,
    *,
    return_lookback:   int = DEFAULT_RETURN_LOOKBACK,
    baseline_lookback: int = DEFAULT_BASELINE_LOOKBACK,
    bullish_z: float = DEFAULT_BULLISH_Z,
    alt_beta:  float = DEFAULT_ALT_BETA,
    weights: dict[str, float] | None = None,
):
    """`series_provider(context)` → dict of {"spy","dxy","gld"} → series."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            ser = series_provider(context) or {}
        except Exception as exc:   # noqa: BLE001
            return SignalScore("cross_asset_macro", symbol, 0.0,
                               valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_cross_asset_score(
            ser, is_alt=_is_alt(symbol),
            return_lookback=return_lookback,
            baseline_lookback=baseline_lookback,
            bullish_z=bullish_z, alt_beta=alt_beta, weights=weights,
        )
        valid = "error" not in meta
        return SignalScore("cross_asset_macro", symbol, score,
                           valid=valid, meta=meta)
    return _fn


def register_cross_asset(registry: SignalRegistry, series_provider, **kwargs):
    registry.register("cross_asset_macro",
                      make_cross_asset_signal_fn(series_provider, **kwargs))


# ─── Multi-asset rolling buffer (in-process) ────────────────────────────────


class CrossAssetSeriesStore:
    """Stores per-asset daily close series; consumed by series_provider."""

    DEFAULT_KEEP_DAYS = 90

    def __init__(self, keep_days: int = DEFAULT_KEEP_DAYS) -> None:
        self.keep_days = max(int(keep_days), 1)
        self._series: dict[str, list[float]] = {"spy": [], "dxy": [], "gld": []}

    def record(self, asset: str, close: float) -> None:
        a = asset.lower()
        if a not in self._series or close is None:
            return
        self._series[a].append(float(close))
        if len(self._series[a]) > self.keep_days:
            del self._series[a][0]

    def series(self, context: dict | None = None) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._series.items()}
