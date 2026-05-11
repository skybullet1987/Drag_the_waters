"""Pulse.apex.signals.btc_onchain — Bitcoin on-chain activity signal.

Source: QC's free `BitcoinMetadata` dataset (Blockchain.com, daily,
since 2009). 23 metrics; we use a subset that has predictive power
for short-term BTC returns:

  - hash_rate              network security; rising = miner conviction
  - n_unique_addresses     daily active addresses; rising = adoption
  - miners_revenue         total USD revenue paid to miners
  - estimated_btc_sent     transaction volume in BTC

Signal logic
------------
For each of the four metrics, compute a 14-day z-score against a
60-day rolling mean. Z-scores > +1 are bullish for BTC (and by
correlation, bullish for the alt market). Z-scores < -1 are bearish.

The four sub-scores are combined with equal weights and clamped to
[-1, +1] for the registry.

This signal is GLOBAL (returns the same score for any symbol).
Symbol-specific signals live in their own modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


# ─── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_SHORT_LOOKBACK   = 14    # days for "recent" mean
DEFAULT_LONG_LOOKBACK    = 60    # days for baseline mean & std
DEFAULT_BULLISH_Z        = 1.0   # |z| beyond which we score ±1
DEFAULT_METRIC_WEIGHTS = {
    "hash_rate":              0.30,
    "n_unique_addresses":     0.30,
    "miners_revenue":         0.20,
    "estimated_btc_sent":     0.20,
}


# ─── Pure-Python core ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BTCOnchainSnapshot:
    """One day of BTC on-chain metrics. None where the metric is missing."""
    hash_rate:              float | None = None
    n_unique_addresses:     float | None = None
    miners_revenue:         float | None = None
    estimated_btc_sent:     float | None = None


def _z_score(series: Sequence[float], short_n: int, long_n: int) -> float | None:
    """Return z-score of recent `short_n`-mean against the `long_n` window.

    Returns None when the series is shorter than long_n or has zero variance.
    """
    if len(series) < long_n:
        return None
    window = list(series[-long_n:])
    if any(x is None for x in window):  # None entries poison the stat
        return None
    if len(series) < short_n:
        return None
    recent = list(series[-short_n:])
    long_mean = sum(window) / long_n
    long_var = sum((x - long_mean) ** 2 for x in window) / long_n
    if long_var <= 0:
        return None
    long_sd = math.sqrt(long_var)
    # Floor: require the relative std (CV) to be at least 0.001 (0.1%)
    # so we don't manufacture huge z-scores from float noise on near-flat
    # series (e.g. test fixtures with steady drift).
    if abs(long_mean) > 0 and (long_sd / abs(long_mean)) < 1e-4:
        return None
    short_mean = sum(recent) / short_n
    return (short_mean - long_mean) / long_sd


def _z_to_unit(z: float | None, bullish_z: float = DEFAULT_BULLISH_Z) -> float:
    """Map a z-score to [-1, +1]: at |z|=bullish_z we hit ±1."""
    if z is None:
        return 0.0
    scaled = z / bullish_z
    return max(-1.0, min(1.0, scaled))


def compute_btc_onchain_score(
    history: dict[str, Sequence[float]],
    *,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    bullish_z:      float = DEFAULT_BULLISH_Z,
    weights:        dict[str, float] | None = None,
) -> tuple[float, dict]:
    """Compose the four metric z-scores into one signal.

    Args
    ----
    history : dict mapping metric name → daily time-series (oldest → newest)
              Missing keys are treated as zero contribution.

    Returns
    -------
    (score, meta) where score ∈ [-1, +1] and meta is a per-metric breakdown.
    """
    weights = weights or DEFAULT_METRIC_WEIGHTS
    contributions: dict[str, float] = {}
    raw_z: dict[str, float | None] = {}
    total = 0.0
    total_weight = 0.0
    for metric, w in weights.items():
        series = history.get(metric, [])
        z = _z_score(series, short_lookback, long_lookback)
        unit = _z_to_unit(z, bullish_z)
        contributions[metric] = unit
        raw_z[metric] = z
        if z is not None:
            total += unit * w
            total_weight += w
    if total_weight <= 0:
        return 0.0, {"raw_z": raw_z, "contributions": contributions,
                     "valid_metrics": 0}
    score = total / total_weight
    return max(-1.0, min(1.0, score)), {
        "raw_z": raw_z,
        "contributions": contributions,
        "valid_metrics": sum(1 for z in raw_z.values() if z is not None),
    }


def make_btc_onchain_signal_fn(
    history_provider,
    *,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    bullish_z:      float = DEFAULT_BULLISH_Z,
    weights:        dict[str, float] | None = None,
):
    """Build a callable suitable for SignalRegistry.register("btc_onchain", ...).

    `history_provider` is callable(symbol, context) → dict of metric→series.
    The signal is GLOBAL, but the contract still expects a symbol arg.
    """
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            history = history_provider(symbol, context) or {}
        except Exception as exc:   # noqa: BLE001
            return SignalScore("btc_onchain", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_btc_onchain_score(
            history,
            short_lookback=short_lookback,
            long_lookback=long_lookback,
            bullish_z=bullish_z,
            weights=weights,
        )
        valid = meta.get("valid_metrics", 0) > 0
        return SignalScore("btc_onchain", symbol, score, valid=valid, meta=meta)
    return _fn


def register_btc_onchain(registry: SignalRegistry, history_provider, **kwargs):
    """Convenience: register the signal under its canonical name."""
    registry.register("btc_onchain",
                      make_btc_onchain_signal_fn(history_provider, **kwargs))


# ─── QC adapter (only meaningful inside QC) ─────────────────────────────────

try:
    from AlgorithmImports import *  # type: ignore  # noqa: F401, F403
    HAS_QC = True
except Exception:
    HAS_QC = False


class BitcoinMetadataHistoryStore:
    """In-process rolling buffer of BitcoinMetadata daily ticks.

    Maintain a deque-like list of the last N days for each tracked metric.
    Pulse.apex.main is responsible for feeding it via `update_from_slice()`.

    This class is testable offline by directly calling `record_bar()` —
    no QC dependency at construction.
    """

    DEFAULT_KEEP_DAYS = 90

    def __init__(self, keep_days: int = DEFAULT_KEEP_DAYS) -> None:
        self.keep_days = max(int(keep_days), 1)
        self._series: dict[str, list[float]] = {
            "hash_rate":          [],
            "n_unique_addresses": [],
            "miners_revenue":     [],
            "estimated_btc_sent": [],
        }

    def record_bar(self, metric: str, value: float) -> None:
        if metric not in self._series:
            return
        self._series[metric].append(float(value))
        if len(self._series[metric]) > self.keep_days:
            del self._series[metric][0]

    def history(self, symbol: str = "", context: dict | None = None) -> dict[str, list[float]]:
        # symbol/context unused; this is a global history
        return {k: list(v) for k, v in self._series.items()}

    if HAS_QC:
        def update_from_slice(self, slice_) -> None:
            """Pull the latest BitcoinMetadata bar from the QC slice (if any)."""
            from QuantConnect.DataSource import BitcoinMetadata  # type: ignore
            # In live/backtest, the slice contains BitcoinMetadata keyed by symbol
            for sym, bar in (slice_.Get(BitcoinMetadata) or {}).items():
                # Map QC field names to our stored series
                if hasattr(bar, "HashRate") and bar.HashRate is not None:
                    self.record_bar("hash_rate", float(bar.HashRate))
                if hasattr(bar, "NumberUniqueAddressesUsed") and bar.NumberUniqueAddressesUsed is not None:
                    self.record_bar("n_unique_addresses",
                                    float(bar.NumberUniqueAddressesUsed))
                if hasattr(bar, "MinersRevenueUsd") and bar.MinersRevenueUsd is not None:
                    self.record_bar("miners_revenue", float(bar.MinersRevenueUsd))
                if hasattr(bar, "EstimatedBtcSent") and bar.EstimatedBtcSent is not None:
                    self.record_bar("estimated_btc_sent",
                                    float(bar.EstimatedBtcSent))
