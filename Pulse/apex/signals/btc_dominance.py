"""Pulse.apex.signals.btc_dominance — BTC.D as alt-season detector.

Source: QC's free `CoinGeckoCryptoMarketCap` dataset (daily, 620 coins).
We compute BTC dominance = BTC market cap / total market cap, then
score the recent dominance trend.

Signal logic
------------
  - dominance falling rapidly  → "alt season" → BULLISH for alts (+1)
  - dominance rising rapidly   → BTC absorbing flows → BEARISH for alts (-1)
  - dominance flat             → neutral (0)

For a BTC entry, the inverse holds.

Per-symbol return logic:
  symbol == BTCUSD                 → score = +rolling_change_z
  symbol in alts                   → score = -rolling_change_z

Where rolling_change_z = z-score of the (7-day - 30-day) dominance Δ
relative to a 90-day baseline.
"""

from __future__ import annotations

import math
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


# ─── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_SHORT_LOOKBACK = 7        # days
DEFAULT_MID_LOOKBACK   = 30
DEFAULT_LONG_LOOKBACK  = 90
DEFAULT_TREND_BULLISH_Z = 1.5     # |z| at which we hit ±1


# ─── Pure-Python core ────────────────────────────────────────────────────────


def compute_btc_dominance_series(btc_caps: Sequence[float],
                                  total_caps: Sequence[float]
                                  ) -> list[float]:
    """Per-day BTC dominance = btc_cap / total_cap. None entries skipped."""
    out: list[float] = []
    for b, t in zip(btc_caps, total_caps):
        if b is None or t is None or t <= 0:
            continue
        out.append(float(b) / float(t))
    return out


def _z_of_delta(short: float, mid: float, baseline_window: Sequence[float]
                ) -> float | None:
    """Z-score of (short - mid) Δ relative to baseline_window's std."""
    if len(baseline_window) < 5:
        return None
    m = sum(baseline_window) / len(baseline_window)
    var = sum((x - m) ** 2 for x in baseline_window) / len(baseline_window)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    # CV floor — see note in cross_asset._z_score
    if abs(m) > 0 and (sd / abs(m)) < 1e-4:
        return None
    return (short - mid) / sd


def compute_btc_dominance_score(
    dominance_series: Sequence[float],
    *,
    is_btc: bool,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    mid_lookback:   int = DEFAULT_MID_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    trend_bullish_z: float = DEFAULT_TREND_BULLISH_Z,
) -> tuple[float, dict]:
    """Compute the dominance trend signal.

    Returns
    -------
    (score, meta) where:
      score is positive when the regime is bullish for `symbol`
      score ∈ [-1, +1]
    """
    if len(dominance_series) < long_lookback:
        return 0.0, {"error": "insufficient_history",
                     "have": len(dominance_series),
                     "need": long_lookback}
    short_mean = sum(dominance_series[-short_lookback:]) / short_lookback
    mid_mean   = sum(dominance_series[-mid_lookback:])   / mid_lookback
    baseline   = list(dominance_series[-long_lookback:])
    z = _z_of_delta(short_mean, mid_mean, baseline)
    if z is None:
        return 0.0, {"error": "z_undefined"}
    # Δ > 0 → BTC dominance rising → bearish for alts, bullish for BTC
    raw = z / trend_bullish_z
    raw = max(-1.0, min(1.0, raw))
    score = raw if is_btc else -raw
    return score, {
        "short_mean": short_mean,
        "mid_mean":   mid_mean,
        "z":          z,
        "raw_signed": raw,
        "is_btc":     is_btc,
    }


# ─── Registry callable factory ──────────────────────────────────────────────

def _is_btc(symbol: str) -> bool:
    s = symbol.upper()
    return s.startswith("BTC") or s in {"XBTUSD", "WBTCUSD"}


def make_btc_dominance_signal_fn(
    series_provider,
    *,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    mid_lookback:   int = DEFAULT_MID_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    trend_bullish_z: float = DEFAULT_TREND_BULLISH_Z,
):
    """`series_provider` is callable(context) → list[float] dominance series."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            series = series_provider(context) or []
        except Exception as exc:   # noqa: BLE001
            return SignalScore("btc_dominance", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_btc_dominance_score(
            series, is_btc=_is_btc(symbol),
            short_lookback=short_lookback, mid_lookback=mid_lookback,
            long_lookback=long_lookback,
            trend_bullish_z=trend_bullish_z,
        )
        valid = "error" not in meta
        return SignalScore("btc_dominance", symbol, score,
                           valid=valid, meta=meta)
    return _fn


def register_btc_dominance(registry: SignalRegistry, series_provider, **kwargs):
    registry.register("btc_dominance",
                      make_btc_dominance_signal_fn(series_provider, **kwargs))


# ─── QC adapter (in-process accumulator) ─────────────────────────────────────


class CoinGeckoDominanceStore:
    """Rolling buffer of (btc_cap, total_cap) pairs → dominance series."""

    DEFAULT_KEEP_DAYS = 120

    def __init__(self, keep_days: int = DEFAULT_KEEP_DAYS) -> None:
        self.keep_days = max(int(keep_days), 1)
        self._btc:   list[float] = []
        self._total: list[float] = []

    def record(self, btc_cap: float, total_cap: float) -> None:
        if btc_cap is None or total_cap is None or total_cap <= 0:
            return
        self._btc.append(float(btc_cap))
        self._total.append(float(total_cap))
        if len(self._btc) > self.keep_days:
            del self._btc[0]
            del self._total[0]

    def dominance_series(self, context: dict | None = None) -> list[float]:
        return compute_btc_dominance_series(self._btc, self._total)
