"""Pulse.apex.data.news_sentiment — pre-computed news sentiment signal.

Source: pre-computed via apex_data/score_news_sentiment.py from the
free CoinDesk + CoinTelegraph + r/cryptocurrency RSS feeds.
Per-headline scoring with VADER (NLTK, free) — no LLM API needed.

CSV format (one row per (date, symbol)):
  Date,Symbol,Articles,SentimentMean,SentimentStd

Signal logic
------------
For each symbol look at the last `lookback_days` (default 3) of the
sentiment series. Compute z-score of the recent mean against a 30-day
baseline.

  z > +1.5 → unusually bullish news flow → +1
  z < -1.5 → unusually bearish news flow → -1

Rationale: VADER's absolute scores are biased toward neutral; the
z-score normalizes per-symbol baseline (e.g. always-controversial coins).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


DEFAULT_RECENT_LOOKBACK   = 3   # days
DEFAULT_BASELINE_LOOKBACK = 30
DEFAULT_BULLISH_Z         = 1.5


def parse_sentiment_csv(content: str) -> dict[str, list[float]]:
    """Returns {symbol: [daily_mean_sentiment, ...]} ordered by date asc.

    Skips header rows and malformed lines.
    """
    rows: list[tuple[str, str, float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("Date") or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            d = parts[0]
            sym = parts[1].upper()
            mean = float(parts[3])
            rows.append((d, sym, mean))
        except (ValueError, IndexError):
            continue
    # Group by symbol, preserving date order (CSV is assumed sorted)
    out: dict[str, list[float]] = {}
    for _, sym, mean in rows:
        out.setdefault(sym, []).append(mean)
    return out


def compute_news_sentiment_score(
    series: Sequence[float],
    *,
    recent_lookback:    int = DEFAULT_RECENT_LOOKBACK,
    baseline_lookback:  int = DEFAULT_BASELINE_LOOKBACK,
    bullish_z:          float = DEFAULT_BULLISH_Z,
) -> tuple[float, dict]:
    if len(series) < baseline_lookback:
        return 0.0, {"error": "insufficient_history",
                     "have": len(series)}
    if len(series) < recent_lookback:
        return 0.0, {"error": "no_recent"}
    recent_mean = sum(series[-recent_lookback:]) / recent_lookback
    baseline = list(series[-baseline_lookback:])
    m = sum(baseline) / baseline_lookback
    var = sum((x - m) ** 2 for x in baseline) / baseline_lookback
    if var <= 0:
        return 0.0, {"error": "no_variance"}
    sd = math.sqrt(var)
    if sd < 1e-6:
        return 0.0, {"error": "near_constant"}
    z = (recent_mean - m) / sd
    score = max(-1.0, min(1.0, z / bullish_z))
    return score, {"z": z, "recent_mean": recent_mean,
                   "baseline_mean": m}


def make_news_sentiment_signal_fn(series_provider, **kwargs):
    """`series_provider(symbol, context)` → list[float] daily mean sentiments."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            series = series_provider(symbol, context) or []
        except Exception as exc:   # noqa: BLE001
            return SignalScore("news_sentiment", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_news_sentiment_score(series, **kwargs)
        valid = "error" not in meta
        return SignalScore("news_sentiment", symbol, score,
                           valid=valid, meta=meta)
    return _fn


def register_news_sentiment(registry: SignalRegistry,
                             series_provider, **kwargs):
    registry.register("news_sentiment",
                      make_news_sentiment_signal_fn(series_provider, **kwargs))


@dataclass
class NewsSentimentStore:
    by_symbol: dict[str, list[float]] = field(default_factory=dict)

    def record(self, symbol: str, daily_mean: float) -> None:
        s = symbol.upper()
        self.by_symbol.setdefault(s, []).append(float(daily_mean))

    def load_csv(self, content: str) -> None:
        loaded = parse_sentiment_csv(content)
        for sym, series in loaded.items():
            self.by_symbol.setdefault(sym, []).extend(series)

    def get(self, symbol: str, context: dict | None = None) -> list[float]:
        return list(self.by_symbol.get(symbol.upper(), []))
