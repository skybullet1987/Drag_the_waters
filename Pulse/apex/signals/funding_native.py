"""Pulse.apex.signals.funding_native — Binance funding rate extreme detector.

Source: QC's `BinanceFundingRate` Lean DataSource (per-symbol, 8h cadence).
Replaces our custom Bybit scrape with a properly-replayable native dataset.

Signal logic
------------
  - rate > +0.05% per 8h (annualized > +55%)  → crowded longs    → -1
  - rate < -0.05% per 8h                      → crowded shorts   → +1 (squeeze)
  - rate between -0.01% and +0.01%            → balanced         →  0
  - linear interpolation between extremes

The signal also supports z-score mode: instead of using fixed thresholds,
score the current rate against its own rolling distribution. Z > +2 = -1,
Z < -2 = +1. Z-mode is more robust across regimes (mature vs young coins
have very different baseline funding scales).

Per-symbol: each Kraken cash spot pair is mapped to its Binance perp
counterpart (BTCUSD → BTCUSDT, etc.). Symbols without a perp counterpart
(e.g. obscure alts) get score=0/valid=False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


# ─── Tunables ────────────────────────────────────────────────────────────────

# Fixed-threshold mode (per-8h decimal)
DEFAULT_DEEP_NEG     = -0.0005      # -0.05% per 8h  → score +1
DEFAULT_LIGHT_NEG    = -0.0001      # -0.01% per 8h  → ~+0.2
DEFAULT_LIGHT_POS    = +0.0001
DEFAULT_DEEP_POS     = +0.0005      # +0.05% per 8h  → score -1

# Z-score mode tunables
DEFAULT_Z_LOOKBACK   = 60      # samples (8h × 60 = 20 days)
DEFAULT_Z_BULLISH    = 2.0     # |z|≥2 → ±1


# ─── Pure-Python core ────────────────────────────────────────────────────────


def _piecewise_score(rate: float,
                     deep_neg: float = DEFAULT_DEEP_NEG,
                     light_neg: float = DEFAULT_LIGHT_NEG,
                     light_pos: float = DEFAULT_LIGHT_POS,
                     deep_pos: float = DEFAULT_DEEP_POS,
                     ) -> float:
    """Map raw funding rate to a piecewise-linear [-1, +1] score."""
    if rate <= deep_neg:
        return +1.0
    if rate >= deep_pos:
        return -1.0
    if light_neg <= rate <= light_pos:
        # Inside the dead-zone — interpolate gently around 0
        if abs(rate) <= 1e-9:
            return 0.0
        # Linear from (light_neg → +0.2) and (light_pos → -0.2)
        if rate < 0:
            return +0.2 * (rate / light_neg)
        return -0.2 * (rate / light_pos)
    # In the slopes
    if rate < light_neg:
        # Between deep_neg (=+1) and light_neg (=+0.2)
        frac = (rate - deep_neg) / (light_neg - deep_neg)
        return 1.0 - 0.8 * frac
    # rate > light_pos
    frac = (rate - light_pos) / (deep_pos - light_pos)
    return -0.2 - 0.8 * frac


def compute_funding_zscore(rate: float, history: Sequence[float],
                           bullish_z: float = DEFAULT_Z_BULLISH
                           ) -> float | None:
    if len(history) < 5:
        return None
    m = sum(history) / len(history)
    var = sum((x - m) ** 2 for x in history) / len(history)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    if sd < 1e-9:    # functionally constant funding rate
        return None
    return (rate - m) / sd


def compute_funding_score(
    rate: float | None,
    *,
    history: Sequence[float] | None = None,
    use_zscore: bool = True,
    bullish_z: float = DEFAULT_Z_BULLISH,
) -> tuple[float, dict]:
    """Returns (score ∈ [-1,1], meta).

    When use_zscore=True and history is sufficient, falls back to the
    piecewise rule only if the rolling window is too short.
    """
    if rate is None:
        return 0.0, {"error": "no_rate"}

    if use_zscore and history is not None and len(history) >= 5:
        z = compute_funding_zscore(rate, history, bullish_z=bullish_z)
        if z is not None:
            unit = -z / bullish_z   # high z = crowded long = bearish
            unit = max(-1.0, min(1.0, unit))
            return unit, {"mode": "zscore", "z": z, "rate": rate}

    score = _piecewise_score(rate)
    return score, {"mode": "piecewise", "rate": rate}


# ─── Symbol → Binance perp ticker mapping ────────────────────────────────────

KRAKEN_TO_BINANCE_PERP = {
    "BTCUSD":  "BTCUSDT",
    "ETHUSD":  "ETHUSDT",
    "SOLUSD":  "SOLUSDT",
    "XRPUSD":  "XRPUSDT",
    "BNBUSD":  "BNBUSDT",
    "ADAUSD":  "ADAUSDT",
    "DOGEUSD": "DOGEUSDT",
    "AVAXUSD": "AVAXUSDT",
    "DOTUSD":  "DOTUSDT",
    "LINKUSD": "LINKUSDT",
    "LTCUSD":  "LTCUSDT",
    "MATICUSD": "MATICUSDT",
    "ATOMUSD": "ATOMUSDT",
    "UNIUSD":  "UNIUSDT",
    "AAVEUSD": "AAVEUSDT",
    "NEARUSD": "NEARUSDT",
    "INJUSD":  "INJUSDT",
    "OPUSD":   "OPUSDT",
    "ARBUSD":  "ARBUSDT",
    "BCHUSD":  "BCHUSDT",
    "TRXUSD":  "TRXUSDT",
    "FETUSD":  "FETUSDT",
    "ICPUSD":  "ICPUSDT",
    "RENDERUSD": "RENDERUSDT",
    "HBARUSD": "HBARUSDT",
}


def kraken_to_perp(symbol: str) -> str | None:
    """Map Kraken cash pair → Binance perp ticker. None if unsupported."""
    return KRAKEN_TO_BINANCE_PERP.get(symbol.upper())


# ─── Registry callable factory ──────────────────────────────────────────────


def make_funding_signal_fn(
    rate_provider,
    *,
    use_zscore: bool = True,
    bullish_z: float = DEFAULT_Z_BULLISH,
):
    """`rate_provider(perp_symbol, context) → (current_rate, history)`."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        perp = kraken_to_perp(symbol)
        if perp is None:
            return SignalScore("funding_extreme", symbol, 0.0, valid=False,
                               meta={"error": "no_perp_mapping",
                                     "symbol": symbol})
        try:
            cur, history = rate_provider(perp, context)
        except Exception as exc:   # noqa: BLE001
            return SignalScore("funding_extreme", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_funding_score(
            cur, history=history,
            use_zscore=use_zscore, bullish_z=bullish_z,
        )
        meta["perp"] = perp
        valid = "error" not in meta
        return SignalScore("funding_extreme", symbol, score,
                           valid=valid, meta=meta)
    return _fn


def register_funding_native(registry: SignalRegistry, rate_provider, **kwargs):
    registry.register("funding_extreme",
                      make_funding_signal_fn(rate_provider, **kwargs))


# ─── In-process funding-rate store ──────────────────────────────────────────


@dataclass
class _PerSymbolFundingHistory:
    rates: list[float]
    keep:  int

    def add(self, r: float) -> None:
        self.rates.append(float(r))
        if len(self.rates) > self.keep:
            del self.rates[0]


class BinanceFundingRateStore:
    DEFAULT_KEEP = 120     # 60×8h ≈ 20 days; 120 = 40 days

    def __init__(self, keep: int = DEFAULT_KEEP) -> None:
        self.keep = keep
        self._by_perp: dict[str, _PerSymbolFundingHistory] = {}

    def record(self, perp_symbol: str, rate: float) -> None:
        if rate is None:
            return
        s = perp_symbol.upper()
        if s not in self._by_perp:
            self._by_perp[s] = _PerSymbolFundingHistory(rates=[], keep=self.keep)
        self._by_perp[s].add(float(rate))

    def get(self, perp_symbol: str, context: dict | None = None
            ) -> tuple[float | None, list[float]]:
        s = perp_symbol.upper()
        h = self._by_perp.get(s)
        if h is None or not h.rates:
            return None, []
        return h.rates[-1], list(h.rates[:-1])
