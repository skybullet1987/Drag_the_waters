"""regime — multi-layer market regime classification for Pulse.

Three independent regime signals composed into a single sizing/gating layer:

1. ``MarketModeDetector``  (ported from Vox/market_mode.py)
   - 5-mode rules-based classification of BTC 4h price action:
     risk_on_trend / pump / chop / selloff / high_vol_reversal

2. ``GoldenCrossRegime``  (ported from HYDRA, used as size mult, not gate)
   - Daily SMA50 > SMA200 + 30d momentum > -5%
   - Returns "bull" / "neutral" / "bear"

3. ``BTCDominanceRegime``
   - Proxy: BTC_30d_return / mean(top10_alts_30d_return)
   - "btc_strong" → reduce alt exposure
   - "alt_strong" → increase alt exposure (rotation incoming)

Convenience:
- ``compose_regime_size_multiplier()`` — combines all three into a single
  [0.0, 1.5] sizing multiplier for any new alt-cap entry

All pure-Python. No QC dependency. Tests work offline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence


# ─── 1. MarketModeDetector (5-mode regime) ──────────────────────────────────

MARKET_MODES = ("risk_on_trend", "pump", "chop", "selloff", "high_vol_reversal")


def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs):
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def detect_market_mode(
    closes: Sequence[float],
    volumes: Sequence[float] | None = None,
) -> str:
    """Pure function — classify the regime from BTC 4h closes (and optional volumes).

    Same logic as Vox's MarketModeDetector but extracted as a pure function
    so it's trivially unit-testable.

    Returns one of MARKET_MODES.
    """
    c = list(closes)
    v = list(volumes) if volumes else []
    if len(c) < 5:
        return "chop"

    ret_4 = (c[-1] - c[-5]) / c[-5] if c[-5] != 0 else 0.0
    ret_1 = (c[-1] - c[-2]) / c[-2] if c[-2] != 0 else 0.0

    if len(c) >= 13:
        ret_12 = (c[-1] - c[-13]) / c[-13] if c[-13] != 0 else 0.0
    else:
        ret_12 = ret_4

    # SMA slope
    if len(c) >= 10:
        sma_now  = _safe_mean(c[-5:])
        sma_prev = _safe_mean(c[-10:-5])
        sma_slope = (sma_now - sma_prev) / sma_prev if sma_prev != 0 else 0.0
    else:
        sma_slope = ret_4

    # Volatility (std of last 8 rets)
    if len(c) >= 9:
        rets = []
        for i in range(-8, 0):
            if c[i - 1] != 0:
                rets.append((c[i] - c[i - 1]) / c[i - 1])
        vol = _safe_std(rets)
    else:
        vol = abs(ret_1)

    # Volume ratio
    vol_ratio = 1.0
    if len(v) >= 6:
        avg_v = _safe_mean(v[-6:-1])
        if avg_v > 0:
            vol_ratio = min(float(v[-1]) / avg_v, 10.0)

    # Range efficiency (trend purity)
    if len(c) >= 9:
        net_move  = abs(c[-1] - c[-9])
        sum_moves = sum(abs(c[i] - c[i - 1]) for i in range(-8, 0))
        range_eff = net_move / sum_moves if sum_moves > 0 else 0.0
    else:
        range_eff = 0.5

    # Classification rules
    if ret_4 < -0.04 and sma_slope < -0.01:
        return "selloff"
    if vol > 0.025 and range_eff < 0.30:
        return "high_vol_reversal"
    if ret_4 > 0.05 and vol_ratio > 2.0:
        return "pump"
    if ret_4 > 0.01 and sma_slope > 0.003 and range_eff > 0.40:
        return "risk_on_trend"
    return "chop"


class MarketModeDetector:
    """Stateful wrapper around `detect_market_mode` — keeps rolling 4h closes."""

    _WINDOW = 24   # 4 days of 4h bars

    def __init__(self):
        self._closes:  deque[float] = deque(maxlen=self._WINDOW + 4)
        self._volumes: deque[float] = deque(maxlen=self._WINDOW + 4)
        self._mode = "chop"

    def update_bar(self, close: float, volume: float = 0.0) -> str:
        self._closes.append(float(close))
        if volume > 0:
            self._volumes.append(float(volume))
        self._mode = detect_market_mode(list(self._closes), list(self._volumes))
        return self._mode

    @property
    def mode(self) -> str:
        return self._mode


# ─── 2. GoldenCrossRegime (HYDRA-style daily trend filter) ──────────────────

@dataclass
class GoldenCrossDecision:
    regime: str           # "bull" | "neutral" | "bear"
    sma_50: float
    sma_200: float
    ret_30d: float
    reason: str


def golden_cross_regime(
    daily_closes: Sequence[float],
    momentum_window: int = 30,
    momentum_floor_pct: float = -0.05,
) -> GoldenCrossDecision:
    """HYDRA-style classification:

        bull    if SMA50 > SMA200 AND 30d return > momentum_floor (-5%)
        bear    if SMA50 < SMA200 AND 30d return < -momentum_floor
        neutral otherwise
    """
    c = list(daily_closes)
    if len(c) < 200:
        return GoldenCrossDecision("neutral", 0, 0, 0, "insufficient_history")

    sma50  = _safe_mean(c[-50:])
    sma200 = _safe_mean(c[-200:])
    ret_30 = (c[-1] - c[-momentum_window]) / c[-momentum_window] \
             if len(c) >= momentum_window and c[-momentum_window] != 0 else 0.0

    bull = c[-1] > sma50 and sma50 > sma200 and ret_30 > momentum_floor_pct
    bear = sma50 < sma200 and ret_30 < -abs(momentum_floor_pct)
    if bull:
        regime = "bull"
        reason = "sma50>sma200 + ret30>floor"
    elif bear:
        regime = "bear"
        reason = "sma50<sma200 + ret30<-floor"
    else:
        regime = "neutral"
        reason = "transition"

    return GoldenCrossDecision(regime, sma50, sma200, ret_30, reason)


def golden_cross_size_multiplier(decision: GoldenCrossDecision) -> float:
    """Size multiplier from regime:
        bull    → 1.0  (full size)
        neutral → 0.5  (half)
        bear    → 0.25 (quarter)
    """
    return {"bull": 1.0, "neutral": 0.5, "bear": 0.25}.get(decision.regime, 0.5)


# ─── 3. BTCDominanceRegime ──────────────────────────────────────────────────

@dataclass
class BTCDominanceDecision:
    """Result of the BTC.D classification."""
    regime: str          # "btc_strong" | "neutral" | "alt_strong"
    btc_ret_30d: float
    alt_mean_ret_30d: float
    ratio: float


def btc_dominance_regime(
    btc_30d_return: float,
    alts_30d_returns: Sequence[float],
    btc_strong_ratio: float = 1.5,
    alt_strong_ratio: float = 0.7,
) -> BTCDominanceDecision:
    """Compute BTC dominance regime as a proxy ratio.

    ratio = btc_return / mean(alt_returns)
    ratio > btc_strong_ratio → BTC outperforming → reduce alt exposure
    ratio < alt_strong_ratio → alts outperforming → boost alt exposure

    Both inputs use 30-day returns by convention but caller can use any horizon.
    """
    if not alts_30d_returns:
        return BTCDominanceDecision("neutral", btc_30d_return, 0.0, 1.0)

    alt_mean = _safe_mean(alts_30d_returns)
    if alt_mean == 0 or btc_30d_return == 0:
        return BTCDominanceDecision("neutral", btc_30d_return, alt_mean, 1.0)

    # Use absolute ratio so direction sign doesn't flip the regime
    ratio = btc_30d_return / alt_mean

    # Both directions matter — ratio = 1.5 means BTC is 1.5x as strong as the
    # alt cohort in the same direction; ratio = -1 means they moved opposite.
    if ratio > btc_strong_ratio:
        regime = "btc_strong"
    elif 0 < ratio < alt_strong_ratio:
        regime = "alt_strong"
    elif ratio <= 0:
        # BTC moved opposite to alts → unclear regime, treat as alt-strong if
        # alts are positive (BTC weakness fuels alt rotation)
        regime = "alt_strong" if alt_mean > 0 else "btc_strong"
    else:
        regime = "neutral"

    return BTCDominanceDecision(regime, btc_30d_return, alt_mean, ratio)


def btc_dominance_alt_size_multiplier(decision: BTCDominanceDecision) -> float:
    """Alt-position size multiplier:
        btc_strong → 0.5 (BTC eating alt liquidity; reduce)
        neutral    → 1.0
        alt_strong → 1.5 (rotation; boost)
    """
    return {"btc_strong": 0.5, "neutral": 1.0, "alt_strong": 1.5}.get(
        decision.regime, 1.0
    )


# ─── Compose all three ──────────────────────────────────────────────────────

def compose_regime_size_multiplier(
    market_mode: str,
    golden_cross_dec: GoldenCrossDecision | None = None,
    btcd_dec: BTCDominanceDecision | None = None,
    is_alt: bool = True,
) -> dict:
    """Combine all 3 regime layers into one [0.0, 1.5] alt-position size mult.

    The composed multiplier is the product of:
        - market_mode multiplier (selloff=0, chop=0.5, risk_on=1.0, pump=1.2, hvr=0.7)
        - golden_cross multiplier (bull=1.0, neutral=0.5, bear=0.25)
        - btcd multiplier (btc_strong=0.5, neutral=1.0, alt_strong=1.5)

    Returns a dict with the final multiplier + each component for logging.
    """
    mm_mult = {
        "risk_on_trend":     1.0,
        "pump":              1.2,
        "chop":              0.5,
        "selloff":           0.0,
        "high_vol_reversal": 0.7,
    }.get(market_mode, 0.5)

    gc_mult = (golden_cross_size_multiplier(golden_cross_dec)
               if golden_cross_dec else 1.0)

    if is_alt and btcd_dec is not None:
        bd_mult = btc_dominance_alt_size_multiplier(btcd_dec)
    else:
        bd_mult = 1.0

    composed = mm_mult * gc_mult * bd_mult
    composed = max(0.0, min(1.5, composed))   # clamp

    return {
        "size_mult":            composed,
        "market_mode_mult":     mm_mult,
        "golden_cross_mult":    gc_mult,
        "btc_dominance_mult":   bd_mult,
        "market_mode":          market_mode,
        "golden_cross_regime": (golden_cross_dec.regime
                                if golden_cross_dec else "n/a"),
        "btc_dominance_regime": (btcd_dec.regime if btcd_dec else "n/a"),
    }
