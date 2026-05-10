"""trend_engine — HYDRA done right (sub-strategy 2 of the multi-strategy portfolio).

HYDRA-100x lost money because:
  - Single-confirmation regime (golden cross alone whipsaws in chop)
  - 4% fixed SL (too tight for daily alts)
  - Bear-flip exited only losers, kept winners (which became next-week losers)
  - 3-coin basket = single-name blow-up risk
  - No correlation cap (basket often held 3 alts all 0.9 corr to BTC)

This module rebuilds with the fixes from PLAN.md §6.A.1:

  1. Dual-confirmation regime: golden cross AND BTC closes above SMA200
     for ≥ 5 consecutive days before flipping to bull
  2. ATR-aware SL: max(4%, 1.5 × daily ATR%); typically 5-9% on alts
  3. Bear-flip = full liquidation (NOT just exit losers)
  4. Basket size 5 with correlation cap (reject if 60d return correlation
     with held position > 0.85)
  5. Risk-adjusted momentum ranking: 30d_return / 30d_realized_vol

Pure-Python compute_basket() returns the recommended portfolio. Caller
(StrategyAllocator) decides how to execute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ─── Defaults (will be walk-forward tuned in Phase 3) ──────────────────────

DEFAULT_BASKET_SIZE             = 5
DEFAULT_MOMENTUM_LOOKBACK_BARS  = 30      # daily bars
DEFAULT_VOL_LOOKBACK_BARS       = 30
DEFAULT_MIN_RISK_ADJ_MOMENTUM   = 0.5     # min ret/vol Sharpe-like
DEFAULT_REGIME_CONFIRM_DAYS     = 5       # bull regime requires N consecutive days
DEFAULT_MAX_CORRELATION_TO_HELD = 0.85    # cap pairwise correlation
DEFAULT_ATR_SL_MULT             = 1.5     # SL = max(4%, 1.5 × daily ATR%)
DEFAULT_FIXED_SL_FLOOR_PCT      = 0.04    # never tighter than 4%
DEFAULT_BEAR_FLIP_LIQUIDATE_ALL = True


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class TrendCandidate:
    """Per-symbol summary for trend ranking."""
    symbol:                 str
    daily_closes:           Sequence[float]
    momentum_30d:           float = 0.0
    realized_vol_30d:       float = 0.0
    risk_adj_momentum:      float = 0.0
    daily_atr_pct:          float = 0.0
    is_above_sma50:         bool = False
    is_above_sma200:        bool = False


@dataclass
class TrendBasket:
    """Recommended portfolio composition + execution metadata."""
    selected:                list[TrendCandidate] = field(default_factory=list)
    weights:                 dict[str, float] = field(default_factory=dict)
    regime:                  str = "neutral"     # bull/bear/neutral/transition
    bull_confirm_days:       int = 0
    should_liquidate_all:    bool = False
    rejection_reasons:       dict[str, str] = field(default_factory=dict)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs):
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _daily_returns(closes: Sequence[float]) -> list[float]:
    if len(closes) < 2:
        return []
    return [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = _safe_mean(xs[-n:])
    my = _safe_mean(ys[-n:])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    sy2 = sum((ys[i] - my) ** 2 for i in range(n))
    den = math.sqrt(sx2 * sy2)
    return num / den if den > 0 else 0.0


# ─── Regime detection (dual-confirmation) ──────────────────────────────────

@dataclass
class TrendRegimeDecision:
    regime:                  str   # bull / bear / neutral / transition
    bull_confirm_days:       int
    should_liquidate_all:    bool


def detect_trend_regime(
    btc_daily_closes: Sequence[float],
    confirm_days: int = DEFAULT_REGIME_CONFIRM_DAYS,
    momentum_floor: float = -0.05,
    momentum_window: int = 30,
) -> TrendRegimeDecision:
    """Dual-confirmation regime per PLAN.md §6.A.1 fix #1.

    Bull: SMA50 > SMA200 AND BTC closes above SMA200 for ≥ confirm_days days
          AND 30d return > momentum_floor (-5%).
    Bear: SMA50 < SMA200 AND 30d return < -momentum_floor.
    Else: neutral / transition.

    The bull confirmation count is the number of consecutive days BTC has
    closed above SMA200 (looking back at most `confirm_days` days).
    """
    c = list(btc_daily_closes)
    if len(c) < 200:
        return TrendRegimeDecision("neutral", 0, False)

    sma50  = _safe_mean(c[-50:])
    sma200 = _safe_mean(c[-200:])

    # Compute the rolling SMA200 history for the last `confirm_days` bars
    confirm_days_count = 0
    for i in range(1, confirm_days + 1):
        if i > len(c):
            break
        # sma200 at bar -i needs c[-i-200:-i]
        if i + 200 > len(c):
            break
        sma200_i = _safe_mean(c[-(i + 200):-i])
        if c[-i] > sma200_i:
            confirm_days_count += 1
        else:
            break   # streak broken

    ret_30 = ((c[-1] - c[-momentum_window]) / c[-momentum_window]
              if len(c) >= momentum_window and c[-momentum_window] != 0 else 0.0)

    is_bull = (
        c[-1] > sma50
        and sma50 > sma200
        and confirm_days_count >= confirm_days
        and ret_30 > momentum_floor
    )
    is_bear = sma50 < sma200 and ret_30 < momentum_floor

    if is_bull:
        return TrendRegimeDecision("bull", confirm_days_count, False)
    if is_bear:
        return TrendRegimeDecision("bear", confirm_days_count,
                                   DEFAULT_BEAR_FLIP_LIQUIDATE_ALL)
    return TrendRegimeDecision("neutral", confirm_days_count, False)


# ─── ATR for daily bars ─────────────────────────────────────────────────────

def daily_atr_pct(highs: Sequence[float], lows: Sequence[float],
                  closes: Sequence[float], period: int = 14) -> float:
    """Average True Range (Wilder) as a fraction of last close.

    Returns ~0.05 = 5% if average daily range is 5%.
    """
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 0.0
    tr_list = []
    for i in range(n - period, n):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            )
        tr_list.append(tr)
    atr = _safe_mean(tr_list)
    return atr / closes[-1] if closes[-1] > 0 else 0.0


def trend_stop_loss_pct(
    daily_atr_fraction: float,
    atr_mult: float = DEFAULT_ATR_SL_MULT,
    floor_pct: float = DEFAULT_FIXED_SL_FLOOR_PCT,
) -> float:
    """SL = max(floor, atr_mult × daily_atr_pct).

    Per PLAN.md §6.A.1 fix #3: never tighter than 4%; typically 5-9% on alts.
    """
    return max(floor_pct, atr_mult * daily_atr_fraction)


# ─── Candidate scoring + ranking ────────────────────────────────────────────

def score_trend_candidate(
    symbol: str,
    daily_closes: Sequence[float],
    daily_highs: Sequence[float] | None = None,
    daily_lows:  Sequence[float] | None = None,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK_BARS,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK_BARS,
) -> TrendCandidate:
    """Compute risk-adjusted momentum + ATR for one candidate."""
    cand = TrendCandidate(symbol=symbol, daily_closes=daily_closes)
    n = len(daily_closes)

    if n < momentum_lookback + 1 or daily_closes[-momentum_lookback] == 0:
        return cand

    cand.momentum_30d = (
        (daily_closes[-1] - daily_closes[-momentum_lookback])
        / daily_closes[-momentum_lookback]
    )

    rets = _daily_returns(daily_closes[-vol_lookback - 1:])
    cand.realized_vol_30d = _safe_std(rets)
    cand.risk_adj_momentum = (
        cand.momentum_30d / cand.realized_vol_30d
        if cand.realized_vol_30d > 0 else 0.0
    )

    if daily_highs and daily_lows:
        cand.daily_atr_pct = daily_atr_pct(daily_highs, daily_lows, daily_closes)

    if n >= 50:
        cand.is_above_sma50 = daily_closes[-1] > _safe_mean(daily_closes[-50:])
    if n >= 200:
        cand.is_above_sma200 = daily_closes[-1] > _safe_mean(daily_closes[-200:])

    return cand


def rank_trend_candidates(
    candidates: list[TrendCandidate],
    min_risk_adj_momentum: float = DEFAULT_MIN_RISK_ADJ_MOMENTUM,
) -> list[TrendCandidate]:
    """Filter + sort candidates by risk-adjusted momentum.

    Filter rules:
      - Must have positive 30d momentum
      - Must be above SMA50 (in trend, not just bouncing)
      - Risk-adjusted momentum ≥ min_risk_adj_momentum
    """
    eligible = [
        c for c in candidates
        if c.momentum_30d > 0
        and c.is_above_sma50
        and c.risk_adj_momentum >= min_risk_adj_momentum
    ]
    eligible.sort(key=lambda c: c.risk_adj_momentum, reverse=True)
    return eligible


# ─── Correlation cap ────────────────────────────────────────────────────────

def _candidate_passes_correlation_cap(
    candidate: TrendCandidate,
    held: list[TrendCandidate],
    max_correlation: float = DEFAULT_MAX_CORRELATION_TO_HELD,
    correlation_window: int = 60,
) -> tuple[bool, str]:
    if not held:
        return True, ""
    cand_rets = _daily_returns(list(candidate.daily_closes)[-correlation_window:])
    if not cand_rets:
        return True, ""   # insufficient data; let through
    for h in held:
        h_rets = _daily_returns(list(h.daily_closes)[-correlation_window:])
        if not h_rets:
            continue
        corr = _pearson_correlation(cand_rets, h_rets)
        if abs(corr) >= max_correlation:
            return False, (
                f"corr_with_{h.symbol}={corr:.2f}>=cap={max_correlation:.2f}"
            )
    return True, ""


# ─── Main composer ──────────────────────────────────────────────────────────

def compute_basket(
    btc_daily_closes: Sequence[float],
    candidates: list[TrendCandidate],
    *,
    basket_size: int = DEFAULT_BASKET_SIZE,
    min_risk_adj_momentum: float = DEFAULT_MIN_RISK_ADJ_MOMENTUM,
    max_correlation: float = DEFAULT_MAX_CORRELATION_TO_HELD,
    confirm_days: int = DEFAULT_REGIME_CONFIRM_DAYS,
) -> TrendBasket:
    """End-to-end: regime gate → rank → correlation filter → vol-weighted basket.

    Returns a TrendBasket with selected coins, target weights, and any
    instruction to liquidate (e.g. on bear flip).
    """
    basket = TrendBasket()

    # 1. Regime gate
    regime_dec = detect_trend_regime(btc_daily_closes, confirm_days=confirm_days)
    basket.regime = regime_dec.regime
    basket.bull_confirm_days = regime_dec.bull_confirm_days
    basket.should_liquidate_all = regime_dec.should_liquidate_all

    if regime_dec.regime != "bull":
        # Not bull → empty basket; caller may liquidate if bear.
        for c in candidates:
            basket.rejection_reasons[c.symbol] = f"regime={regime_dec.regime}"
        return basket

    # 2. Rank by risk-adjusted momentum
    ranked = rank_trend_candidates(candidates, min_risk_adj_momentum)
    not_eligible = {c.symbol for c in candidates} - {c.symbol for c in ranked}
    for sym in not_eligible:
        basket.rejection_reasons[sym] = "weak_momentum_or_below_sma50"

    # 3. Correlation cap — greedy: pick best, then add only if uncorrelated
    selected: list[TrendCandidate] = []
    for cand in ranked:
        if len(selected) >= basket_size:
            break
        ok, reason = _candidate_passes_correlation_cap(
            cand, selected, max_correlation=max_correlation,
        )
        if ok:
            selected.append(cand)
        else:
            basket.rejection_reasons[cand.symbol] = reason
    basket.selected = selected

    # 4. Vol-weighted allocation (inverse-vol, cap 35% per coin)
    if selected:
        inv_vols = {c.symbol: (1.0 / c.realized_vol_30d
                                if c.realized_vol_30d > 0 else 1.0)
                     for c in selected}
        total = sum(inv_vols.values())
        weights = {sym: (iv / total if total > 0 else 1.0 / len(selected))
                   for sym, iv in inv_vols.items()}
        # Iteratively cap+redistribute so no single weight exceeds 0.35.
        # Capped weights are frozen; their excess is redistributed
        # proportionally among the still-uncapped coins.
        for _ in range(10):
            over = {s: w for s, w in weights.items() if w > 0.35 + 1e-9}
            if not over:
                break
            excess = sum(w - 0.35 for w in over.values())
            for s in over:
                weights[s] = 0.35
            uncapped = {s: w for s, w in weights.items() if s not in over}
            uncapped_sum = sum(uncapped.values())
            if uncapped_sum > 0:
                for s in uncapped:
                    weights[s] = weights[s] + excess * (weights[s] / uncapped_sum)
            else:
                # Everyone capped — equal distribution of the leftover
                share = excess / len(weights)
                for s in weights:
                    weights[s] = min(0.35, weights[s] + share)
        basket.weights = weights

    return basket
