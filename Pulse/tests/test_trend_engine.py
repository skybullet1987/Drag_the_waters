"""Tests for Pulse.trend_engine."""

from __future__ import annotations

import math

import pytest

from Pulse.trend_engine import (
    DEFAULT_BASKET_SIZE, DEFAULT_REGIME_CONFIRM_DAYS,
    TrendCandidate, TrendBasket, TrendRegimeDecision,
    detect_trend_regime, daily_atr_pct, trend_stop_loss_pct,
    score_trend_candidate, rank_trend_candidates,
    compute_basket,
    _pearson_correlation,
)


# ───────────────────────────────────────────────────────────────────────────────
# Regime detection (dual-confirmation)
# ───────────────────────────────────────────────────────────────────────────────

def test_regime_neutral_short_history():
    d = detect_trend_regime([100] * 50)
    assert d.regime == "neutral"
    assert d.bull_confirm_days == 0


def test_regime_bull_only_with_5_day_confirmation():
    """Steady uptrend long enough that BTC has been above SMA200 for 5+ days."""
    closes = [100 + i * 0.5 for i in range(250)]
    d = detect_trend_regime(closes, confirm_days=5)
    assert d.regime == "bull"
    assert d.bull_confirm_days >= 5


def test_regime_neutral_when_just_crossed():
    """BTC just popped above SMA200 today — confirm_days < 5 → not bull yet."""
    # Long downtrend then a 1-day pop above the slow SMA
    closes = [200 - i * 0.4 for i in range(220)] + [200.0]
    d = detect_trend_regime(closes, confirm_days=5)
    assert d.bull_confirm_days <= 1
    assert d.regime != "bull"


def test_regime_bear_on_sustained_downtrend():
    closes = [200 - i * 0.5 for i in range(250)]
    d = detect_trend_regime(closes)
    assert d.regime == "bear"
    assert d.should_liquidate_all  # PLAN.md fix #3


def test_regime_neutral_in_chop():
    """Choppy sideways — neither bull nor bear."""
    closes = [100 + 5 * math.sin(i * 0.2) for i in range(300)]
    d = detect_trend_regime(closes)
    assert d.regime == "neutral"


# ───────────────────────────────────────────────────────────────────────────────
# Daily ATR
# ───────────────────────────────────────────────────────────────────────────────

def test_atr_zero_on_short_history():
    assert daily_atr_pct([100], [99], [99.5]) == 0.0


def test_atr_basic_5pct():
    """20 bars with 5% daily range → ATR ≈ 5%."""
    n = 30
    closes = [100] * n
    highs  = [102.5] * n
    lows   = [97.5] * n
    atr = daily_atr_pct(highs, lows, closes, period=14)
    assert atr == pytest.approx(0.05, rel=0.05)


def test_trend_stop_loss_floor_4pct():
    """Even with tiny ATR, SL never tighter than 4%."""
    sl = trend_stop_loss_pct(daily_atr_fraction=0.005)   # 0.5% ATR
    assert sl == 0.04


def test_trend_stop_loss_atr_aware():
    """ATR 8% × 1.5 mult = 12% SL (above 4% floor)."""
    sl = trend_stop_loss_pct(daily_atr_fraction=0.08)
    assert sl == pytest.approx(0.12)


# ───────────────────────────────────────────────────────────────────────────────
# Score / rank candidates
# ───────────────────────────────────────────────────────────────────────────────

def _bull_candidate(symbol: str, vol_amp: float = 0.005, n: int = 250):
    """Synthetic bull-trend candidate."""
    import random
    rnd = random.Random(hash(symbol) % 2**32)
    closes = [100 + i * 0.5 + rnd.gauss(0, 100 * vol_amp) for i in range(n)]
    highs  = [c * 1.005 for c in closes]
    lows   = [c * 0.995 for c in closes]
    return score_trend_candidate(symbol, closes, highs, lows)


def test_score_returns_zero_on_short_history():
    c = score_trend_candidate("BTC", [100, 101, 102])
    assert c.momentum_30d == 0.0


def test_score_computes_momentum_and_vol():
    c = _bull_candidate("BTC")
    assert c.momentum_30d > 0
    assert c.realized_vol_30d > 0
    assert c.risk_adj_momentum > 0


def test_score_is_above_sma_flags():
    c = _bull_candidate("BTC")
    assert c.is_above_sma50
    assert c.is_above_sma200


def test_rank_filters_negative_momentum():
    closes = [100 - i * 0.5 for i in range(250)]   # downtrend
    c = score_trend_candidate("LOSER", closes, closes, closes)
    ranked = rank_trend_candidates([c])
    assert ranked == []


def test_rank_orders_by_risk_adj_momentum():
    """High vol candidates rank below low-vol same-return candidates."""
    smooth = _bull_candidate("SMOOTH", vol_amp=0.005)
    spiky  = _bull_candidate("SPIKY",  vol_amp=0.05)
    ranked = rank_trend_candidates([spiky, smooth])
    # Smooth has higher Sharpe-like ratio → ranks first
    if ranked:
        assert ranked[0].symbol == "SMOOTH"


# ───────────────────────────────────────────────────────────────────────────────
# Correlation
# ───────────────────────────────────────────────────────────────────────────────

def test_pearson_correlation_perfect():
    xs = [1, 2, 3, 4, 5]
    ys = [1, 2, 3, 4, 5]
    assert _pearson_correlation(xs, ys) == pytest.approx(1.0)


def test_pearson_correlation_inverse():
    xs = [1, 2, 3, 4, 5]
    ys = [5, 4, 3, 2, 1]
    assert _pearson_correlation(xs, ys) == pytest.approx(-1.0)


def test_pearson_correlation_zero_when_independent():
    xs = [1, 2, 3, 4, 5]
    ys = [3, 1, 4, 1, 5]    # roughly independent
    c = _pearson_correlation(xs, ys)
    assert -0.5 < c < 0.5


# ───────────────────────────────────────────────────────────────────────────────
# compute_basket end-to-end
# ───────────────────────────────────────────────────────────────────────────────

def test_compute_basket_empty_in_bear_regime():
    """Bear regime → empty basket + liquidate flag."""
    btc_closes = [200 - i * 0.5 for i in range(250)]
    candidates = [_bull_candidate(f"S{i}") for i in range(5)]
    b = compute_basket(btc_closes, candidates)
    assert b.regime == "bear"
    assert b.selected == []
    assert b.should_liquidate_all
    # All candidates rejected with regime reason
    for c in candidates:
        assert "regime=bear" in b.rejection_reasons.get(c.symbol, "")


def test_compute_basket_picks_top_5_in_bull_regime():
    btc_closes = [100 + i * 0.5 for i in range(250)]
    candidates = [_bull_candidate(f"COIN{i}") for i in range(8)]
    b = compute_basket(btc_closes, candidates)
    assert b.regime == "bull"
    assert len(b.selected) <= DEFAULT_BASKET_SIZE
    # Weights should sum to ~1.0
    assert sum(b.weights.values()) == pytest.approx(1.0, rel=0.01)


def test_compute_basket_correlation_cap_excludes():
    """Two perfectly correlated candidates: only one ends up in basket."""
    btc_closes = [100 + i * 0.5 for i in range(250)]
    # SOL and ETH made identical (will have correlation = 1.0)
    sol = score_trend_candidate(
        "SOL", [100 + i * 0.5 for i in range(250)],
        [101 + i * 0.5 for i in range(250)],
        [99 + i * 0.5 for i in range(250)],
    )
    eth = score_trend_candidate(
        "ETH", [100 + i * 0.5 for i in range(250)],
        [101 + i * 0.5 for i in range(250)],
        [99 + i * 0.5 for i in range(250)],
    )
    # Add a few uncorrelated bulls
    others = [_bull_candidate(f"DIVERSE{i}", vol_amp=0.02) for i in range(3)]
    b = compute_basket(btc_closes, [sol, eth] + others, basket_size=3)
    syms = [c.symbol for c in b.selected]
    # Either SOL or ETH made it; the other should be excluded for correlation
    assert not (("SOL" in syms) and ("ETH" in syms))


def test_compute_basket_weights_inv_vol_capped_at_35pct():
    """No coin should get more than 35% weight even if it has the lowest vol."""
    btc_closes = [100 + i * 0.5 for i in range(250)]
    cands = [_bull_candidate(f"CN{i}", vol_amp=0.005 + i * 0.01) for i in range(5)]
    b = compute_basket(btc_closes, cands)
    for w in b.weights.values():
        assert w <= 0.35 + 1e-6


def test_compute_basket_neutral_regime_empty():
    """Neutral regime (just-crossed bull) → no entries."""
    closes = [100 - i * 0.4 for i in range(220)] + [100.0]
    cands = [_bull_candidate(f"CN{i}") for i in range(5)]
    b = compute_basket(closes, cands)
    assert b.regime != "bull"
    assert b.selected == []


def test_compute_basket_bull_confirm_days_visible():
    btc_closes = [100 + i * 0.5 for i in range(250)]
    b = compute_basket(btc_closes, [])
    assert b.bull_confirm_days >= DEFAULT_REGIME_CONFIRM_DAYS
