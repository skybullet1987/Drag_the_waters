"""Tests for Pulse.scalp_engine."""

from __future__ import annotations

import random

import pytest

from Pulse.scalp_engine import (
    SymbolBars, MarketContext, ScalpScore,
    compute_scalp_score, rank_candidates,
    DEFAULT_ENTRY_THRESHOLD, DEFAULT_HIGH_CONVICTION_THRES,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers — synthesize realistic OHLCV bars
# ───────────────────────────────────────────────────────────────────────────────

def _flat_bars(symbol="SOLUSD", n=80, price=100.0, volume=1000) -> SymbolBars:
    """Dead-flat market: low score expected."""
    return SymbolBars(
        symbol=symbol,
        opens=[price]   * n,
        highs=[price * 1.0001] * n,
        lows=[price  * 0.9999] * n,
        closes=[price] * n,
        volumes=[volume] * n,
    )


def _strong_uptrend_with_volume_burst(symbol="SOLUSD", n=80,
                                       start=100.0) -> SymbolBars:
    """Steady uptrend + volume spike on the latest bar — should fire.

    Also includes some normal volume variability so the z-score has a non-zero
    denominator (live data is never perfectly flat).
    """
    rnd = random.Random(0)
    closes = [start + 0.4 * i + rnd.gauss(0, 0.1) for i in range(n)]
    # Slow down the trend in the last 20 bars so RSI cools below 70
    for i in range(60, n - 1):
        closes[i] = closes[i - 1] + rnd.gauss(0.02, 0.15)
    opens  = [c - 0.2 for c in closes]
    highs  = [max(o, c) + 0.3 for o, c in zip(opens, closes)]
    lows   = [min(o, c) - 0.2 for o, c in zip(opens, closes)]
    # Volumes: realistic background variability + final-bar burst
    volumes = [1000 + rnd.gauss(0, 100) for _ in range(n - 1)] + [6000.0]
    return SymbolBars(symbol, opens, highs, lows, closes, volumes)


def _overbought_downturn(symbol="SOLUSD", n=80) -> SymbolBars:
    """Pump that's already exhausted — RSI overbought, CVD negative."""
    rnd = random.Random(1)
    # Long uptrend followed by 5 bars rolling over (bear cross happens, RSI eases)
    closes = [100 + 0.6 * i for i in range(60)] + [
        140 - 0.4 * i for i in range(20)
    ]
    closes = [c + rnd.gauss(0, 0.1) for c in closes]
    opens  = [c - 0.1 for c in closes]
    highs  = [max(o, c) + 0.2 for o, c in zip(opens, closes)]
    lows   = [min(o, c) - 0.2 for o, c in zip(opens, closes)]
    volumes = [1000] * n
    return SymbolBars(symbol, opens, highs, lows, closes, volumes)


# ───────────────────────────────────────────────────────────────────────────────
# Component sanity
# ───────────────────────────────────────────────────────────────────────────────

def test_flat_market_low_score():
    """Dead-flat OHLCV should score near zero."""
    bars = _flat_bars()
    s = compute_scalp_score(bars)
    assert s.score < DEFAULT_ENTRY_THRESHOLD
    assert not s.enter
    assert not s.high_conviction


def test_strong_setup_high_score():
    """Strong uptrend + volume burst should clear the entry threshold."""
    bars = _strong_uptrend_with_volume_burst()
    s = compute_scalp_score(bars)
    assert s.score >= DEFAULT_ENTRY_THRESHOLD
    assert s.enter
    # Should hit several individual components
    fired = sum(1 for x in [s.cvd_score, s.vol_ignition, s.micro_trend,
                            s.vwap_signal] if x > 0)
    assert fired >= 3


def test_overbought_chase_blocked_by_rsi():
    """A coin pumped to RSI > 70 should not score high — RSI component zeros."""
    # Use a freshly-pumped series (no rollover yet)
    n = 80
    closes = [100 + 0.5 * i for i in range(n)]
    bars = SymbolBars(
        "SOLUSD", [c - 0.1 for c in closes], [c + 0.2 for c in closes],
        [c - 0.2 for c in closes], closes,
        [1000] * n,
    )
    s = compute_scalp_score(bars)
    if s.rsi > 70:
        assert s.rsi_filter == 0.0


def test_score_clamped_to_unit_interval():
    """Even with all signals firing, score must stay in [0, 1]."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(
        symbol_recent_returns={f"S{i}": 0.05 for i in range(10)},
    )
    s = compute_scalp_score(bars, ctx)
    assert 0.0 <= s.score <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Spillover boost
# ───────────────────────────────────────────────────────────────────────────────

def test_spillover_boost_applied_to_laggard():
    """A laggard candidate gets a +0.10 boost when 5+ alts are pumping."""
    bars = _flat_bars("INJUSD")
    ctx = MarketContext(
        symbol_recent_returns={
            "BTCUSD": 0.03, "ETHUSD": 0.04, "SOLUSD": 0.05,
            "XRPUSD": 0.025, "DOGEUSD": 0.022,
            "INJUSD": 0.005,   # the laggard
        },
    )
    s = compute_scalp_score(bars, ctx)
    assert s.spillover_boost == 0.10


def test_spillover_no_boost_for_already_pumping():
    bars = _flat_bars("BTCUSD")
    ctx = MarketContext(
        symbol_recent_returns={
            "BTCUSD": 0.05, "ETHUSD": 0.04, "SOLUSD": 0.05,
            "XRPUSD": 0.025, "DOGEUSD": 0.022,
        },
    )
    s = compute_scalp_score(bars, ctx)
    assert s.spillover_boost == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# Size multipliers
# ───────────────────────────────────────────────────────────────────────────────

def test_default_multipliers_close_to_one():
    """No history / no context → multipliers default to 1.0."""
    bars = _flat_bars()
    s = compute_scalp_score(bars)
    assert s.kyle_size_mult == 1.0     # no history
    assert s.rv_size_mult == 1.0       # no history
    # Regime mult: no btc_closes → market_mode='chop', no GC → 0.5*1.0 (no btcd)
    assert 0 < s.regime_size_mult <= 1.0
    assert s.fg_size_mult == 1.0       # no FG value


def test_extreme_greed_blocks_entry_via_fg():
    """F&G=95 blocks new entries even on a perfect score."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(fg_value=95.0)
    s = compute_scalp_score(bars, ctx)
    assert s.fg_regime == "extreme_greed"
    assert not s.enter


def test_extreme_fear_boosts_size():
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(fg_value=20.0)
    s = compute_scalp_score(bars, ctx)
    assert s.fg_size_mult == 1.2


def test_selloff_market_mode_zeros_regime_mult():
    """When BTC is in a clear selloff, regime_size_mult must be 0 → no alt entries."""
    bars = _strong_uptrend_with_volume_burst("INJUSD")
    # BTC 4h closes that classify as selloff
    btc_closes = [50000, 49000, 48000, 47000, 46000,
                  45000, 44000, 43000, 42000, 41000, 40000, 39000, 38000]
    ctx = MarketContext(btc_4h_closes=btc_closes)
    s = compute_scalp_score(bars, ctx)
    assert s.market_mode == "selloff"
    assert s.regime_size_mult == 0.0


def test_composed_mult_is_product_of_components():
    bars = _flat_bars("INJUSD")
    ctx = MarketContext(fg_value=20.0)   # extreme_fear → 1.2 size mult
    s = compute_scalp_score(bars, ctx)
    expected = (
        s.kyle_size_mult * s.rv_size_mult
        * s.regime_size_mult * s.fg_size_mult
    )
    assert s.composed_size_mult == pytest.approx(expected, rel=1e-9)


# ───────────────────────────────────────────────────────────────────────────────
# rank_candidates batch
# ───────────────────────────────────────────────────────────────────────────────

def test_rank_candidates_filters_non_enter():
    """Only candidates with enter==True should appear in the result."""
    cands = [
        _flat_bars("BTCUSD"),
        _strong_uptrend_with_volume_burst("SOLUSD"),
        _flat_bars("LTCUSD"),
    ]
    ranked = rank_candidates(cands)
    syms = [r.symbol for r in ranked]
    assert "SOLUSD" in syms
    assert "BTCUSD" not in syms
    assert "LTCUSD" not in syms


def test_rank_candidates_sorted_by_score_desc():
    """Multiple winners must be sorted highest-score first."""
    cands = [
        _strong_uptrend_with_volume_burst("SOLUSD"),
        _strong_uptrend_with_volume_burst("INJUSD"),
        _strong_uptrend_with_volume_burst("BTCUSD"),
    ]
    ranked = rank_candidates(cands)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_returns_empty_when_no_signal():
    """No signal → empty result list, no crash."""
    cands = [_flat_bars(f"SYM{i}") for i in range(5)]
    ranked = rank_candidates(cands)
    assert ranked == []


# ───────────────────────────────────────────────────────────────────────────────
# Reporting / serialization
# ───────────────────────────────────────────────────────────────────────────────

def test_as_dict_serialization_complete():
    """as_dict() must include score, components, multipliers, diagnostics."""
    bars = _strong_uptrend_with_volume_burst()
    s = compute_scalp_score(bars)
    d = s.as_dict()
    assert d["symbol"] == "SOLUSD"
    assert "score" in d
    assert "components" in d
    assert "size_multipliers" in d
    assert "diag" in d
    # All components present
    for k in ("cvd", "vol_ignition", "micro_trend", "rsi_filter",
              "vwap_signal", "spillover"):
        assert k in d["components"]


def test_high_conviction_flag_set_above_07():
    """Score >= 0.70 sets high_conviction=True."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(
        symbol_recent_returns={
            "BTCUSD": 0.03, "ETHUSD": 0.04, "XRPUSD": 0.05,
            "DOGEUSD": 0.025, "ADAUSD": 0.022,
        },
    )
    s = compute_scalp_score(bars, ctx)
    if s.score >= DEFAULT_HIGH_CONVICTION_THRES:
        assert s.high_conviction
    else:
        assert not s.high_conviction


# ───────────────────────────────────────────────────────────────────────────────
# Tier C.4 funding rate integration
# ───────────────────────────────────────────────────────────────────────────────

def test_scalp_default_funding_mult_is_one():
    """No funding rate in context → funding mult = 1.0 (neutral)."""
    bars = _flat_bars()
    s = compute_scalp_score(bars)
    assert s.funding_size_mult == 1.0
    assert s.funding_regime == "balanced"


def test_scalp_deep_short_squeeze_boosts_size():
    """Funding -0.10% per 8h → 1.20× size boost."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(funding_rate=-0.001)
    s = compute_scalp_score(bars, ctx)
    assert s.funding_regime == "deep_short_squeeze"
    assert s.funding_size_mult == 1.20


def test_scalp_deep_long_crowd_halves_size():
    """Funding +0.06% per 8h → 0.50× size penalty."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(funding_rate=0.0006)
    s = compute_scalp_score(bars, ctx)
    assert s.funding_regime == "deep_long_crowd"
    assert s.funding_size_mult == 0.5


def test_scalp_panic_funding_blocks_entry():
    """Funding +0.15% per 8h → block new entries even on a perfect score."""
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(funding_rate=0.0015)
    s = compute_scalp_score(bars, ctx)
    assert s.funding_regime == "deep_long_crowd"
    assert not s.enter   # blocked


def test_composed_size_mult_includes_funding():
    """The composed multiplier is now the product of FIVE multipliers."""
    bars = _flat_bars("INJUSD")
    ctx = MarketContext(fg_value=20.0, funding_rate=-0.001)
    s = compute_scalp_score(bars, ctx)
    expected = (
        s.kyle_size_mult * s.rv_size_mult
        * s.regime_size_mult * s.fg_size_mult * s.funding_size_mult
    )
    assert s.composed_size_mult == pytest.approx(expected, rel=1e-9)


def test_as_dict_includes_funding_fields():
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext(funding_rate=-0.001)
    s = compute_scalp_score(bars, ctx)
    d = s.as_dict()
    assert "funding" in d["size_multipliers"]
    assert "funding_regime" in d["diag"]
    assert d["size_multipliers"]["funding"] == 1.20
    assert d["diag"]["funding_regime"] == "deep_short_squeeze"


# ───────────────────────────────────────────────────────────────────────────────
# Tier D.3 X mention boost integration
# ───────────────────────────────────────────────────────────────────────────────

def test_scalp_x_mention_burst_adds_boost():
    """Symbol with X mention rate ≥ 3× baseline → +0.10 score boost."""
    bars = _strong_uptrend_with_volume_burst("SOLUSD")
    # Baseline mentions: 100/hr; recent: 350/hr → 3.5× = buzzy
    ctx = MarketContext(x_mention_rates={"SOLUSD": (350.0, 100.0)})
    s = compute_scalp_score(bars, ctx)
    assert s.x_mention_boost == 0.10


def test_scalp_x_mention_no_boost_below_threshold():
    bars = _strong_uptrend_with_volume_burst("SOLUSD")
    ctx = MarketContext(x_mention_rates={"SOLUSD": (200.0, 100.0)})  # 2× only
    s = compute_scalp_score(bars, ctx)
    assert s.x_mention_boost == 0.0


def test_scalp_x_mention_default_no_data_no_boost():
    bars = _strong_uptrend_with_volume_burst()
    ctx = MarketContext()   # no X data at all
    s = compute_scalp_score(bars, ctx)
    assert s.x_mention_boost == 0.0


def test_scalp_x_mention_appears_in_as_dict():
    bars = _strong_uptrend_with_volume_burst("SOLUSD")
    ctx = MarketContext(x_mention_rates={"SOLUSD": (350.0, 100.0)})
    s = compute_scalp_score(bars, ctx)
    d = s.as_dict()
    assert "x_mention" in d["components"]
    assert d["components"]["x_mention"] == 0.10
