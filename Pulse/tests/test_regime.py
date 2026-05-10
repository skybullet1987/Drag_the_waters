"""Tests for Pulse.regime."""

from __future__ import annotations

import pytest

from Pulse.regime import (
    MARKET_MODES,
    detect_market_mode,
    MarketModeDetector,
    GoldenCrossDecision,
    golden_cross_regime,
    golden_cross_size_multiplier,
    BTCDominanceDecision,
    btc_dominance_regime,
    btc_dominance_alt_size_multiplier,
    compose_regime_size_multiplier,
)


# ───────────────────────────────────────────────────────────────────────────────
# detect_market_mode (5 modes)
# ───────────────────────────────────────────────────────────────────────────────

def test_5_market_modes_defined():
    assert MARKET_MODES == (
        "risk_on_trend", "pump", "chop", "selloff", "high_vol_reversal",
    )


def test_market_mode_returns_chop_on_insufficient_data():
    assert detect_market_mode([100, 101]) == "chop"


def test_market_mode_chop_on_flat():
    closes = [100 + ((i % 2) * 0.05) for i in range(20)]   # tiny oscillation
    assert detect_market_mode(closes) == "chop"


def test_market_mode_selloff_on_steep_drop():
    """ret_4 < -4% AND sma_slope < -1%."""
    closes = [100, 99, 98, 95, 92, 89, 86, 83, 80, 77, 74, 71, 68]
    assert detect_market_mode(closes) == "selloff"


def test_market_mode_risk_on_trend_on_steady_uptrend():
    """ret_4 > 1%, sma_slope > 0.3%, range_eff > 40%."""
    closes = [100 + i * 0.5 for i in range(20)]   # steady +0.5/bar
    mode = detect_market_mode(closes)
    assert mode == "risk_on_trend"


def test_market_mode_pump_on_fast_rise_with_volume():
    """ret_4 > 5% AND vol_ratio > 2.

    vol_ratio = current / mean(previous 5). Need to make the most-recent
    5-bar average tame so that the very latest bar's volume looks like a
    huge spike on top of it.
    """
    closes  = [100, 100, 100, 100, 100, 102, 104, 106, 108, 110]
    # Last 6 volumes used for ratio; v[-1] = 1500 is the spike, prior 5 are 100s
    volumes = [100] * 9 + [1500]
    mode = detect_market_mode(closes, volumes)
    assert mode == "pump"


def test_market_mode_high_vol_reversal_on_choppy_swings():
    """High vol + low range_eff (lots of motion, no net direction)."""
    closes = [100, 105, 100, 105, 100, 105, 100, 105, 100, 105]
    mode = detect_market_mode(closes)
    assert mode == "high_vol_reversal"


def test_market_mode_detector_class_stateful():
    d = MarketModeDetector()
    for c in [100 + i * 0.5 for i in range(20)]:
        d.update_bar(c, 100)
    assert d.mode == "risk_on_trend"


def test_market_mode_detector_initial_chop():
    d = MarketModeDetector()
    assert d.mode == "chop"


# ───────────────────────────────────────────────────────────────────────────────
# Golden cross regime
# ───────────────────────────────────────────────────────────────────────────────

def test_gc_neutral_on_short_history():
    d = golden_cross_regime([100] * 50)
    assert d.regime == "neutral"
    assert d.reason == "insufficient_history"


def test_gc_bull_on_uptrend():
    """Steady uptrend: SMA50 > SMA200 + positive 30d return."""
    closes = [100 + i * 0.5 for i in range(250)]
    d = golden_cross_regime(closes)
    assert d.regime == "bull"
    assert d.sma_50 > d.sma_200


def test_gc_bear_on_downtrend():
    """Steady downtrend with notable losses."""
    closes = [200 - i * 0.5 for i in range(250)]
    d = golden_cross_regime(closes)
    assert d.regime == "bear"
    assert d.sma_50 < d.sma_200


def test_gc_neutral_during_transition():
    """Just-crossed below SMA200: SMA50 above but recent drop too sharp."""
    closes = [100 + i * 0.4 for i in range(200)] + [180 - i * 2.5 for i in range(50)]
    d = golden_cross_regime(closes)
    assert d.regime in ("neutral", "bear")


def test_gc_size_multiplier_mapping():
    bull   = GoldenCrossDecision("bull", 110, 100, 0.10, "x")
    neutral = GoldenCrossDecision("neutral", 100, 100, 0.0, "x")
    bear   = GoldenCrossDecision("bear", 90, 100, -0.10, "x")
    assert golden_cross_size_multiplier(bull) == 1.0
    assert golden_cross_size_multiplier(neutral) == 0.5
    assert golden_cross_size_multiplier(bear) == 0.25


# ───────────────────────────────────────────────────────────────────────────────
# BTC dominance regime
# ───────────────────────────────────────────────────────────────────────────────

def test_btcd_neutral_on_no_alt_data():
    d = btc_dominance_regime(btc_30d_return=0.10, alts_30d_returns=[])
    assert d.regime == "neutral"


def test_btcd_btc_strong_when_btc_dominates():
    """BTC +20%, alts only +5%. Ratio = 4.0 → btc_strong."""
    d = btc_dominance_regime(0.20, [0.05] * 5)
    assert d.regime == "btc_strong"
    assert d.ratio == pytest.approx(4.0)


def test_btcd_alt_strong_when_alts_dominate():
    """BTC +5%, alts +15%. Ratio = 0.33 → alt_strong (< 0.7)."""
    d = btc_dominance_regime(0.05, [0.15] * 5)
    assert d.regime == "alt_strong"


def test_btcd_alt_strong_when_btc_falls_alts_rise():
    """BTC -2%, alts +10%. Ratio negative + alts up → alt_strong (rotation)."""
    d = btc_dominance_regime(-0.02, [0.10] * 5)
    assert d.regime == "alt_strong"


def test_btcd_alt_size_multiplier_mapping():
    btc_strong = BTCDominanceDecision("btc_strong", 0.20, 0.05, 4.0)
    neutral    = BTCDominanceDecision("neutral", 0.05, 0.05, 1.0)
    alt_strong = BTCDominanceDecision("alt_strong", 0.05, 0.15, 0.33)
    assert btc_dominance_alt_size_multiplier(btc_strong) == 0.5
    assert btc_dominance_alt_size_multiplier(neutral) == 1.0
    assert btc_dominance_alt_size_multiplier(alt_strong) == 1.5


# ───────────────────────────────────────────────────────────────────────────────
# compose_regime_size_multiplier
# ───────────────────────────────────────────────────────────────────────────────

def test_compose_pump_bull_alt_strong_max_size():
    """Best-case alt environment: pump (1.2) × bull (1.0) × alt_strong (1.5) = 1.8 → clamped to 1.5."""
    gc = GoldenCrossDecision("bull", 110, 100, 0.10, "x")
    bd = BTCDominanceDecision("alt_strong", 0.05, 0.15, 0.33)
    out = compose_regime_size_multiplier("pump", gc, bd)
    assert out["size_mult"] == 1.5  # clamped


def test_compose_selloff_zeros_out_size():
    """Selloff regime should kill alt entries entirely."""
    gc = GoldenCrossDecision("bull", 110, 100, 0.10, "x")
    bd = BTCDominanceDecision("alt_strong", 0.05, 0.15, 0.33)
    out = compose_regime_size_multiplier("selloff", gc, bd)
    assert out["size_mult"] == 0.0


def test_compose_chop_neutral_btc_strong_small_size():
    """Worst-case alt environment: chop (0.5) × neutral (0.5) × btc_strong (0.5) = 0.125."""
    gc = GoldenCrossDecision("neutral", 100, 100, 0.01, "x")
    bd = BTCDominanceDecision("btc_strong", 0.20, 0.05, 4.0)
    out = compose_regime_size_multiplier("chop", gc, bd)
    assert out["size_mult"] == pytest.approx(0.125)


def test_compose_no_btcd_for_btc_position():
    """When the candidate IS BTC (is_alt=False), btc_dominance is irrelevant."""
    gc = GoldenCrossDecision("bull", 110, 100, 0.10, "x")
    out = compose_regime_size_multiplier(
        "risk_on_trend", gc, btcd_dec=None, is_alt=False
    )
    assert out["size_mult"] == 1.0  # 1.0 × 1.0 × 1.0


def test_compose_returns_breakdown_dict():
    out = compose_regime_size_multiplier("pump", None, None)
    assert "size_mult" in out
    assert "market_mode_mult" in out
    assert "golden_cross_mult" in out
    assert "btc_dominance_mult" in out
    assert out["market_mode_mult"] == 1.2
