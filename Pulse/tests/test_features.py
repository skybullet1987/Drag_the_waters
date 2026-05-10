"""Tests for Pulse.features."""

from __future__ import annotations

import math

import pytest

from Pulse.features import (
    signed_bar_volume,
    cumulative_volume_delta, cvd_slope, cvd_signal_score,
    kyle_lambda, kyle_regime_multiplier,
    yang_zhang_realized_variance, realized_vol_bps, realized_vol_size_scaler,
    trade_rate_burst_zscore, volume_ignition_signal,
    VWAPState, vwap_band_position,
    cross_symbol_momentum_spillover, cross_symbol_score_boost,
    ema, rsi,
)


# ───────────────────────────────────────────────────────────────────────────────
# CVD
# ───────────────────────────────────────────────────────────────────────────────

def test_signed_bar_volume_at_high_is_full_buy():
    """Close at the high → full buy pressure."""
    sv = signed_bar_volume(open_p=100, high=110, low=100, close=110, volume=1000)
    assert sv == pytest.approx(1000)  # full +1 × volume


def test_signed_bar_volume_at_low_is_full_sell():
    sv = signed_bar_volume(open_p=110, high=110, low=100, close=100, volume=1000)
    assert sv == pytest.approx(-1000)


def test_signed_bar_volume_at_midpoint_is_zero():
    sv = signed_bar_volume(open_p=100, high=110, low=100, close=105, volume=1000)
    assert sv == pytest.approx(0.0)


def test_signed_bar_volume_zero_range_returns_zero():
    """Doji-flat bar (high==low) shouldn't divide by zero."""
    sv = signed_bar_volume(open_p=100, high=100, low=100, close=100, volume=1000)
    assert sv == 0.0


def test_signed_bar_volume_zero_volume_returns_zero():
    sv = signed_bar_volume(open_p=100, high=110, low=100, close=110, volume=0)
    assert sv == 0.0


def test_cumulative_volume_delta_running_sum():
    opens   = [100, 101, 102]
    highs   = [102, 103, 104]
    lows    = [99, 100, 101]
    closes  = [102, 103, 104]   # all closes at high → all positive bars
    volumes = [1000, 1000, 1000]
    cvd = cumulative_volume_delta(opens, highs, lows, closes, volumes)
    assert len(cvd) == 3
    assert cvd[0] > 0
    assert cvd[1] > cvd[0]
    assert cvd[2] > cvd[1]


def test_cvd_slope_positive_on_uptrend():
    cvd = [100 + 10 * i for i in range(20)]
    s = cvd_slope(cvd, lookback=12)
    assert s == pytest.approx(10.0, rel=1e-9)


def test_cvd_slope_negative_on_downtrend():
    cvd = [100 - 5 * i for i in range(20)]
    assert cvd_slope(cvd, lookback=12) == pytest.approx(-5.0, rel=1e-9)


def test_cvd_slope_zero_on_flat():
    cvd = [100] * 20
    assert cvd_slope(cvd, lookback=12) == 0.0


def test_cvd_slope_insufficient_data_returns_zero():
    assert cvd_slope([1, 2, 3], lookback=12) == 0.0


def test_cvd_signal_score_normalized_to_unit():
    """Even an extreme slope should clip to ±1."""
    cvd = [10_000_000 * i for i in range(30)]
    score = cvd_signal_score(cvd, lookback=12)
    assert -1.0 <= score <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Kyle's Lambda
# ───────────────────────────────────────────────────────────────────────────────

def test_kyle_lambda_basic():
    """100 bars, |Δprice| ≈ 1, vol = 100 → λ = 1/100 = 0.01."""
    closes = [100 + (i % 2) for i in range(100)]   # oscillates 100/101
    vols   = [100] * 100
    lam = kyle_lambda(closes, vols, lookback=20)
    assert lam == pytest.approx(0.01, rel=0.05)


def test_kyle_lambda_higher_when_volume_thin():
    """Same price moves with thinner volume → higher λ."""
    closes = [100 + (i % 2) for i in range(100)]
    lam_thick = kyle_lambda(closes, [100] * 100, lookback=20)
    lam_thin  = kyle_lambda(closes, [10]  * 100, lookback=20)
    assert lam_thin > lam_thick


def test_kyle_lambda_zero_on_no_volume():
    assert kyle_lambda([100, 101, 102], [0, 0, 0], lookback=2) == 0.0


def test_kyle_regime_multiplier_high_lambda_reduces_size():
    history = [0.001 * i for i in range(20)]   # 0.0..0.019
    mult = kyle_regime_multiplier(current_lambda=0.018, rolling_lambda_history=history)
    assert 0.5 <= mult < 1.0


def test_kyle_regime_multiplier_low_lambda_boosts_size():
    history = [0.001 * i for i in range(20)]
    mult = kyle_regime_multiplier(current_lambda=0.001, rolling_lambda_history=history)
    assert 1.0 < mult <= 1.5


def test_kyle_regime_multiplier_no_history_returns_one():
    assert kyle_regime_multiplier(current_lambda=0.05, rolling_lambda_history=[]) == 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Yang-Zhang realized variance
# ───────────────────────────────────────────────────────────────────────────────

def test_yz_returns_zero_on_insufficient_data():
    assert yang_zhang_realized_variance([100], [101], [99], [100]) == 0.0


def test_yz_low_vol_when_prices_flat():
    n = 30
    o = [100.0] * n
    h = [100.05] * n
    l = [99.95]  * n
    c = [100.0] * n
    var = yang_zhang_realized_variance(o, h, l, c, lookback=20)
    assert var >= 0
    assert var < 1e-4   # very small


def test_yz_higher_vol_when_prices_volatile():
    n = 30
    o = [100.0 + (i * 0.1) for i in range(n)]
    h = [v * 1.01 for v in o]
    l = [v * 0.99 for v in o]
    c = [v * (1 + 0.005 * (-1) ** i) for i, v in enumerate(o)]
    quiet_var = yang_zhang_realized_variance(
        [100]*30, [100.05]*30, [99.95]*30, [100]*30, lookback=20,
    )
    loud_var = yang_zhang_realized_variance(o, h, l, c, lookback=20)
    assert loud_var > quiet_var


def test_realized_vol_bps_returns_positive():
    n = 30
    o = [100.0 + (i * 0.1) for i in range(n)]
    h = [v * 1.005 for v in o]
    l = [v * 0.995 for v in o]
    c = o
    bps = realized_vol_bps(o, h, l, c, lookback=20)
    assert bps > 0


def test_rv_size_scaler_high_vol_reduces():
    history = list(range(20, 200, 10))   # 20..200 bps
    mult = realized_vol_size_scaler(current_vol_bps=180, rolling_vol_bps_history=history)
    assert 0.5 <= mult < 1.0


def test_rv_size_scaler_low_vol_boosts():
    history = list(range(20, 200, 10))
    mult = realized_vol_size_scaler(current_vol_bps=25, rolling_vol_bps_history=history)
    assert 1.0 < mult <= 1.5


# ───────────────────────────────────────────────────────────────────────────────
# Trade-rate burst
# ───────────────────────────────────────────────────────────────────────────────

def test_trade_rate_z_zero_on_flat_volume():
    z = trade_rate_burst_zscore([100] * 100, lookback=60)
    assert z == 0.0


def test_trade_rate_z_high_on_4x_burst():
    history = [100] * 100
    history[-1] = 500   # 4x
    z = trade_rate_burst_zscore(history, lookback=60)
    # No std → 0; need some variance in history first
    assert z == 0.0


def test_trade_rate_z_high_on_burst_with_realistic_history():
    import random
    rnd = random.Random(0)
    history = [100 + rnd.gauss(0, 10) for _ in range(60)] + [350.0]
    z = trade_rate_burst_zscore(history, lookback=60)
    assert z >= 2.0


def test_volume_ignition_signal_thresholds():
    import random
    rnd = random.Random(0)
    base = [100 + rnd.gauss(0, 10) for _ in range(60)]   # mean~100, std~10
    # Strong burst (z >= 2) → 0.20
    assert volume_ignition_signal(base + [400.0], lookback=60) == 0.20
    # Partial burst (1 <= z < 2) → 0.10  →  ~15 above mean (1.5σ)
    assert volume_ignition_signal(base + [115.0], lookback=60) == 0.10
    # No burst (z < 1) → 0.0
    assert volume_ignition_signal(base + [101.0], lookback=60) == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# VWAP ± σ band position
# ───────────────────────────────────────────────────────────────────────────────

def test_vwap_state_basic():
    s = VWAPState()
    s.update(100.0, 1000)
    s.update(102.0, 1000)
    assert s.vwap() == pytest.approx(101.0)
    assert s.std() > 0


def test_vwap_state_ignores_zero():
    s = VWAPState()
    s.update(100, 1000)
    s.update(0, 1000)        # ignored
    s.update(100, 0)         # ignored
    assert s.vwap() == 100.0


def test_vwap_state_reset():
    s = VWAPState()
    s.update(100, 1000)
    s.reset()
    assert s.vwap() == 0.0


def test_vwap_band_position_inside():
    assert vwap_band_position(price=100, vwap=100, std=1) == 0


def test_vwap_band_position_above_2sigma():
    assert vwap_band_position(price=103, vwap=100, std=1) == 2


def test_vwap_band_position_below_2sigma():
    assert vwap_band_position(price=97, vwap=100, std=1) == -2


def test_vwap_band_position_in_1sigma_band():
    assert vwap_band_position(price=101.5, vwap=100, std=1) == 1
    assert vwap_band_position(price=98.5,  vwap=100, std=1) == -1


def test_vwap_band_position_zero_std_safe():
    assert vwap_band_position(price=100, vwap=100, std=0) == 0


# ───────────────────────────────────────────────────────────────────────────────
# Cross-symbol momentum spillover
# ───────────────────────────────────────────────────────────────────────────────

def test_spillover_no_pump():
    rets = {f"S{i}": 0.01 for i in range(10)}
    out = cross_symbol_momentum_spillover(rets, threshold_pct=0.02, min_count=5)
    assert out["n_pumping"] == 0
    assert not out["spillover_active"]


def test_spillover_active_when_5_alts_pump():
    rets = {f"S{i}": 0.03 if i < 6 else 0.005 for i in range(10)}
    out = cross_symbol_momentum_spillover(rets, threshold_pct=0.02, min_count=5)
    assert out["n_pumping"] == 6
    assert out["spillover_active"]


def test_spillover_score_boost_only_for_laggard():
    """When 5+ alts pump, give a boost to a NON-pumping candidate."""
    rets = {"BTC": 0.03, "ETH": 0.04, "SOL": 0.05, "XRP": 0.025, "DOGE": 0.022,
            "INJ": 0.005}
    boost_inj = cross_symbol_score_boost("INJ", rets,
                                         threshold_pct=0.02, min_count=5,
                                         boost=0.10)
    boost_btc = cross_symbol_score_boost("BTC", rets,
                                         threshold_pct=0.02, min_count=5,
                                         boost=0.10)
    assert boost_inj == 0.10  # laggard gets boost
    assert boost_btc == 0.0   # already pumping → no extra boost


def test_spillover_no_boost_when_inactive():
    rets = {"INJ": 0.005, "BTC": 0.01}
    assert cross_symbol_score_boost("INJ", rets, min_count=5) == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# EMA + RSI
# ───────────────────────────────────────────────────────────────────────────────

def test_ema_constant_series_returns_constant():
    assert ema([5, 5, 5, 5, 5], period=3) == pytest.approx(5)


def test_ema_short_data_returns_mean():
    assert ema([1, 2, 3], period=10) == pytest.approx(2.0)


def test_ema_responds_to_recent():
    """Recent uptrend should make EMA > mean."""
    e = ema([1, 1, 1, 5, 9], period=3)
    assert e > 3.4   # mean would be 3.4, EMA weights recent more


def test_rsi_short_data_returns_50():
    assert rsi([1, 2, 3]) == 50.0


def test_rsi_pure_uptrend_high():
    closes = list(range(20))   # 0,1,2,...,19 — pure uptrend
    r = rsi(closes, period=14)
    assert r >= 99.9  # all gains, no losses


def test_rsi_pure_downtrend_low():
    closes = list(range(20, 0, -1))   # pure downtrend
    r = rsi(closes, period=14)
    assert r <= 1.0


def test_rsi_neutral_oscillation_around_50():
    closes = [100 + (i % 2) for i in range(30)]
    r = rsi(closes, period=14)
    assert 40 < r < 60
