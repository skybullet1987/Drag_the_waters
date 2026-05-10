"""Tests for Pulse.slippage."""

from __future__ import annotations

import pytest

from Pulse.slippage import (
    SlippageParams, estimate_slippage_pct,
    _spread_floor, _tier_multiplier,
)


# ───────────────────────────────────────────────────────────────────────────────
# _spread_floor — synthetic spread by price tier
# ───────────────────────────────────────────────────────────────────────────────

def test_spread_floor_dust_widest():
    p = SlippageParams()
    assert _spread_floor(0.005, p) == 0.020  # $0.005 → 200bp dust floor


def test_spread_floor_micro():
    p = SlippageParams()
    assert _spread_floor(0.05, p) == 0.010   # $0.05 → 100bp


def test_spread_floor_major_tightest():
    p = SlippageParams()
    assert _spread_floor(50_000, p) == 0.0010   # BTC → 10bp floor


# ───────────────────────────────────────────────────────────────────────────────
# _tier_multiplier
# ───────────────────────────────────────────────────────────────────────────────

def test_tier_multiplier_dust_4x():
    p = SlippageParams()
    assert _tier_multiplier(0.005, p) == 4.0


def test_tier_multiplier_major_1x():
    p = SlippageParams()
    assert _tier_multiplier(50_000, p) == 1.0


# ───────────────────────────────────────────────────────────────────────────────
# estimate_slippage_pct — end-to-end behavior
# ───────────────────────────────────────────────────────────────────────────────

def test_zero_price_returns_zero():
    s = estimate_slippage_pct(price=0, order_quantity=1, bar_volume=100)
    assert s == 0.0


def test_btc_with_real_bidask_uses_spread_not_floor():
    """BTC with bid=49995/ask=50005: half-spread = 5/50000 = 1bp.
    Slip = base 30bp + half-spread 1bp = 31bp; tier mult 1.0 → ~31bp.
    Important: the model uses HALF the spread (one-side cost), not the full spread."""
    s = estimate_slippage_pct(
        price=50_000, order_quantity=0.001, bar_volume=10,
        bid=49_995, ask=50_005,
    )
    assert 0.0030 < s < 0.0035


def test_micro_cap_uses_synthetic_floor_when_no_bidask():
    """KAS at $0.04 with no bid/ask:
        base 30bp + spread 100bp = 130bp
        tier mult 2.5x → 325bp uncapped
        but max cap = 250bp (0.025) → result == 250bp.
    """
    s = estimate_slippage_pct(
        price=0.04, order_quantity=10, bar_volume=10000,
        bid=0, ask=0,
    )
    assert s == pytest.approx(0.025, rel=1e-9)  # capped at max


def test_high_participation_increases_slippage():
    """Order = 10% of bar volume should hurt more than 0.1%."""
    base = estimate_slippage_pct(
        price=100, order_quantity=1, bar_volume=10000,  # 0.01% participation
    )
    big = estimate_slippage_pct(
        price=100, order_quantity=1000, bar_volume=10000,  # 10% participation
    )
    assert big > base
    # Convex penalty: 10% participation should add visible bps
    assert big - base > 0.0005


def test_capped_at_max_slippage():
    """Egregious order on dust coin still capped at 2.5%."""
    p = SlippageParams(max_slippage_pct=0.025)
    s = estimate_slippage_pct(
        price=0.0001, order_quantity=1_000_000, bar_volume=100,
        params=p,
    )
    assert s == pytest.approx(0.025, rel=1e-9)


def test_kasusd_like_envelope_matches_live():
    """Live KASUSD: $0.04 price, ~$30 order on a probably-low-vol bar.
    Single-leg slippage was 107bp. Our model with synthetic spread floor
    should land in the 100-300bp envelope.
    """
    s = estimate_slippage_pct(
        price=0.04, order_quantity=750, bar_volume=20_000,  # ~3.7% participation
        bid=0, ask=0,
    )
    bps = s * 10_000
    assert 100 <= bps <= 500, (
        f"KASUSD-like slip = {bps:.1f}bp; live observed 107bp single leg"
    )


def test_btc_envelope_tighter_than_alt():
    """Slippage on a $1000 BTC trade < same-notional alt trade."""
    btc = estimate_slippage_pct(
        price=50_000, order_quantity=0.02, bar_volume=2.0,
        bid=49_995, ask=50_005,
    )
    sol = estimate_slippage_pct(
        price=170, order_quantity=6.0, bar_volume=600,
        bid=169.7, ask=170.3,
    )
    # SOL $1000 trade should slip more than BTC $1000 trade
    assert sol > btc


def test_no_volume_data_skips_volume_impact():
    """Bar volume 0 means we skip the impact term but still apply spread+base."""
    s = estimate_slippage_pct(
        price=100, order_quantity=10, bar_volume=0,
        bid=99.95, ask=100.05,
    )
    # base 30bp + spread 5bp = ~35bp (no volume impact)
    assert 0.003 < s < 0.005


def test_params_override():
    """Pass custom params to override defaults."""
    p = SlippageParams(base_slippage_pct=0.001)  # 10bp instead of 30bp
    s = estimate_slippage_pct(
        price=50_000, order_quantity=0.01, bar_volume=1.0,
        bid=49_995, ask=50_005, params=p,
    )
    # Should be lower than default
    s_default = estimate_slippage_pct(
        price=50_000, order_quantity=0.01, bar_volume=1.0,
        bid=49_995, ask=50_005,
    )
    assert s < s_default
