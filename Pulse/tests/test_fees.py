"""Tests for Pulse.fees."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.fees import (
    KRAKEN_FEE_TIERS, lookup_kraken_tier,
    compute_fee_pct, compute_flat_fee_pct,
    VolumeTracker,
)


# ───────────────────────────────────────────────────────────────────────────────
# Tier table
# ───────────────────────────────────────────────────────────────────────────────

def test_tier_table_descending_min_volume():
    """Tiers must be sorted high → low so first-match-wins works."""
    mins = [t[0] for t in KRAKEN_FEE_TIERS]
    assert mins == sorted(mins, reverse=True)


def test_tier_table_8_levels():
    assert len(KRAKEN_FEE_TIERS) == 8


def test_tier_table_taker_always_higher_than_maker():
    for min_vol, maker, taker in KRAKEN_FEE_TIERS:
        assert taker >= maker, f"tier ${min_vol} maker={maker} > taker={taker}"


def test_tier_table_lower_volume_higher_fees():
    """As you go from highest tier to lowest, both rates should monotonically
    increase (or at least be non-decreasing)."""
    rates = [(t[1], t[2]) for t in KRAKEN_FEE_TIERS]
    for i in range(1, len(rates)):
        assert rates[i][0] >= rates[i-1][0]
        assert rates[i][1] >= rates[i-1][1]


# ───────────────────────────────────────────────────────────────────────────────
# lookup_kraken_tier
# ───────────────────────────────────────────────────────────────────────────────

def test_lookup_zero_volume_lowest_tier():
    """At $0 volume → the worst (highest fee) tier."""
    maker, taker = lookup_kraken_tier(0.0)
    assert (maker, taker) == (0.0040, 0.0080)


def test_lookup_above_500k_best_tier():
    maker, taker = lookup_kraken_tier(1_000_000)
    assert (maker, taker) == (0.0008, 0.0018)


def test_lookup_50k_mid_tier():
    """Just at the $50K threshold."""
    maker, taker = lookup_kraken_tier(50_000)
    assert (maker, taker) == (0.0014, 0.0024)


def test_lookup_just_under_threshold_uses_lower_tier():
    """$49,999 should use the $25K tier, not the $50K tier."""
    maker, taker = lookup_kraken_tier(49_999)
    assert (maker, taker) == (0.0020, 0.0035)


# ───────────────────────────────────────────────────────────────────────────────
# compute_fee_pct (tiered)
# ───────────────────────────────────────────────────────────────────────────────

def test_market_order_pays_taker():
    fee = compute_fee_pct(is_limit_order=False, monthly_volume_usd=0)
    assert fee == 0.0080


def test_limit_order_pays_blended():
    """Default 25% taker / 75% maker on limit at $0 volume.
    Expected: 0.75*0.0040 + 0.25*0.0080 = 0.0050."""
    fee = compute_fee_pct(is_limit_order=True, monthly_volume_usd=0)
    assert fee == pytest.approx(0.0050, rel=1e-9)


def test_compound_to_higher_tier_reduces_fees():
    """A $1M trader pays much less per trade than a $0 trader."""
    f0  = compute_fee_pct(is_limit_order=False, monthly_volume_usd=0)
    f1m = compute_fee_pct(is_limit_order=False, monthly_volume_usd=1_000_000)
    assert f1m < f0
    assert f1m == 0.0018  # best taker rate


def test_limit_taker_ratio_override():
    """Override defaults to test sensitivity."""
    f_default = compute_fee_pct(is_limit_order=True, monthly_volume_usd=0)
    f_aggro   = compute_fee_pct(is_limit_order=True, monthly_volume_usd=0,
                                limit_taker_ratio=1.0)  # 100% taker
    assert f_aggro > f_default
    assert f_aggro == 0.0080  # full taker


# ───────────────────────────────────────────────────────────────────────────────
# compute_flat_fee_pct (MG36 MakerTaker)
# ───────────────────────────────────────────────────────────────────────────────

def test_flat_market_order_full_taker():
    f = compute_flat_fee_pct(is_limit_order=False)
    assert f == 0.0040


def test_flat_limit_order_blended_default_40_taker():
    """0.6 * 0.0025 + 0.4 * 0.0040 = 0.0031."""
    f = compute_flat_fee_pct(is_limit_order=True)
    assert f == pytest.approx(0.0031, rel=1e-9)


# ───────────────────────────────────────────────────────────────────────────────
# VolumeTracker
# ───────────────────────────────────────────────────────────────────────────────

def test_tracker_starts_empty():
    t = VolumeTracker()
    assert t.estimated_30d_volume(datetime(2026, 1, 1)) == 0.0


def test_tracker_records_and_projects():
    """$10K of volume in 1 day → projected $300K/month."""
    t = VolumeTracker()
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t.record(t0, 10_000)
    t1 = t0 + timedelta(days=1)
    monthly = t.estimated_30d_volume(t1)
    assert monthly == pytest.approx(10_000 * 30 / 1, rel=1e-9)  # $300K


def test_tracker_zero_or_neg_value_ignored():
    t = VolumeTracker()
    t.record(datetime(2026, 1, 1), 0)
    t.record(datetime(2026, 1, 1), -100)
    assert t.cumulative_volume_usd == 0.0
    assert t.start_time is None


def test_tracker_handles_same_day_estimate():
    """Same-day query: elapsed_days is clamped to 1 to avoid div-by-zero."""
    t = VolumeTracker()
    t0 = datetime(2026, 1, 1, 12)
    t.record(t0, 5_000)
    same_day = t.estimated_30d_volume(t0)  # zero days elapsed
    assert same_day == 5_000 * 30  # treated as 1 day → $150K/mo


def test_tracker_multi_record_accumulates():
    t = VolumeTracker()
    t0 = datetime(2026, 1, 1)
    for i in range(10):
        t.record(t0 + timedelta(hours=i), 1_000)
    assert t.cumulative_volume_usd == 10_000
    assert t.start_time == t0


def test_tracker_used_with_compute_fee_pct():
    """Integration: trader does $100K in 30 days → tier $100K (0.22% taker)."""
    t = VolumeTracker()
    t0 = datetime(2026, 1, 1)
    for d in range(30):
        t.record(t0 + timedelta(days=d), 3_400)   # ~$102K total
    monthly = t.estimated_30d_volume(t0 + timedelta(days=30))
    assert monthly >= 100_000
    fee = compute_fee_pct(is_limit_order=False, monthly_volume_usd=monthly)
    assert fee == 0.0022   # $100K tier taker
