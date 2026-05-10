"""Tests for Pulse.optimal_execution."""

from __future__ import annotations

import math
import pytest

from Pulse.optimal_execution import (
    DEFAULT_N_SLICES, DEFAULT_LARGE_ORDER_THRESHOLD_BPS,
    SliceStep, SlicePlan,
    _slip_bps_for_slice, twap_total_slip_bps,
    almgren_chriss_schedule, build_slice_plan,
)


# ───────────────────────────────────────────────────────────────────────────────
# _slip_bps_for_slice
# ───────────────────────────────────────────────────────────────────────────────

def test_slip_zero_on_zero_qty():
    assert _slip_bps_for_slice(0, 1000, 0.30, 0.10) == 0.0


def test_slip_zero_on_zero_volume():
    assert _slip_bps_for_slice(100, 0, 0.30, 0.10) == 0.0


def test_slip_increases_with_participation():
    """Higher participation rate → more slip."""
    s1 = _slip_bps_for_slice(10, 1000, eta=0.30, gamma=0.10)   # 1%
    s2 = _slip_bps_for_slice(100, 1000, eta=0.30, gamma=0.10)  # 10%
    assert s2 > s1


def test_slip_quadratic_term():
    """Doubling participation more than doubles slip (gamma is quadratic)."""
    s_low  = _slip_bps_for_slice(50, 1000, eta=0.30, gamma=1.0)
    s_high = _slip_bps_for_slice(100, 1000, eta=0.30, gamma=1.0)
    assert s_high > 2 * s_low


# ───────────────────────────────────────────────────────────────────────────────
# almgren_chriss_schedule
# ───────────────────────────────────────────────────────────────────────────────

def test_ac_schedule_empty_on_zero_slices():
    assert almgren_chriss_schedule(100, 0) == []


def test_ac_schedule_single_slice():
    """1 slice → all qty in one chunk."""
    sched = almgren_chriss_schedule(100, 1)
    assert sched == [100]


def test_ac_schedule_twap_when_zero_risk_aversion():
    """risk_aversion=0 → equal slices."""
    sched = almgren_chriss_schedule(100, 5, risk_aversion=0)
    assert all(s == pytest.approx(20) for s in sched)


def test_ac_schedule_sums_to_total():
    sched = almgren_chriss_schedule(100, 5, risk_aversion=1e-3)
    assert sum(sched) == pytest.approx(100, rel=1e-6)


def test_ac_schedule_front_loads_with_high_risk_aversion():
    """High risk aversion → bigger first slice (close out faster)."""
    sched = almgren_chriss_schedule(100, 5, risk_aversion=1.0,
                                    sigma=0.05, eta=0.01)
    assert sched[0] > sched[-1]


def test_ac_schedule_falls_back_to_twap_on_invalid_eta():
    """eta=0 → degenerate; should fall back to TWAP."""
    sched = almgren_chriss_schedule(100, 5, risk_aversion=1.0, eta=0)
    assert all(s == pytest.approx(20) for s in sched)


# ───────────────────────────────────────────────────────────────────────────────
# twap_total_slip_bps
# ───────────────────────────────────────────────────────────────────────────────

def test_twap_zero_on_zero_slices():
    assert twap_total_slip_bps(100, 0, 1000, 0.30, 0.10) == 0.0


def test_twap_more_slices_less_total_slip():
    """5 slices @ 20 each should slip less than 1 slice @ 100 due to gamma."""
    one_shot = twap_total_slip_bps(100, 1, 1000, eta=0.30, gamma=1.0)
    sliced   = twap_total_slip_bps(100, 5, 1000, eta=0.30, gamma=1.0)
    assert sliced < one_shot


# ───────────────────────────────────────────────────────────────────────────────
# build_slice_plan — top-level
# ───────────────────────────────────────────────────────────────────────────────

def test_build_plan_skips_zero_qty():
    plan = build_slice_plan(0, 1000)
    assert plan.skipped_reason == "zero_or_negative_quantity"
    assert plan.steps == []


def test_build_plan_single_slice_when_n_slices_1():
    plan = build_slice_plan(100, 1000, n_slices=1)
    assert plan.n_slices == 1
    assert len(plan.steps) == 1
    assert plan.steps[0].quantity == 100
    assert plan.skipped_reason == "n_slices<=1"


def test_build_plan_skips_when_no_volume():
    plan = build_slice_plan(100, 0, n_slices=5)
    assert plan.skipped_reason == "no_volume_estimate"


def test_build_plan_skips_when_below_threshold():
    """Tiny order vs huge bar volume → no slicing needed."""
    plan = build_slice_plan(1, 1_000_000, n_slices=5,
                            large_order_threshold_bps=25)
    assert plan.skipped_reason == "below_large_order_threshold"
    assert plan.n_slices == 1


def test_build_plan_slices_when_above_threshold():
    """Large order vs small bar volume → real slicing."""
    plan = build_slice_plan(500, 1000, n_slices=5,
                            eta=0.30, gamma=1.0,
                            large_order_threshold_bps=10)
    assert plan.skipped_reason is None
    assert len(plan.steps) == 5
    assert plan.expected_total_slip_bps > 0


def test_build_plan_quantities_sum_to_total():
    plan = build_slice_plan(500, 1000, n_slices=5,
                            eta=0.30, gamma=1.0,
                            large_order_threshold_bps=10)
    total = sum(s.quantity for s in plan.steps)
    assert total == pytest.approx(500, rel=1e-6)


def test_build_plan_saves_bps_vs_one_shot():
    """Sliced execution should slip less in total than one-shot (when gamma > 0)."""
    plan = build_slice_plan(500, 1000, n_slices=5,
                            risk_aversion=0,   # TWAP for predictable savings
                            eta=0.30, gamma=1.0,
                            large_order_threshold_bps=10)
    one_shot = _slip_bps_for_slice(500, 1000, eta=0.30, gamma=1.0)
    assert plan.expected_total_slip_bps < one_shot


def test_build_plan_to_dict_serializable():
    plan = build_slice_plan(500, 1000, n_slices=5,
                            eta=0.30, gamma=1.0,
                            large_order_threshold_bps=10)
    d = plan.to_dict()
    assert d["total_quantity"] == 500
    assert d["n_slices"] == 5
    assert "steps" in d
    assert len(d["steps"]) == 5
    for s in d["steps"]:
        assert "qty" in s and "frac" in s and "slip_bps" in s


def test_build_plan_envelope_caps_extreme_slices():
    """Even with crazy params, no single slice exceeds max_frac of total."""
    plan = build_slice_plan(500, 1000, n_slices=5,
                            risk_aversion=100.0,   # very front-loaded
                            sigma=1.0, eta=0.01,
                            min_slice_frac=0.05, max_slice_frac=0.50,
                            large_order_threshold_bps=10)
    if plan.steps:
        for s in plan.steps:
            assert s.qty_fraction >= 0.05 - 1e-6
            assert s.qty_fraction <= 0.50 + 1e-6


def test_build_plan_btc_realistic_example():
    """Realistic Pulse example: $5K BTC entry on a 0.5 BTC/min average bar.
    BTC at $50K, qty = 0.1 BTC. 5-minute slice (10 bars later), bar vol ~0.5.
    Slicing into 5 of 0.02 each should be feasible."""
    plan = build_slice_plan(
        total_quantity=0.1, bar_volume_estimate=0.5, n_slices=5,
        eta=0.30, gamma=0.10,
        large_order_threshold_bps=20,
    )
    # 0.1 / 0.5 = 20% participation one-shot → big slip
    # 5 slices of 0.02 each = 4% participation each → much less slip
    assert plan.skipped_reason is None
    assert len(plan.steps) == 5
    one_shot_slip = _slip_bps_for_slice(0.1, 0.5, eta=0.30, gamma=0.10)
    assert plan.expected_total_slip_bps < one_shot_slip
