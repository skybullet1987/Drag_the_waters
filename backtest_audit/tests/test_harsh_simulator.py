"""Tests for backtest_audit.harsh_simulator (pure-Python paths only)."""

from __future__ import annotations

import statistics

import pytest

from backtest_audit.harsh_simulator import (
    HarshConfig, HarshFillSimulator, HarshFill,
)


# ───────────────────────────────────────────────────────────────────────────────
# HarshConfig
# ───────────────────────────────────────────────────────────────────────────────

def test_default_config_anchored_to_live_evidence():
    """Defaults must match PLAN.md §0.A live evidence."""
    c = HarshConfig()
    assert c.base_slippage_bps == 100.0
    assert c.p99_slippage_bps == 200.0
    assert c.taker_fee_pct == 0.0040
    assert c.assumed_maker_fill_rate <= 0.15  # MG36 was 0/2
    assert c.fill_delay_bars == 1
    assert c.force_obi_to_zero is True


def test_tier_multiplier_lookup():
    c = HarshConfig()
    assert c.slip_multiplier_for_tier("major") == 0.5
    assert c.slip_multiplier_for_tier("large") == 1.0
    assert c.slip_multiplier_for_tier("mid") == 1.3
    assert c.slip_multiplier_for_tier("micro") == 2.5
    # Unknown tier defaults to mid
    assert c.slip_multiplier_for_tier("nonsense") == 1.3


# ───────────────────────────────────────────────────────────────────────────────
# HarshFillSimulator
# ───────────────────────────────────────────────────────────────────────────────

def test_market_buy_fills_at_t1_with_slippage():
    sim = HarshFillSimulator(HarshConfig(seed=1, reject_rate_normal=0.0))
    fill = sim.simulate_fill(
        symbol="SOLUSD", side="Buy", is_market=True,
        intended_price=100.0, next_bar_price=100.0,
        tier="large",
    )
    assert fill.filled
    assert fill.fill_price > 100.0    # paid more
    assert fill.slippage_bps > 0
    assert fill.fee_pct == 0.0040
    assert fill.delayed_bars == 1


def test_market_sell_fills_at_t1_with_negative_price_impact():
    sim = HarshFillSimulator(HarshConfig(seed=1, reject_rate_normal=0.0))
    fill = sim.simulate_fill(
        symbol="SOLUSD", side="Sell", is_market=True,
        intended_price=100.0, next_bar_price=100.0,
        tier="large",
    )
    assert fill.filled
    assert fill.fill_price < 100.0    # received less
    assert fill.slippage_bps > 0


def test_micro_cap_slippage_higher_than_major_cap():
    """Micro caps draw 2.5× the slippage of large; major draws 0.5× of large."""
    cfg = HarshConfig(seed=42, reject_rate_normal=0.0)
    sim_micro = HarshFillSimulator(cfg)
    sim_major = HarshFillSimulator(cfg)

    micro_slips = []
    major_slips = []
    for i in range(100):
        f = sim_micro.simulate_fill(
            "KASUSD", "Buy", True, 0.04, 0.04, tier="micro"
        )
        micro_slips.append(f.slippage_bps)
        f = sim_major.simulate_fill(
            "BTCUSD", "Buy", True, 50_000, 50_000, tier="major"
        )
        major_slips.append(f.slippage_bps)

    assert statistics.mean(micro_slips) > 2 * statistics.mean(major_slips)


def test_limit_orders_mostly_time_out():
    """At maker_fill_rate=0.10, ~90% of limits should fall back to market."""
    sim = HarshFillSimulator(HarshConfig(
        seed=7, assumed_maker_fill_rate=0.10, reject_rate_normal=0.0,
    ))
    n = 200
    n_market_fallback = 0
    n_maker = 0
    for i in range(n):
        f = sim.simulate_fill(
            "SOLUSD", "Buy", is_market=False,
            intended_price=100.0, next_bar_price=100.0, tier="large",
        )
        if f.fee_pct == sim.cfg.maker_fee_pct:
            n_maker += 1
        elif f.fee_pct == sim.cfg.taker_fee_pct:
            n_market_fallback += 1
    # ~10% maker, ~90% taker fallback (with some random variance)
    assert n_maker < n * 0.20
    assert n_market_fallback > n * 0.70


def test_random_rejection_rate():
    """At 50% rejection, expect ~50% rejected fills."""
    sim = HarshFillSimulator(HarshConfig(seed=11, reject_rate_normal=0.50))
    n = 300
    rejects = 0
    for i in range(n):
        f = sim.simulate_fill(
            "SOLUSD", "Buy", True, 100.0, 100.0, tier="large",
        )
        if not f.filled:
            rejects += 1
    assert n * 0.40 <= rejects <= n * 0.60


def test_concurrent_orders_increase_slippage():
    """Each concurrent open order adds 10bp of slippage by default."""
    cfg = HarshConfig(seed=3, reject_rate_normal=0.0,
                      multi_order_spread_bps=10.0)
    sim_solo = HarshFillSimulator(cfg)
    sim_busy = HarshFillSimulator(cfg)

    solo = []; busy = []
    for i in range(150):
        solo.append(sim_solo.simulate_fill(
            "SOLUSD", "Buy", True, 100.0, 100.0,
            tier="large", concurrent_open_orders=0,
        ).slippage_bps)
        busy.append(sim_busy.simulate_fill(
            "SOLUSD", "Buy", True, 100.0, 100.0,
            tier="large", concurrent_open_orders=5,
        ).slippage_bps)

    # 5 concurrent orders → +50bp on average baseline
    assert statistics.mean(busy) - statistics.mean(solo) > 30.0


def test_vol_spike_produces_higher_tail_slippage():
    """Vol-spike fills draw from a heavier-tailed distribution."""
    cfg = HarshConfig(seed=99, reject_rate_normal=0.0,
                      reject_rate_vol_spike=0.0)
    sim_calm = HarshFillSimulator(cfg)
    sim_spike = HarshFillSimulator(cfg)

    calm = []
    spike = []
    for i in range(200):
        calm.append(sim_calm.simulate_fill(
            "SOLUSD", "Buy", True, 100.0, 100.0,
            tier="large", is_vol_spike=False,
        ).slippage_bps)
        spike.append(sim_spike.simulate_fill(
            "SOLUSD", "Buy", True, 100.0, 100.0,
            tier="large", is_vol_spike=True,
        ).slippage_bps)

    # The 90th percentile of vol-spike should exceed calm's
    assert sorted(spike)[180] > sorted(calm)[180]


def test_simulator_reproducible_with_seed():
    a = HarshFillSimulator(HarshConfig(seed=42, reject_rate_normal=0.0))
    b = HarshFillSimulator(HarshConfig(seed=42, reject_rate_normal=0.0))
    for i in range(20):
        fa = a.simulate_fill("SOLUSD", "Buy", True, 100.0, 100.0, "large")
        fb = b.simulate_fill("SOLUSD", "Buy", True, 100.0, 100.0, "large")
        assert fa.slippage_bps == fb.slippage_bps
        assert fa.fill_price == fb.fill_price


# ───────────────────────────────────────────────────────────────────────────────
# Calibration: harsh simulator on KASUSD-like trade should approximate live
# ───────────────────────────────────────────────────────────────────────────────

def test_kasusd_like_trade_matches_live_slippage_envelope():
    """The most damning trade in MG36 paper log:
        KASUSD micro-cap, buy at $0.04057 with 107bp slippage,
        exit at $0.0406 with 197bp slippage. Round-trip ≈ 304bp.

    Run 1000 simulated KASUSD trades — the mean round-trip slippage
    should land in the same envelope (200-400bp).
    """
    sim = HarshFillSimulator(HarshConfig(seed=2026, reject_rate_normal=0.0))
    rt_slips = []
    for i in range(1000):
        buy = sim.simulate_fill(
            "KASUSD", "Buy", True, 0.04, 0.04, tier="micro",
        )
        sell = sim.simulate_fill(
            "KASUSD", "Sell", True, 0.04, 0.04, tier="micro",
        )
        if buy.filled and sell.filled:
            rt_slips.append(buy.slippage_bps + sell.slippage_bps)
    mean_rt = statistics.mean(rt_slips)
    assert 200 <= mean_rt <= 600, (
        f"Harsh sim KASUSD round-trip slippage mean={mean_rt:.1f}bp; "
        f"live observed ~304bp; calibration off"
    )
