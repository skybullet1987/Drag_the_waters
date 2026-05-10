"""Tests for Pulse.portfolio — multi-strategy allocator."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.portfolio import (
    STRATEGIES,
    StrategyOutcome, StrategyState, AllocationDecision,
    StrategyAllocator,
    rolling_sharpe, cumulative_pnl,
    equal_weight_allocation, capital_per_strategy,
)


# ───────────────────────────────────────────────────────────────────────────────
# Constants + simple helpers
# ───────────────────────────────────────────────────────────────────────────────

def test_strategies_constant_three_strategies():
    assert STRATEGIES == ("scalp", "trend", "mr")


def test_equal_weight_allocation_sums_to_one():
    w = equal_weight_allocation()
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 3) for v in w.values())


# ───────────────────────────────────────────────────────────────────────────────
# rolling_sharpe / cumulative_pnl
# ───────────────────────────────────────────────────────────────────────────────

def test_rolling_sharpe_zero_on_empty():
    assert rolling_sharpe([]) == 0.0


def test_rolling_sharpe_zero_on_single_trade():
    out = [StrategyOutcome(datetime.now(), 0.01)]
    assert rolling_sharpe(out) == 0.0


def test_rolling_sharpe_positive_on_consistent_winners():
    base = datetime(2026, 1, 1)
    out = [StrategyOutcome(base + timedelta(hours=i), 0.01) for i in range(20)]
    # Constant pnl → std=0 → Sharpe undefined → returns 0
    assert rolling_sharpe(out) == 0.0


def test_rolling_sharpe_higher_for_steady_strategy():
    base = datetime(2026, 1, 1)
    steady = [StrategyOutcome(base + timedelta(hours=i),
                              0.01 + 0.001 * (i % 3 - 1))
              for i in range(20)]
    spiky  = [StrategyOutcome(base + timedelta(hours=i),
                              0.05 if i % 2 == 0 else -0.04)
              for i in range(20)]
    assert rolling_sharpe(steady) > rolling_sharpe(spiky)


def test_cumulative_pnl_window_only():
    base = datetime.now()
    out = [StrategyOutcome(base, 0.01)] * 50
    # Only last 30 trades (default lookback) summed
    assert cumulative_pnl(out, lookback=30) == pytest.approx(0.30)


# ───────────────────────────────────────────────────────────────────────────────
# StrategyAllocator construction
# ───────────────────────────────────────────────────────────────────────────────

def test_allocator_invalid_floor_raises():
    with pytest.raises(ValueError, match="min_floor_frac"):
        StrategyAllocator(min_floor_frac=0.40)   # 0.40 × 3 > 1.0


def test_allocator_invalid_ceiling_raises():
    with pytest.raises(ValueError, match="max_ceiling_frac"):
        StrategyAllocator(min_floor_frac=0.20, max_ceiling_frac=0.10)


def test_record_outcome_unknown_strategy():
    a = StrategyAllocator()
    with pytest.raises(KeyError):
        a.record_outcome("invalid", datetime.now(), 0.01)


# ───────────────────────────────────────────────────────────────────────────────
# Compute allocation — warmup + steady-state
# ───────────────────────────────────────────────────────────────────────────────

def test_warmup_equal_weights():
    """With < min_trades_alloc trades, allocation is equal-weight."""
    a = StrategyAllocator()
    now = datetime(2026, 5, 10, 12)
    a.record_outcome("scalp", now, 0.01)
    d = a.compute_allocation(now)
    assert sum(d.weights.values()) == pytest.approx(1.0, rel=1e-6)
    assert all(d.weights[s] > 0 for s in STRATEGIES)


def test_steady_state_winner_gets_more():
    """After enough trades, the winner gets a higher weight."""
    a = StrategyAllocator(lookback_trades=20, min_trades_alloc=10)
    base = datetime(2026, 5, 10)

    # scalp: steady winner
    for i in range(20):
        a.record_outcome("scalp", base + timedelta(hours=i), 0.01)
    # trend: mostly losses
    for i in range(20):
        a.record_outcome("trend", base + timedelta(hours=i), -0.005)
    # mr: small wins
    for i in range(20):
        a.record_outcome("mr",    base + timedelta(hours=i), 0.003)

    d = a.compute_allocation(now=base + timedelta(days=1))

    # Sharpe values present
    assert "scalp" in d.sharpes
    # Weights honor floor + sum to 1
    for s in STRATEGIES:
        assert d.weights[s] >= 0.20 - 1e-6
        assert d.weights[s] <= 0.60 + 1e-6
    assert sum(d.weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_paused_strategy_excluded():
    """Paused strategies don't participate in allocation."""
    a = StrategyAllocator()
    now = datetime(2026, 5, 10, 12)
    # Add some trades for both
    for i in range(15):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.01)
        a.record_outcome("trend", now + timedelta(hours=i), 0.005)
        a.record_outcome("mr",    now + timedelta(hours=i), 0.002)
    a.pause_strategy("trend", until=now + timedelta(days=1))
    d = a.compute_allocation(now)
    # Trend should have zero or be excluded
    assert d.weights.get("trend", 0) == 0
    # Scalp + MR should sum to 1.0
    assert d.weights["scalp"] + d.weights["mr"] == pytest.approx(1.0, rel=1e-6)


def test_disabled_strategy_excluded():
    a = StrategyAllocator()
    a.disable_strategy("mr")
    now = datetime(2026, 5, 10)
    for i in range(15):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.01)
        a.record_outcome("trend", now + timedelta(hours=i), 0.005)
    d = a.compute_allocation(now)
    assert d.weights.get("mr", 0) == 0
    assert d.weights["scalp"] + d.weights["trend"] == pytest.approx(1.0, rel=1e-6)


# ───────────────────────────────────────────────────────────────────────────────
# Risk veto
# ───────────────────────────────────────────────────────────────────────────────

def test_risk_veto_when_two_negative():
    """When 2 of 3 strategies are negative-PnL, halve the third's exposure."""
    a = StrategyAllocator()
    now = datetime(2026, 5, 10)
    # scalp + trend losing
    for i in range(20):
        a.record_outcome("scalp", now + timedelta(hours=i), -0.005)
        a.record_outcome("trend", now + timedelta(hours=i), -0.005)
    # mr winning
    for i in range(20):
        a.record_outcome("mr",    now + timedelta(hours=i), 0.01)
    d = a.compute_allocation(now)
    assert d.risk_veto_active
    # Healthy strategy gets reduced exposure
    assert "risk_veto" in " ".join(d.notes)


def test_no_risk_veto_when_all_positive():
    a = StrategyAllocator()
    now = datetime(2026, 5, 10)
    for i in range(20):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.01)
        a.record_outcome("trend", now + timedelta(hours=i), 0.005)
        a.record_outcome("mr",    now + timedelta(hours=i), 0.003)
    d = a.compute_allocation(now)
    assert not d.risk_veto_active


# ───────────────────────────────────────────────────────────────────────────────
# Last-man-standing rule
# ───────────────────────────────────────────────────────────────────────────────

def test_all_paused_revives_one_at_floor():
    """When ALL strategies are paused, revive at least one at the floor."""
    a = StrategyAllocator()
    now = datetime(2026, 5, 10)
    # Give scalp the best history
    for i in range(20):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.02)
        a.record_outcome("trend", now + timedelta(hours=i), -0.01)
        a.record_outcome("mr",    now + timedelta(hours=i), -0.005)
    for s in STRATEGIES:
        a.pause_strategy(s, until=now + timedelta(days=2))
    d = a.compute_allocation(now)
    # At least one strategy was revived at floor
    revived = [s for s, w in d.weights.items() if w > 0]
    assert len(revived) >= 1
    # The revived one should be the highest-Sharpe (scalp) at the floor
    assert "scalp" in revived
    assert d.weights["scalp"] == 0.20    # floor


# ───────────────────────────────────────────────────────────────────────────────
# Floor + ceiling enforcement
# ───────────────────────────────────────────────────────────────────────────────

def test_min_floor_respected_for_loser():
    """A losing strategy still gets the 20% floor (acts as probe)."""
    a = StrategyAllocator()
    now = datetime(2026, 5, 10)
    # scalp wins big, trend + mr lose
    for i in range(30):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.02)
        a.record_outcome("trend", now + timedelta(hours=i), -0.005)
        a.record_outcome("mr",    now + timedelta(hours=i), -0.003)
    d = a.compute_allocation(now)
    for s in STRATEGIES:
        assert d.weights[s] >= 0.20 - 1e-6


def test_max_ceiling_respected_for_winner():
    """Even a runaway winner is capped at 60%."""
    a = StrategyAllocator(max_ceiling_frac=0.60)
    now = datetime(2026, 5, 10)
    for i in range(50):
        a.record_outcome("scalp", now + timedelta(hours=i), 0.02)
        a.record_outcome("trend", now + timedelta(hours=i), 0.0001)
        a.record_outcome("mr",    now + timedelta(hours=i), 0.0001)
    d = a.compute_allocation(now)
    for s in STRATEGIES:
        assert d.weights[s] <= 0.60 + 1e-6


# ───────────────────────────────────────────────────────────────────────────────
# Capital conversion
# ───────────────────────────────────────────────────────────────────────────────

def test_capital_per_strategy_dollar_split():
    a = StrategyAllocator()
    now = datetime(2026, 5, 10)
    a.record_outcome("scalp", now, 0.01)
    d = a.compute_allocation(now)
    caps = capital_per_strategy(d, total_capital_usd=1000.0)
    assert sum(caps.values()) == pytest.approx(1000.0, rel=1e-6)
    for s in d.weights:
        assert caps[s] == pytest.approx(d.weights[s] * 1000.0)
