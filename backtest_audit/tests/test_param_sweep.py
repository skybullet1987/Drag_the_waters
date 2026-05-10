"""Tests for backtest_audit.param_sweep."""

from __future__ import annotations

import pytest

from backtest_audit.param_sweep import (
    DEFAULT_PARAM_RANGES, INT_PARAMS,
    latin_hypercube, map_to_param_space,
    generate_param_sweep, pulse_param_sweep,
    lhs_coverage_metric,
)


# ───────────────────────────────────────────────────────────────────────────────
# Default parameter ranges
# ───────────────────────────────────────────────────────────────────────────────

def test_default_ranges_has_12_params():
    assert len(DEFAULT_PARAM_RANGES) == 12


def test_default_ranges_all_have_low_lt_high():
    for name, (lo, hi) in DEFAULT_PARAM_RANGES.items():
        assert lo < hi, f"{name}: lo={lo} >= hi={hi}"


def test_int_params_subset_of_param_ranges():
    for ip in INT_PARAMS:
        assert ip in DEFAULT_PARAM_RANGES


# ───────────────────────────────────────────────────────────────────────────────
# latin_hypercube
# ───────────────────────────────────────────────────────────────────────────────

def test_lhs_invalid_args():
    with pytest.raises(ValueError):
        latin_hypercube(n_samples=0, n_dims=5)
    with pytest.raises(ValueError):
        latin_hypercube(n_samples=10, n_dims=0)


def test_lhs_returns_correct_shape():
    samples = latin_hypercube(n_samples=20, n_dims=12)
    assert len(samples) == 20
    assert all(len(s) == 12 for s in samples)


def test_lhs_values_in_unit_interval():
    samples = latin_hypercube(n_samples=30, n_dims=10)
    for s in samples:
        for v in s:
            assert 0.0 <= v <= 1.0


def test_lhs_seed_reproducible():
    a = latin_hypercube(n_samples=20, n_dims=5, seed=42)
    b = latin_hypercube(n_samples=20, n_dims=5, seed=42)
    assert a == b


def test_lhs_different_seeds_different_samples():
    a = latin_hypercube(n_samples=20, n_dims=5, seed=1)
    b = latin_hypercube(n_samples=20, n_dims=5, seed=2)
    assert a != b


def test_lhs_each_dim_has_exactly_one_sample_per_bin():
    """Critical LHS property: orthogonal projection has full coverage."""
    n = 20
    samples = latin_hypercube(n_samples=n, n_dims=4, seed=42)
    bin_size = 1.0 / n
    for d in range(4):
        bins = [int(s[d] / bin_size) for s in samples]
        assert len(set(bins)) == n   # exactly one sample per bin


def test_lhs_distribution_roughly_uniform():
    """Sample mean per dimension should be ~0.5 (uniform)."""
    samples = latin_hypercube(n_samples=200, n_dims=5, seed=42)
    for d in range(5):
        col = [s[d] for s in samples]
        mean = sum(col) / len(col)
        assert abs(mean - 0.5) < 0.05    # within 5% of uniform mean


# ───────────────────────────────────────────────────────────────────────────────
# map_to_param_space
# ───────────────────────────────────────────────────────────────────────────────

def test_map_to_param_space_returns_named_dicts():
    samples = latin_hypercube(n_samples=5, n_dims=12, seed=42)
    out = map_to_param_space(samples)
    assert len(out) == 5
    assert all(isinstance(d, dict) for d in out)
    for d in out:
        assert set(d.keys()) == set(DEFAULT_PARAM_RANGES.keys())


def test_map_values_within_named_ranges():
    samples = latin_hypercube(n_samples=20, n_dims=12, seed=42)
    out = map_to_param_space(samples)
    for d in out:
        for name, val in d.items():
            lo, hi = DEFAULT_PARAM_RANGES[name]
            assert lo <= val <= hi, f"{name}={val} not in [{lo}, {hi}]"


def test_map_int_params_are_integers():
    samples = latin_hypercube(n_samples=10, n_dims=12, seed=42)
    out = map_to_param_space(samples)
    for d in out:
        for ip in INT_PARAMS:
            assert isinstance(d[ip], int)


def test_map_dim_mismatch_raises():
    samples = [[0.1, 0.2, 0.3]]   # only 3 dims
    with pytest.raises(ValueError, match="dims but param_ranges"):
        map_to_param_space(samples)   # default ranges has 12 dims


# ───────────────────────────────────────────────────────────────────────────────
# generate_param_sweep + pulse_param_sweep top-level
# ───────────────────────────────────────────────────────────────────────────────

def test_generate_param_sweep_returns_id_param_pairs():
    out = generate_param_sweep(n_samples=10)
    assert len(out) == 10
    for pid, params in out:
        assert pid.startswith("sweep_")
        assert isinstance(params, dict)


def test_generate_param_sweep_unique_ids():
    out = generate_param_sweep(n_samples=50)
    ids = [pid for pid, _ in out]
    assert len(set(ids)) == 50


def test_pulse_param_sweep_default_size_50():
    out = pulse_param_sweep()
    assert len(out) == 50
    for pid, params in out:
        assert "entry_threshold" in params
        assert "max_position_usd" in params


def test_pulse_sweep_compatible_with_regime_runner():
    """Output must work as input to backtest_audit.regime_runner.sweep()."""
    from backtest_audit.regime_runner import sweep, REGIME_WINDOWS, WindowResult

    param_sets = pulse_param_sweep(n_samples=3)
    # Stub backtest_fn that just echoes a constant return
    def fake_bt(window, params):
        return WindowResult(
            window=window.name, net_return_pct=10.0, drawdown_pct=5.0,
            sharpe=1.0, win_rate_pct=50.0, trades=100,
        )
    sw = sweep(param_sets, fake_bt)
    assert len(sw.results) == 3


# ───────────────────────────────────────────────────────────────────────────────
# Coverage diagnostics
# ───────────────────────────────────────────────────────────────────────────────

def test_lhs_coverage_metric_basic():
    samples = latin_hypercube(n_samples=10, n_dims=3, seed=42)
    cov = lhs_coverage_metric(samples)
    assert cov["n_samples"] == 10
    assert cov["n_dims"] == 3
    # Perfect LHS → exactly 1 sample per bin per dim
    assert cov["min_bin_count_per_dim"] == 1
    assert cov["max_bin_count_per_dim"] == 1


def test_lhs_coverage_empty():
    cov = lhs_coverage_metric([])
    assert cov["n_samples"] == 0


def test_lhs_coverage_large_sweep():
    """100-sample / 12-dim sweep — diagnostic should report perfect coverage."""
    samples = latin_hypercube(n_samples=100, n_dims=12, seed=42)
    cov = lhs_coverage_metric(samples)
    assert cov["min_bin_count_per_dim"] == 1
    assert cov["max_bin_count_per_dim"] == 1
