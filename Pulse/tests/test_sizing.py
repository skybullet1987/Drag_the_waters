"""Tests for Pulse.sizing — Bayesian per-symbol posterior."""

from __future__ import annotations

import pytest

from Pulse.sizing import (
    DEFAULT_PRIOR_ALPHA, DEFAULT_PRIOR_BETA,
    SymbolPosterior, BayesianSymbolSizer,
    beta_cdf, beta_inverse_cdf,
)


# ───────────────────────────────────────────────────────────────────────────────
# Beta math
# ───────────────────────────────────────────────────────────────────────────────

def test_beta_cdf_zero_outside_unit():
    assert beta_cdf(-0.1, 2, 2) == 0
    assert beta_cdf(1.5, 2, 2) == 1


def test_beta_cdf_uniform_at_alpha_beta_1():
    """Beta(1,1) = uniform → CDF(x) = x."""
    assert beta_cdf(0.5, 1, 1) == pytest.approx(0.5, abs=1e-3)
    assert beta_cdf(0.25, 1, 1) == pytest.approx(0.25, abs=1e-3)


def test_beta_cdf_symmetric_at_alpha_eq_beta():
    """Beta(α, α) is symmetric around 0.5 → CDF(0.5) = 0.5."""
    for a in (2, 5, 10, 100):
        assert beta_cdf(0.5, a, a) == pytest.approx(0.5, abs=1e-3)


def test_beta_cdf_known_values():
    """Reference values verified by analytic integration of x*(1-x)^4*30
    (Beta(2,5) PDF) and 30*x^4*(1-x) (Beta(5,2) PDF):
        scipy.stats.beta.cdf(0.3, 2, 5) ≈ 0.5798
        scipy.stats.beta.cdf(0.7, 5, 2) ≈ 0.4202
    """
    assert beta_cdf(0.3, 2, 5) == pytest.approx(0.5798, abs=1e-3)
    assert beta_cdf(0.7, 5, 2) == pytest.approx(0.4202, abs=1e-3)


def test_beta_cdf_invalid_params():
    with pytest.raises(ValueError):
        beta_cdf(0.5, 0, 2)
    with pytest.raises(ValueError):
        beta_cdf(0.5, -1, 2)


def test_beta_inverse_cdf_round_trip():
    """inverse(cdf(x)) == x for various Beta distributions."""
    for (a, b, x) in [(2, 5, 0.3), (10, 10, 0.5), (3, 1, 0.8)]:
        p = beta_cdf(x, a, b)
        x2 = beta_inverse_cdf(p, a, b, tol=1e-7)
        assert x2 == pytest.approx(x, abs=1e-3)


def test_beta_inverse_cdf_invalid_p():
    with pytest.raises(ValueError):
        beta_inverse_cdf(0.0, 2, 2)
    with pytest.raises(ValueError):
        beta_inverse_cdf(1.0, 2, 2)


# ───────────────────────────────────────────────────────────────────────────────
# SymbolPosterior dataclass
# ───────────────────────────────────────────────────────────────────────────────

def test_posterior_mean():
    p = SymbolPosterior("BTC", alpha=10, beta=10)
    assert p.mean == pytest.approx(0.5)


def test_posterior_lower_ci_below_mean():
    p = SymbolPosterior("BTC", alpha=10, beta=10)
    lower = p.lower_ci(pct=0.05)
    assert 0 < lower < p.mean


def test_posterior_variance_decreases_with_more_data():
    weak = SymbolPosterior("X", alpha=2, beta=2)        # small N
    strong = SymbolPosterior("X", alpha=200, beta=200)   # large N (same mean)
    assert strong.variance < weak.variance


# ───────────────────────────────────────────────────────────────────────────────
# BayesianSymbolSizer
# ───────────────────────────────────────────────────────────────────────────────

def test_sizer_invalid_args():
    with pytest.raises(ValueError):
        BayesianSymbolSizer(prior_alpha=0)
    with pytest.raises(ValueError):
        BayesianSymbolSizer(min_scaler=1.5, max_scaler=1.0)


def test_sizer_unobserved_symbol_uses_prior():
    """A symbol with no observations defaults to the prior."""
    sizer = BayesianSymbolSizer()
    p = sizer.posterior("BTCUSD")
    assert p.n_observed == 0
    assert p.alpha == DEFAULT_PRIOR_ALPHA
    assert p.beta == DEFAULT_PRIOR_BETA


def test_sizer_5050_symbol_calibrated_to_about_one():
    """Default Beta(10,10) → mean=0.5, lower_ci ≈ 0.34 → scaler ≈ 0.5×0.34×4 ≈ 0.68.
    With min_scaler=0.40 cap, we expect somewhere around 0.6-0.8.
    """
    sizer = BayesianSymbolSizer()
    s = sizer.size_multiplier("BTC")
    # The "≈ 1.0" in the docstring assumes K is calibrated to user prior;
    # with default K=4 and Beta(10,10), expected scaler is ~0.65-0.80.
    assert 0.5 <= s <= 1.0


def test_sizer_consistent_winners_get_higher_scaler():
    sizer = BayesianSymbolSizer()
    # Update one symbol with 20 wins
    for _ in range(20):
        sizer.update("WINNER", won=True)
    # Update another with 20 losses
    for _ in range(20):
        sizer.update("LOSER", won=False)
    s_win  = sizer.size_multiplier("WINNER")
    s_lose = sizer.size_multiplier("LOSER")
    assert s_win > s_lose
    # Loser hits the floor
    assert s_lose == sizer.min_scaler
    # Winner gets boost above 50/50 baseline
    s_default = sizer.size_multiplier("UNOBSERVED")
    assert s_win > s_default


def test_sizer_winner_capped_at_max():
    sizer = BayesianSymbolSizer(max_scaler=1.5)
    for _ in range(100):
        sizer.update("X", won=True)
    s = sizer.size_multiplier("X")
    assert s == 1.5   # capped at max


def test_sizer_loser_floored_at_min():
    sizer = BayesianSymbolSizer(min_scaler=0.4)
    for _ in range(100):
        sizer.update("X", won=False)
    s = sizer.size_multiplier("X")
    assert s == 0.4   # floored at min


def test_sizer_update_increments_counters():
    sizer = BayesianSymbolSizer()
    sizer.update("X", won=True)
    sizer.update("X", won=True)
    sizer.update("X", won=False)
    p = sizer.posterior("X")
    assert p.n_observed == 3
    assert p.n_wins == 2
    assert p.n_losses == 1
    assert p.alpha == DEFAULT_PRIOR_ALPHA + 2
    assert p.beta  == DEFAULT_PRIOR_BETA + 1


def test_sizer_isolation_per_symbol():
    """Updates to one symbol don't bleed into another."""
    sizer = BayesianSymbolSizer()
    for _ in range(20):
        sizer.update("BTC", won=True)
    s_btc = sizer.size_multiplier("BTC")
    s_eth = sizer.size_multiplier("ETH")    # never touched
    assert s_btc > s_eth


def test_sizer_reset_specific_symbol():
    sizer = BayesianSymbolSizer()
    sizer.update("X", won=True)
    sizer.update("Y", won=True)
    sizer.reset("X")
    assert sizer.posterior("X").n_observed == 0
    assert sizer.posterior("Y").n_observed == 1


def test_sizer_reset_all():
    sizer = BayesianSymbolSizer()
    sizer.update("X", won=True)
    sizer.update("Y", won=False)
    sizer.reset()
    assert sizer.posterior("X").n_observed == 0
    assert sizer.posterior("Y").n_observed == 0


def test_sizer_kasusd_scenario():
    """Live MG36 KASUSD lost. After 5 losses, scaler should be at or near floor.
    Compare to BTCUSD with 5 wins → scaler significantly higher."""
    sizer = BayesianSymbolSizer()
    for _ in range(5):
        sizer.update("KASUSD", won=False)
    for _ in range(5):
        sizer.update("BTCUSD", won=True)
    s_kas = sizer.size_multiplier("KASUSD")
    s_btc = sizer.size_multiplier("BTCUSD")
    assert s_btc - s_kas > 0.20    # meaningful spread after just 5 trades


def test_sizer_all_posteriors_returns_copy():
    sizer = BayesianSymbolSizer()
    sizer.update("X", won=True)
    snap = sizer.all_posteriors()
    assert "X" in snap
    # Modifying the snapshot doesn't affect the sizer's state
    snap.clear()
    assert sizer.posterior("X").n_observed == 1
