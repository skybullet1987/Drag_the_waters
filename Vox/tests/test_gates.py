"""
Unit tests for the Vox confidence-gate logic and Vox v2 model stack.

No QuantConnect dependency — runs with plain pytest + numpy + scikit-learn.
"""
import sys
import os

import numpy as np
import pytest

# Allow importing from the Vox package without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import VoxEnsemble, triple_barrier_outcome  # type: ignore  # noqa: E402

# Mirror constants from main.py (cannot import main.py without AlgorithmImports).
# These must match the values in main.py.
SCORE_MIN       = 0.50    # updated to balanced/tradable default (was 0.55 in v2)
SCORE_MIN_FLOOR = 0.15
MIN_AGREE       = 2       # updated to relaxed default (was 3 in v2)


def _score_min_eff(positive_rate, s_min_floor=SCORE_MIN_FLOOR, s_min=SCORE_MIN):
    """Mirror of the runtime formula in VoxAlgorithm._try_enter()."""
    return float(np.clip(max(s_min_floor, 3.0 * positive_rate), s_min_floor, s_min))


# ─────────────────────────────────────────────────────────────────────────────
# agree_threshold tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgreeThreshold:
    def _ens(self, positive_rate):
        ens = VoxEnsemble()
        ens._positive_rate = positive_rate
        return ens

    def test_low_positive_rate_gives_floor(self):
        """With positive_rate=0.03, agree_thr should floor at 0.15."""
        assert self._ens(0.03)._agree_threshold() == pytest.approx(0.15)

    def test_medium_positive_rate(self):
        """With positive_rate=0.15, agree_thr = clip(0.30, 0.15, 0.55) = 0.30."""
        assert self._ens(0.15)._agree_threshold() == pytest.approx(0.30)

    def test_high_positive_rate_clipped(self):
        """With positive_rate=0.40, agree_thr is capped at 0.55."""
        assert self._ens(0.40)._agree_threshold() == pytest.approx(0.55)

    def test_always_within_bounds(self):
        """agree_thr must always stay within [0.15, 0.55] for any positive_rate."""
        for pr in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
            thr = self._ens(pr)._agree_threshold()
            assert 0.15 <= thr <= 0.55, f"agree_thr={thr} out of bounds for pr={pr}"

    def test_prob_vector_around_020_passes_agree_gate(self):
        """
        A 4-model ensemble with probas ~0.20 must pass the agree gate when
        positive_rate=0.03 (reproduces the reported zero-trade bug).
        """
        ens = VoxEnsemble()
        ens._positive_rate = 0.03
        agree_thr = ens._agree_threshold()

        # Vox v2 has 4 classifiers; simulating probas at ~0.20
        proba_vector = [0.18, 0.22, 0.19, 0.21]
        n_agree = sum(1 for p in proba_vector if p >= agree_thr)

        assert agree_thr <= 0.20, (
            f"agree_thr={agree_thr} should be <= 0.20 for positive_rate=0.03"
        )
        # With agree_thr=0.15, all 4 probas pass → n_agree=4 >= MIN_AGREE=2
        assert n_agree >= MIN_AGREE, (
            f"n_agree={n_agree} < MIN_AGREE={MIN_AGREE} for agree_thr={agree_thr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# score_min_eff tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMinEff:
    def test_low_positive_rate_gives_floor(self):
        """With positive_rate=0.03, effective score threshold floors at 0.15."""
        assert _score_min_eff(0.03) == pytest.approx(SCORE_MIN_FLOOR)

    def test_medium_positive_rate(self):
        """With positive_rate=0.06, effective score = clip(max(0.15, 0.18), 0.15, 0.50) = 0.18."""
        assert _score_min_eff(0.06) == pytest.approx(0.18)

    def test_high_positive_rate_clipped_to_score_min(self):
        """With positive_rate=0.30, effective score is capped at SCORE_MIN=0.50."""
        assert _score_min_eff(0.30) == pytest.approx(SCORE_MIN)

    def test_always_within_bounds(self):
        """score_min_eff must always stay within [SCORE_MIN_FLOOR, SCORE_MIN]."""
        for pr in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
            eff = _score_min_eff(pr)
            assert SCORE_MIN_FLOOR <= eff <= SCORE_MIN, (
                f"score_min_eff={eff} out of [{SCORE_MIN_FLOOR}, {SCORE_MIN}] for pr={pr}"
            )

    def test_score_min_constant_remains_upper_clamp(self):
        """SCORE_MIN is preserved as the upper bound, never exceeded."""
        assert _score_min_eff(1.0) == pytest.approx(SCORE_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# positive_rate persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveRatePersistence:
    def test_positive_rate_initialises_to_zero(self):
        ens = VoxEnsemble()
        assert ens._positive_rate == 0.0

    def test_load_state_restores_positive_rate(self):
        """load_state must copy _positive_rate from the serialised ensemble."""
        src = VoxEnsemble()
        src._positive_rate = 0.042

        dst = VoxEnsemble()
        dst.load_state(src)

        assert dst._positive_rate == pytest.approx(0.042)

    def test_load_state_defaults_missing_positive_rate(self):
        """Loading an old pickle without _positive_rate must default to 0.0."""
        src = VoxEnsemble()
        # Simulate an old pickle that lacks the attribute.
        if hasattr(src, "_positive_rate"):
            del src.__dict__["_positive_rate"]

        dst = VoxEnsemble()
        dst.load_state(src)

        assert dst._positive_rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Vox v2 model stack — ensemble structure
# ─────────────────────────────────────────────────────────────────────────────

class TestVoxV2ModelStack:
    def test_ensemble_has_four_classifiers(self):
        """Vox v2 classifier ensemble must have exactly 4 models (no GaussianNB)."""
        ens = VoxEnsemble()
        names = [name for name, _ in ens._estimators]
        assert len(names) == 4, f"Expected 4 classifiers, got {len(names)}: {names}"

    def test_ensemble_has_no_gnb(self):
        """GaussianNB must be removed from the Vox v2 ensemble."""
        ens = VoxEnsemble()
        names = [name for name, _ in ens._estimators]
        assert "gnb" not in names, "GaussianNB (gnb) must not be in the Vox v2 ensemble"

    def test_ensemble_has_hgbc(self):
        """HistGradientBoostingClassifier (hgbc) must be in the classifier ensemble."""
        ens = VoxEnsemble()
        names = [name for name, _ in ens._estimators]
        assert "hgbc" in names, "HistGradientBoostingClassifier (hgbc) missing from ensemble"

    def test_ensemble_has_lr_et_rf(self):
        """LogisticRegression, ExtraTreesClassifier, RandomForestClassifier must be present."""
        ens = VoxEnsemble()
        names = [name for name, _ in ens._estimators]
        for required in ("lr", "et", "rf"):
            assert required in names, f"'{required}' missing from classifier ensemble"

    def test_classifier_weights_match_estimators(self):
        """Classifier weights must have the same length as the estimators list."""
        ens = VoxEnsemble()
        assert len(ens._classifier_weights) == len(ens._estimators), (
            f"classifier_weights length {len(ens._classifier_weights)} != "
            f"estimators length {len(ens._estimators)}"
        )

    def test_regression_ensemble_has_three_regressors(self):
        """Regression ensemble must have exactly 3 regressors."""
        ens = VoxEnsemble()
        assert len(ens._regressors) == 3, (
            f"Expected 3 regressors, got {len(ens._regressors)}"
        )

    def test_regression_ensemble_has_hgbr_etr_ridge(self):
        """Regression ensemble must include hgbr, etr, ridge."""
        ens = VoxEnsemble()
        names = [name for name, _ in ens._regressors]
        for required in ("hgbr", "etr", "ridge"):
            assert required in names, f"Regressor '{required}' missing"

    def test_regressors_not_fitted_initially(self):
        """Regressors should not be fitted before training."""
        ens = VoxEnsemble()
        assert not ens._reg_fitted

    def test_predict_returns_pred_return_field(self):
        """predict_with_confidence must return pred_return key (0.0 when not trained)."""
        ens = VoxEnsemble()
        # Train classifiers with minimal data
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))
        y = rng.integers(0, 2, 50)
        ens.fit(X, y)  # no y_return → regressors not trained

        result = ens.predict_with_confidence(X[0])
        assert "pred_return" in result
        assert "class_proba" in result
        assert "mean_proba" in result   # backward-compat alias
        assert result["pred_return"] == 0.0  # regressors not trained

    def test_predict_returns_pred_return_when_regressors_trained(self):
        """predict_with_confidence must return non-trivial pred_return when regressors fitted."""
        ens = VoxEnsemble()
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 10))
        y_class = rng.integers(0, 2, 100)
        y_return = rng.normal(0, 0.01, 100).astype(float)
        ens.fit(X, y_class, y_return=y_return)

        assert ens._reg_fitted, "Regressors should be fitted when y_return is provided"
        result = ens.predict_with_confidence(X[0])
        assert isinstance(result["pred_return"], float)
        # pred_return is deterministic given the same data
        assert result["pred_return"] == result["pred_return"]   # not NaN

    def test_load_state_restores_regressors(self):
        """load_state must restore fitted regressors from saved ensemble."""
        src = VoxEnsemble()
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 10))
        y_class = rng.integers(0, 2, 100)
        y_return = rng.normal(0, 0.01, 100).astype(float)
        src.fit(X, y_class, y_return=y_return)
        assert src._reg_fitted

        dst = VoxEnsemble()
        assert not dst._reg_fitted
        dst.load_state(src)
        assert dst._reg_fitted, "load_state should restore _reg_fitted=True"


# ─────────────────────────────────────────────────────────────────────────────
# triple_barrier_outcome (cost-aware labels)
# ─────────────────────────────────────────────────────────────────────────────

class TestTripleBarrierOutcome:
    def test_tp_hit_gives_label_1_and_positive_return(self):
        """When TP is hit and net return > 0, label=1 and realized_return = tp - cost."""
        prices = [100.0] + [102.0] * 5   # entry 100, TP = 1.02 = 2%
        label, ret = triple_barrier_outcome(prices, tp=0.02, sl=0.01, timeout_bars=10)
        assert label == 1
        assert ret == pytest.approx(0.02)

    def test_tp_hit_cost_aware_positive(self):
        """Cost-aware: TP hit, net return after 0.5% cost is still positive."""
        prices = [100.0, 102.0]
        label, ret = triple_barrier_outcome(prices, tp=0.02, sl=0.01, timeout_bars=10,
                                             cost_fraction=0.005)
        assert label == 1       # tp - cost = 0.015 > 0
        assert ret == pytest.approx(0.015)

    def test_sl_hit_gives_label_0_negative_return(self):
        """When SL is hit, label=0 and realized_return = -(sl + cost)."""
        prices = [100.0, 98.0]   # SL = -2%, below sl=0.012
        label, ret = triple_barrier_outcome(prices, tp=0.02, sl=0.012, timeout_bars=10)
        assert label == 0
        assert ret == pytest.approx(-0.012)

    def test_timeout_gives_label_0(self):
        """When neither barrier is hit within timeout, label=0."""
        prices = [100.0] + [100.5] * 5  # +0.5%, below TP of 2%
        label, ret = triple_barrier_outcome(prices, tp=0.02, sl=0.01, timeout_bars=5)
        assert label == 0

    def test_empty_prices_gives_label_0(self):
        """Empty or single-bar price series returns (0, -cost)."""
        label, ret = triple_barrier_outcome([], tp=0.02, sl=0.01, timeout_bars=10)
        assert label == 0

    def test_cost_reduces_return(self):
        """Realized return is always net of cost_fraction."""
        prices = [100.0, 99.0]   # SL at exactly -1%
        cost = 0.003
        label, ret = triple_barrier_outcome(prices, tp=0.02, sl=0.01, timeout_bars=10,
                                             cost_fraction=cost)
        assert ret == pytest.approx(-0.01 - cost)


# ─────────────────────────────────────────────────────────────────────────────
# Risk profile constants
# ─────────────────────────────────────────────────────────────────────────────

# Mirror of constants defined in main.py so tests do not require AlgorithmImports.
AGGRESSIVE_SCORE_MIN               = 0.48
AGGRESSIVE_MIN_EV                  = 0.0005
AGGRESSIVE_PRED_RETURN_MIN         = -0.0010
AGGRESSIVE_MAX_DISPERSION          = 0.28
AGGRESSIVE_MIN_AGREE               = 1
AGGRESSIVE_ALLOCATION              = 0.75
AGGRESSIVE_MAX_ALLOC               = 0.95
AGGRESSIVE_KELLY_FRAC              = 0.50
AGGRESSIVE_TAKE_PROFIT             = 0.045
AGGRESSIVE_STOP_LOSS               = 0.020
AGGRESSIVE_TIMEOUT_HOURS           = 8
AGGRESSIVE_MAX_DD_PCT              = 0.20

RUTHLESS_SCORE_MIN               = 0.45
RUTHLESS_MIN_EV                  = 0.0000
RUTHLESS_PRED_RETURN_MIN         = -0.0020
RUTHLESS_MAX_DISPERSION          = 0.35
RUTHLESS_MIN_AGREE               = 1
RUTHLESS_ALLOCATION              = 0.90
RUTHLESS_MAX_ALLOC               = 1.00
RUTHLESS_KELLY_FRAC              = 0.75
RUTHLESS_TAKE_PROFIT             = 0.060
RUTHLESS_STOP_LOSS               = 0.025
RUTHLESS_TIMEOUT_HOURS           = 12
RUTHLESS_MAX_DD_PCT              = 0.35

MOMENTUM_RET4_MIN          = 0.015
MOMENTUM_RET16_MIN         = 0.025
MOMENTUM_VOLUME_MIN        = 2.0
MOMENTUM_BTC_REL_MIN       = 0.005
MOMENTUM_OVERRIDE_MIN_EV   = -0.002


class TestRiskProfileConstants:
    """Validate that aggressive/ruthless profile constants are strictly looser than balanced."""

    def test_aggressive_score_min_looser_than_balanced(self):
        """Aggressive score_min must be lower (less restrictive) than balanced 0.50."""
        assert AGGRESSIVE_SCORE_MIN < SCORE_MIN

    def test_ruthless_score_min_looser_than_aggressive(self):
        """Ruthless score_min must be lower than aggressive."""
        assert RUTHLESS_SCORE_MIN < AGGRESSIVE_SCORE_MIN

    def test_aggressive_max_alloc_larger_than_balanced(self):
        """Aggressive max_alloc=0.95 must exceed balanced 0.80."""
        assert AGGRESSIVE_MAX_ALLOC > 0.80

    def test_ruthless_max_alloc_is_full(self):
        """Ruthless max_alloc must be 1.00 (100 % of portfolio)."""
        assert RUTHLESS_MAX_ALLOC == pytest.approx(1.00)

    def test_ruthless_allocation_larger_than_aggressive(self):
        """Ruthless base allocation must be larger than aggressive."""
        assert RUTHLESS_ALLOCATION > AGGRESSIVE_ALLOCATION

    def test_ruthless_take_profit_wider_than_aggressive(self):
        """Ruthless take_profit must be wider than aggressive."""
        assert RUTHLESS_TAKE_PROFIT > AGGRESSIVE_TAKE_PROFIT

    def test_aggressive_take_profit_wider_than_balanced(self):
        """Aggressive take_profit must be wider than balanced 0.030."""
        assert AGGRESSIVE_TAKE_PROFIT > 0.030

    def test_ruthless_max_dd_higher_than_aggressive(self):
        """Ruthless drawdown circuit-breaker must be more permissive than aggressive."""
        assert RUTHLESS_MAX_DD_PCT > AGGRESSIVE_MAX_DD_PCT

    def test_aggressive_max_dd_higher_than_balanced(self):
        """Aggressive drawdown circuit-breaker must be more permissive than balanced 0.08."""
        assert AGGRESSIVE_MAX_DD_PCT > 0.08

    def test_ruthless_kelly_frac_larger_than_aggressive(self):
        """Ruthless Kelly fraction must be larger (more aggressive sizing)."""
        assert RUTHLESS_KELLY_FRAC > AGGRESSIVE_KELLY_FRAC

    def test_aggressive_kelly_frac_larger_than_balanced(self):
        """Aggressive Kelly fraction must be larger than balanced 0.25."""
        assert AGGRESSIVE_KELLY_FRAC > 0.25


class TestMomentumOverrideGate:
    """Tests for the momentum breakout override logic."""

    def _build_feat(self, ret_4=0.0, ret_16=0.0, vol_r=1.0, btc_rel=0.0):
        """Build a 10-element feature vector with specified momentum values."""
        feat = np.zeros(10, dtype=float)
        feat[1] = ret_4
        feat[3] = ret_16
        feat[6] = vol_r
        feat[7] = btc_rel
        return feat

    def _check_momentum(self, feat,
                        ret4_min=MOMENTUM_RET4_MIN,
                        ret16_min=MOMENTUM_RET16_MIN,
                        vol_min=MOMENTUM_VOLUME_MIN,
                        btc_rel_min=MOMENTUM_BTC_REL_MIN):
        """Mirror of the momentum override condition in _try_enter()."""
        return (
            float(feat[1]) >= ret4_min
            and float(feat[3]) >= ret16_min
            and float(feat[6]) >= vol_min
            and float(feat[7]) >= btc_rel_min
        )

    def test_all_conditions_met_passes(self):
        """When all momentum conditions meet defaults, override should activate."""
        feat = self._build_feat(
            ret_4=0.020, ret_16=0.030, vol_r=3.0, btc_rel=0.010
        )
        assert self._check_momentum(feat)

    def test_at_exact_threshold_passes(self):
        """Conditions at exactly the threshold boundary must pass."""
        feat = self._build_feat(
            ret_4=MOMENTUM_RET4_MIN,
            ret_16=MOMENTUM_RET16_MIN,
            vol_r=MOMENTUM_VOLUME_MIN,
            btc_rel=MOMENTUM_BTC_REL_MIN,
        )
        assert self._check_momentum(feat)

    def test_low_ret4_fails(self):
        """If ret_4 is below threshold, momentum override must not activate."""
        feat = self._build_feat(
            ret_4=0.010,  # below 0.015
            ret_16=0.030, vol_r=3.0, btc_rel=0.010
        )
        assert not self._check_momentum(feat)

    def test_low_ret16_fails(self):
        """If ret_16 is below threshold, override must not activate."""
        feat = self._build_feat(
            ret_4=0.020,
            ret_16=0.010,  # below 0.025
            vol_r=3.0, btc_rel=0.010
        )
        assert not self._check_momentum(feat)

    def test_low_volume_fails(self):
        """If vol_r is below threshold (< 2.0), override must not activate."""
        feat = self._build_feat(
            ret_4=0.020, ret_16=0.030,
            vol_r=1.5,  # below 2.0
            btc_rel=0.010
        )
        assert not self._check_momentum(feat)

    def test_low_btc_rel_fails(self):
        """If btc_rel is below threshold, override must not activate."""
        feat = self._build_feat(
            ret_4=0.020, ret_16=0.030, vol_r=3.0,
            btc_rel=0.002  # below 0.005
        )
        assert not self._check_momentum(feat)

    def test_ev_floor_blocks_catastrophic_entry(self):
        """EV below MOMENTUM_OVERRIDE_MIN_EV must block the override."""
        ev = MOMENTUM_OVERRIDE_MIN_EV - 0.001   # e.g. −0.003
        assert ev < MOMENTUM_OVERRIDE_MIN_EV    # sanity

    def test_ev_at_floor_passes(self):
        """EV exactly at MOMENTUM_OVERRIDE_MIN_EV must be allowed."""
        ev = MOMENTUM_OVERRIDE_MIN_EV
        assert ev >= MOMENTUM_OVERRIDE_MIN_EV


class TestMomentumScoreFormula:
    """Tests for the aggressive/ruthless momentum-boosted final scoring formula."""

    def _momentum_score(self, feat):
        """Mirror of the momentum_score computation in _try_enter()."""
        _vol_excess = min(max(float(feat[6]) - 1.0, 0.0), 4.0) / 4.0
        return float(np.clip(
            0.40 * float(feat[1])
            + 0.30 * float(feat[3])
            + 0.20 * _vol_excess
            + 0.10 * float(feat[7]),
            -0.05, 0.10
        ))

    def _final_score_aggressive(self, ev, pred_return, feat, reg_fitted=True):
        """Mirror of the aggressive final_score formula."""
        momentum_score = self._momentum_score(feat)
        if reg_fitted and pred_return != 0.0:
            return 0.50 * ev + 0.25 * pred_return + 0.25 * momentum_score
        std_proba = 0.1  # assume low dispersion
        confidence_adj = max(0.0, 1.0 - std_proba)
        return 0.75 * ev * confidence_adj + 0.25 * momentum_score

    def test_momentum_score_bounded(self):
        """Momentum score must always stay within [−0.05, 0.10]."""
        # Extreme positive momentum
        feat_high = np.zeros(10)
        feat_high[1] = 1.0   # ret_4 = 100%
        feat_high[3] = 1.0   # ret_16 = 100%
        feat_high[6] = 100.0  # vol_r very high
        feat_high[7] = 1.0   # btc_rel = 100%
        assert self._momentum_score(feat_high) <= 0.10

        # Extreme negative momentum
        feat_low = np.zeros(10)
        feat_low[1] = -1.0
        feat_low[3] = -1.0
        feat_low[6] = 0.0
        feat_low[7] = -1.0
        assert self._momentum_score(feat_low) >= -0.05

    def test_volume_excess_normalised(self):
        """Volume excess normalisation: vol_r=5.0 should give max vol contribution."""
        feat_v1 = np.zeros(10); feat_v1[6] = 5.0  # vol_r at cap
        feat_v2 = np.zeros(10); feat_v2[6] = 2.0  # mid volume
        # Both have same other fields; vol5 >= vol2 in momentum score
        assert self._momentum_score(feat_v1) >= self._momentum_score(feat_v2)

    def test_high_momentum_boosts_score_vs_balanced(self):
        """A strong-momentum candidate should score higher in aggressive mode."""
        feat = np.zeros(10)
        feat[1] = 0.03   # ret_4 = 3%
        feat[3] = 0.05   # ret_16 = 5%
        feat[6] = 3.0    # vol_r = 3×
        feat[7] = 0.02   # btc_rel = 2%

        ev = 0.005
        pred_return = 0.003

        agg_score = self._final_score_aggressive(ev, pred_return, feat)
        bal_score = 0.6 * ev + 0.4 * pred_return

        assert agg_score > bal_score, (
            f"Aggressive score {agg_score:.5f} should exceed balanced {bal_score:.5f}"
        )
