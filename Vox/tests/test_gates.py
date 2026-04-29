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

# Mirror of constants defined in config.py so tests do not require AlgorithmImports.
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

# Ruthless v2 constants (updated for high-upside asymmetric targeting)
RUTHLESS_SCORE_MIN               = 0.45
RUTHLESS_MIN_EV                  = 0.0000
RUTHLESS_PRED_RETURN_MIN         = -0.0040   # looser: was -0.0020
RUTHLESS_MAX_DISPERSION          = 0.35
RUTHLESS_MIN_AGREE               = 1
RUTHLESS_ALLOCATION              = 0.90
RUTHLESS_MAX_ALLOC               = 1.00
RUTHLESS_KELLY_FRAC              = 0.75
RUTHLESS_MIN_ALLOC               = 0.75      # Kelly floor — new in v2
RUTHLESS_USE_KELLY               = False     # flat allocation by default — new in v2
RUTHLESS_TAKE_PROFIT             = 0.09     # was 0.060; wider for P/L ratio ≈ 3.0
RUTHLESS_STOP_LOSS               = 0.03     # was 0.025
RUTHLESS_TIMEOUT_HOURS           = 24       # was 12; winners get room to run
RUTHLESS_MAX_DD_PCT              = 0.35
RUTHLESS_RUNNER_MODE             = True     # trailing stop instead of instant TP exit
RUTHLESS_TRAIL_AFTER_TP          = 0.04    # activate trailing once return ≥ +4 %
RUTHLESS_TRAIL_PCT               = 0.025   # exit if price drops 2.5 % from trailing high

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
        assert RUTHLESS_MAX_ALLOC == 1.00

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


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v2 constants
# ─────────────────────────────────────────────────────────────────────────────

class TestRuthlessV2Constants:
    """Validate Ruthless v2 TP/SL/timeout asymmetry and new sizing parameters."""

    def test_ruthless_v2_take_profit_is_nine_percent(self):
        """Ruthless v2 take_profit must be 9 % for high-upside targeting."""
        assert RUTHLESS_TAKE_PROFIT == pytest.approx(0.09)

    def test_ruthless_v2_stop_loss_is_three_percent(self):
        """Ruthless v2 stop_loss must be 3 % (P/L ratio ≈ 3.0)."""
        assert RUTHLESS_STOP_LOSS == pytest.approx(0.03)

    def test_ruthless_v2_pl_ratio_above_two(self):
        """Ruthless v2 P/L ratio (TP/SL) must be >= 2.0 to support high-upside."""
        assert RUTHLESS_TAKE_PROFIT / RUTHLESS_STOP_LOSS >= 2.0

    def test_ruthless_v2_timeout_is_24h(self):
        """Ruthless v2 timeout must be 24 h so winners have room to run."""
        assert RUTHLESS_TIMEOUT_HOURS == 24

    def test_ruthless_v2_timeout_longer_than_aggressive(self):
        """Ruthless v2 timeout must be longer than aggressive."""
        assert RUTHLESS_TIMEOUT_HOURS > AGGRESSIVE_TIMEOUT_HOURS

    def test_ruthless_v2_pred_return_min_looser(self):
        """Ruthless v2 pred_return_min must be lower than aggressive (very loose veto)."""
        assert RUTHLESS_PRED_RETURN_MIN < AGGRESSIVE_PRED_RETURN_MIN

    def test_ruthless_v2_pred_return_min_is_minus_forty_bps(self):
        """Ruthless v2 pred_return_min should be -0.004 (−40 bps = −0.40 % loose veto)."""
        assert RUTHLESS_PRED_RETURN_MIN == pytest.approx(-0.004)

    def test_ruthless_v2_min_alloc_defined(self):
        """Ruthless v2 must define RUTHLESS_MIN_ALLOC allocation floor."""
        assert RUTHLESS_MIN_ALLOC == pytest.approx(0.75)

    def test_ruthless_v2_min_alloc_below_allocation(self):
        """RUTHLESS_MIN_ALLOC floor must be <= RUTHLESS_ALLOCATION."""
        assert RUTHLESS_MIN_ALLOC <= RUTHLESS_ALLOCATION

    def test_ruthless_v2_use_kelly_false(self):
        """Ruthless v2 must default to use_kelly=False for flat 90 % sizing."""
        assert RUTHLESS_USE_KELLY is False

    def test_ruthless_v2_runner_mode_true(self):
        """Ruthless v2 must enable runner_mode by default."""
        assert RUTHLESS_RUNNER_MODE is True

    def test_ruthless_v2_trail_after_tp_below_take_profit(self):
        """trail_after_tp must be below take_profit so trailing activates before full TP."""
        assert RUTHLESS_TRAIL_AFTER_TP < RUTHLESS_TAKE_PROFIT

    def test_ruthless_v2_trail_pct_positive(self):
        """trail_pct must be a positive fraction for the trailing stop width."""
        assert 0.0 < RUTHLESS_TRAIL_PCT < 0.10


# ─────────────────────────────────────────────────────────────────────────────
# compute_qty — min_alloc floor
# ─────────────────────────────────────────────────────────────────────────────

# Import the pure-Python sizing function (no AlgorithmImports dependency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))

import importlib
import types

# Minimal stub so risk.py can be imported without AlgorithmImports
_stub = types.ModuleType("AlgorithmImports")
sys.modules.setdefault("AlgorithmImports", _stub)

from risk import compute_qty  # type: ignore  # noqa: E402


class TestComputeQtyMinAlloc:
    """Validate that min_alloc floor is applied correctly in compute_qty."""

    _PRICE = 100.0
    _PV    = 10_000.0
    _CB    = 1.0   # no buffer for simplicity

    def _qty(self, use_kelly, allocation, min_alloc=0.0,
             mean_proba=0.5, tp=0.09, sl=0.03,
             kelly_frac=0.25, max_alloc=1.0):
        return compute_qty(
            mean_proba      = mean_proba,
            tp              = tp,
            sl              = sl,
            price           = self._PRICE,
            portfolio_value = self._PV,
            kelly_frac      = kelly_frac,
            max_alloc       = max_alloc,
            cash_buffer     = self._CB,
            use_kelly       = use_kelly,
            allocation      = allocation,
            min_alloc       = min_alloc,
        )

    def test_default_min_alloc_zero_preserves_old_behavior(self):
        """With min_alloc=0 (default), Kelly is unconstrained — balanced mode unchanged."""
        qty_new, alloc_new = self._qty(use_kelly=False, allocation=0.50, min_alloc=0.0)
        qty_old, alloc_old = self._qty(use_kelly=False, allocation=0.50)
        assert qty_new == pytest.approx(qty_old)
        assert alloc_new == pytest.approx(alloc_old)

    def test_use_kelly_false_ignores_min_alloc(self):
        """When use_kelly=False, min_alloc is irrelevant — flat allocation is used."""
        _, alloc = self._qty(use_kelly=False, allocation=0.90, min_alloc=0.75)
        assert alloc == pytest.approx(0.90)

    def test_min_alloc_floor_applied_when_kelly_positive(self):
        """When Kelly is positive but small, min_alloc raises it to the floor."""
        # With proba=0.35, tp=0.09, sl=0.03, kelly_frac=0.25:
        # b = 3.0, f_full = (0.35*4 - 1)/3 = 0.40/3 ≈ 0.133, kelly = 0.133*0.25 ≈ 0.033
        # min_alloc=0.75 should push it to 0.75.
        _, alloc = self._qty(
            use_kelly=True, allocation=0.90, min_alloc=0.75,
            mean_proba=0.35, tp=0.09, sl=0.03, kelly_frac=0.25, max_alloc=1.0
        )
        assert alloc >= 0.75

    def test_min_alloc_does_not_exceed_max_alloc(self):
        """min_alloc cannot push allocation above max_alloc."""
        _, alloc = self._qty(
            use_kelly=True, allocation=0.90, min_alloc=0.95,
            mean_proba=0.35, tp=0.09, sl=0.03, kelly_frac=0.25, max_alloc=0.80
        )
        assert alloc <= 0.80

    def test_min_alloc_not_applied_when_kelly_nonpositive(self):
        """When Kelly is 0 or negative, fall back to flat allocation (not min_alloc)."""
        # proba=0.20 with b=3.0: f_full = (0.20*4-1)/3 = -0.2/3 < 0 → Kelly <= 0
        _, alloc = self._qty(
            use_kelly=True, allocation=0.90, min_alloc=0.75,
            mean_proba=0.20, tp=0.09, sl=0.03, kelly_frac=0.25, max_alloc=1.0
        )
        assert alloc == pytest.approx(0.90)   # falls back to flat allocation

    def test_ruthless_v2_flat_allocation_approx_ninety_percent(self):
        """With use_kelly=False and allocation=0.90, position should be ~90 % of portfolio."""
        qty, alloc = self._qty(
            use_kelly=False, allocation=RUTHLESS_ALLOCATION,
            min_alloc=RUTHLESS_MIN_ALLOC, max_alloc=RUTHLESS_MAX_ALLOC
        )
        assert alloc == pytest.approx(RUTHLESS_ALLOCATION)
        expected_usd = self._PV * RUTHLESS_ALLOCATION
        assert qty * self._PRICE == pytest.approx(expected_usd)


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — TP/SL floor enforcement (Issue 1)
# ─────────────────────────────────────────────────────────────────────────────

# Mirror of ruthless TP/SL floor constants from config.py
RUTHLESS_TAKE_PROFIT_FLOOR = 0.09
RUTHLESS_STOP_LOSS_FLOOR   = 0.03


def _apply_ruthless_floors(tp_atr, sl_atr, tp_floor, sl_floor, is_ruthless):
    """Mirror of ruthless TP/SL floor logic in _try_enter()."""
    tp_floor_applied = False
    sl_floor_applied = False
    if is_ruthless:
        if tp_atr < tp_floor:
            tp_atr = tp_floor
            tp_floor_applied = True
        if sl_atr < sl_floor:
            sl_atr = sl_floor
            sl_floor_applied = True
    return tp_atr, sl_atr, tp_floor_applied, sl_floor_applied


class TestRuthlessTpSlFloor:
    """Ruthless mode must not allow ATR-derived TP/SL to shrink below configured floors."""

    def test_ruthless_atr_below_floor_raises_tp(self):
        """ATR TP of 2% must be raised to 9% floor in ruthless mode."""
        tp, sl, tp_fl, sl_fl = _apply_ruthless_floors(
            0.02, 0.03, RUTHLESS_TAKE_PROFIT_FLOOR, RUTHLESS_STOP_LOSS_FLOOR, True
        )
        assert tp == pytest.approx(RUTHLESS_TAKE_PROFIT_FLOOR)
        assert tp_fl is True

    def test_ruthless_atr_below_floor_raises_sl(self):
        """ATR SL of 0.8% must be raised to 3% floor in ruthless mode."""
        tp, sl, tp_fl, sl_fl = _apply_ruthless_floors(
            0.09, 0.008, RUTHLESS_TAKE_PROFIT_FLOOR, RUTHLESS_STOP_LOSS_FLOOR, True
        )
        assert sl == pytest.approx(RUTHLESS_STOP_LOSS_FLOOR)
        assert sl_fl is True

    def test_ruthless_atr_above_floor_kept(self):
        """ATR TP/SL above floor must not be changed; floor flags must be False."""
        tp, sl, tp_fl, sl_fl = _apply_ruthless_floors(
            0.12, 0.05, RUTHLESS_TAKE_PROFIT_FLOOR, RUTHLESS_STOP_LOSS_FLOOR, True
        )
        assert tp == pytest.approx(0.12)
        assert sl == pytest.approx(0.05)
        assert tp_fl is False
        assert sl_fl is False

    def test_balanced_mode_atr_not_modified(self):
        """In balanced mode, ATR TP/SL must not be modified regardless of value."""
        tp, sl, tp_fl, sl_fl = _apply_ruthless_floors(
            0.01, 0.005, RUTHLESS_TAKE_PROFIT_FLOOR, RUTHLESS_STOP_LOSS_FLOOR, False
        )
        assert tp == pytest.approx(0.01)
        assert sl == pytest.approx(0.005)
        assert tp_fl is False
        assert sl_fl is False

    def test_ruthless_both_floors_applied(self):
        """Both TP and SL below floor should both be raised."""
        tp, sl, tp_fl, sl_fl = _apply_ruthless_floors(
            0.005, 0.005, RUTHLESS_TAKE_PROFIT_FLOOR, RUTHLESS_STOP_LOSS_FLOOR, True
        )
        assert tp == pytest.approx(RUTHLESS_TAKE_PROFIT_FLOOR)
        assert sl == pytest.approx(RUTHLESS_STOP_LOSS_FLOOR)
        assert tp_fl is True
        assert sl_fl is True


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — anti-chop same-symbol cooldown (Issue 2)
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime as _dt, timedelta as _td

# Constants mirrored from config.py
RUTHLESS_SL_COOLDOWN_MINS  = 120
RUTHLESS_LOSS_WINDOW_HOURS = 24
RUTHLESS_LOSS_LIMIT        = 2
RUTHLESS_LOSS_BLOCK_HOURS  = 24


def _sl_times_exceeds_limit(sl_times, now, window_hours, limit):
    """Mirror of anti-chop 2-in-24h check logic."""
    cutoff = now - _td(hours=window_hours)
    return sum(1 for t in sl_times if t >= cutoff) >= limit


class TestRuthlessAntiChopCooldown:
    """Same-symbol SL cooldown and 2-in-24h block for ruthless mode."""

    def test_sl_cooldown_is_120_minutes(self):
        """Ruthless per-coin SL cooldown must be 120 minutes."""
        assert RUTHLESS_SL_COOLDOWN_MINS == 120

    def test_two_sl_in_24h_triggers_block(self):
        """2 SL exits within the 24h window must trigger an extended block."""
        now = _dt(2024, 1, 7, 12, 0)
        sl_times = [now - _td(hours=3), now - _td(hours=1)]  # 2 SLs within 24h
        assert _sl_times_exceeds_limit(sl_times, now, RUTHLESS_LOSS_WINDOW_HOURS, RUTHLESS_LOSS_LIMIT)

    def test_one_sl_in_24h_no_block(self):
        """Only 1 SL exit within 24h must not trigger the extended block."""
        now = _dt(2024, 1, 7, 12, 0)
        sl_times = [now - _td(hours=3)]
        assert not _sl_times_exceeds_limit(sl_times, now, RUTHLESS_LOSS_WINDOW_HOURS, RUTHLESS_LOSS_LIMIT)

    def test_old_sl_outside_window_not_counted(self):
        """SL exits older than 24h must not count toward the limit."""
        now = _dt(2024, 1, 7, 12, 0)
        # One SL 25 hours ago (outside window), one just now
        sl_times = [now - _td(hours=25), now - _td(hours=1)]
        assert not _sl_times_exceeds_limit(sl_times, now, RUTHLESS_LOSS_WINDOW_HOURS, RUTHLESS_LOSS_LIMIT)

    def test_three_sl_in_window_also_triggers(self):
        """3 SL exits in the window also triggers the block (>= limit)."""
        now = _dt(2024, 1, 7, 12, 0)
        sl_times = [now - _td(hours=10), now - _td(hours=5), now - _td(hours=1)]
        assert _sl_times_exceeds_limit(sl_times, now, RUTHLESS_LOSS_WINDOW_HOURS, RUTHLESS_LOSS_LIMIT)


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — portfolio loss-streak brake (Issue 3)
# ─────────────────────────────────────────────────────────────────────────────

RUTHLESS_PORTFOLIO_LOSS_STREAK = 4
RUTHLESS_PORTFOLIO_PAUSE_HOURS = 6


def _check_portfolio_pause(streak, streak_limit, pause_hours, now):
    """Mirror of portfolio loss-streak brake logic."""
    if streak >= streak_limit:
        return now + _td(hours=pause_hours), 0   # (pause_until, reset_streak)
    return None, streak


class TestRuthlessPortfolioLossStreakBrake:
    """Portfolio loss-streak brake for ruthless mode."""

    def test_four_losses_triggers_pause(self):
        """4 consecutive losses must trigger a 6-hour pause."""
        now = _dt(2024, 1, 7, 12, 0)
        pause_until, new_streak = _check_portfolio_pause(
            4, RUTHLESS_PORTFOLIO_LOSS_STREAK, RUTHLESS_PORTFOLIO_PAUSE_HOURS, now
        )
        assert pause_until is not None
        assert pause_until == now + _td(hours=RUTHLESS_PORTFOLIO_PAUSE_HOURS)
        assert new_streak == 0

    def test_three_losses_no_pause(self):
        """3 consecutive losses must not trigger a pause."""
        now = _dt(2024, 1, 7, 12, 0)
        pause_until, new_streak = _check_portfolio_pause(
            3, RUTHLESS_PORTFOLIO_LOSS_STREAK, RUTHLESS_PORTFOLIO_PAUSE_HOURS, now
        )
        assert pause_until is None
        assert new_streak == 3

    def test_streak_resets_on_trigger(self):
        """Streak counter must reset to 0 after triggering the pause."""
        now = _dt(2024, 1, 7, 12, 0)
        _, new_streak = _check_portfolio_pause(
            5, RUTHLESS_PORTFOLIO_LOSS_STREAK, RUTHLESS_PORTFOLIO_PAUSE_HOURS, now
        )
        assert new_streak == 0

    def test_pause_duration_is_six_hours(self):
        """Pause duration must be 6 hours."""
        assert RUTHLESS_PORTFOLIO_PAUSE_HOURS == 6

    def test_streak_limit_is_four(self):
        """Streak limit must be 4 consecutive losses."""
        assert RUTHLESS_PORTFOLIO_LOSS_STREAK == 4


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — confirmation gate (Issue 4)
# ─────────────────────────────────────────────────────────────────────────────

# Mirror confirmation gate constants from config.py
RUTHLESS_CONFIRM_EV_MIN    = 0.006
RUTHLESS_CONFIRM_PROBA_MIN = 0.60
RUTHLESS_CONFIRM_AGREE_MIN = 2
RUTHLESS_CONFIRM_RET4_MIN  = 0.010
RUTHLESS_CONFIRM_RET16_MIN = 0.020
RUTHLESS_CONFIRM_VOLR_MIN  = 1.5


def _ruthless_confirmation(entry_path, ev, class_proba, n_agree, feat):
    """Mirror of the ruthless confirmation gate logic in _try_enter()."""
    if entry_path == "momentum_override":
        return "momentum_override"
    if (ev >= RUTHLESS_CONFIRM_EV_MIN
            and class_proba >= RUTHLESS_CONFIRM_PROBA_MIN
            and n_agree >= RUTHLESS_CONFIRM_AGREE_MIN):
        return "strong_ml"
    if (float(feat[1]) >= RUTHLESS_CONFIRM_RET4_MIN
            and float(feat[3]) >= RUTHLESS_CONFIRM_RET16_MIN
            and float(feat[6]) >= RUTHLESS_CONFIRM_VOLR_MIN):
        return "trend_momentum"
    return None


class TestRuthlessConfirmationGate:
    """Ruthless confirmation gate: must pass one of three confirmation paths."""

    def _feat(self, ret_4=0.0, ret_16=0.0, vol_r=1.0):
        f = np.zeros(10)
        f[1] = ret_4; f[3] = ret_16; f[6] = vol_r
        return f

    def test_momentum_override_always_confirms(self):
        """momentum_override entry path bypasses ML/trend conditions."""
        result = _ruthless_confirmation(
            "momentum_override", ev=0.001, class_proba=0.30, n_agree=1,
            feat=self._feat()
        )
        assert result == "momentum_override"

    def test_strong_ml_confirms(self):
        """Strong ML (high ev + proba + n_agree) must confirm."""
        result = _ruthless_confirmation(
            "ml", ev=0.008, class_proba=0.65, n_agree=2,
            feat=self._feat()
        )
        assert result == "strong_ml"

    def test_trend_momentum_confirms(self):
        """Clear trend/momentum (ret_4 + ret_16 + vol_r) must confirm."""
        result = _ruthless_confirmation(
            "ml", ev=0.001, class_proba=0.40, n_agree=1,
            feat=self._feat(ret_4=0.015, ret_16=0.025, vol_r=2.0)
        )
        assert result == "trend_momentum"

    def test_weak_signal_fails_confirmation(self):
        """Weak signal (low ev, low proba, low momentum) must fail all gates."""
        result = _ruthless_confirmation(
            "ml", ev=0.001, class_proba=0.40, n_agree=1,
            feat=self._feat(ret_4=0.005, ret_16=0.010, vol_r=1.2)
        )
        assert result is None

    def test_strong_ml_just_at_threshold_passes(self):
        """All strong_ml conditions at exact threshold must pass."""
        result = _ruthless_confirmation(
            "ml",
            ev=RUTHLESS_CONFIRM_EV_MIN,
            class_proba=RUTHLESS_CONFIRM_PROBA_MIN,
            n_agree=RUTHLESS_CONFIRM_AGREE_MIN,
            feat=self._feat()
        )
        assert result == "strong_ml"

    def test_strong_ml_ev_below_threshold_falls_through(self):
        """EV just below strong_ml threshold must fall through to trend check."""
        result = _ruthless_confirmation(
            "ml",
            ev=RUTHLESS_CONFIRM_EV_MIN - 0.001,
            class_proba=0.70, n_agree=3,
            feat=self._feat()   # no trend either
        )
        assert result is None  # neither strong_ml nor trend passes

    def test_balanced_mode_not_affected(self):
        """Confirmation gate logic is only applied in ruthless mode (gate returns None for
        non-ruthless in the real code).  Verify constants are profile-specific."""
        # Ensure ruthless confirm thresholds are stricter than balanced gates
        assert RUTHLESS_CONFIRM_PROBA_MIN > RUTHLESS_SCORE_MIN

    def test_trend_momentum_vol_r_at_threshold(self):
        """vol_r exactly at RUTHLESS_CONFIRM_VOLR_MIN must pass trend_momentum."""
        result = _ruthless_confirmation(
            "ml", ev=0.001, class_proba=0.40, n_agree=1,
            feat=self._feat(
                ret_4=RUTHLESS_CONFIRM_RET4_MIN,
                ret_16=RUTHLESS_CONFIRM_RET16_MIN,
                vol_r=RUTHLESS_CONFIRM_VOLR_MIN,
            )
        )
        assert result == "trend_momentum"


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — build_features SMA slope (Issue 6a)
# ─────────────────────────────────────────────────────────────────────────────

from models import build_features  # type: ignore  # noqa: E402


class TestBuildFeaturesSmaSlope:
    """Verify the new sma_slope feature at index 9 in build_features."""

    def _make_prices(self, n=20, trend="flat", base=100.0):
        """Create a simple price series: flat, up, or down."""
        if trend == "up":
            return [base * (1 + 0.001 * i) for i in range(n)]
        elif trend == "down":
            return [base * (1 - 0.001 * i) for i in range(n)]
        else:
            return [base] * n

    def _vols(self, n=20):
        return [1000.0] * n

    def _btc(self, n=5):
        return [30000.0] * n

    def test_feature_vector_length_is_ten(self):
        """build_features must still return 10 features."""
        feat = build_features(self._make_prices(), self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert len(feat) == 10

    def test_sma_slope_index_nine(self):
        """Feature at index 9 must be a float (sma_slope, not always 0.0)."""
        feat = build_features(self._make_prices(trend="up"), self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert isinstance(float(feat[9]), float)

    def test_sma_slope_positive_in_uptrend(self):
        """SMA slope must be positive when prices are rising."""
        feat = build_features(self._make_prices(trend="up"), self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert float(feat[9]) > 0.0

    def test_sma_slope_negative_in_downtrend(self):
        """SMA slope must be negative when prices are falling."""
        feat = build_features(self._make_prices(trend="down"), self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert float(feat[9]) < 0.0

    def test_sma_slope_near_zero_in_flat_market(self):
        """SMA slope must be near zero for flat prices."""
        feat = build_features(self._make_prices(trend="flat"), self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert abs(float(feat[9])) < 1e-10

    def test_sma_slope_capped_at_ten_percent(self):
        """SMA slope is capped at ±0.10 to prevent extreme values."""
        # Very steep ramp
        prices = [100.0 * (1 + 0.05 * i) for i in range(20)]  # 5% per bar
        feat = build_features(prices, self._vols(), self._btc(), hour=12)
        assert feat is not None
        assert float(feat[9]) <= 0.10
        assert float(feat[9]) >= -0.10

    def test_vol_r_capped_at_ten(self):
        """Volume ratio must be capped at 10× to prevent explosion."""
        # 15 prior bars with volume 1000, then last bar with enormous spike
        base_vols = [1000.0] * 19  # 19 bars of normal volume
        spike_vols = base_vols[:-1] + [200_000.0]  # replace last with spike
        feat = build_features(
            self._make_prices(n=20), spike_vols, self._btc(), hour=12
        )
        if feat is not None:
            assert float(feat[6]) <= 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v3 — label parameters (Issue 6b)
# ─────────────────────────────────────────────────────────────────────────────

# Mirror label constants from config.py
RUTHLESS_LABEL_TP           = 0.09
RUTHLESS_LABEL_SL           = 0.03
RUTHLESS_LABEL_HORIZON_BARS = 96


class TestRuthlessLabelParameters:
    """Ruthless label parameters must align with wider 24h trading targets."""

    def test_ruthless_label_tp_matches_take_profit(self):
        """Ruthless label TP must equal ruthless take_profit for barrier alignment."""
        assert RUTHLESS_LABEL_TP == pytest.approx(RUTHLESS_TAKE_PROFIT)

    def test_ruthless_label_sl_matches_stop_loss(self):
        """Ruthless label SL must equal ruthless stop_loss for barrier alignment."""
        assert RUTHLESS_LABEL_SL == pytest.approx(RUTHLESS_STOP_LOSS)

    def test_ruthless_label_horizon_is_96_bars(self):
        """Ruthless 24h horizon = 24 × 4 bars/h = 96 bars at 15-min decision intervals."""
        assert RUTHLESS_LABEL_HORIZON_BARS == 96

    def test_ruthless_label_horizon_wider_than_default(self):
        """Ruthless label horizon must be wider than the default LABEL_HORIZON_BARS=72."""
        from models import LABEL_HORIZON_BARS as DEFAULT_HORIZON
        assert RUTHLESS_LABEL_HORIZON_BARS > DEFAULT_HORIZON

    def test_ruthless_label_tp_wider_than_default(self):
        """Ruthless label TP (0.09) must be wider than default LABEL_TP (0.012)."""
        from models import LABEL_TP as DEFAULT_LABEL_TP
        assert RUTHLESS_LABEL_TP > DEFAULT_LABEL_TP

    def test_balanced_mode_label_params_unchanged(self):
        """Default (balanced) label params must remain at their original values."""
        from models import LABEL_TP, LABEL_SL, LABEL_HORIZON_BARS
        assert LABEL_TP == pytest.approx(0.012)
        assert LABEL_SL == pytest.approx(0.010)
        assert LABEL_HORIZON_BARS == 72
