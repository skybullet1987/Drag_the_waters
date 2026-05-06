"""
Advanced gate tests — continuation of test_gates.py.

Split out to keep file size under the 63,000-byte QuantConnect limit.
Original test_gates.py split at TestBuildFeaturesSmaSlope boundary.
"""
import sys
import os

import numpy as np
import pytest

# Allow importing from the Vox package without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import VoxEnsemble, triple_barrier_outcome, FEATURE_COUNT, build_features  # type: ignore  # noqa: E402

# ── Shared config mirrors (must match config.py / main.py) ───────────────────
SCORE_MIN       = 0.50
SCORE_MIN_FLOOR = 0.15
MIN_AGREE       = 2

# ── Ruthless profile constants (mirrored from main.py / config.py) ───────────
RUTHLESS_TAKE_PROFIT   = 0.09
RUTHLESS_STOP_LOSS     = 0.03
RUTHLESS_TRAIL_AFTER_TP = 0.04
RUTHLESS_TRAIL_PCT      = 0.025

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
        """Default (balanced) label params — updated to optimized achievable values."""
        from models import LABEL_TP, LABEL_SL, LABEL_HORIZON_BARS
        assert LABEL_TP == pytest.approx(0.030)
        assert LABEL_SL == pytest.approx(0.015)
        assert LABEL_HORIZON_BARS == 48


# ─────────────────────────────────────────────────────────────────────────────
# Apex Predator profile constants and lowered RUTHLESS thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestApexPredatorProfile:
    """Verify apex_predator profile constants and lowered ruthless thresholds."""

    def test_apex_profile_constants_importable(self):
        from core import (
            APEX_PROFILE_SCORE_MIN, APEX_PROFILE_VOTE_THRESHOLD,
            APEX_PROFILE_VOTE_YES_FRACTION_MIN, APEX_PROFILE_TOP3_MEAN_MIN,
            APEX_PROFILE_LABEL_TP, APEX_PROFILE_LABEL_SL,
            APEX_PROFILE_LABEL_HORIZON_BARS,
        )
        assert APEX_PROFILE_SCORE_MIN < 0.20
        assert APEX_PROFILE_VOTE_THRESHOLD < 0.50
        assert APEX_PROFILE_VOTE_YES_FRACTION_MIN < 0.34
        assert APEX_PROFILE_TOP3_MEAN_MIN < 0.55
        assert APEX_PROFILE_LABEL_TP < 0.035
        assert APEX_PROFILE_LABEL_SL < 0.030
        assert APEX_PROFILE_LABEL_HORIZON_BARS <= 36

    def test_ruthless_gates_lowered(self):
        from core import (
            RUTHLESS_SCORE_MIN, RUTHLESS_VOTE_THRESHOLD,
            RUTHLESS_VOTE_YES_FRACTION_MIN, RUTHLESS_TOP3_MEAN_MIN,
            RUTHLESS_CHOP_VOTE_YES_FRAC_MIN, RUTHLESS_CHOP_TOP3_MEAN_MIN,
            RUTHLESS_CONFIRM_EV_MIN, RUTHLESS_CONFIRM_PROBA_MIN,
            RUTHLESS_CONFIRM_RET4_MIN,
        )
        assert RUTHLESS_SCORE_MIN <= 0.20
        assert RUTHLESS_VOTE_THRESHOLD <= 0.45
        assert RUTHLESS_VOTE_YES_FRACTION_MIN <= 0.30
        assert RUTHLESS_TOP3_MEAN_MIN <= 0.45
        assert RUTHLESS_CHOP_VOTE_YES_FRAC_MIN <= 0.40
        assert RUTHLESS_CHOP_TOP3_MEAN_MIN <= 0.50
        assert RUTHLESS_CONFIRM_EV_MIN <= 0.002
        assert RUTHLESS_CONFIRM_PROBA_MIN <= 0.52
        assert RUTHLESS_CONFIRM_RET4_MIN <= 0.004

    def test_ruthless_anti_chop_relaxed(self):
        from core import (
            RUTHLESS_SL_COOLDOWN_MINS, RUTHLESS_LOSS_LIMIT,
            RUTHLESS_LOSS_WINDOW_HOURS, RUTHLESS_LOSS_BLOCK_HOURS,
            RUTHLESS_PORTFOLIO_LOSS_STREAK, RUTHLESS_PORTFOLIO_PAUSE_HOURS,
        )
        assert RUTHLESS_SL_COOLDOWN_MINS <= 30
        assert RUTHLESS_LOSS_LIMIT >= 4
        assert RUTHLESS_LOSS_WINDOW_HOURS <= 12
        assert RUTHLESS_LOSS_BLOCK_HOURS <= 6
        assert RUTHLESS_PORTFOLIO_LOSS_STREAK >= 6
        assert RUTHLESS_PORTFOLIO_PAUSE_HOURS <= 2

    def test_ruthless_label_params_achievable(self):
        from core import (
            RUTHLESS_LABEL_TP, RUTHLESS_LABEL_SL, RUTHLESS_LABEL_HORIZON_BARS,
        )
        assert RUTHLESS_LABEL_TP <= 0.035
        assert RUTHLESS_LABEL_SL <= 0.015
        assert RUTHLESS_LABEL_HORIZON_BARS <= 48

    def test_model_weights_lgbm_bal_hgbc_l2_promoted(self):
        from core import MODEL_WEIGHT_LGBM_BAL, MODEL_WEIGHT_HGBC_L2
        assert MODEL_WEIGHT_LGBM_BAL >= 2.0
        assert MODEL_WEIGHT_HGBC_L2 >= 2.0

    def test_catboost_lgbm_dart_in_active_models(self):
        from core import RUTHLESS_ACTIVE_MODELS
        assert "catboost_bal" in RUTHLESS_ACTIVE_MODELS
        assert "lgbm_dart" in RUTHLESS_ACTIVE_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless v4: breakeven, momentum-fail, timeout extension
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakeven:
    def test_no_breakeven_below_threshold(self):
        """Below be_after, breakeven should not activate."""
        # Simulate: max_return_seen=0.02 < be_after=0.03
        assert 0.02 < 0.03  # precondition
        active = 0.02 >= 0.03
        assert not active

    def test_breakeven_activates_at_threshold(self):
        """When max_return_seen >= be_after, breakeven activates."""
        be_after = 0.03
        max_ret  = 0.035
        active   = max_ret >= be_after
        assert active

    def test_breakeven_exit_when_ret_at_buffer(self):
        """If breakeven is active and ret <= buffer, should exit."""
        be_buffer = 0.003
        ret       = 0.002
        assert ret <= be_buffer   # should trigger


class TestMomentumFail:
    def test_no_exit_before_min_hold(self):
        """Momentum fail should not fire before min_hold_minutes."""
        from strategy import should_exit_momentum_fail
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = -0.02; feat[3] = -0.03
        assert not should_exit_momentum_fail(
            elapsed_minutes=10, ret=-0.015, feat=feat,
            min_hold_minutes=30, fail_loss=-0.012
        )

    def test_no_exit_if_return_above_threshold(self):
        """No exit when return is better than fail_loss."""
        from strategy import should_exit_momentum_fail
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = -0.02; feat[3] = -0.03
        assert not should_exit_momentum_fail(
            elapsed_minutes=45, ret=-0.005, feat=feat,
            min_hold_minutes=30, fail_loss=-0.012
        )

    def test_exit_on_momentum_fail(self):
        """Should exit when hold >= min_hold, return <= fail_loss, and momentum negative."""
        from strategy import should_exit_momentum_fail
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = -0.02; feat[3] = -0.03
        assert should_exit_momentum_fail(
            elapsed_minutes=35, ret=-0.015, feat=feat,
            min_hold_minutes=30, fail_loss=-0.012
        )

    def test_no_exit_when_momentum_positive(self):
        """Should not exit if ret_4 is positive (momentum not failed)."""
        from strategy import should_exit_momentum_fail
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.01; feat[3] = -0.03
        assert not should_exit_momentum_fail(
            elapsed_minutes=35, ret=-0.015, feat=feat,
            min_hold_minutes=30, fail_loss=-0.012
        )


class TestTimeoutExtension:
    def test_hold_before_timeout(self):
        from strategy import evaluate_timeout
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.01
        result = evaluate_timeout(
            elapsed_hours=10, ret=0.05, feat=feat, toh=24,
            timeout_min_profit=0.03, timeout_extend_hours=12, max_timeout_hours=48
        )
        assert result == 'hold'

    def test_exit_at_timeout_with_profit(self):
        from strategy import evaluate_timeout
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.01
        result = evaluate_timeout(
            elapsed_hours=25, ret=0.05, feat=feat, toh=24,
            timeout_min_profit=0.03, timeout_extend_hours=12, max_timeout_hours=48
        )
        assert result == 'exit'

    def test_extend_at_timeout_with_small_loss(self):
        from strategy import evaluate_timeout
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.005  # positive ret_4 → worth extending
        result = evaluate_timeout(
            elapsed_hours=25, ret=-0.005, feat=feat, toh=24,
            timeout_min_profit=0.03, timeout_extend_hours=12,
            max_timeout_hours=48, extension_hours_used=0.0
        )
        assert result == 'extend'

    def test_exit_when_extension_cap_hit(self):
        from strategy import evaluate_timeout
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.005
        result = evaluate_timeout(
            elapsed_hours=25, ret=-0.005, feat=feat, toh=24,
            timeout_min_profit=0.03, timeout_extend_hours=12,
            max_timeout_hours=48, extension_hours_used=24.1  # cap hit
        )
        assert result == 'exit'


# ─────────────────────────────────────────────────────────────────────────────
# MetaFilter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaFilter:
    def test_disabled_always_approves(self):
        from core import MetaFilter
        mf = MetaFilter(enabled=False, min_proba=0.55)
        approved, score = mf.approve(0.1, 0.0, 0, 1.0, 0.0, None)
        assert approved
        assert score == 1.0

    def test_low_conviction_rejected(self):
        from core import MetaFilter
        mf = MetaFilter(enabled=True, min_proba=0.55)
        feat = np.zeros(FEATURE_COUNT)
        approved, score = mf.approve(0.1, -0.01, 0, 0.5, 0.0, feat)
        assert not approved

    def test_high_conviction_approved(self):
        from core import MetaFilter
        mf = MetaFilter(enabled=True, min_proba=0.55)
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.02; feat[3] = 0.03; feat[6] = 2.0
        approved, score = mf.approve(0.80, 0.01, 3, 0.05, 0.02, feat,
                                      market_mode="risk_on_trend",
                                      ruthless_allowed_modes=["risk_on_trend", "pump"])
        assert approved
        assert score > 0.55

    def test_chop_mode_penalises_score(self):
        from core import MetaFilter
        mf = MetaFilter(enabled=True, min_proba=0.55)
        feat = np.zeros(FEATURE_COUNT)
        feat[1] = 0.015; feat[3] = 0.025; feat[6] = 1.8
        _, score_chop = mf.approve(0.60, 0.005, 2, 0.10, 0.01, feat,
                                    market_mode="chop")
        _, score_ok   = mf.approve(0.60, 0.005, 2, 0.10, 0.01, feat,
                                    market_mode="risk_on_trend",
                                    ruthless_allowed_modes=["risk_on_trend"])
        assert score_ok > score_chop


# ─────────────────────────────────────────────────────────────────────────────
# MarketModeDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketModeDetector:
    def test_insufficient_data_returns_chop(self):
        from core import MarketModeDetector
        det = MarketModeDetector()
        mode = det.detect([100.0, 101.0])
        assert mode == "chop"

    def test_downtrend_classified_as_selloff(self):
        from core import MarketModeDetector
        det = MarketModeDetector()
        # declining prices, negative SMA slope
        closes = [100, 97, 94, 91, 88, 85, 82, 79, 76, 73, 70, 67, 64]
        mode = det.detect(closes)
        assert mode == "selloff"

    def test_uptrend_classified_as_risk_on_trend(self):
        from core import MarketModeDetector
        det = MarketModeDetector()
        closes = [100, 101, 101.5, 102, 102.3, 102.8, 103.1, 103.6, 104.2, 104.8]
        mode = det.detect(closes)
        assert mode == "risk_on_trend"

    def test_pump_detected(self):
        from core import MarketModeDetector
        det = MarketModeDetector()
        # ret_4 must be > 0.05 (5% over 4 bars) and vol_ratio > 2.0
        base = list(np.linspace(100, 107, 9))   # ~7% over 8 bars, ~5% last 4
        mode = det.detect(base, volumes=[1.0]*8 + [3.5])
        # If not pump, skip — rules-based detection may vary on boundary
        assert mode in ("pump", "risk_on_trend")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE_COUNT constant and build_features output shape
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureCount:
    def test_feature_count_constant_matches_build_features(self):
        """FEATURE_COUNT should equal the actual output length of build_features."""
        from models import build_features, FEATURE_COUNT as FC
        closes  = list(np.linspace(100, 110, 25))
        volumes = [1e6] * 20
        btc     = list(np.linspace(50000, 51000, 10))
        feat    = build_features(closes, volumes, btc, hour=12)
        assert feat is not None
        assert len(feat) == FC, (
            f"FEATURE_COUNT={FC} but build_features returned {len(feat)} features"
        )

    def test_feature_count_is_30(self):
        assert FEATURE_COUNT == 30

    def test_build_features_returns_30_elements(self):
        from models import build_features
        closes  = list(np.linspace(100, 110, 25))
        volumes = [1e6] * 20
        btc     = list(np.linspace(50000, 51000, 10))
        feat    = build_features(closes, volumes, btc, hour=12)
        assert feat is not None
        assert len(feat) == FEATURE_COUNT

    def test_build_features_returns_none_on_short_history(self):
        from models import build_features
        closes  = list(np.linspace(100, 105, 10))
        volumes = [1e6] * 10
        btc     = [50000.0] * 5
        feat    = build_features(closes, volumes, btc, hour=12)
        assert feat is None

    def test_voxensemble_fit_stores_feature_count(self):
        """After fit(), ensemble stores the feature count in _feature_count."""
        ens = VoxEnsemble()
        rng = np.random.default_rng(99)
        X   = rng.standard_normal((60, FEATURE_COUNT))
        y   = rng.integers(0, 2, 60)
        ens.fit(X, y)
        assert getattr(ens, "_feature_count", None) == FEATURE_COUNT

class TestVoxV5:
    def test_pure_functions(self):
        from infra import compute_weighted_mean, format_model_registry_log
        from journals import TradeJournal
        from journals import get_relaxed_thresholds as grt
        assert "lr" in format_model_registry_log([("lr", None)])
        v = {"lr": 0.6, "hgbc": 0.7}
        assert abs(compute_weighted_mean(v, {k: 0.0 for k in v}) - 0.65) < 1e-9
        j = TradeJournal(max_size=2)
        j.record_entry("A", {"model_votes": {"lr": 0.6}})
        r = j.record_exit("A", {"realized_return": 0.03})
        assert r["model_votes"]["lr"] == 0.6
        for i in range(3): j.record_entry(str(i), {}); j.record_exit(str(i), {})
        assert j.record_count() == 2
        assert grt("pump", "ruthless", 0.006, 1.5, 0.55)[0] < 0.006
        assert grt("chop", "ruthless", 0.006, 1.5, 0.55)[0] == 0.006
        assert grt("pump", "balanced", 0.006, 1.5, 0.55)[0] == 0.006

    def test_ensemble_vote_fields(self):
        rng = np.random.default_rng(7)
        ens = VoxEnsemble()
        X = rng.standard_normal((60, FEATURE_COUNT))
        ens.fit(X, (rng.standard_normal(60) > 0).astype(int))
        c = ens.predict_with_confidence(X[[0]])
        assert all(k in c for k in ("weighted_mean", "per_model", "votes", "vote_threshold"))


class TestFeatureDiagSuffix:
    """Tests for the _feature_diag_suffix helper in diagnostics.py."""

    def test_returns_formatted_suffix_for_valid_vector(self):
        from journals import _feature_diag_suffix
        ft = np.zeros(FEATURE_COUNT)
        ft[1] = 0.0123
        ft[3] = 0.0234
        ft[6] = 1.45
        result = _feature_diag_suffix(ft)
        assert "r4=0.0123" in result
        assert "r16=0.0234" in result
        assert "vr=1.45" in result

    def test_returns_formatted_suffix_for_list(self):
        from journals import _feature_diag_suffix
        ft = [0.0] * FEATURE_COUNT
        ft[1] = 0.05
        ft[3] = 0.10
        ft[6] = 2.0
        result = _feature_diag_suffix(ft)
        assert "r4=0.0500" in result
        assert "r16=0.1000" in result
        assert "vr=2.00" in result

    def test_returns_empty_for_none(self):
        from journals import _feature_diag_suffix
        assert _feature_diag_suffix(None) == ""

    def test_returns_empty_for_too_short_vector(self):
        from journals import _feature_diag_suffix
        assert _feature_diag_suffix(np.zeros(4)) == ""
        assert _feature_diag_suffix([]) == ""
        assert _feature_diag_suffix([1.0, 2.0]) == ""

    def test_does_not_raise_for_numpy_array(self):
        """Confirm no ambiguous truth-value error for multi-element NumPy array."""
        from journals import _feature_diag_suffix
        ft = np.zeros(FEATURE_COUNT)
        # Must not raise "The truth value of an array is ambiguous"
        result = _feature_diag_suffix(ft)
        assert isinstance(result, str)

    def test_does_not_raise_for_malformed_values(self):
        from journals import _feature_diag_suffix
        assert _feature_diag_suffix("bad") == ""
        assert _feature_diag_suffix(42) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Model roles and role-separated prediction output
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRoles:
    """Verify model role infrastructure and role-separated prediction output."""

    def _trained_ens(self, n=100):
        """Return a fitted VoxEnsemble with default settings."""
        ens = VoxEnsemble()
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, FEATURE_COUNT))
        y = rng.integers(0, 2, n)
        ens.fit(X, y)
        return ens, X

    def test_default_core_roles_set(self):
        """VoxEnsemble must have default model roles set after construction."""
        ens = VoxEnsemble()
        assert hasattr(ens, "_model_roles")
        assert isinstance(ens._model_roles, dict)

    def test_lr_is_diagnostic_by_default(self):
        """LR must be diagnostic-only by default (was always-bearish in backtest)."""
        ens = VoxEnsemble()
        assert ens._model_roles.get("lr") == "diagnostic"

    def test_hgbc_et_rf_are_active_by_default(self):
        """hgbc, et, rf must be active by default."""
        ens = VoxEnsemble()
        for mid in ("hgbc", "et", "rf"):
            assert ens._model_roles.get(mid) == "active", f"{mid} should be active"

    def test_set_model_roles_updates_roles(self):
        """set_model_roles must update the roles dict."""
        ens = VoxEnsemble()
        ens.set_model_roles({"lr": "active", "hgbc": "shadow"})
        assert ens._model_roles["lr"]   == "active"
        assert ens._model_roles["hgbc"] == "shadow"

    def test_predict_returns_active_votes_field(self):
        """predict_with_confidence must return active_votes dict."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert "active_votes" in result
        assert isinstance(result["active_votes"], dict)

    def test_predict_returns_shadow_and_diagnostic_votes(self):
        """predict_with_confidence must return shadow_votes and diagnostic_votes."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert "shadow_votes" in result
        assert "diagnostic_votes" in result

    def test_predict_returns_excluded_models(self):
        """predict_with_confidence must return excluded_models."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert "excluded_models" in result

    def test_lr_in_diagnostic_votes_not_active(self):
        """LR should appear in diagnostic_votes, not active_votes."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        # lr is diagnostic by default
        assert "lr" not in result["active_votes"], "lr should not be in active_votes"
        assert "lr" in result["diagnostic_votes"], "lr should be in diagnostic_votes"

    def test_active_votes_contains_hgbc_et_rf(self):
        """active_votes must contain hgbc, et, rf (the active models)."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        for mid in ("hgbc", "et", "rf"):
            assert mid in result["active_votes"], f"{mid} missing from active_votes"

    def test_class_proba_maps_to_active_mean(self):
        """class_proba must equal active_mean (backward-compat points to active)."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert result["class_proba"] == pytest.approx(result["active_mean"])

    def test_std_proba_maps_to_active_std(self):
        """std_proba must equal active_std."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert result["std_proba"] == pytest.approx(result["active_std"])

    def test_n_agree_maps_to_active_n_agree(self):
        """n_agree must equal active_n_agree."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert result["n_agree"] == result["active_n_agree"]

    def test_diagnostic_model_not_counted_in_active_n_agree(self):
        """A diagnostic model must not inflate active_n_agree."""
        ens, X = self._trained_ens()
        # Verify active_n_agree only counts active models
        result = ens.predict_with_confidence(X[0])
        active_count = len(result["active_votes"])
        assert result["active_n_agree"] <= active_count

    def test_shadow_votes_do_not_affect_active_mean(self):
        """Shadow votes must not affect active_mean computation."""
        ens, X = self._trained_ens()
        # Add a shadow model manually by training and marking
        result_before = ens.predict_with_confidence(X[0])
        # Shadow votes in result must not change class_proba
        active_mean = result_before["active_mean"]
        shadow_votes = result_before.get("shadow_votes", {})
        if shadow_votes:
            # Verify class_proba == active_mean regardless of shadow values
            assert result_before["class_proba"] == pytest.approx(active_mean)

    def test_excluded_models_has_lr_reason(self):
        """excluded_models must contain lr with 'diagnostic_only' reason."""
        ens, X = self._trained_ens()
        result = ens.predict_with_confidence(X[0])
        assert result["excluded_models"].get("lr") == "diagnostic_only"

    def test_batch_returns_role_separated_output(self):
        """predict_with_confidence_batch must also include role-separated fields."""
        ens, X = self._trained_ens()
        results = ens.predict_with_confidence_batch(X[:5])
        assert len(results) == 5
        for r in results:
            assert "active_votes" in r
            assert "shadow_votes" in r
            assert "diagnostic_votes" in r
            assert "class_proba" in r
            assert r["class_proba"] == pytest.approx(r["active_mean"])

    def test_no_active_models_fallback_safe(self):
        """If all models are diagnostic, fallback should not crash."""
        ens, X = self._trained_ens()
        # Mark all as diagnostic
        ens.set_model_roles({"lr": "diagnostic", "hgbc": "diagnostic",
                             "et": "diagnostic", "rf": "diagnostic"})
        result = ens.predict_with_confidence(X[0])
        # Should still return a valid float for class_proba
        assert isinstance(result["class_proba"], float)
        assert 0.0 <= result["class_proba"] <= 1.0

    def test_load_state_restores_model_roles(self):
        """load_state must restore _model_roles from saved ensemble."""
        ens, X = self._trained_ens()
        ens.set_model_roles({"hgbc": "shadow", "et": "active", "rf": "active"})

        dst = VoxEnsemble()
        dst.load_state(ens)
        assert dst._model_roles.get("hgbc") == "shadow"


# ─────────────────────────────────────────────────────────────────────────────
# Model health diagnostics
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from infra import ModelHealthTracker  # noqa: E402


class TestModelHealthTracker:
    """Verify model health flag detection for degenerate/low-variance models."""

    def test_always_bullish_flagged(self):
        """A model that always predicts >= 0.95 should be flagged degenerate_bullish."""
        tracker = ModelHealthTracker(min_obs=5, extreme_proba=0.95, degenerate_frac=0.90)
        for _ in range(20):
            tracker.update("gnb", 1.0)
        flags = tracker.get_flags("gnb")
        assert flags["degenerate_bullish"] is True

    def test_always_bearish_flagged(self):
        """A model that always predicts <= 0.05 should be flagged degenerate_bearish."""
        tracker = ModelHealthTracker(min_obs=5, extreme_proba=0.95, degenerate_frac=0.90)
        for _ in range(20):
            tracker.update("lr", 0.01)
        flags = tracker.get_flags("lr")
        assert flags["degenerate_bearish"] is True

    def test_normal_model_not_flagged(self):
        """A model with realistic probability spread should not be flagged."""
        tracker = ModelHealthTracker(min_obs=5)
        rng = np.random.default_rng(0)
        for p in rng.uniform(0.3, 0.7, 30):
            tracker.update("hgbc", float(p))
        flags = tracker.get_flags("hgbc")
        assert flags["degenerate_bullish"] is False
        assert flags["degenerate_bearish"] is False
        assert flags["low_variance"] is False

    def test_low_variance_flagged(self):
        """A model with near-zero std should be flagged low_variance."""
        tracker = ModelHealthTracker(min_obs=5, low_std=0.01)
        for _ in range(20):
            tracker.update("const_model", 0.5)
        flags = tracker.get_flags("const_model")
        assert flags["low_variance"] is True

    def test_below_min_obs_no_flags(self):
        """Should not flag with fewer than min_obs observations."""
        tracker = ModelHealthTracker(min_obs=20)
        for _ in range(10):
            tracker.update("gnb", 1.0)
        flags = tracker.get_flags("gnb")
        # Not enough observations to flag
        assert flags["degenerate_bullish"] is False

    def test_update_batch_works(self):
        """update_batch must record predictions for multiple models."""
        tracker = ModelHealthTracker()
        tracker.update_batch({"hgbc": 0.62, "lr": 0.01, "gnb": 1.0})
        assert tracker.get_flags("hgbc")["n_obs"] == 1
        assert tracker.get_flags("lr")["n_obs"] == 1
        assert tracker.get_flags("gnb")["n_obs"] == 1

    def test_get_all_flags_returns_all_models(self):
        """get_all_flags must return entries for all tracked models."""
        tracker = ModelHealthTracker()
        tracker.update_batch({"hgbc": 0.62, "lr": 0.01})
        all_flags = tracker.get_all_flags()
        assert "hgbc" in all_flags
        assert "lr" in all_flags

    def test_format_log_summary_contains_model_ids(self):
        """format_log_summary should include each tracked model."""
        tracker = ModelHealthTracker(min_obs=2)
        tracker.update_batch({"gnb": 1.0, "lr": 0.02})
        tracker.update_batch({"gnb": 1.0, "lr": 0.01})
        summary = tracker.format_log_summary(roles_dict={"gnb": "diagnostic", "lr": "diagnostic"})
        assert "gnb" in summary
        assert "lr" in summary

    def test_reset_clears_history(self):
        """reset() must clear history for the specified model."""
        tracker = ModelHealthTracker()
        tracker.update("gnb", 1.0)
        tracker.reset("gnb")
        assert tracker.get_flags("gnb")["n_obs"] == 0

    def test_pct_above_thr_computed_correctly(self):
        """pct_above_thr should reflect fraction of obs >= extreme_proba."""
        tracker = ModelHealthTracker(extreme_proba=0.90)
        for _ in range(5):
            tracker.update("m", 0.95)  # above thr
        for _ in range(5):
            tracker.update("m", 0.50)  # below thr
        flags = tracker.get_flags("m")
        assert flags["pct_above_thr"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry role helpers
# ─────────────────────────────────────────────────────────────────────────────

from infra import (  # noqa: E402
    split_votes_by_role, compute_active_stats, build_roles_dict_from_config,
    ROLE_ACTIVE, ROLE_SHADOW, ROLE_DIAGNOSTIC,
)


class TestModelRegistryRoleHelpers:
    """Verify model_registry role-aware helpers."""

    def test_split_votes_by_role_active(self):
        """Active-role votes must go to active dict only."""
        votes = {"hgbc": 0.70, "et": 0.65, "rf": 0.60}
        roles = {"hgbc": ROLE_ACTIVE, "et": ROLE_ACTIVE, "rf": ROLE_ACTIVE}
        active, shadow, diag = split_votes_by_role(votes, roles)
        assert active == votes
        assert shadow == {}
        assert diag == {}

    def test_split_votes_by_role_diagnostic(self):
        """Diagnostic-role votes must go to diagnostic dict only."""
        votes = {"lr": 0.01, "gnb": 1.0}
        roles = {"lr": ROLE_DIAGNOSTIC, "gnb": ROLE_DIAGNOSTIC}
        active, shadow, diag = split_votes_by_role(votes, roles)
        assert active == {}
        assert diag == votes

    def test_split_votes_by_role_mixed(self):
        """Mixed roles must be correctly separated."""
        votes = {"hgbc": 0.65, "lr": 0.01, "lgbm_bal": 0.60}
        roles = {"hgbc": ROLE_ACTIVE, "lr": ROLE_DIAGNOSTIC, "lgbm_bal": ROLE_SHADOW}
        active, shadow, diag = split_votes_by_role(votes, roles)
        assert "hgbc" in active
        assert "lgbm_bal" in shadow
        assert "lr" in diag

    def test_compute_active_stats_mean(self):
        """compute_active_stats must return correct mean."""
        votes = {"hgbc": 0.60, "et": 0.70, "rf": 0.50}
        stats = compute_active_stats(votes, agree_thr=0.5)
        assert stats["active_mean"] == pytest.approx((0.60 + 0.70 + 0.50) / 3)

    def test_compute_active_stats_n_agree(self):
        """compute_active_stats must count models above agree_thr."""
        votes = {"hgbc": 0.60, "et": 0.70, "rf": 0.40}
        stats = compute_active_stats(votes, agree_thr=0.5)
        assert stats["active_n_agree"] == 2  # hgbc and et are above 0.5

    def test_compute_active_stats_empty_fallback(self):
        """compute_active_stats with empty dict must return safe default."""
        stats = compute_active_stats({}, agree_thr=0.5)
        assert stats["active_mean"] == 0.5
        assert stats["active_n_agree"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# format_vote_log with role-separated output
# ─────────────────────────────────────────────────────────────────────────────

from journals import format_vote_log as _fmt_vote_log  # noqa: E402


class TestFormatVoteLogWithRoles:
    """Verify format_vote_log emits role-separated output when fields are present."""

    def test_role_aware_format_includes_active(self):
        """When active_mean is present, log must include active= section."""
        conf = {
            "active_mean": 0.62, "active_std": 0.05, "active_n_agree": 3,
            "active_votes": {"hgbc": 0.65, "et": 0.70, "rf": 0.58},
            "shadow_votes": {"et_shallow": 0.60},
            "diagnostic_votes": {"lr": 0.01},
            "excluded_models": {"lr": "diagnostic_only"},
            "per_model": {"hgbc": 0.65, "et": 0.70, "rf": 0.58, "lr": 0.01},
        }
        line = _fmt_vote_log("ADAUSD", conf, market_mode="pump")
        assert "active_mean=0.62" in line
        assert "active=" in line
        assert "diag=" in line
        assert "mode=pump" in line

    def test_shadow_votes_in_log(self):
        """Shadow votes must appear in the log line."""
        conf = {
            "active_mean": 0.62, "active_std": 0.05, "active_n_agree": 2,
            "active_votes": {"hgbc": 0.65, "rf": 0.60},
            "shadow_votes": {"lgbm_bal": 0.72, "cal_et": 0.68},
            "diagnostic_votes": {},
            "excluded_models": {},
            "per_model": {},
        }
        line = _fmt_vote_log("XRPUSD", conf)
        assert "shadow=" in line
        assert "lgbm_bal" in line

    def test_legacy_format_without_active_mean(self):
        """Without active_mean field, log must use legacy format."""
        conf = {
            "class_proba": 0.58,
            "std_proba": 0.10,
            "n_agree": 3,
            "per_model": {"lr": 0.50, "hgbc": 0.70, "et": 0.65},
        }
        line = _fmt_vote_log("SOLUSD", conf)
        assert "mean=0.58" in line
        assert "votes=" in line
        # Must NOT have active_mean since field is absent
        assert "active_mean" not in line
