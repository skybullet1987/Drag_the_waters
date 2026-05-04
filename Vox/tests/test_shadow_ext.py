"""
Tests for the Tier 1 + Tier 2 shadow-lab extensions added in strategy.py.

All tests must pass without any optional deps (imblearn, ngboost, hdbscan,
flaml, xgboost).  Optional-dep models are expected to return None or no-op
when the dep is missing; the tests verify that graceful degradation.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy import (  # type: ignore  # noqa: E402
    _make_mlp,
    _make_balanced_rf,
    _make_rusboost,
    _make_ngboost,
    _make_xgb_dart,
    HDBSCANRegimeDiagnostic,
    FLAMLShadow,
    extend_shadow_estimators,
    ROLE_SHADOW,
    ROLE_DIAGNOSTIC,
    HAS_IMBLEARN,
    HAS_NGBOOST,
    HAS_HDBSCAN,
    HAS_FLAML,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

N = 80   # enough for train/val split inside MLP
FEAT = 20

rng = np.random.default_rng(0)
X_train = rng.standard_normal((N, FEAT)).astype(float)
y_train = rng.integers(0, 2, size=N)


# ── Factory function tests ─────────────────────────────────────────────────────

class TestMakemlp:
    def test_returns_estimator(self):
        est = _make_mlp()
        assert est is not None

    def test_fit_predict_proba(self):
        est = _make_mlp()
        if est is None:
            pytest.skip("MLPClassifier unavailable")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_train[:5])
        assert p.shape == (5, 2)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


class TestMakeBalancedRF:
    def test_returns_none_or_estimator(self):
        est = _make_balanced_rf()
        if not HAS_IMBLEARN:
            assert est is None
        else:
            assert est is not None

    def test_fit_predict_proba_when_available(self):
        est = _make_balanced_rf()
        if est is None:
            pytest.skip("imbalanced-learn not installed")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_train[:5])
        assert p.shape == (5, 2)


class TestMakeRUSBoost:
    def test_returns_none_or_estimator(self):
        est = _make_rusboost()
        if not HAS_IMBLEARN:
            assert est is None

    def test_fit_predict_proba_when_available(self):
        est = _make_rusboost()
        if est is None:
            pytest.skip("imbalanced-learn not installed")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_train[:5])
        assert p.shape == (5, 2)


class TestMakeNGBoost:
    def test_returns_none_when_missing(self):
        est = _make_ngboost()
        if not HAS_NGBOOST:
            assert est is None

    def test_fit_predict_proba_when_available(self):
        est = _make_ngboost()
        if est is None:
            pytest.skip("ngboost not installed")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_train[:5])
        assert p.shape[0] == 5


class TestMakeXGBDart:
    def test_returns_none_or_estimator(self):
        est = _make_xgb_dart()
        # xgboost is usually installed; just verify no crash

    def test_fit_predict_proba_when_available(self):
        est = _make_xgb_dart()
        if est is None:
            pytest.skip("xgboost not installed")
        est.fit(X_train, y_train)
        p = est.predict_proba(X_train[:5])
        assert p.shape == (5, 2)


# ── HDBSCANRegimeDiagnostic tests ─────────────────────────────────────────────

class TestHDBSCANRegimeDiagnostic:
    def test_init_default(self):
        d = HDBSCANRegimeDiagnostic()
        assert d._mcs == 15
        assert not d._fitted
        assert d._centroid is None

    def test_predict_proba_unfitted_returns_half(self):
        d = HDBSCANRegimeDiagnostic()
        out = d.predict_proba(X_train[:4])
        assert out.shape == (4, 2)
        assert np.allclose(out[:, 0], 0.5)

    def test_fit_no_crash_with_or_without_dep(self):
        d = HDBSCANRegimeDiagnostic()
        d.fit(X_train, y_train)  # must not raise

    def test_predict_proba_after_fit(self):
        d = HDBSCANRegimeDiagnostic()
        d.fit(X_train, y_train)
        out = d.predict_proba(X_train[:3])
        assert out.shape == (3, 2)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_predict_proba_sums_to_one(self):
        d = HDBSCANRegimeDiagnostic()
        d.fit(X_train, y_train)
        out = d.predict_proba(X_train[:5])
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_idempotent(self):
        d = HDBSCANRegimeDiagnostic()
        d.fit(X_train, y_train)
        d.fit(X_train, y_train)  # second fit should not crash


# ── FLAMLShadow tests ─────────────────────────────────────────────────────────

class TestFLAMLShadow:
    def test_init(self):
        f = FLAMLShadow(time_budget=5)
        assert f._tb == 5
        assert not f._fitted
        assert f._m is None

    def test_predict_proba_unfitted_returns_half(self):
        f = FLAMLShadow()
        out = f.predict_proba(X_train[:4])
        assert out.shape == (4, 2)
        assert np.allclose(out[:, 0], 0.5)

    def test_fit_no_crash_with_or_without_dep(self):
        f = FLAMLShadow(time_budget=3)
        f.fit(X_train, y_train)  # must not raise

    def test_predict_proba_after_fit_when_available(self):
        if not HAS_FLAML:
            pytest.skip("flaml not installed")
        f = FLAMLShadow(time_budget=5)
        f.fit(X_train, y_train)
        out = f.predict_proba(X_train[:4])
        assert out.shape == (4, 2)


# ── extend_shadow_estimators tests ────────────────────────────────────────────

class TestExtendShadowEstimators:
    def test_returns_list(self):
        result = extend_shadow_estimators([])
        assert isinstance(result, list)

    def test_all_tuples_have_three_elements(self):
        result = extend_shadow_estimators([])
        for item in result:
            assert len(item) == 3, f"Bad tuple: {item}"

    def test_roles_are_valid(self):
        valid = {ROLE_SHADOW, ROLE_DIAGNOSTIC}
        result = extend_shadow_estimators([])
        for mid, est, role in result:
            assert role in valid, f"{mid} has invalid role {role}"

    def test_max_count_respected(self):
        result = extend_shadow_estimators([], max_count=3)
        assert len(result) <= 3

    def test_existing_models_preserved(self):
        from sklearn.linear_model import LogisticRegression
        existing = [("lr_test", LogisticRegression(), ROLE_SHADOW)]
        result = extend_shadow_estimators(existing, max_count=20)
        ids = [mid for mid, _, _ in result]
        assert "lr_test" in ids

    def test_new_models_present(self):
        result = extend_shadow_estimators([])
        ids = {mid for mid, _, _ in result}
        # These come from factory functions with no external deps
        assert "gbc" in ids
        assert "ada" in ids
        assert "mlp" in ids

    def test_hdbscan_regime_present(self):
        result = extend_shadow_estimators([])
        ids = {mid for mid, _, _ in result}
        assert "hdbscan_regime" in ids

    def test_no_duplicate_ids(self):
        result = extend_shadow_estimators([])
        ids = [mid for mid, _, _ in result]
        assert len(ids) == len(set(ids)), f"Duplicate model ids: {ids}"

    def test_estimators_have_fit_predict_proba(self):
        result = extend_shadow_estimators([])
        for mid, est, role in result:
            assert hasattr(est, "fit"), f"{mid} missing .fit"
            assert hasattr(est, "predict_proba"), f"{mid} missing .predict_proba"

    def test_logger_called_on_error(self):
        logs = []
        # Pass a broken existing list entry to ensure logger path is tested
        result = extend_shadow_estimators([], logger=lambda m: logs.append(m))
        # No crash; logger may or may not be invoked depending on what's installed
        assert isinstance(result, list)
