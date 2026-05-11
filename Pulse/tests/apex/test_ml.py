"""Tests for Pulse.apex.ml.* (Phase 4 ML decision layer)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
import pytest

from Pulse.apex.config import APEX_FEATURE_DIM
from Pulse.apex.ml.dataset import (
    DEFAULT_FORWARD_HOURS, DEFAULT_POS_THRESHOLD,
    Sample, WalkForwardFold,
    build_dataset, compute_forward_return, lookup_forward_price,
    to_xy, to_xy_returns, walk_forward_splits,
)
from Pulse.apex.ml.predict import ApexInference, _fallback_prob
from Pulse.apex.ml.train import (
    HAS_SKLEARN, HAS_NUMPY,
    DEFAULT_HIGH_PROB_CUT,
    fit_ensemble, predict_ensemble, soft_vote,
    walk_forward_train_eval, passes_acceptance_gate,
    serialize_ensemble, load_ensemble,
)


# ───────────────────────────────────────────────────────────────────────────────
# dataset.lookup_forward_price
# ───────────────────────────────────────────────────────────────────────────────

def _hourly_series(start: datetime, hours: int, start_price: float = 100.0,
                   step_pct: float = 0.001
                   ) -> list[tuple[datetime, float]]:
    out = []
    p = start_price
    for h in range(hours):
        out.append((start + timedelta(hours=h), p))
        p *= (1 + step_pct)
    return out


def test_lookup_forward_returns_close_match():
    series = _hourly_series(datetime(2025, 1, 1), 48)
    fwd = lookup_forward_price(series, datetime(2025, 1, 1), forward_hours=24)
    assert fwd is not None
    assert fwd == pytest.approx(series[24][1])


def test_lookup_forward_returns_none_when_too_far():
    series = _hourly_series(datetime(2025, 1, 1), 5)
    fwd = lookup_forward_price(series, datetime(2025, 1, 1), forward_hours=24)
    # Closest point is 4h away from target → > 1h tolerance → None
    assert fwd is None


def test_lookup_forward_picks_nearest_within_tolerance():
    # Series at 12:00 and 13:30 only
    series = [
        (datetime(2025, 1, 1, 12, 0), 100.0),
        (datetime(2025, 1, 1, 13, 30), 102.0),
    ]
    fwd = lookup_forward_price(series, datetime(2025, 1, 1, 13, 0),
                                forward_hours=0)
    # Target = 13:00; closest is 13:30 (30min away), within ±60min
    assert fwd == 102.0


# ───────────────────────────────────────────────────────────────────────────────
# dataset.compute_forward_return
# ───────────────────────────────────────────────────────────────────────────────

def test_forward_return_basic():
    series = [(datetime(2025, 1, 1), 100.0),
              (datetime(2025, 1, 2), 105.0)]
    r = compute_forward_return(series, datetime(2025, 1, 1), 100.0,
                                forward_hours=24)
    assert r == pytest.approx(0.05)


def test_forward_return_handles_zero_start():
    series = [(datetime(2025, 1, 1), 0.0),
              (datetime(2025, 1, 2), 100.0)]
    r = compute_forward_return(series, datetime(2025, 1, 1), 0.0,
                                forward_hours=24)
    assert r is None


# ───────────────────────────────────────────────────────────────────────────────
# build_dataset
# ───────────────────────────────────────────────────────────────────────────────

def test_build_dataset_drops_records_with_no_price_series():
    feature_records = [(datetime(2025, 1, 1), "BTCUSD", [0.5] * 30, 100.0)]
    samples = build_dataset(feature_records, {})
    assert samples == []


def test_build_dataset_skips_when_forward_price_missing():
    feature_records = [(datetime(2025, 1, 1), "BTCUSD", [0.5] * 30, 100.0)]
    series = {"BTCUSD": [(datetime(2025, 1, 1), 100.0)]}  # no forward
    samples = build_dataset(feature_records, series)
    assert samples == []


def test_build_dataset_label_pos_when_above_threshold():
    feature_records = [(datetime(2025, 1, 1), "BTCUSD", [0.5] * 30, 100.0)]
    series = {"BTCUSD": [(datetime(2025, 1, 1), 100.0),
                          (datetime(2025, 1, 2), 105.0)]}
    samples = build_dataset(feature_records, series, pos_threshold=0.02)
    assert len(samples) == 1
    assert samples[0].label_pos == 1
    assert samples[0].forward_return == pytest.approx(0.05)


def test_build_dataset_label_neg_when_below_threshold():
    feature_records = [(datetime(2025, 1, 1), "BTCUSD", [0.5] * 30, 100.0)]
    series = {"BTCUSD": [(datetime(2025, 1, 1), 100.0),
                          (datetime(2025, 1, 2), 95.0)]}
    samples = build_dataset(feature_records, series,
                             pos_threshold=0.02, neg_threshold=-0.02)
    assert samples[0].label_pos == 0
    assert samples[0].label_neg == 1


def test_to_xy_returns_aligned_columns():
    samples = [
        Sample(datetime(2025, 1, 1), "BTC", [1.0, 2.0], 0.05, 1, 0),
        Sample(datetime(2025, 1, 2), "BTC", [3.0, 4.0], -0.05, 0, 1),
    ]
    X, y = to_xy(samples)
    assert X == [[1.0, 2.0], [3.0, 4.0]]
    assert y == [1, 0]


def test_to_xy_returns_continuous_targets():
    samples = [
        Sample(datetime(2025, 1, 1), "BTC", [1.0], 0.05, 1, 0),
        Sample(datetime(2025, 1, 2), "BTC", [2.0], -0.03, 0, 0),
    ]
    X, y = to_xy_returns(samples)
    assert y == [0.05, -0.03]


# ───────────────────────────────────────────────────────────────────────────────
# walk_forward_splits
# ───────────────────────────────────────────────────────────────────────────────

def _make_samples(start: datetime, days: int, per_day: int = 10) -> list[Sample]:
    out = []
    for d in range(days):
        for i in range(per_day):
            ts = start + timedelta(days=d, minutes=i * 60)
            out.append(Sample(ts, "BTCUSD", [0.5] * 30, 0.0, 0, 0))
    return out


def test_walk_forward_splits_creates_non_overlapping_folds():
    samples = _make_samples(datetime(2024, 1, 1), 200, per_day=5)
    folds = walk_forward_splits(samples, train_window_days=60,
                                 test_window_days=30, step_days=30,
                                 min_test_size=20)
    assert len(folds) >= 2
    # Each fold's test must NOT overlap with its train
    for f in folds:
        assert f.train_end == f.test_start


def test_walk_forward_splits_drops_undersized_folds():
    samples = _make_samples(datetime(2024, 1, 1), 100, per_day=5)
    folds = walk_forward_splits(samples, train_window_days=60,
                                 test_window_days=30, step_days=30,
                                 min_test_size=10000)  # impossibly high
    assert folds == []


def test_walk_forward_splits_handles_empty_input():
    folds = walk_forward_splits([])
    assert folds == []


# ───────────────────────────────────────────────────────────────────────────────
# train.soft_vote
# ───────────────────────────────────────────────────────────────────────────────

def test_soft_vote_averages_columnwise():
    out = soft_vote([[0.2, 0.6], [0.4, 0.8]])
    assert out == [pytest.approx(0.3), pytest.approx(0.7)]


def test_soft_vote_empty_input():
    assert soft_vote([]) == []


# ───────────────────────────────────────────────────────────────────────────────
# train.fit_ensemble + predict_ensemble (need sklearn)
# ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_fit_ensemble_trains_at_least_logreg():
    X = [[float(i % 5)] * 30 for i in range(100)]
    y = [int(i % 2) for i in range(100)]
    bundle, members = fit_ensemble(X, y)
    assert "logreg" in members
    assert len(bundle) >= 1


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_fit_ensemble_skips_models_when_one_class_only():
    X = [[1.0] * 30 for _ in range(50)]
    y = [1] * 50      # all-same class
    bundle, _ = fit_ensemble(X, y)
    # Only logreg has the cold-start fallback path that handles single class
    # Other learners should be skipped (xgb/lgbm need 2 classes).
    assert "xgb" not in bundle
    assert "lgbm" not in bundle


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_predict_ensemble_returns_in_unit_range():
    X_tr = [[float(i % 7) / 7.0] * 30 for i in range(80)]
    y_tr = [int((i % 7) > 3) for i in range(80)]
    bundle, _ = fit_ensemble(X_tr, y_tr)
    X_te = [[0.5] * 30 for _ in range(5)]
    probs = predict_ensemble(bundle, X_te)
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_ensemble_empty_bundle_returns_neutral():
    X_te = [[0.0] * 30 for _ in range(3)]
    probs = predict_ensemble({}, X_te)
    assert probs == [0.5, 0.5, 0.5]


# ───────────────────────────────────────────────────────────────────────────────
# train.walk_forward_train_eval (integration)
# ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_walk_forward_train_eval_produces_report():
    """End-to-end: build a synthetic dataset where label correlates with
    feature[0] > 0.5, walk-forward, expect ROC-AUC > 0.6."""
    samples = []
    seed = 42
    for d in range(180):
        for i in range(8):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            f0 = (seed / 0x7FFFFFFF)
            features = [f0] + [0.0] * 29
            label_pos = 1 if f0 > 0.6 else 0
            ts = datetime(2024, 1, 1) + timedelta(days=d, minutes=i * 60)
            samples.append(Sample(ts, "BTCUSD", features,
                                   forward_return=0.0,
                                   label_pos=label_pos, label_neg=0))
    folds = walk_forward_splits(samples, train_window_days=60,
                                 test_window_days=30, step_days=30,
                                 min_test_size=50)
    assert len(folds) >= 2
    report, bundle = walk_forward_train_eval(samples, folds)
    assert report.n_folds == len(folds)
    assert report.mean_roc_auc > 0.6   # signal IS in the data
    assert "logreg" in report.members_used


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_walk_forward_train_eval_random_data_low_auc():
    """Pure noise → AUC near 0.5 (no edge to find)."""
    seed = 123
    samples = []
    for d in range(180):
        for i in range(5):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            f0 = (seed / 0x7FFFFFFF)
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            label = int((seed / 0x7FFFFFFF) > 0.5)
            ts = datetime(2024, 1, 1) + timedelta(days=d, minutes=i * 60)
            samples.append(Sample(ts, "BTCUSD", [f0] + [0.0] * 29,
                                   0.0, label, 0))
    folds = walk_forward_splits(samples, train_window_days=60,
                                 test_window_days=30, step_days=30,
                                 min_test_size=50)
    report, _ = walk_forward_train_eval(samples, folds)
    assert 0.40 <= report.mean_roc_auc <= 0.60


# ───────────────────────────────────────────────────────────────────────────────
# train.passes_acceptance_gate
# ───────────────────────────────────────────────────────────────────────────────

def test_acceptance_gate_passes_when_metrics_ok():
    from Pulse.apex.ml.train import TrainReport
    rep = TrainReport(n_folds=3, mean_roc_auc=0.62, mean_precision=0.60)
    ok, msg = passes_acceptance_gate(rep)
    assert ok is True


def test_acceptance_gate_fails_on_low_auc():
    from Pulse.apex.ml.train import TrainReport
    rep = TrainReport(n_folds=3, mean_roc_auc=0.50, mean_precision=0.60)
    ok, msg = passes_acceptance_gate(rep)
    assert ok is False
    assert "roc_auc" in msg


def test_acceptance_gate_fails_on_low_precision():
    from Pulse.apex.ml.train import TrainReport
    rep = TrainReport(n_folds=3, mean_roc_auc=0.62, mean_precision=0.50)
    ok, msg = passes_acceptance_gate(rep)
    assert ok is False
    assert "precision_high" in msg


# ───────────────────────────────────────────────────────────────────────────────
# train.serialize / load roundtrip
# ───────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_serialize_load_roundtrip(tmp_path):
    from Pulse.apex.ml.train import TrainReport
    X = [[float(i % 3) / 3.0] * 30 for i in range(60)]
    y = [int((i % 3) > 1) for i in range(60)]
    bundle, _ = fit_ensemble(X, y)
    rep = TrainReport(n_folds=0, mean_roc_auc=0.7, mean_precision=0.65)
    out_path = tmp_path / "apex.joblib"
    serialize_ensemble(bundle, rep, str(out_path))
    assert out_path.exists()
    assert (tmp_path / "apex.joblib.json").exists()
    loaded = load_ensemble(str(out_path))
    # Same set of member keys
    assert set(loaded.keys()) == set(bundle.keys())


# ───────────────────────────────────────────────────────────────────────────────
# predict.ApexInference
# ───────────────────────────────────────────────────────────────────────────────

def test_inference_fallback_neutral_when_no_model():
    inf = ApexInference(model_path=None)
    p = inf.predict_one([0.0] * APEX_FEATURE_DIM)
    assert 0.0 <= p <= 1.0
    assert inf.in_fallback_mode is True


def test_inference_fallback_only_skips_load(tmp_path):
    inf = ApexInference(model_path="nonexistent.joblib", fallback_only=True)
    p = inf.predict_one([0.5] * APEX_FEATURE_DIM)
    assert 0.0 <= p <= 1.0
    assert inf.in_fallback_mode is True


def test_inference_handles_short_feature_vector():
    inf = ApexInference(model_path=None)
    p = inf.predict_one([0.5] * 5)   # too short
    assert 0.0 <= p <= 1.0


def test_inference_handles_long_feature_vector():
    inf = ApexInference(model_path=None)
    p = inf.predict_one([0.5] * 100)  # too long
    assert 0.0 <= p <= 1.0


def test_inference_predict_many_returns_correct_length():
    inf = ApexInference(model_path=None)
    out = inf.predict_many([[0.5] * APEX_FEATURE_DIM for _ in range(5)])
    assert len(out) == 5
    assert all(0.0 <= p <= 1.0 for p in out)


def test_inference_predict_many_empty():
    inf = ApexInference(model_path=None)
    assert inf.predict_many([]) == []


def test_inference_load_failure_falls_back(tmp_path):
    bad_path = tmp_path / "no_such_model.joblib"
    inf = ApexInference(model_path=str(bad_path))
    p = inf.predict_one([0.5] * APEX_FEATURE_DIM)
    assert 0.0 <= p <= 1.0
    assert inf.in_fallback_mode is True
    assert inf.load_error is not None


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
def test_inference_loads_real_model(tmp_path):
    from Pulse.apex.ml.train import TrainReport
    X = [[float(i % 3) / 3.0] * 30 for i in range(60)]
    y = [int((i % 3) > 1) for i in range(60)]
    bundle, _ = fit_ensemble(X, y)
    rep = TrainReport(n_folds=0, mean_roc_auc=0.7, mean_precision=0.65)
    out = tmp_path / "apex.joblib"
    serialize_ensemble(bundle, rep, str(out))
    inf = ApexInference(model_path=str(out))
    p = inf.predict_one([0.5] * APEX_FEATURE_DIM)
    assert 0.0 <= p <= 1.0
    assert inf.in_fallback_mode is False


# ───────────────────────────────────────────────────────────────────────────────
# predict._fallback_prob
# ───────────────────────────────────────────────────────────────────────────────

def test_fallback_prob_neutral_at_zero_vector():
    p = _fallback_prob([0.0] * APEX_FEATURE_DIM)
    assert p == pytest.approx(0.5)


def test_fallback_prob_bullish_when_positive_signals():
    """Set every signal's raw score to +1 → high prob."""
    from Pulse.apex.feature_vector import FEATURES_PER_SIGNAL
    from Pulse.apex.feature_vector import DEFAULT_SIGNAL_ORDER
    vec = [0.0] * APEX_FEATURE_DIM
    for i in range(len(DEFAULT_SIGNAL_ORDER)):
        idx = i * FEATURES_PER_SIGNAL    # raw position
        if idx < APEX_FEATURE_DIM:
            vec[idx] = 1.0
    p = _fallback_prob(vec)
    assert p > 0.7


def test_fallback_prob_bearish_when_negative_signals():
    from Pulse.apex.feature_vector import FEATURES_PER_SIGNAL
    from Pulse.apex.feature_vector import DEFAULT_SIGNAL_ORDER
    vec = [0.0] * APEX_FEATURE_DIM
    for i in range(len(DEFAULT_SIGNAL_ORDER)):
        idx = i * FEATURES_PER_SIGNAL
        if idx < APEX_FEATURE_DIM:
            vec[idx] = -1.0
    p = _fallback_prob(vec)
    assert p < 0.3
