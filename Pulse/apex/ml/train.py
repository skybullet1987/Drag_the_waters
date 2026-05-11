"""Pulse.apex.ml.train — soft-vote ensemble trainer (XGB + LGBM + LogReg).

Training pipeline
-----------------
For each walk-forward fold:
  1. Train each available base learner on (X_train, y_train).
  2. Score y_test → predict_proba; soft-vote average.
  3. Compute fold metrics: ROC-AUC, precision @ prob>0.65, log-loss.
  4. Refit on the full dataset → final ensemble model.
  5. Joblib-dump the ensemble + a metadata sidecar.

Optional XGBoost and LightGBM are guarded by try/except. When neither
is installed (e.g. fresh QC env without the ML extras), the pipeline
falls back to LogReg-only — still functional, just less expressive.

The trainer does NOT touch QC. It reads pre-built (X, y) from
Pulse.apex.ml.dataset and writes a portable joblib that the inference
side (predict.py) loads at QC algorithm start.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

# ─── Optional dependencies (degrade gracefully) ─────────────────────────────

try:
    import numpy as _np                # type: ignore
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, log_loss, precision_score
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as _xgb              # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as _lgb             # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_HIGH_PROB_CUT = 0.65       # for precision@high
DEFAULT_RANDOM_STATE  = 42
DEFAULT_LGBM_KWARGS = {
    "n_estimators":     200,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 20,
    "objective":        "binary",
    "verbose":          -1,
}
DEFAULT_XGB_KWARGS = {
    "n_estimators":   200,
    "learning_rate":  0.05,
    "max_depth":      6,
    "objective":      "binary:logistic",
    "eval_metric":    "logloss",
    "verbosity":      0,
}


# ─── Result containers ──────────────────────────────────────────────────────


@dataclass
class FoldMetrics:
    fold_idx:        int
    train_size:      int
    test_size:       int
    roc_auc:         float
    log_loss:        float
    precision_high:  float
    n_high_prob:     int
    base_rate:       float


@dataclass
class TrainReport:
    n_folds:           int
    fold_metrics:      list[FoldMetrics] = field(default_factory=list)
    mean_roc_auc:      float = 0.0
    mean_precision:    float = 0.0
    members_used:      list[str] = field(default_factory=list)
    feature_dim:       int = 0
    random_state:      int = DEFAULT_RANDOM_STATE

    def to_dict(self) -> dict:
        return {
            "n_folds":         self.n_folds,
            "mean_roc_auc":    round(self.mean_roc_auc, 4),
            "mean_precision":  round(self.mean_precision, 4),
            "members_used":    list(self.members_used),
            "feature_dim":     self.feature_dim,
            "random_state":    self.random_state,
            "fold_metrics": [
                {
                    "fold_idx":       m.fold_idx,
                    "train_size":     m.train_size,
                    "test_size":      m.test_size,
                    "roc_auc":        round(m.roc_auc, 4),
                    "log_loss":       round(m.log_loss, 4),
                    "precision_high": round(m.precision_high, 4),
                    "n_high_prob":    m.n_high_prob,
                    "base_rate":      round(m.base_rate, 4),
                }
                for m in self.fold_metrics
            ],
        }


# ─── Per-member training ─────────────────────────────────────────────────────


def _train_logreg(X_train, y_train, random_state=DEFAULT_RANDOM_STATE):
    if not HAS_SKLEARN:
        return None, None
    if len(set(y_train)) < 2:
        # sklearn LogReg cannot fit on a single-class label set; skip.
        return None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    base = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0,
                              random_state=random_state)
    try:
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(Xs, y_train)
        return scaler, clf
    except Exception:
        base.fit(Xs, y_train)
        return scaler, base


def _train_xgb(X_train, y_train, random_state=DEFAULT_RANDOM_STATE):
    if not HAS_XGB or not HAS_NUMPY:
        return None
    if len(set(y_train)) < 2:
        return None
    kwargs = dict(DEFAULT_XGB_KWARGS)
    kwargs["random_state"] = random_state
    clf = _xgb.XGBClassifier(**kwargs)
    clf.fit(_np.array(X_train), _np.array(y_train))
    return clf


def _train_lgbm(X_train, y_train, random_state=DEFAULT_RANDOM_STATE):
    if not HAS_LGBM or not HAS_NUMPY:
        return None
    if len(set(y_train)) < 2:
        return None
    kwargs = dict(DEFAULT_LGBM_KWARGS)
    kwargs["random_state"] = random_state
    clf = _lgb.LGBMClassifier(**kwargs)
    clf.fit(_np.array(X_train), _np.array(y_train))
    return clf


# ─── Soft-vote ─────────────────────────────────────────────────────────────


def _proba_from(model, scaler, X) -> list[float]:
    """Return P(label=1) for each row."""
    if HAS_NUMPY:
        Xa = _np.array(X)
    else:
        Xa = X
    if scaler is not None:
        Xa = scaler.transform(Xa)
    p = model.predict_proba(Xa)
    if hasattr(p, "tolist"):
        p = p.tolist()
    return [row[1] if len(row) > 1 else float(row[0]) for row in p]


def soft_vote(probs_list: list[list[float]]) -> list[float]:
    if not probs_list:
        return []
    n = len(probs_list[0])
    avg = [0.0] * n
    for p in probs_list:
        for i, v in enumerate(p):
            avg[i] += v
    return [v / len(probs_list) for v in avg]


# ─── Public training API ─────────────────────────────────────────────────────


def fit_ensemble(X_train, y_train, *,
                 random_state: int = DEFAULT_RANDOM_STATE
                 ) -> tuple[dict, list[str]]:
    """Train all available base learners; return (model_bundle, members_used)."""
    bundle: dict = {}
    members: list[str] = []

    scaler_lr, lr = _train_logreg(X_train, y_train, random_state)
    if lr is not None:
        bundle["logreg"] = {"scaler": scaler_lr, "model": lr}
        members.append("logreg")

    xgb = _train_xgb(X_train, y_train, random_state)
    if xgb is not None:
        bundle["xgb"] = {"scaler": None, "model": xgb}
        members.append("xgb")

    lgbm = _train_lgbm(X_train, y_train, random_state)
    if lgbm is not None:
        bundle["lgbm"] = {"scaler": None, "model": lgbm}
        members.append("lgbm")

    return bundle, members


def predict_ensemble(bundle: dict, X) -> list[float]:
    if not bundle:
        # Cold-start fallback: return neutral 0.5
        return [0.5] * len(X)
    probs = []
    for name, member in bundle.items():
        try:
            p = _proba_from(member["model"], member["scaler"], X)
            probs.append(p)
        except Exception:
            continue
    if not probs:
        return [0.5] * len(X)
    return soft_vote(probs)


# ─── Walk-forward driver ────────────────────────────────────────────────────


def walk_forward_train_eval(
    samples,
    folds,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    high_prob_cut: float = DEFAULT_HIGH_PROB_CUT,
) -> tuple[TrainReport, dict]:
    """Walk-forward train+eval; return (report, final_full_fit_bundle).

    The final bundle is trained on ALL samples — recommended for live
    deployment after walk-forward metrics pass the gate.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn is required for walk_forward_train_eval")

    fold_metrics: list[FoldMetrics] = []
    members_seen: list[str] = []
    feature_dim = len(samples[0].features) if samples else 0

    for fi, fold in enumerate(folds):
        X_tr = [samples[i].features for i in fold.train_idx]
        y_tr = [samples[i].label_pos for i in fold.train_idx]
        X_te = [samples[i].features for i in fold.test_idx]
        y_te = [samples[i].label_pos for i in fold.test_idx]
        bundle, members = fit_ensemble(X_tr, y_tr, random_state=random_state)
        for m in members:
            if m not in members_seen:
                members_seen.append(m)
        probs = predict_ensemble(bundle, X_te)
        try:
            roc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.5
        except Exception:
            roc = 0.5
        try:
            ll = log_loss(y_te, probs, labels=[0, 1])
        except Exception:
            ll = float("nan")
        high = [(int(p >= high_prob_cut), int(y)) for p, y in zip(probs, y_te)]
        n_high = sum(1 for h, _ in high if h)
        if n_high >= 5:
            tp = sum(1 for h, y in high if h and y == 1)
            prec = tp / n_high
        else:
            prec = float("nan")
        base_rate = sum(y_te) / len(y_te) if y_te else 0.0
        fold_metrics.append(FoldMetrics(
            fold_idx=fi,
            train_size=len(X_tr), test_size=len(X_te),
            roc_auc=roc, log_loss=ll, precision_high=prec,
            n_high_prob=n_high, base_rate=base_rate,
        ))

    # Aggregate
    valid_aucs = [m.roc_auc for m in fold_metrics
                  if not (m.roc_auc is None or math.isnan(m.roc_auc))]
    valid_precs = [m.precision_high for m in fold_metrics
                   if not (m.precision_high is None
                           or math.isnan(m.precision_high))]
    mean_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.5
    mean_prec = sum(valid_precs) / len(valid_precs) if valid_precs else 0.0

    report = TrainReport(
        n_folds=len(fold_metrics),
        fold_metrics=fold_metrics,
        mean_roc_auc=mean_auc, mean_precision=mean_prec,
        members_used=members_seen, feature_dim=feature_dim,
        random_state=random_state,
    )

    # Final fit on ALL samples
    X_all = [s.features for s in samples]
    y_all = [s.label_pos for s in samples]
    final_bundle, _ = fit_ensemble(X_all, y_all, random_state=random_state)

    return report, final_bundle


def passes_acceptance_gate(report: TrainReport, *,
                            min_roc_auc: float = 0.55,
                            min_precision: float = 0.55) -> tuple[bool, str]:
    """The PLAN gate: ROC-AUC ≥ 0.55 AND precision@0.65 ≥ 0.55."""
    if report.mean_roc_auc < min_roc_auc:
        return False, (f"roc_auc {report.mean_roc_auc:.3f} "
                       f"< gate {min_roc_auc}")
    if report.mean_precision < min_precision:
        return False, (f"precision_high {report.mean_precision:.3f} "
                       f"< gate {min_precision}")
    return True, "gates passed"


def serialize_ensemble(bundle: dict, report: TrainReport, path: str) -> None:
    """Joblib dump bundle + sidecar JSON with the report."""
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required to serialize models") from exc
    joblib.dump(bundle, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_ensemble(path: str) -> dict:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("joblib is required to load models") from exc
    return joblib.load(path)
