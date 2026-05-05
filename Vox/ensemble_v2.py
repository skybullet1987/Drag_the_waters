# ensemble_v2.py — Cutting-edge 15-model ensemble factory for gatling mode
#
# All models verified available on QuantConnect (DockerfileLeanFoundation):
#   scikit-learn 1.6.1, lightgbm 4.6.0, xgboost 3.0.5, catboost 1.2.8,
#   pytorch-tabnet 4.1.0, ngboost 0.5.6, interpret 0.7.2,
#   imbalanced-learn 0.14.1

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV


def _cal(est, method="isotonic", cv=2):
    return CalibratedClassifierCV(est, method=method, cv=cv)


def make_v2_estimators(logger=None, use_calibration=True):
    """Build the 15-model V2 ensemble.

    Returns list of (id, estimator, role, weight) tuples.
    Role: "active" | "shadow" | "veto"
    """
    models = []

    def _try_add(model_id, make_fn, role="active", weight=1.0):
        try:
            est = make_fn()
            models.append((model_id, est, role, weight))
        except Exception as exc:
            if logger:
                logger(f"[ensemble_v2] {model_id} init failed: {exc}")

    # ── TIER 1: Primary boosters (weight=1.5) ───────────────────────────────

    # 1. CatBoost — ordered boosting prevents target leakage
    def _catboost():
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            auto_class_weights="Balanced", verbose=0, random_seed=42,
            l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8,
        )
    _try_add("catboost", _catboost, weight=1.5)

    # 2. XGBoost histogram — fast GPU-accelerated boosting
    def _xgb_hist():
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            tree_method="hist", eval_metric="logloss",
            scale_pos_weight=2.0, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1,
            use_label_encoder=False,
        )
    _try_add("xgb_hist", _xgb_hist, weight=1.5)

    # 3. LightGBM GOSS — gradient-based one-side sampling
    def _lgbm_goss():
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            boosting_type="goss", class_weight="balanced",
            top_rate=0.2, other_rate=0.1, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=1,
        )
    _try_add("lgbm_goss", _lgbm_goss, weight=1.5)

    # 4. HistGradientBoosting — sklearn-native baseline
    def _hgbc():
        try:
            return HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4,
                min_samples_leaf=20, l2_regularization=0.3,
                random_state=42, class_weight="balanced",
            )
        except TypeError:
            return HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4,
                min_samples_leaf=20, l2_regularization=0.3,
                random_state=42,
            )
    _try_add("hgbc", _hgbc, weight=1.5)

    # ── TIER 2: Diversity layer (weight=1.0) ─────────────────────────────────

    # 5. TabNet — attention-based neural network for tabular data
    def _tabnet():
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.5,
            n_independent=1, n_shared=1,
            lambda_sparse=1e-3, momentum=0.3,
            verbose=0, seed=42,
        )
    _try_add("tabnet", _tabnet, weight=1.0)

    # 6. EBM — Explainable Boosting Machine (glass-box GAM)
    def _ebm():
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(
            max_bins=64, interactions=0, learning_rate=0.01,
            min_samples_leaf=5, max_rounds=300, random_state=42,
        )
    _try_add("ebm", _ebm, weight=1.0)

    # 7. NGBoost — natural gradient boosting with uncertainty
    def _ngboost():
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from sklearn.tree import DecisionTreeRegressor
        base = DecisionTreeRegressor(max_depth=3, random_state=42)
        return NGBClassifier(
            Base=base, Dist=Bernoulli, n_estimators=150,
            learning_rate=0.04, minibatch_frac=0.8,
            verbose=False, random_state=42,
        )
    _try_add("ngboost", _ngboost, weight=1.0)

    # 8. BalancedRandomForest — handles class imbalance natively
    def _rf_bal():
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=10,
            sampling_strategy="auto", replacement=False,
            random_state=42, n_jobs=1,
        )
    _try_add("rf_bal", _rf_bal, weight=1.0)

    # ── TIER 3: Anti-correlation layer (weight=0.75) ─────────────────────────

    # 9. Shallow ExtraTrees — regularized via depth limit
    def _et_d5():
        est = ExtraTreesClassifier(
            n_estimators=120, max_depth=5, min_samples_leaf=15,
            n_jobs=1, random_state=43, class_weight="balanced",
        )
        return _cal(est) if use_calibration else est
    _try_add("et_depth5", _et_d5, weight=0.75)

    # 10. Calibrated Ridge — pure linear, max diversity vs trees
    def _ridge_cal():
        return _cal(
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
            method="sigmoid", cv=3,
        )
    _try_add("ridge_cal", _ridge_cal, weight=0.75)

    # 11. Calibrated SVC(RBF) — kernel method, different geometry
    def _svc_rbf():
        return _cal(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
            method="sigmoid", cv=3,
        )
    _try_add("svc_rbf", _svc_rbf, weight=0.75)

    # 12. MLP — neural network with 2 hidden layers
    def _mlp():
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", alpha=0.001, learning_rate="adaptive",
            max_iter=300, early_stopping=True, validation_fraction=0.15,
            random_state=42,
        )
    _try_add("mlp", _mlp, weight=0.75)

    # ── TIER 4: Veto / meta (weight=1.0) ─────────────────────────────────────

    # 13. IsolationForest — anomaly veto (weird market = don't trade)
    def _iforest():
        return IsolationForest(
            n_estimators=100, contamination=0.1,
            random_state=42, n_jobs=1,
        )
    _try_add("iforest_veto", _iforest, role="veto", weight=1.0)

    # 14. LightGBM DART — dropout-regularized boosting
    def _lgbm_dart():
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            boosting_type="dart", class_weight="balanced",
            drop_rate=0.15, skip_drop=0.5,
            random_state=43, verbose=-1, n_jobs=1,
        )
    _try_add("lgbm_dart", _lgbm_dart, weight=1.0)

    # 15. Stacking meta-learner — learns which models to trust
    #     Uses LogisticRegression as meta-classifier over base model outputs.
    #     Trained separately after base models (see fit_stack_meta).
    #     Initially shadow until validated.
    def _stack_placeholder():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=500, C=0.5, class_weight="balanced", random_state=42,
        )
    _try_add("stack_meta", _stack_placeholder, role="shadow", weight=0.0)

    if logger:
        ids = [m[0] for m in models]
        logger(f"[ensemble_v2] initialized {len(models)} models: {ids}")

    return models


def fit_v2_models(models, X, y, logger=None):
    """Train all V2 models. Returns (fitted_models, n_ok).

    Special handling:
      - TabNet needs numpy arrays and separate fit API
      - IsolationForest is unsupervised (fit on X only)
      - stack_meta needs predictions from other models (deferred)
    """
    fitted = []
    n_ok = 0
    base_predictions = {}  # for stack_meta

    for model_id, est, role, weight in models:
        if model_id == "stack_meta":
            fitted.append((model_id, est, role, weight))
            continue

        try:
            if model_id == "iforest_veto":
                est.fit(X)
            elif model_id == "tabnet":
                X_f = np.asarray(X, dtype=np.float32)
                y_i = np.asarray(y, dtype=np.int64)
                est.fit(X_f, y_i, eval_set=[(X_f, y_i)],
                        max_epochs=50, patience=10, batch_size=256)
            elif model_id == "ngboost":
                est.fit(X, y)
            else:
                est.fit(X, y)
            n_ok += 1

            if role == "active" and model_id != "iforest_veto":
                try:
                    if hasattr(est, "predict_proba"):
                        base_predictions[model_id] = est.predict_proba(X)[:, 1]
                except Exception:
                    pass

        except Exception as exc:
            if logger:
                logger(f"[ensemble_v2] {model_id} train failed: {exc}")

        fitted.append((model_id, est, role, weight))

    # Train stack_meta on base model predictions
    if base_predictions and len(base_predictions) >= 3:
        meta_X = np.column_stack(list(base_predictions.values()))
        for i, (mid, est, role, w) in enumerate(fitted):
            if mid == "stack_meta":
                try:
                    est.fit(meta_X, y)
                    n_ok += 1
                    fitted[i] = (mid, est, "shadow", w)
                    if logger:
                        logger(f"[ensemble_v2] stack_meta trained on {len(base_predictions)} model outputs")
                except Exception as exc:
                    if logger:
                        logger(f"[ensemble_v2] stack_meta train failed: {exc}")
                break

    if logger:
        logger(f"[ensemble_v2] trained {n_ok}/{len(models)} models")

    return fitted, n_ok


def predict_v2_models(fitted_models, X_row, logger=None):
    """Get predictions from all V2 models for a single sample.

    Returns dict of {model_id: probability} for each model.
    IsolationForest returns anomaly score mapped to [0,1].
    """
    X_arr = np.asarray(X_row).reshape(1, -1)
    predictions = {}

    base_preds = {}
    for model_id, est, role, weight in fitted_models:
        if model_id == "stack_meta":
            continue
        try:
            if model_id == "iforest_veto":
                score = est.score_samples(X_arr)[0]
                p = 1.0 / (1.0 + np.exp(-score))
                predictions[model_id] = float(p)
            elif model_id == "tabnet":
                X_f = np.asarray(X_arr, dtype=np.float32)
                p = float(est.predict_proba(X_f)[0, 1])
                predictions[model_id] = p
            elif model_id == "ngboost":
                p = float(est.predict_proba(X_arr)[0, 1])
                predictions[model_id] = p
            else:
                p = float(est.predict_proba(X_arr)[0, 1])
                predictions[model_id] = p

            if role == "active":
                base_preds[model_id] = predictions[model_id]

        except Exception as exc:
            if logger:
                logger(f"[ensemble_v2] {model_id} predict failed: {exc}")

    # Stack meta prediction
    for model_id, est, role, weight in fitted_models:
        if model_id == "stack_meta" and base_preds and len(base_preds) >= 3:
            try:
                meta_X = np.array(list(base_preds.values())).reshape(1, -1)
                p = float(est.predict_proba(meta_X)[0, 1])
                predictions[model_id] = p
            except Exception:
                pass

    return predictions


# ── V2 Model ID list for gatling config ──────────────────────────────────────

V2_MODEL_IDS = [
    "catboost", "xgb_hist", "lgbm_goss", "hgbc",
    "tabnet", "ebm", "ngboost", "rf_bal",
    "et_depth5", "ridge_cal", "svc_rbf", "mlp",
    "iforest_veto", "lgbm_dart", "stack_meta",
]

V2_ACTIVE_IDS = [
    "catboost", "xgb_hist", "lgbm_goss", "hgbc",
    "tabnet", "ebm", "ngboost", "rf_bal",
    "et_depth5", "ridge_cal", "svc_rbf", "mlp",
    "lgbm_dart",
]

V2_VETO_IDS = ["iforest_veto"]
V2_SHADOW_IDS = ["stack_meta"]

V2_WEIGHTS = {
    "catboost": 1.5, "xgb_hist": 1.5, "lgbm_goss": 1.5, "hgbc": 1.5,
    "tabnet": 1.0, "ebm": 1.0, "ngboost": 1.0, "rf_bal": 1.0,
    "et_depth5": 0.75, "ridge_cal": 0.75, "svc_rbf": 0.75, "mlp": 0.75,
    "iforest_veto": 1.0, "lgbm_dart": 1.0, "stack_meta": 0.0,
}
