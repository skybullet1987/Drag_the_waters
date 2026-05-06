# ── shadow_diagnostics.py: shadow and diagnostic model classes ────────────────
#
# Moved from strategy.py to keep strategy.py under the QuantConnect 63KB limit.
# Re-exported from strategy.py for backward compatibility.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from infra import ROLE_SHADOW, ROLE_DIAGNOSTIC  # noqa: F401 (re-exported by strategy.py)


# ── Optional HMM import ───────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# ── Optional Tier 1 / Tier 2 shadow-lab imports ───────────────────────────────
try:
    from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

try:
    from ngboost import NGBClassifier
    HAS_NGBOOST = True
except Exception:
    HAS_NGBOOST = False

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except Exception:
    HAS_EBM = False

try:
    from catboost import CatBoostClassifier as _CatBoostCls
    HAS_CATBOOST_V2 = True
except Exception:
    HAS_CATBOOST_V2 = False

# HDBSCAN is not available in the QuantConnect runtime (compiled C extension).
# Keep permanently disabled so the optional diagnostic model does not prevent
# algorithm import.  HDBSCANRegimeDiagnostic falls back to neutral 0.5 probas.
_hdbscan_lib = None
HAS_HDBSCAN = False

HAS_FLAML = False
_FLAMLAutoML = None


# ── Buy-probability shadow models ─────────────────────────────────────────────

def _make_gbc(logger=None):
    """Build a compact GradientBoostingClassifier shadow model."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8,
            random_state=42,
        )
    except Exception as exc:
        if logger:
            logger(f"[shadow_lab] gbc init failed: {exc}")
        return None


def _make_ada(logger=None):
    """Build a compact AdaBoostClassifier shadow model."""
    try:
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(
            n_estimators=80, learning_rate=0.05,
            random_state=42,
        )
    except Exception as exc:
        if logger:
            logger(f"[shadow_lab] ada init failed: {exc}")
        return None


def _make_mlp(logger=None):
    try:
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=42)
    except Exception as exc:
        if logger: logger(f"[shadow_lab] mlp: {exc}")
        return None

def _make_balanced_rf(logger=None):
    if not HAS_IMBLEARN: return None
    try: return BalancedRandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=1, random_state=42)
    except Exception as exc:
        if logger: logger(f"[shadow_lab] bal_rf: {exc}")
        return None

def _make_rusboost(logger=None):
    if not HAS_IMBLEARN: return None
    try: return RUSBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    except Exception as exc:
        if logger: logger(f"[shadow_lab] rusboost: {exc}")
        return None

def _make_ngboost(logger=None):
    if not HAS_NGBOOST: return None
    try: return NGBClassifier(n_estimators=100, verbose=False, random_state=42)
    except Exception as exc:
        if logger: logger(f"[shadow_lab] ngboost: {exc}")
        return None

def _make_xgb_dart(logger=None):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                             booster="dart", rate_drop=0.1, eval_metric="logloss",
                             random_state=42, n_jobs=1)
    except Exception as exc:
        if logger: logger(f"[shadow_lab] xgb_dart: {exc}")
        return None


class HDBSCANRegimeDiagnostic:
    _fi = [1, 3, 5, 6, 7]

    def __init__(self, min_cluster_size=15):
        self._mcs = min_cluster_size
        self._fitted = False
        self._centroid = None

    def _x(self, X):
        X = np.atleast_2d(X); return X[:, self._fi] if X.shape[1] > max(self._fi) else X

    def fit(self, X, y_class):
        if not HAS_HDBSCAN: return
        try:
            Xr = self._x(X).astype(float)
            lbl = _hdbscan_lib.HDBSCAN(min_cluster_size=self._mcs).fit_predict(Xr)
            uk = set(lbl) - {-1}
            best = max(uk, key=lambda k: np.mean(Xr[lbl == k, 0])) if uk else None
            self._centroid = Xr[lbl == best].mean(0) if best is not None else None
            self._fitted = True
        except Exception: self._fitted = False

    def predict_proba(self, X):
        n = len(np.atleast_2d(X)); fb = np.column_stack([np.full(n, .5), np.full(n, .5)])
        if not self._fitted or self._centroid is None: return fb
        try:
            Xr = self._x(X).astype(float); d = np.linalg.norm(Xr - self._centroid, axis=1)
            p = 1.0 / (1.0 + np.exp(d - d.mean())); return np.column_stack([1.0 - p, p])
        except Exception: return fb


class FLAMLShadow:
    def __init__(self, time_budget=20):
        self._tb = time_budget
        self._m = None
        self._fitted = False

    def fit(self, X, y):
        if not HAS_FLAML: return
        try:
            m = _FLAMLAutoML()
            m.fit(np.array(X, dtype=float), np.array(y),
                  task="classification", time_budget=self._tb, verbose=0)
            self._m = m; self._fitted = True
        except Exception: self._fitted = False

    def predict_proba(self, X):
        n = len(np.atleast_2d(X)); fb = np.column_stack([np.full(n, .5), np.full(n, .5)])
        if not self._fitted or self._m is None: return fb
        try: return self._m.predict_proba(np.array(X, dtype=float))
        except Exception: return fb


# ── Regime diagnostic models ───────────────────────────────────────────────────
#
# These are sklearn wrappers that output a probability-like score or cluster
# state rather than a buy probability. They are tagged ROLE_DIAGNOSTIC and
# never affect active trading decisions.

class MarkovRegimeDiagnostic:
    """Lightweight Markov-inspired regime diagnostic."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._clf = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42,
        )
        self._fitted = False
        self._feat_idx = [1, 3, 5, 6, 7]  # ret_4, ret_16, atr_pct, vol_ratio, btc_rel

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def _make_labels(self, X, y_class):
        X2 = self._extract(X)
        ret4   = X2[:, 0]
        ret16  = X2[:, 1]
        vol_r  = X2[:, 3] if X2.shape[1] > 3 else np.zeros(len(X2))
        # 0 = uptrend, 1 = chop, 2 = downtrend
        labels = np.ones(len(X2), dtype=int)  # default chop
        labels[(ret4 > 0.005) & (ret16 > 0.010)] = 0   # uptrend
        labels[(ret4 < -0.005) & (ret16 < -0.010)] = 2  # downtrend
        return labels

    def fit(self, X, y_class):
        try:
            labels = self._make_labels(X, y_class)
            self._clf.fit(self._extract(X), labels)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        if not self._fitted:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        try:
            probs = self._clf.predict_proba(self._extract(X))
            # class 0 = uptrend → col 1 carries the uptrend probability
            n_classes = probs.shape[1]
            up_col = 0  # class 0 = uptrend
            up_prob = probs[:, up_col] if n_classes > 0 else np.full(len(probs), 0.5)
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class HMMRegimeDiagnostic:
    """Optional HMM-based regime diagnostic (requires hmmlearn).

    If hmmlearn is not installed, falls back to MarkovRegimeDiagnostic.
    Tagged ROLE_DIAGNOSTIC; outputs probability of being in the "up" state.
    """

    def __init__(self, n_components=3):
        self._n = n_components
        self._hmm = None
        self._fitted = False
        self._fallback = MarkovRegimeDiagnostic()
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        if not HMM_AVAILABLE:
            self._fallback.fit(X, y_class)
            return
        try:
            Xr = self._extract(X).astype(float)
            self._hmm = GaussianHMM(
                n_components=self._n, covariance_type="diag",
                n_iter=50, random_state=42,
            )
            self._hmm.fit(Xr)
            self._fitted = True
        except Exception:
            self._fallback.fit(X, y_class)
            self._fitted = False

    def predict_proba(self, X):
        if not (HMM_AVAILABLE and self._fitted and self._hmm is not None):
            return self._fallback.predict_proba(X)
        try:
            Xr = self._extract(X).astype(float)
            # Use log-probability of state 0 as a risk-on proxy
            log_p = self._hmm.predict_proba(Xr)
            up_col = 0
            up_prob = log_p[:, up_col] if log_p.shape[1] > 0 else np.full(len(Xr), 0.5)
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            return self._fallback.predict_proba(X)


class KMeansRegimeDiagnostic:
    """KMeans-based regime clustering diagnostic.

    Clusters the feature space into N regimes and assigns a regime label.
    The probability output is based on distance to the "risk-on" cluster
    (identified heuristically as the cluster with highest mean ret_4).
    """

    def __init__(self, n_clusters=4):
        self._n = n_clusters
        self._km = None
        self._risk_on_cluster = 0
        self._fitted = False
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        try:
            from sklearn.cluster import KMeans
            Xr = self._extract(X).astype(float)
            self._km = KMeans(
                n_clusters=self._n, random_state=42, n_init=5,
            )
            labels = self._km.fit_predict(Xr)
            # Identify "risk-on" cluster: highest mean ret_4 among clusters
            ret4 = Xr[:, 0]
            cluster_means = [
                np.mean(ret4[labels == k]) if np.any(labels == k) else 0.0
                for k in range(self._n)
            ]
            self._risk_on_cluster = int(np.argmax(cluster_means))
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        if not self._fitted or self._km is None:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        try:
            Xr = self._extract(X).astype(float)
            dists = self._km.transform(Xr)  # shape (n, n_clusters)
            # Softmax inverse-distance to risk-on cluster
            inv_d = 1.0 / (dists + 1e-6)
            inv_sum = inv_d.sum(axis=1, keepdims=True)
            probs = inv_d / inv_sum
            up_prob = probs[:, self._risk_on_cluster]
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class IsoForestRiskDiagnostic:
    """IsolationForest anomaly/risk diagnostic.

    Flags unusual/outlier market setups via anomaly score.
    Higher score (closer to 1.0) = more anomalous = higher risk.

    Output in col-1 position is the anomaly probability (re-scaled from
    IsolationForest decision_function → [0, 1]).
    """

    def __init__(self):
        self._iso = None
        self._fitted = False
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        try:
            from sklearn.ensemble import IsolationForest
            Xr = self._extract(X).astype(float)
            self._iso = IsolationForest(
                n_estimators=80, contamination=0.05, random_state=42, n_jobs=1,
            )
            self._iso.fit(Xr)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        """Return anomaly probability: col-1 = P(anomaly) in [0, 1]."""
        if not self._fitted or self._iso is None:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])
        try:
            Xr = self._extract(X).astype(float)
            # decision_function: negative = anomaly, positive = normal
            scores = self._iso.decision_function(Xr)
            # Re-scale to [0, 1]: anomaly probability increases as score goes negative.
            # Scale factor controls steepness of the sigmoid mapping.
            _ANOMALY_SIGMOID_SCALE = 10.0
            anom_prob = 1.0 / (1.0 + np.exp(_ANOMALY_SIGMOID_SCALE * scores))
            return np.column_stack([1.0 - anom_prob, anom_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])


# ── NEW MODEL FACTORIES (non-tree diversity layer) ───────────────────────────

def _make_svc_cal(logger=None):
    """Calibrated SVM with RBF kernel — different decision geometry from trees."""
    try:
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
            method="sigmoid", cv=3,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] svc_cal: {exc}")
        return None


def _make_ridge_cal(logger=None):
    """Calibrated Ridge classifier — pure linear model, max diversity vs trees."""
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
            method="sigmoid", cv=3,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] ridge_cal: {exc}")
        return None


def _make_ebm(logger=None):
    """Explainable Boosting Machine — glass-box GAM, excellent calibration."""
    if not HAS_EBM:
        return None
    try:
        return ExplainableBoostingClassifier(
            max_bins=64, interactions=0, learning_rate=0.01,
            min_samples_leaf=5, max_rounds=200, random_state=42,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] ebm: {exc}")
        return None


def _make_catboost_shallow(logger=None):
    """CatBoost depth=3 — ordered boosting prevents target leakage."""
    if not HAS_CATBOOST_V2:
        return None
    try:
        return _CatBoostCls(
            iterations=120, depth=3, learning_rate=0.05,
            auto_class_weights="Balanced", verbose=0, random_seed=44,
            l2_leaf_reg=5.0, bootstrap_type="Bernoulli", subsample=0.8,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] catboost_d3: {exc}")
        return None


def _make_lgbm_goss(logger=None):
    """LightGBM GOSS — gradient-based one-side sampling, focuses on hard samples."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            boosting_type="goss", class_weight="balanced",
            top_rate=0.2, other_rate=0.1, min_child_samples=20,
            random_state=44, verbose=-1, n_jobs=1,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] lgbm_goss: {exc}")
        return None


def _make_knn_cal(logger=None):
    """Calibrated KNN — instance-based, totally different from all tree/linear models."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(
            KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1),
            method="isotonic", cv=3,
        )
    except Exception as exc:
        if logger: logger(f"[shadow_lab] knn_cal: {exc}")
        return None


def extend_shadow_estimators(existing, max_count=20, logger=None):
    """Extend shadow list: Tier-1 (gbc/ada/mlp/bal_rf/rusboost/ngboost) +
    Tier-2 (xgb_dart/flaml) + diagnostics (markov/hmm/kmeans/isoforest/hdbscan)."""
    shadows = list(existing)

    def _try_add(model_id, factory_fn, role):
        if len(shadows) >= max_count: return
        est = factory_fn(logger)
        if est is not None:
            shadows.append((model_id, est, role))

    def _try_init(mid, cls, role):
        if len(shadows) >= max_count: return
        try: shadows.append((mid, cls(), role))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] {mid}: {exc}")

    _try_add("gbc",         _make_gbc,             ROLE_SHADOW)
    _try_add("ada",         _make_ada,             ROLE_SHADOW)
    _try_add("mlp",         _make_mlp,             ROLE_SHADOW)
    _try_add("bal_rf",      _make_balanced_rf,     ROLE_SHADOW)
    _try_add("rusboost",    _make_rusboost,        ROLE_SHADOW)
    _try_add("ngboost",     _make_ngboost,         ROLE_SHADOW)
    _try_add("xgb_dart",    _make_xgb_dart,       ROLE_SHADOW)
    # ── NEW: non-tree diversity models ───────────────────────────────────
    _try_add("svc_cal",     _make_svc_cal,         ROLE_SHADOW)
    _try_add("ridge_cal",   _make_ridge_cal,       ROLE_SHADOW)
    _try_add("ebm",         _make_ebm,             ROLE_SHADOW)
    _try_add("catboost_d3", _make_catboost_shallow, ROLE_SHADOW)
    _try_add("lgbm_goss",   _make_lgbm_goss,      ROLE_SHADOW)
    _try_add("knn_cal",     _make_knn_cal,         ROLE_SHADOW)
    if len(shadows) < max_count and HAS_FLAML:
        try: shadows.append(("flaml", FLAMLShadow(time_budget=20), ROLE_SHADOW))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] flaml: {exc}")
    _try_init("markov_regime",  MarkovRegimeDiagnostic,  ROLE_DIAGNOSTIC)
    _try_init("hmm_regime",     HMMRegimeDiagnostic,     ROLE_DIAGNOSTIC)
    _try_init("kmeans_regime",  KMeansRegimeDiagnostic,  ROLE_DIAGNOSTIC)
    _try_init("isoforest_risk", IsoForestRiskDiagnostic, ROLE_DIAGNOSTIC)
    _try_init("hdbscan_regime", HDBSCANRegimeDiagnostic, ROLE_DIAGNOSTIC)
    return shadows[:max_count]
