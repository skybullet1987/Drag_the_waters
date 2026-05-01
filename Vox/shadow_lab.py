# ── Vox Shadow Lab — Extended Shadow and Diagnostic Models ───────────────────
#
# This module provides additional shadow and diagnostic models that are loaded
# by _make_shadow_estimators in models.py.
#
# Shadow models (gbc, ada):
#   - Predicted and logged, never affect trading decisions.
#   - Provides buy-probability score for post-hoc attribution.
#
# Diagnostic/regime models (markov_regime, hmm_regime, kmeans_regime,
#                            isoforest_risk):
#   - Produce regime state / risk overlay scores.
#   - Persisted in diagnostic_scores / regime_model_state fields.
#   - Never used as direct buy probability.
#
# All models are optional and fail silently if sklearn dependencies are missing.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ── Optional HMM import ───────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

ROLE_SHADOW     = "shadow"
ROLE_DIAGNOSTIC = "diagnostic"


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


# ── Regime diagnostic models ───────────────────────────────────────────────────
#
# These are sklearn wrappers that output a probability-like score or cluster
# state rather than a buy probability. They are tagged ROLE_DIAGNOSTIC and
# never affect active trading decisions.

class MarkovRegimeDiagnostic:
    """Lightweight Markov-inspired regime diagnostic.

    Uses a multinomial logistic regression trained on regime-related features
    (ret_4, ret_16, volatility, volume_ratio) to estimate probabilities of
    being in uptrend/downtrend/chop/high_vol states.

    Outputs a pseudo-probability for the "risk_on" regime (states 0=up, 1=chop,
    2=down) rather than a direct buy probability.  Persisted as a diagnostic
    probability.
    """

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._clf = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42,
        )
        self._fitted = False
        # Feature indices used from the full feature vector:
        # [0]=ret_1, [1]=ret_4, [2]=ret_8, [3]=ret_16, [4]=rsi_14,
        # [5]=atr_pct, [6]=vol_ratio, [7]=btc_rel,  …
        self._feat_idx = [1, 3, 5, 6, 7]  # ret_4, ret_16, atr_pct, vol_ratio, btc_rel

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def _make_labels(self, X, y_class):
        """Build regime labels from features heuristically."""
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
        """Return P(uptrend) as a scalar in a 2-column array (col 1 = P(uptrend))."""
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


# ── Public API: extend existing shadow list ────────────────────────────────────

def extend_shadow_estimators(existing, max_count=12, logger=None):
    """Extend an existing shadow estimator list with gbc, ada, and regime diagnostics.

    Parameters
    ----------
    existing  : list[(id, estimator, role)]
    max_count : int — hard cap on total models returned
    logger    : callable or None

    Returns
    -------
    list[(id, estimator, role)] — extended list, capped at max_count
    """
    shadows = list(existing)

    def _try_add(model_id, factory_fn, role):
        if len(shadows) >= max_count:
            return
        est = factory_fn(logger)
        if est is not None:
            shadows.append((model_id, est, role))

    # ── Buy-probability shadow models ─────────────────────────────────────────
    _try_add("gbc", _make_gbc, ROLE_SHADOW)
    _try_add("ada", _make_ada, ROLE_SHADOW)

    # ── Regime/risk diagnostic models ─────────────────────────────────────────
    if len(shadows) < max_count:
        try:
            shadows.append(("markov_regime", MarkovRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] markov_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("hmm_regime", HMMRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] hmm_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("kmeans_regime", KMeansRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] kmeans_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("isoforest_risk", IsoForestRiskDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] isoforest_risk init failed: {exc}")

    return shadows[:max_count]
