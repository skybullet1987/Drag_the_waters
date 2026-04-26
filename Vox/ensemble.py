# ── Vox Ensemble ──────────────────────────────────────────────────────────────
#
# Heterogeneous soft-voting ensemble with probability calibration.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Optional LightGBM — fall back to GradientBoostingClassifier if not installed
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBMClassifier   = None
    _LGBM_AVAILABLE   = False


def _make_estimators(logger=None):
    """
    Build the list of (name, estimator) tuples for the VotingClassifier.

    Tree-based models are wrapped in ``CalibratedClassifierCV`` to obtain
    reliable probability estimates (Platt scaling / isotonic regression).
    Logistic regression and GaussianNB are already probabilistic.

    Parameters
    ----------
    logger : callable or None
        If provided, called with a single string message (e.g. algorithm.log).
    """
    def _warn(msg):
        if logger:
            logger(msg)

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, C=1.0)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=200, max_depth=5, n_jobs=1, random_state=42
        ),
        method="isotonic", cv=3,
    )

    # ── LightGBM (or GradientBoosting fallback) ───────────────────────────────
    if _LGBM_AVAILABLE:
        _base_lgbm = _LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
            random_state=42,
        )
        lgbm_name = "lgbm"
    else:
        _warn(
            "[ensemble] LightGBM not available; "
            "falling back to GradientBoostingClassifier"
        )
        _base_lgbm = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        lgbm_name = "lgbm_gb_fallback"

    lgbm = CalibratedClassifierCV(
        _base_lgbm, method="isotonic", cv=3
    )

    # ── Extra Trees ───────────────────────────────────────────────────────────
    et = CalibratedClassifierCV(
        ExtraTreesClassifier(
            n_estimators=200, max_depth=5, n_jobs=1, random_state=42
        ),
        method="isotonic", cv=3,
    )

    # ── Gaussian Naive Bayes ──────────────────────────────────────────────────
    gnb = GaussianNB()

    return [
        ("lr",   lr),
        ("rf",   rf),
        (lgbm_name, lgbm),
        ("et",   et),
        ("gnb",  gnb),
    ]


class VoxEnsemble:
    """
    Heterogeneous soft-voting ensemble classifier for the Vox strategy.

    Combines five diverse models:
    - **LogisticRegression** — linear baseline, fast and interpretable.
    - **RandomForestClassifier** — bagged trees, handles non-linearity.
    - **LGBMClassifier** (GradientBoostingClassifier fallback) — boosted trees,
      high signal-to-noise on tabular data.
    - **ExtraTreesClassifier** — extremely randomised trees, low correlation
      with RF, adds diversity.
    - **GaussianNB** — probabilistic baseline, calibrated out-of-the-box.

    All tree models are wrapped in ``CalibratedClassifierCV(method="isotonic")``
    to ensure that ``predict_proba`` outputs are well-calibrated.

    Usage
    -----
    >>> ens = VoxEnsemble()
    >>> ens.fit(X_train, y_train)
    >>> result = ens.predict_with_confidence(X_new)
    >>> print(result["mean_proba"], result["n_agree"])
    """

    def __init__(self, logger=None):
        """
        Parameters
        ----------
        logger : callable or None
            Optional logging function (e.g. ``algorithm.log``).
        """
        self._logger     = logger
        self._estimators = _make_estimators(logger)
        self._model      = VotingClassifier(
            estimators=self._estimators, voting="soft"
        )
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit the ensemble on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        self._model.fit(X, y)
        self._fitted = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_with_confidence(self, X):
        """
        Return per-model and aggregate probability estimates for *X*.

        Parameters
        ----------
        X : array-like of shape (1, n_features) or (n_features,)
            Single feature vector (or batch — only first row is reported).

        Returns
        -------
        dict with keys:
            ``mean_proba``  — float in [0, 1]: average P(class=1) across models.
            ``std_proba``   — float: standard deviation of per-model probabilities.
            ``n_agree``     — int: number of models with P(class=1) >= 0.5.
            ``per_model``   — dict[str, float]: model_name -> P(class=1).

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")

        X_arr = np.atleast_2d(X)

        probas = {}
        for name, est in self._model.named_estimators_.items():
            try:
                p = float(est.predict_proba(X_arr)[0, 1])
            except Exception:
                p = 0.5   # neutral fallback on individual model failure
            probas[name] = p

        vals      = list(probas.values())
        mean_p    = float(np.mean(vals))
        std_p     = float(np.std(vals))
        n_agree   = int(sum(1 for p in vals if p >= 0.5))

        return {
            "mean_proba": mean_p,
            "std_proba":  std_p,
            "n_agree":    n_agree,
            "per_model":  probas,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self):
        """True if the ensemble has been trained at least once."""
        return self._fitted
