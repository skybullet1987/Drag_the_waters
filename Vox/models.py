# ── Vox Models ────────────────────────────────────────────────────────────────
#
# Consolidated module for feature engineering, triple-barrier labeling, the
# voting-ensemble classifier, and the walk-forward training pipeline.
# Previously split across features.py, labeling.py, ensemble.py, training.py.
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
from sklearn.model_selection import TimeSeriesSplit

# Optional LightGBM — fall back to GradientBoostingClassifier if not installed
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBMClassifier  = None
    _LGBM_AVAILABLE  = False

# ── Module-level tuning constants ─────────────────────────────────────────────

# Step size for the label-generation loop in build_training_data.
# Using stride=3 reduces dataset size ~3× with minimal coverage loss
# (entry decisions happen every 15 min; training every 5-min bar is oversampling).
TRAIN_STRIDE = 3

# When False, raw tree estimators are returned without CalibratedClassifierCV
# wrapping. Roughly halves inference cost; useful during diagnostic / iteration
# phases. Overridable at runtime via the use_calibration constructor argument.
USE_CALIBRATION = True

# Hard cap on training rows; keeps fit time bounded regardless of history length.
MAX_TRAIN_SAMPLES = 20000

# Set True for research-mode walk-forward CV scoring (adds ~5× training cost).
VOX_ENABLE_CV = False

# Number of CV folds used when VOX_ENABLE_CV is True.
CV_SPLITS = 3

# ── Label-specific triple-barrier parameters ──────────────────────────────────
#
# These govern what gets labelled "1" during training and are intentionally
# decoupled from the live-execution TAKE_PROFIT / STOP_LOSS / TIMEOUT_HOURS.
# Looser barriers here increase the positive rate, improving model calibration.
# Overridable at runtime via QC parameters: label_tp, label_sl, label_horizon_bars.
LABEL_TP           = 0.012   # take-profit fraction for training labels  (+1.2 %)
LABEL_SL           = 0.010   # stop-loss fraction for training labels    (−1.0 %)
LABEL_HORIZON_BARS = 72      # max bars to hold at train time (≈6h at 5-min bars)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(highs, lows, closes, period=14):
    """
    Compute the Average True Range (Wilder's method) over *period* bars.

    Parameters
    ----------
    highs  : array-like of float, length >= period + 1
    lows   : array-like of float, length >= period + 1
    closes : array-like of float, length >= period + 1  (index 0 = oldest)
    period : int, default 14

    Returns
    -------
    float
        ATR value, or 0.0 when there is insufficient data.
    """
    h = np.asarray(highs,  dtype=float)
    l = np.asarray(lows,   dtype=float)
    c = np.asarray(closes, dtype=float)

    if len(c) < period + 1:
        return 0.0

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1]),
        ),
    )
    atr = float(np.mean(tr[-period:]))
    return atr


def build_features(closes, volumes, btc_closes, hour):
    """
    Build the 10-element feature vector for one decision bar.

    Parameters
    ----------
    closes     : array-like of float, length >= 17
    volumes    : array-like of float, length >= 16
    btc_closes : array-like of float, length >= 5
    hour       : int, 0–23  (UTC hour of current bar)

    Returns
    -------
    numpy.ndarray of shape (10,) or None when insufficient history.

    Feature layout
    --------------
    0  ret_1     — 1-bar return
    1  ret_4     — 4-bar return
    2  ret_8     — 8-bar return
    3  ret_16    — 16-bar return
    4  rsi_14    — RSI(14) normalised to [0, 1]
    5  atr_n     — ATR(14) / close[-1]
    6  vol_r     — volume ratio: current / 15-bar mean
    7  btc_rel   — 4-bar symbol return minus 4-bar BTC return
    8  hour_of_day — hour normalised to [0, 1]
    9  (reserved — zero-padded for future use)
    """
    c  = np.asarray(closes,    dtype=float)
    v  = np.asarray(volumes,   dtype=float)
    bc = np.asarray(btc_closes, dtype=float)

    if len(c) < 17 or len(v) < 16 or len(bc) < 5:
        return None

    last = c[-1]
    if last == 0.0:
        return None

    # ── Returns at multiple horizons ──────────────────────────────────────────
    def _ret(n):
        return (c[-1] - c[-1 - n]) / c[-1 - n] if c[-1 - n] != 0 else 0.0

    ret_1  = _ret(1)
    ret_4  = _ret(4)
    ret_8  = _ret(8)
    ret_16 = _ret(16)

    # ── RSI(14) — simple (non-smoothed) ───────────────────────────────────────
    deltas = np.diff(c[-15:])
    gains  = float(np.sum(deltas[deltas > 0]))
    losses = float(np.sum(-deltas[deltas < 0]))
    avg_g  = gains  / 14.0
    avg_l  = losses / 14.0
    rsi    = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    # ── ATR(14) normalised by close ───────────────────────────────────────────
    if len(c) >= 16:
        tr_proxy = np.abs(np.diff(c[-15:]))
        atr_val  = float(np.mean(tr_proxy))
    else:
        atr_val = 0.0
    atr_n = atr_val / last if last != 0 else 0.0

    # ── Volume ratio ──────────────────────────────────────────────────────────
    prior_avg = float(np.mean(v[-16:-1]))
    vol_r     = (v[-1] / prior_avg) if prior_avg > 0 else 1.0

    # ── BTC-relative return (4-bar) ───────────────────────────────────────────
    btc_ret_4 = (bc[-1] - bc[-5]) / bc[-5] if len(bc) >= 5 and bc[-5] != 0 else 0.0
    btc_rel   = ret_4 - btc_ret_4

    return np.array([
        ret_1,
        ret_4,
        ret_8,
        ret_16,
        rsi / 100.0,          # normalise to [0, 1]
        atr_n,
        vol_r,
        btc_rel,
        float(hour) / 23.0,   # normalise to [0, 1]
        0.0,                   # reserved
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIPLE-BARRIER LABELING
# ═══════════════════════════════════════════════════════════════════════════════
#
# ALIGNMENT CONSTRAINT
# ────────────────────
# The tp, sl, and timeout_bars values used here at training time MUST exactly
# match the TP, SL, and TIMEOUT_HOURS / DECISION_INTERVAL_MIN values used in
# live/backtest execution.  Misalignment causes the model to optimise for a
# target it never sees in production.

def triple_barrier_label(prices, tp, sl, timeout_bars):
    """
    Assign a binary label to a trade starting at ``prices[0]``.

    Barriers:
      - Upper: entry × (1 + tp)        → label = 1 (win)
      - Lower: entry × (1 − sl)        → label = 0 (loss)
      - Vertical: bar index == timeout  → label = 0 (timeout)

    Parameters
    ----------
    prices       : array-like of float — price series from entry bar onward.
    tp           : float — take-profit fraction  (e.g. 0.020 for +2 %).
    sl           : float — stop-loss fraction    (e.g. 0.012 for −1.2 %).
    timeout_bars : int   — maximum bars to hold.

    Returns
    -------
    int  — 1 if TP hit first; 0 otherwise.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return 0

    entry = prices[0]
    if entry == 0.0:
        return 0

    upper = entry * (1.0 + tp)
    lower = entry * (1.0 - sl)

    limit = min(len(prices) - 1, timeout_bars)
    for i in range(1, limit + 1):
        px = prices[i]
        if px >= upper:
            return 1
        if px <= lower:
            return 0

    return 0   # vertical barrier reached


# ═══════════════════════════════════════════════════════════════════════════════
# VOTING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def _make_estimators(logger=None, use_calibration=True):
    """
    Build the list of (name, estimator) tuples for the VotingClassifier.

    Tree-based models are wrapped in CalibratedClassifierCV to obtain reliable
    probability estimates.  Logistic regression and GaussianNB are inherently
    probabilistic.

    Parameters
    ----------
    logger          : callable or None — for diagnostic warnings.
    use_calibration : bool — when False, return raw tree estimators without
                      CalibratedClassifierCV wrapping (halves inference cost).
    """
    def _warn(msg):
        if logger:
            logger(msg)

    def _maybe_calibrate(est):
        if use_calibration:
            return CalibratedClassifierCV(est, method="isotonic", cv=2)
        return est

    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    rf = _maybe_calibrate(
        RandomForestClassifier(
            n_estimators=100, max_depth=5, n_jobs=1, random_state=42,
            class_weight="balanced",
        )
    )

    if _LGBM_AVAILABLE:
        _base_lgbm = _LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8,
            deterministic=True, force_row_wise=True,
            n_jobs=1, verbose=-1, random_state=42,
            class_weight="balanced",
        )
        lgbm_name = "lgbm"
    else:
        _warn(
            "[ensemble] LightGBM not available; "
            "falling back to GradientBoostingClassifier"
        )
        _base_lgbm = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        )
        lgbm_name = "lgbm_gb_fallback"

    lgbm = _maybe_calibrate(_base_lgbm)

    et = _maybe_calibrate(
        ExtraTreesClassifier(
            n_estimators=100, max_depth=5, n_jobs=1, random_state=42,
            class_weight="balanced",
        )
    )

    gnb = GaussianNB(priors=None)

    return [
        ("lr",       lr),
        ("rf",       rf),
        (lgbm_name,  lgbm),
        ("et",       et),
        ("gnb",      gnb),
    ]


class VoxEnsemble:
    """
    Heterogeneous soft-voting ensemble classifier for the Vox strategy.

    Models
    ------
    - **LogisticRegression** — linear baseline, fast and interpretable.
    - **RandomForestClassifier** — bagged trees, handles non-linearity.
    - **LGBMClassifier** (GradientBoostingClassifier fallback) — boosted
      trees, high signal-to-noise on tabular data.
    - **ExtraTreesClassifier** — extremely randomised trees, decorrelated
      from RF, adds diversity.
    - **GaussianNB** — probabilistic baseline, calibrated out-of-the-box.

    All tree models are wrapped in CalibratedClassifierCV(method="isotonic")
    to ensure well-calibrated predict_proba outputs.
    """

    def __init__(self, logger=None, use_calibration=True):
        self._logger          = logger
        self._use_calibration = use_calibration
        self._estimators      = _make_estimators(logger, use_calibration=use_calibration)
        self._model           = VotingClassifier(
            estimators=self._estimators, voting="soft"
        )
        self._fitted        = False
        self._positive_rate = 0.0   # updated on every fit(); persisted via pickle

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the ensemble on training data (X, y)."""
        # Record positive rate before fitting so it's available immediately.
        if len(y) > 0:
            self._positive_rate = float(np.mean(y == 1))

        self._model.fit(X, y)

        # GaussianNB does not support class_weight; re-fit it with per-sample
        # weights derived from class frequencies so it isn't dominated by the
        # majority class (positive_rate ≈ 1–5 % produces extreme imbalance).
        if "gnb" in self._model.named_estimators_:
            classes, counts = np.unique(y, return_counts=True)
            sw = np.ones(len(y), dtype=float)
            if len(classes) == 2:
                n = float(len(y))
                for cls, cnt in zip(classes, counts):
                    sw[y == cls] = n / (2.0 * float(cnt))
            self._model.named_estimators_["gnb"].fit(X, y, sample_weight=sw)

        self._fitted = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def _agree_threshold(self):
        """Base-rate-aware per-model probability threshold for agreement counting.

        With a positive_rate of ~1–5 % the calibrated models rarely exceed 0.5,
        so a hard 0.5 gate would reject everything.  This scales the threshold
        proportionally to the positive rate and clips to [0.15, 0.55].
        """
        return float(np.clip(2.0 * self._positive_rate, 0.15, 0.55))

    def predict_with_confidence(self, X):
        """
        Return per-model and aggregate probability estimates for *X*.

        Parameters
        ----------
        X : array-like of shape (1, n_features)

        Returns
        -------
        dict
            ``mean_proba``  — float in [0, 1]: average P(class=1).
            ``std_proba``   — float: standard deviation across models.
            ``n_agree``     — int: models with P(class=1) >= 0.5.
            ``per_model``   — dict[str, float]: model_name -> P(class=1).

        Raises
        ------
        RuntimeError  if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")

        X_arr  = np.atleast_2d(X)
        probas = {}
        for name, est in self._model.named_estimators_.items():
            try:
                p = float(est.predict_proba(X_arr)[0, 1])
            except Exception:
                p = 0.5
            probas[name] = p

        vals    = list(probas.values())
        mean_p  = float(np.mean(vals))
        std_p   = float(np.std(vals))
        n_agree = int(sum(1 for p in vals if p >= self._agree_threshold()))

        return {
            "mean_proba": mean_p,
            "std_proba":  std_p,
            "n_agree":    n_agree,
            "per_model":  probas,
        }

    def predict_with_confidence_batch(self, X):
        """Vectorised version of predict_with_confidence over a batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        list[dict]
            One dict per row with keys: mean_proba, std_proba, n_agree, per_model.
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")
        X = np.atleast_2d(X)
        proba_per_model = {}
        for name, est in self._model.named_estimators_.items():
            try:
                proba_per_model[name] = est.predict_proba(X)[:, 1].astype(float)
            except Exception:
                proba_per_model[name] = np.full(len(X), 0.5, dtype=float)
        model_names = list(proba_per_model.keys())
        arr    = np.stack([proba_per_model[m] for m in model_names], axis=1)  # (N, M)
        means  = arr.mean(axis=1)
        stds   = arr.std(axis=1)
        agree_thr = self._agree_threshold()
        agrees = (arr >= agree_thr).sum(axis=1)
        out = []
        for i in range(len(X)):
            out.append({
                "mean_proba": float(means[i]),
                "std_proba":  float(stds[i]),
                "n_agree":    int(agrees[i]),
                "per_model":  {m: float(proba_per_model[m][i]) for m in model_names},
            })
        return out

    # ── State management ──────────────────────────────────────────────────────

    def load_state(self, saved):
        """Copy fitted state from a previously serialised VoxEnsemble."""
        self._model         = saved._model
        self._fitted        = saved._fitted
        self._positive_rate = getattr(saved, "_positive_rate", 0.0)

    def set_logger(self, logger):
        """Attach (or replace) the logger callable. Not persisted."""
        self._logger = logger

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_logger"] = None     # CLR MethodBinding is not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._logger = None         # caller can reattach via set_logger()

    @property
    def is_fitted(self):
        """True if the ensemble has been trained at least once."""
        return self._fitted


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_data(
    algorithm,
    symbols,
    state_dict,
    tp,
    sl,
    timeout_bars,
    decision_interval_min,
    label_tp=None,
    label_sl=None,
    label_horizon_bars=None,
):
    """
    Construct a labelled feature matrix from per-symbol state history.

    Parameters
    ----------
    algorithm             : QCAlgorithm — used for logging.
    symbols               : list[Symbol]
    state_dict            : dict[Symbol, dict]
        Per-symbol deques keyed by "closes", "highs", "lows", "volumes".
    tp                    : float — execution take-profit (not used for labeling).
    sl                    : float — execution stop-loss  (not used for labeling).
    timeout_bars          : int   — execution timeout bars (kept for backward compat).
    decision_interval_min : int   — bar cadence in minutes (for logging).
    label_tp              : float or None — TP fraction for labels; defaults to LABEL_TP.
    label_sl              : float or None — SL fraction for labels; defaults to LABEL_SL.
    label_horizon_bars    : int   or None — timeout bars for labels; defaults to LABEL_HORIZON_BARS.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or (None, None)
        (X, y) arrays of shape (n_samples, n_features) and (n_samples,).
    """
    # Use dedicated label params so training targets are decoupled from execution.
    _label_tp      = label_tp           if label_tp           is not None else LABEL_TP
    _label_sl      = label_sl           if label_sl           is not None else LABEL_SL
    _label_horizon = label_horizon_bars if label_horizon_bars is not None else LABEL_HORIZON_BARS

    btc_sym = next(
        (s for s in symbols if s.value.upper().startswith("BTC")), None
    )

    X_rows, y_rows = [], []

    for sym in symbols:
        st = state_dict.get(sym)
        if st is None:
            continue

        closes  = list(st.get("closes",  []))
        volumes = list(st.get("volumes", []))
        btc_closes = (
            list(state_dict[btc_sym].get("closes", []))
            if btc_sym and btc_sym in state_dict else []
        )

        n = len(closes)
        min_len = 17 + _label_horizon
        if n < min_len:
            continue

        WINDOW     = 17                   # build_features needs the last 17 bars
        BTC_WINDOW = max(WINDOW, 5)       # btc_closes only needs last 5

        for i in range(WINDOW, n - _label_horizon, TRAIN_STRIDE):
            feat = build_features(
                closes     = closes[i - WINDOW + 1 : i + 1],
                volumes    = volumes[i - WINDOW + 1 : i + 1],
                btc_closes = (btc_closes[i - BTC_WINDOW + 1 : i + 1] if btc_closes else []),
                hour       = 0,   # hour unknown from deque; neutral value
            )
            if feat is None:
                continue

            label = triple_barrier_label(
                prices       = closes[i : i + _label_horizon + 1],
                tp           = _label_tp,
                sl           = _label_sl,
                timeout_bars = _label_horizon,
            )
            X_rows.append(feat)
            y_rows.append(label)

    if not X_rows:
        algorithm.log("[training] build_training_data: no usable rows")
        return None, None

    if len(X_rows) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(X_rows), MAX_TRAIN_SAMPLES, replace=False))
        X_rows = np.array(X_rows, dtype=float)[idx]
        y_rows = np.array(y_rows, dtype=int)[idx]
        algorithm.log(
            f"[training] Subsampled to {MAX_TRAIN_SAMPLES} rows (from larger pool)."
        )

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    algorithm.log(
        f"[training] Dataset: {X.shape[0]} samples, "
        f"{X.shape[1]} features, "
        f"positive_rate={float(y.mean()):.3f}"
    )
    return X, y


def walk_forward_train(ensemble, X, y):
    """Fit *ensemble* on the full dataset.

    NOTE: the prior walk-forward CV scoring loop was removed because it
    multiplied training cost by ~5× and was diagnostic-only. To re-enable
    CV scoring during research, set VOX_ENABLE_CV=True at module level.
    """
    np.random.seed(42)

    if VOX_ENABLE_CV:
        try:
            tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
            fold_scores = []
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                if len(np.unique(y_tr)) < 2:
                    continue
                ensemble.fit(X_tr, y_tr)
                proba = ensemble._model.predict_proba(X_te)[:, 1]
                preds = (proba >= 0.5).astype(int)
                fold_scores.append(float(np.mean(preds == y_te)))
            if fold_scores and hasattr(ensemble, "_logger") and ensemble._logger:
                ensemble._logger(
                    f"[training] CV scores: "
                    f"{[round(s, 3) for s in fold_scores]} "
                    f"mean={float(np.mean(fold_scores)):.3f}"
                )
        except Exception as exc:
            if hasattr(ensemble, "_logger") and ensemble._logger:
                ensemble._logger(f"[training] CV scoring failed: {exc}")

    if len(np.unique(y)) >= 2:
        ensemble.fit(X, y)
    return ensemble
