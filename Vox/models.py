# ── Vox Models ────────────────────────────────────────────────────────────────
#
# Consolidated module for feature engineering, triple-barrier labeling, the
# voting-ensemble classifier + regression ensemble, and the walk-forward
# training pipeline.
# Previously split across features.py, labeling.py, ensemble.py, training.py.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

# ── Optional external ML models with safe import guards ──────────────────────
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

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

# Current output dimension of build_features.
# Changed from 10 (Vox v2) to 20 (Vox v4) by adding 10 trend/chop features.
# Backward-compatibility: load_state() detects mismatches and leaves the
# ensemble unfitted so the next cycle retrains on the new feature set.
FEATURE_COUNT = 20

# ── Label-specific triple-barrier parameters ──────────────────────────────────
#
# These govern what gets labelled "1" during training and are intentionally
# decoupled from the live-execution TAKE_PROFIT / STOP_LOSS / TIMEOUT_HOURS.
# Looser barriers here increase the positive rate, improving model calibration.
# Overridable at runtime via QC parameters: label_tp, label_sl, label_horizon_bars.
LABEL_TP           = 0.012   # take-profit fraction for training labels  (+1.2 %)
LABEL_SL           = 0.010   # stop-loss fraction for training labels    (−1.0 %)
LABEL_HORIZON_BARS = 72      # max bars to hold at train time (≈6h at 5-min bars)

# Estimated round-trip cost used when generating cost-aware training labels.
# Set lower than live COST_BPS to generate conservative-but-not-too-strict targets.
# Overridable by passing cost_bps to build_training_data().
LABEL_COST_BPS = 30          # basis points — 0.30 % round-trip for label generation

# Minimum samples required to fit regressors (avoids degenerate fits).
MIN_REGRESSOR_SAMPLES = 50


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
    """Build the FEATURE_COUNT-element feature vector for one decision bar.

    Parameters
    ----------
    closes     : array-like of float, length >= 17
    volumes    : array-like of float, length >= 16
    btc_closes : array-like of float, length >= 5
    hour       : int, 0–23  (UTC hour of current bar)

    Returns
    -------
    numpy.ndarray of shape (FEATURE_COUNT,) or None when insufficient history.

    Feature layout
    --------------
    0  ret_1            — 1-bar return
    1  ret_4            — 4-bar return
    2  ret_8            — 8-bar return
    3  ret_16           — 16-bar return
    4  rsi_14           — RSI(14) normalised to [0, 1]
    5  atr_n            — ATR(14) / close[-1]
    6  vol_r            — volume ratio: current / 15-bar mean (capped at 10)
    7  btc_rel          — 4-bar symbol return minus 4-bar BTC return
    8  hour_of_day      — hour normalised to [0, 1]
    9  sma_slope        — 8-bar SMA slope (non-overlapping windows), capped [-0.10, 0.10]
    10 range_eff        — 16-bar range efficiency (trend purity)
    11 sma_fast_slope   — 4-bar SMA slope (fast), capped [-0.10, 0.10]
    12 price_vs_sma_fast — price relative to 4-bar SMA
    13 price_vs_sma_slow — price relative to 8-bar SMA
    14 recent_high_breakout — distance above 16-bar prior high, capped [-0.10, 0.10]
    15 vol_zscore       — volume z-score over 16 bars, capped [-3, 3]
    16 reversal_frac    — fraction of sign changes in last 8 bars [0, 1]
    17 green_bar_ratio  — fraction of up-bars in last 8 bars [0, 1]
    18 atr_expansion    — recent ATR vs lagged ATR ratio minus 1, capped [-1, 2]
    19 btc_ret_1        — 1-bar BTC return
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

    # ── Volume ratio (capped to prevent explosion on low-liquidity spikes) ────
    prior_avg = float(np.mean(v[-16:-1]))
    vol_r     = (v[-1] / prior_avg) if prior_avg > 0 else 1.0
    vol_r     = min(vol_r, 10.0)

    # ── BTC-relative return (4-bar) ───────────────────────────────────────────
    btc_ret_4 = (bc[-1] - bc[-5]) / bc[-5] if len(bc) >= 5 and bc[-5] != 0 else 0.0
    btc_rel   = ret_4 - btc_ret_4

    # ── Short SMA slope (non-overlapping 8-bar windows) ──────────────────────
    sma_last  = float(np.mean(c[-8:]))
    sma_prev  = float(np.mean(c[-16:-8]))
    sma_slope = (sma_last - sma_prev) / sma_prev if sma_prev != 0 else 0.0
    sma_slope = float(np.clip(sma_slope, -0.10, 0.10))

    # ── Range efficiency (trend purity) ──────────────────────────────────────
    c16 = c[-17:]   # 16-bar window
    net_move_16  = abs(c16[-1] - c16[0])
    sum_moves_16 = float(np.sum(np.abs(np.diff(c16))))
    range_eff = net_move_16 / sum_moves_16 if sum_moves_16 > 0 else 0.5

    # ── Fast SMA slope (4-bar window) ─────────────────────────────────────────
    sma_fast_now  = float(np.mean(c[-4:]))
    sma_fast_prev = float(np.mean(c[-8:-4]))
    sma_fast_slope = (sma_fast_now - sma_fast_prev) / sma_fast_prev if sma_fast_prev != 0 else 0.0
    sma_fast_slope = float(np.clip(sma_fast_slope, -0.10, 0.10))

    # ── Price vs SMA-fast and SMA-slow ────────────────────────────────────────
    price_vs_sma_fast = (last - sma_fast_now) / sma_fast_now if sma_fast_now != 0 else 0.0
    price_vs_sma_slow = (last - sma_last) / sma_last if sma_last != 0 else 0.0

    # ── Recent high breakout ──────────────────────────────────────────────────
    recent_high = float(np.max(c[-17:-1]))
    recent_high_breakout = float(np.clip(
        (last / recent_high - 1.0) if recent_high > 0 else 0.0, -0.10, 0.10
    ))

    # ── Volume z-score (normalised spike detection) ───────────────────────────
    vol_mean   = float(np.mean(v[-16:]))
    vol_std    = float(np.std(v[-16:]))
    vol_zscore = float(np.clip(
        (v[-1] - vol_mean) / vol_std if vol_std > 0 else 0.0, -3.0, 3.0
    ))

    # ── Reversal count (sign changes in last 8 bars) ──────────────────────────
    bar_rets = np.diff(c[-9:])
    signs = np.sign(bar_rets)
    reversals = int(np.sum(signs[1:] != signs[:-1]))
    reversal_frac = reversals / 7.0   # normalise to [0, 1]

    # ── Green bar ratio (fraction of up-bars in last 8 bars) ──────────────────
    green_bar_ratio = float(np.sum(bar_rets > 0)) / 8.0

    # ── ATR expansion (current vs lagged) ─────────────────────────────────────
    if len(c) >= 9:
        atr_recent = float(np.mean(np.abs(np.diff(c[-5:]))))
        atr_older  = float(np.mean(np.abs(np.diff(c[-9:-5]))))
        atr_expansion = float(np.clip(
            (atr_recent / atr_older - 1.0) if atr_older > 0 else 0.0, -1.0, 2.0
        ))
    else:
        atr_expansion = 0.0

    # ── 1-bar BTC return ──────────────────────────────────────────────────────
    btc_ret_1 = (bc[-1] - bc[-2]) / bc[-2] if len(bc) >= 2 and bc[-2] != 0 else 0.0

    return np.array([
        ret_1,
        ret_4,
        ret_8,
        ret_16,
        rsi / 100.0,
        atr_n,
        vol_r,
        btc_rel,
        float(hour) / 23.0,
        sma_slope,             # feature 9
        range_eff,             # feature 10 (NEW)
        sma_fast_slope,        # feature 11 (NEW)
        price_vs_sma_fast,     # feature 12 (NEW)
        price_vs_sma_slow,     # feature 13 (NEW)
        recent_high_breakout,  # feature 14 (NEW)
        vol_zscore,            # feature 15 (NEW)
        reversal_frac,         # feature 16 (NEW)
        green_bar_ratio,       # feature 17 (NEW)
        atr_expansion,         # feature 18 (NEW)
        btc_ret_1,             # feature 19 (NEW)
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


def triple_barrier_outcome(prices, tp, sl, timeout_bars, cost_fraction=0.0):
    """
    Evaluate the triple-barrier outcome and return both a cost-aware binary
    label and the realised net return.

    Cost-aware label: ``1`` only if the TP barrier is hit **and** the net
    return after estimated costs is positive (``tp - cost_fraction > 0``).

    Parameters
    ----------
    prices        : array-like of float — price series from entry bar onward.
    tp            : float — take-profit fraction (e.g. 0.012 for +1.2 %).
    sl            : float — stop-loss fraction   (e.g. 0.010 for −1.0 %).
    timeout_bars  : int   — maximum bars to hold.
    cost_fraction : float — estimated round-trip fee/slippage fraction
                    (e.g. 0.003 for 30 bps).

    Returns
    -------
    tuple[int, float]
        ``(label, realized_net_return)``

        - ``label``              — 1 if TP hit and net profit is positive; 0 otherwise.
        - ``realized_net_return``— return net of costs at the first barrier:
            * TP hit: ``+tp − cost_fraction``
            * SL hit: ``−sl − cost_fraction``
            * Timeout: ``(final_price − entry) / entry − cost_fraction``
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return 0, -cost_fraction

    entry = prices[0]
    if entry == 0.0:
        return 0, -cost_fraction

    upper = entry * (1.0 + tp)
    lower = entry * (1.0 - sl)

    limit = min(len(prices) - 1, timeout_bars)
    for i in range(1, limit + 1):
        px = prices[i]
        if px >= upper:
            net_ret = tp - cost_fraction
            label = 1 if net_ret > 0 else 0
            return label, net_ret
        if px <= lower:
            return 0, -sl - cost_fraction

    # Vertical barrier: use actual price at the horizon
    final_px = prices[min(len(prices) - 1, timeout_bars)]
    net_ret = (final_px - entry) / entry - cost_fraction if entry > 0 else -cost_fraction
    return 0, net_ret   # timeout is always label=0


# ═══════════════════════════════════════════════════════════════════════════════
# VOTING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

# Vox v2 classifier weights (must match estimator order in _make_estimators).
# HistGradientBoosting is the strongest model; LR is the lightest baseline.
CLASSIFIER_WEIGHTS = [0.20, 0.35, 0.25, 0.20]  # lr, hgbc, et, rf

# Vox v2 regressor weights (must match order in _make_regressors).
REGRESSOR_WEIGHTS = [0.40, 0.35, 0.25]          # hgbr, etr, ridge


def _make_estimators(logger=None, use_calibration=True):
    """
    Build the list of (name, estimator) tuples for the VotingClassifier.

    Vox v2 model stack (sklearn-native, no external dependencies):
      - LogisticRegression (lr)              — linear baseline
      - HistGradientBoostingClassifier (hgbc)— strong boosted trees, no calibration needed
      - ExtraTreesClassifier (et)            — randomised trees, adds diversity
      - RandomForestClassifier (rf)          — bagged trees

    Tree models (ET, RF) are wrapped in CalibratedClassifierCV(isotonic, cv=2)
    for reliable probability estimates.  HistGradientBoosting has good built-in
    calibration and is used directly.  GaussianNB is removed (too naive for
    crypto features and dominates ensembles at low positive rates).

    Parameters
    ----------
    logger          : callable or None — for diagnostic warnings.
    use_calibration : bool — when False, return raw tree estimators without
                      CalibratedClassifierCV wrapping (halves inference cost).
    """
    def _maybe_calibrate(est):
        if use_calibration:
            return CalibratedClassifierCV(est, method="isotonic", cv=2)
        return est

    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    # HistGradientBoostingClassifier: sklearn-native boosted trees, replaces
    # LightGBM/GradientBoostingClassifier.  Well-calibrated by default;
    # skip CalibratedClassifierCV to avoid cv-split issues on small datasets.
    try:
        hgbc = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42, class_weight="balanced",
        )
    except TypeError:
        # Older sklearn may not support class_weight on HistGradientBoosting
        hgbc = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=0.1,
            random_state=42,
        )

    et = _maybe_calibrate(
        ExtraTreesClassifier(
            n_estimators=100, max_depth=5, n_jobs=1, random_state=42,
            class_weight="balanced",
        )
    )

    rf = _maybe_calibrate(
        RandomForestClassifier(
            n_estimators=100, max_depth=5, n_jobs=1, random_state=42,
            class_weight="balanced",
        )
    )

    return [
        ("lr",   lr),
        ("hgbc", hgbc),
        ("et",   et),
        ("rf",   rf),
    ]


def _make_regressors():
    """
    Build the list of (name, estimator) tuples for the regression ensemble.

    Vox v2 regression stack (predicts expected forward return net of costs):
      - HistGradientBoostingRegressor (hgbr) — strong sklearn-native model
      - ExtraTreesRegressor (etr)            — non-linear, diverse from HGBR
      - Ridge (ridge)                        — linear baseline, stable on small data

    Returns
    -------
    list[tuple[str, estimator]]
    """
    hgbr = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, l2_regularization=0.1,
        random_state=42,
    )
    etr = ExtraTreesRegressor(
        n_estimators=100, max_depth=5, n_jobs=1, random_state=42,
    )
    ridge = Ridge(alpha=1.0)

    return [
        ("hgbr",  hgbr),
        ("etr",   etr),
        ("ridge", ridge),
    ]


class VoxEnsemble:
    """
    Heterogeneous soft-voting ensemble for the Vox v2 strategy.

    Classifiers (weighted soft voting)
    -----------------------------------
    - **LogisticRegression** (weight 0.20) — linear baseline.
    - **HistGradientBoostingClassifier** (weight 0.35) — strong sklearn-native booster.
    - **ExtraTreesClassifier** (weight 0.25) — randomised trees, diverse.
    - **RandomForestClassifier** (weight 0.20) — bagged trees.

    GaussianNB is intentionally removed: at typical positive rates of 1–5 %
    it dominates the soft vote with extreme probabilities and degrades calibration.

    Regressors (weighted average of predicted return)
    -------------------------------------------------
    - **HistGradientBoostingRegressor** (weight 0.40)
    - **ExtraTreesRegressor** (weight 0.35)
    - **Ridge** (weight 0.25)

    Trained on the cost-aware realised return from ``triple_barrier_outcome``.
    Only used if ``y_return`` is provided to ``fit()``.

    Tree classifiers are wrapped in CalibratedClassifierCV(method="isotonic", cv=2)
    for reliable probability estimates.  HistGradientBoosting is well-calibrated
    by design and is not additionally wrapped.
    """

    def __init__(self, logger=None, use_calibration=True):
        self._logger          = logger
        self._use_calibration = use_calibration
        self._estimators      = _make_estimators(logger, use_calibration=use_calibration)
        self._classifier_weights = list(CLASSIFIER_WEIGHTS)
        self._model           = VotingClassifier(
            estimators=self._estimators,
            voting="soft",
            weights=self._classifier_weights,
        )
        self._fitted        = False
        self._positive_rate = 0.0   # updated on every fit(); persisted via pickle

        # Regression ensemble (trained when y_return is provided to fit())
        self._regressors        = _make_regressors()
        self._regressor_weights = list(REGRESSOR_WEIGHTS)
        self._reg_fitted        = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y_class, y_return=None):
        """
        Fit the ensemble on training data.

        Parameters
        ----------
        X        : np.ndarray of shape (n_samples, n_features)
        y_class  : np.ndarray of shape (n_samples,) — binary classification labels.
        y_return : np.ndarray or None — float regression targets (net return).
                   When provided, the regression ensemble is also trained.
        """
        # Record positive rate before fitting so it's available immediately.
        if len(y_class) > 0:
            self._positive_rate = float(np.mean(y_class == 1))

        self._model.fit(X, y_class)
        self._fitted = True
        # Store feature count so load_state can detect stale models after FEATURE_COUNT bumps.
        self._feature_count = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else FEATURE_COUNT

        # Train regression ensemble when return targets are available.
        if y_return is not None and len(y_return) >= MIN_REGRESSOR_SAMPLES:
            self._reg_fitted = False
            for name, reg in self._regressors:
                try:
                    reg.fit(X, y_return)
                except Exception as exc:
                    if self._logger:
                        self._logger(f"[ensemble] regressor {name} fit failed: {exc}")
            self._reg_fitted = True

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
        Return per-model and aggregate probability and return estimates for *X*.

        Parameters
        ----------
        X : array-like of shape (1, n_features)

        Returns
        -------
        dict
            ``class_proba``        — float in [0, 1]: weighted P(class=1) (alias: ``mean_proba``).
            ``mean_proba``         — same as ``class_proba`` (backward-compatible alias).
            ``std_proba``          — float: std-dev of per-model probabilities.
            ``n_agree``            — int: models with P(class=1) >= agree_threshold.
            ``pred_return``        — float: weighted-average predicted net return (0.0 if regressors not trained).
            ``return_dispersion``  — float: std-dev of regressor predictions (0.0 if unavailable).
            ``per_model``          — dict[str, float]: model_name -> P(class=1).

        Raises
        ------
        RuntimeError  if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")

        X_arr  = np.atleast_2d(X)

        # ── Classifier probabilities ──────────────────────────────────────────
        probas = {}
        for name, est in self._model.named_estimators_.items():
            try:
                p = float(est.predict_proba(X_arr)[0, 1])
            except Exception:
                p = 0.5
            probas[name] = p

        # Use VotingClassifier's weighted prediction for the aggregate.
        try:
            mean_p = float(self._model.predict_proba(X_arr)[0, 1])
        except Exception:
            mean_p = float(np.mean(list(probas.values())))

        std_p   = float(np.std(list(probas.values())))
        n_agree = int(sum(1 for p in probas.values() if p >= self._agree_threshold()))

        # ── Regression predictions ────────────────────────────────────────────
        pred_return       = 0.0
        return_dispersion = 0.0
        if self._reg_fitted:
            reg_preds  = []
            reg_ws     = []
            for (name, reg), w in zip(self._regressors, self._regressor_weights):
                try:
                    p = float(reg.predict(X_arr)[0])
                    reg_preds.append(p)
                    reg_ws.append(w)
                except Exception:
                    pass
            if reg_preds:
                total_w       = sum(reg_ws)
                pred_return   = sum(p * w for p, w in zip(reg_preds, reg_ws)) / total_w
                return_dispersion = float(np.std(reg_preds)) if len(reg_preds) > 1 else 0.0

        return {
            "class_proba":       mean_p,   # v2 primary key
            "mean_proba":        mean_p,   # backward-compat alias
            "std_proba":         std_p,
            "n_agree":           n_agree,
            "pred_return":       pred_return,
            "return_dispersion": return_dispersion,
            "per_model":         probas,
        }

    def predict_with_confidence_batch(self, X):
        """Vectorised version of predict_with_confidence over a batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        list[dict]
            One dict per row with keys:
            class_proba, mean_proba, std_proba, n_agree,
            pred_return, return_dispersion, per_model.
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")
        X = np.atleast_2d(X)
        N = len(X)

        # ── Feature count guard: mark unfitted if shape mismatch ──────────────
        if X.shape[1] != FEATURE_COUNT:
            if self._logger:
                self._logger(
                    f"[VoxEnsemble] Feature count mismatch: got {X.shape[1]}, "
                    f"expected {FEATURE_COUNT}. Clearing fitted state."
                )
            self._fitted = False
            raise RuntimeError(
                f"VoxEnsemble feature count mismatch: got {X.shape[1]}, "
                f"expected {FEATURE_COUNT}. Model will retrain next cycle."
            )

        # ── Classifier probabilities ──────────────────────────────────────────
        proba_per_model = {}
        for name, est in self._model.named_estimators_.items():
            try:
                proba_per_model[name] = est.predict_proba(X)[:, 1].astype(float)
            except Exception:
                proba_per_model[name] = np.full(N, 0.5, dtype=float)

        model_names = list(proba_per_model.keys())
        arr         = np.stack([proba_per_model[m] for m in model_names], axis=1)  # (N, M)
        stds        = arr.std(axis=1)
        agree_thr   = self._agree_threshold()
        agrees      = (arr >= agree_thr).sum(axis=1)

        # VotingClassifier weighted predictions
        try:
            means = self._model.predict_proba(X)[:, 1].astype(float)
        except Exception:
            means = arr.mean(axis=1)

        # ── Regression predictions (batch) ────────────────────────────────────
        reg_preds_list = []   # list of (array_of_length_N, weight)
        if self._reg_fitted:
            for (name, reg), w in zip(self._regressors, self._regressor_weights):
                try:
                    preds = reg.predict(X).astype(float)
                    reg_preds_list.append((preds, w))
                except Exception:
                    pass

        if reg_preds_list:
            total_rw = sum(w for _, w in reg_preds_list)
            pred_returns = sum(p * w for p, w in reg_preds_list) / total_rw
            if len(reg_preds_list) > 1:
                reg_arr       = np.stack([p for p, _ in reg_preds_list], axis=1)
                ret_dispersions = reg_arr.std(axis=1)
            else:
                ret_dispersions = np.zeros(N)
        else:
            pred_returns    = np.zeros(N)
            ret_dispersions = np.zeros(N)

        out = []
        for i in range(N):
            out.append({
                "class_proba":       float(means[i]),
                "mean_proba":        float(means[i]),   # backward-compat
                "std_proba":         float(stds[i]),
                "n_agree":           int(agrees[i]),
                "pred_return":       float(pred_returns[i]),
                "return_dispersion": float(ret_dispersions[i]),
                "per_model":         {m: float(proba_per_model[m][i]) for m in model_names},
            })
        return out

    # ── State management ──────────────────────────────────────────────────────

    def load_state(self, saved):
        """Copy fitted state from a previously serialised VoxEnsemble.

        If the saved model was trained on a different feature count (i.e., before
        FEATURE_COUNT was bumped from 10 → 20), the fitted state is rejected and
        the caller will trigger a retrain on the next cycle.
        """
        saved_fc = getattr(saved, "_feature_count", None)
        if saved_fc is not None and saved_fc != FEATURE_COUNT:
            if self._logger:
                self._logger(
                    f"[VoxEnsemble] Stale model (trained on {saved_fc} features, "
                    f"current FEATURE_COUNT={FEATURE_COUNT}). Discarding saved state."
                )
            return   # leave self._fitted = False so caller retrains
        self._model         = saved._model
        self._fitted        = saved._fitted
        self._positive_rate = getattr(saved, "_positive_rate", 0.0)
        # Load regression ensemble if it was saved (v2+).
        # Older pickles without regressors default to unfitted state.
        if getattr(saved, "_reg_fitted", False) and getattr(saved, "_regressors", None):
            self._regressors    = saved._regressors
            self._reg_fitted    = True

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

    @property
    def base_rate(self):
        """Positive-class base rate from the most recent training run.

        Returns the fraction of label-1 samples seen during the last fit().
        Used by the caller (main.py) to compute an adaptive confidence threshold
        that scales with the model's training base rate.
        """
        return self._positive_rate


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
    cost_bps=None,
):
    """
    Construct a labelled feature matrix from per-symbol state history.

    Labels are **cost-aware**: a sample gets label 1 only if the TP barrier
    is hit *and* the net return after estimated costs is positive.

    Return targets (y_return) are the realized net-of-costs trade return at
    each triple-barrier outcome — suitable for regression ensemble training.

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
    cost_bps              : float or None — round-trip cost in basis points for label generation;
                            defaults to LABEL_COST_BPS.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray] or (None, None, None)
        (X, y_class, y_return) arrays:
        - X        : shape (n_samples, n_features)
        - y_class  : shape (n_samples,) int — cost-aware binary labels
        - y_return : shape (n_samples,) float — realised net return at barrier
    """
    # Use dedicated label params so training targets are decoupled from execution.
    _label_tp      = label_tp           if label_tp           is not None else LABEL_TP
    _label_sl      = label_sl           if label_sl           is not None else LABEL_SL
    _label_horizon = label_horizon_bars if label_horizon_bars is not None else LABEL_HORIZON_BARS
    _cost_bps      = cost_bps           if cost_bps           is not None else LABEL_COST_BPS
    cost_fraction  = _cost_bps * 1e-4

    btc_sym = next(
        (s for s in symbols if s.value.upper().startswith("BTC")), None
    )

    X_rows, y_rows, r_rows = [], [], []

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

            label, realized_return = triple_barrier_outcome(
                prices        = closes[i : i + _label_horizon + 1],
                tp            = _label_tp,
                sl            = _label_sl,
                timeout_bars  = _label_horizon,
                cost_fraction = cost_fraction,
            )
            X_rows.append(feat)
            y_rows.append(label)
            r_rows.append(realized_return)

    if not X_rows:
        algorithm.log("[training] build_training_data: no usable rows")
        return None, None, None

    if len(X_rows) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(X_rows), MAX_TRAIN_SAMPLES, replace=False))
        X_rows = np.array(X_rows, dtype=float)[idx]
        y_rows = np.array(y_rows, dtype=int)[idx]
        r_rows = np.array(r_rows, dtype=float)[idx]
        algorithm.log(
            f"[training] Subsampled to {MAX_TRAIN_SAMPLES} rows (from larger pool)."
        )

    X        = np.array(X_rows, dtype=float)
    y_class  = np.array(y_rows, dtype=int)
    y_return = np.array(r_rows, dtype=float)
    algorithm.log(
        f"[training] Dataset: {X.shape[0]} samples, "
        f"{X.shape[1]} features, "
        f"positive_rate={float(y_class.mean()):.3f} "
        f"mean_return={float(y_return.mean()):.4f} "
        f"cost_bps={_cost_bps}"
    )
    return X, y_class, y_return


def walk_forward_train(ensemble, X, y_class, y_return=None):
    """Fit *ensemble* on the full dataset.

    Trains both the classifier ensemble and (when y_return is provided) the
    regression ensemble.

    NOTE: the prior walk-forward CV scoring loop was removed because it
    multiplied training cost by ~5× and was diagnostic-only. To re-enable
    CV scoring during research, set VOX_ENABLE_CV=True at module level.

    Parameters
    ----------
    ensemble  : VoxEnsemble
    X         : np.ndarray of shape (n_samples, n_features)
    y_class   : np.ndarray of shape (n_samples,) — binary classification labels.
    y_return  : np.ndarray or None — float regression targets.
    """
    np.random.seed(42)

    if VOX_ENABLE_CV:
        try:
            tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
            fold_scores = []
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y_class[train_idx], y_class[test_idx]
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

    if len(np.unique(y_class)) >= 2:
        ensemble.fit(X, y_class, y_return=y_return)
    return ensemble
