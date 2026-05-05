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
from infra import compute_vote_score as _compute_vote_score

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

# ── Version constants ──────────────────────────────────────────────────────────
#
# Bump MODEL_VERSION when the ensemble architecture changes (new estimators,
# calibration changes, etc.) — persisted pickles with older versions are discarded.
# Bump FEATURE_VERSION when build_features() output changes (FEATURE_COUNT or
# feature semantics) — old pickles with a different FEATURE_COUNT are already
# discarded by load_state(); this version adds an explicit semantic check.
# Bump LABEL_VERSION when triple_barrier_outcome() semantics change materially.
# Bump CONFIG_VERSION when risk-profile constants change in a breaking way.
MODEL_VERSION   = "v4.0"
FEATURE_VERSION = "v4.0"   # corresponds to FEATURE_COUNT=20 (Vox v4 feature set)
LABEL_VERSION   = "v2.0"   # triple-barrier with cost_fraction
CONFIG_VERSION  = "v1.0"

# ── Label-specific triple-barrier parameters ──────────────────────────────────
#
# These govern what gets labelled "1" during training and are intentionally
# decoupled from the live-execution TAKE_PROFIT / STOP_LOSS / TIMEOUT_HOURS.
# Looser barriers here increase the positive rate, improving model calibration.
# Overridable at runtime via QC parameters: label_tp, label_sl, label_horizon_bars.
LABEL_TP           = 0.030   # take-profit fraction for training labels  (+3.0 %)
LABEL_SL           = 0.015   # stop-loss fraction for training labels    (−1.5 %)
LABEL_HORIZON_BARS = 48      # max bars to hold at train time (≈4h at 5-min bars)

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
# LR is diagnostic-only (always-bearish bias on crypto; zero active weight).
# HGBC is the strongest model; ET and RF provide diversity.
# NOTE: VotingClassifier still trains LR (for diagnostic_votes output), but
# its zero weight means it does not influence class_proba or active voting.
CLASSIFIER_WEIGHTS = [0.00, 0.45, 0.30, 0.25]  # lr, hgbc, et, rf

# Vox v2 regressor weights (must match order in _make_regressors).
REGRESSOR_WEIGHTS = [0.40, 0.35, 0.25]          # hgbr, etr, ridge


def _make_estimators(logger=None, use_calibration=True):
    """Build (name, estimator) tuples for the base VotingClassifier."""
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


def _make_shadow_estimators(use_calibration=True, max_count=12, logger=None):
    """Build (id, estimator, role) tuples for shadow-only models (never affect trading)."""
    from infra import ROLE_SHADOW, ROLE_DIAGNOSTIC
    shadows = []

    def _cal(est):
        if use_calibration:
            return CalibratedClassifierCV(est, method="isotonic", cv=2)
        return est

    # ── et_shallow (faster, less overfit) ───────────────────────────────────
    try:
        shadows.append(("et_shallow", _cal(
            ExtraTreesClassifier(
                n_estimators=80, max_depth=3, n_jobs=1, random_state=43,
                class_weight="balanced",
            )
        ), ROLE_SHADOW))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] et_shallow init failed: {exc}")

    # ── rf_shallow ───────────────────────────────────────────────────────────
    try:
        shadows.append(("rf_shallow", _cal(
            RandomForestClassifier(
                n_estimators=80, max_depth=3, n_jobs=1, random_state=43,
                class_weight="balanced",
            )
        ), ROLE_SHADOW))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] rf_shallow init failed: {exc}")

    # ── HGBC with stronger L2 regularization ─────────────────────────────────
    try:
        _hgbc_l2 = None
        try:
            _hgbc_l2 = HistGradientBoostingClassifier(
                max_iter=80, learning_rate=0.05, max_depth=4,
                min_samples_leaf=25, l2_regularization=0.5,
                random_state=43, class_weight="balanced",
            )
        except TypeError:
            _hgbc_l2 = HistGradientBoostingClassifier(
                max_iter=80, learning_rate=0.05, max_depth=4,
                min_samples_leaf=25, l2_regularization=0.5,
                random_state=43,
            )
        shadows.append(("hgbc_l2", _hgbc_l2, ROLE_SHADOW))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] hgbc_l2 init failed: {exc}")

    # ── cal_et ───────────────────────────────────────────────────────────────
    try:
        shadows.append(("cal_et", CalibratedClassifierCV(
            ExtraTreesClassifier(
                n_estimators=100, max_depth=5, n_jobs=1, random_state=44,
                class_weight="balanced",
            ), method="isotonic", cv=3
        ), ROLE_SHADOW))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] cal_et init failed: {exc}")

    # ── cal_rf ───────────────────────────────────────────────────────────────
    try:
        shadows.append(("cal_rf", CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=100, max_depth=5, n_jobs=1, random_state=44,
                class_weight="balanced",
            ), method="isotonic", cv=3
        ), ROLE_SHADOW))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] cal_rf init failed: {exc}")

    # ── Balanced LR variant (diagnostic: linear, good calibration reference) ──
    try:
        shadows.append(("lr_bal", LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced",
        ), ROLE_DIAGNOSTIC))
    except Exception as exc:
        if logger: logger(f"[shadow_lab] lr_bal init failed: {exc}")

    # ── Optional external models (graceful fallback if not installed) ─────────
    if len(shadows) < max_count and HAS_LGBM:
        try:
            shadows.append(("lgbm_bal", LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                class_weight="balanced", random_state=42, verbose=-1,
                n_jobs=1,
            ), ROLE_SHADOW))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] lgbm_bal init failed: {exc}")

    if len(shadows) < max_count and HAS_XGB:
        try:
            shadows.append(("xgb_bal", XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=1,
            ), ROLE_SHADOW))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] xgb_bal init failed: {exc}")


    if len(shadows) < max_count and HAS_CATBOOST:
        try:
            shadows.append(("catboost_bal", CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                auto_class_weights="Balanced", verbose=0, random_seed=42,
            ), ROLE_SHADOW))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] catboost_bal init failed: {exc}")

    if len(shadows) < max_count and HAS_LGBM:
        try:
            shadows.append(("lgbm_dart", LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                boosting_type="dart", class_weight="balanced",
                random_state=43, verbose=-1, n_jobs=1,
            ), ROLE_SHADOW))
        except Exception as exc:
            if logger: logger(f"[shadow_lab] lgbm_dart init failed: {exc}")

    try:
        from strategy import extend_shadow_estimators as _ext
        shadows = _ext(shadows, max_count=max_count, logger=logger)
    except Exception:
        pass
    return shadows[:max_count]


# Default roles: lr=diagnostic (always-bearish), gnb=diagnostic (always-bullish)
_DEFAULT_CORE_ROLES = {
    "lr":   "diagnostic",
    "hgbc": "active",
    "et":   "active",
    "rf":   "active",
    "gnb":  "diagnostic",  # always-bullish; not in VotingClassifier but tracked
}


class VoxEnsemble:
    """Heterogeneous soft-voting ensemble for the Vox v2 strategy.

    Classifiers (weighted soft voting): LR(0.00), HGBC(0.45), ET(0.30), RF(0.25).
    LR weight is 0.00: it is diagnostic-only (always-bearish bias on crypto data).
    Regressors (weighted avg predicted return): HGBC_R(0.40), ETR(0.35), Ridge(0.25).
    GaussianNB excluded: always-bullish on crypto data; degrades calibration.
    Tree classifiers wrapped in CalibratedClassifierCV(isotonic, cv=2).

    Shadow Model Lab: when shadow_lab_enabled=True, extra models are trained and
    predicted but never affect trading. See predict_with_confidence() for output.
    Model roles: active | shadow | diagnostic | disabled.
    """

    def __init__(self, logger=None, use_calibration=True,
                 shadow_lab_enabled=True, shadow_max_count=20):
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

        # Optional per-model user weights for weighted mean computation.
        # Default: empty dict → unweighted mean (same as current behavior).
        # Set via set_model_weights().  Keys are model IDs ("lr", "hgbc", etc.)
        self._user_model_weights = {}

        # ── Model roles ───────────────────────────────────────────────────────
        # Maps model_id -> role string (active/shadow/diagnostic/disabled).
        # Governs which models count for active_mean/active_n_agree.
        # Populated from config via set_model_roles(); defaults applied otherwise.
        self._model_roles = dict(_DEFAULT_CORE_ROLES)

        # ── Shadow model lab ──────────────────────────────────────────────────
        # List of (id, estimator, role) for shadow/diagnostic-only models.
        # These are trained and predicted but never affect trading.
        self._shadow_lab_enabled  = shadow_lab_enabled
        self._shadow_max_count    = int(shadow_max_count)
        self._shadow_models       = []   # list of (id, estimator, role)
        self._shadow_fitted       = False
        self._v2_models           = None  # set by load_v2_ensemble()
        if shadow_lab_enabled:
            try:
                self._shadow_models = _make_shadow_estimators(
                    use_calibration=use_calibration,
                    max_count=shadow_max_count,
                    logger=logger,
                )
            except Exception as exc:
                if logger:
                    logger(f"[shadow_lab] init failed: {exc}")
                self._shadow_models = []

    def load_v2_ensemble(self):
        """Replace shadow models with V2 cutting-edge ensemble."""
        try:
            from ensemble_v2 import make_v2_estimators
            v2 = make_v2_estimators(logger=self._logger)
            from infra import ROLE_SHADOW
            self._shadow_models = []
            self._v2_models = v2
            for mid, est, role, weight in v2:
                self._shadow_models.append((mid, est, ROLE_SHADOW))
                self._model_roles[mid] = "active" if role == "active" else (
                    "diagnostic" if role == "veto" else "shadow")
            if self._logger:
                ids = [m[0] for m in v2]
                self._logger(f"[ensemble_v2] loaded {len(v2)} V2 models: {ids}")
                self._logger(f"[ensemble_v2] roles: { {m[0]: self._model_roles.get(m[0],'?') for m in v2} }")
        except Exception as exc:
            if self._logger:
                import traceback
                self._logger(f"[ensemble_v2] load FAILED: {exc}")
                self._logger(f"[ensemble_v2] traceback: {traceback.format_exc()[-300:]}")

    def set_model_weights(self, weights_dict):
        """Set optional per-model weights for the weighted mean computation.

        Only affects the ``weighted_mean`` field in the confidence output.
        The VotingClassifier's internal weights (CLASSIFIER_WEIGHTS) are not
        changed; the ``class_proba`` field remains the VotingClassifier result.

        Parameters
        ----------
        weights_dict : dict[str, float] — model_id -> weight.
                       A weight of 0.0 zeroes out that model's contribution.
                       If all weights are 0 or dict is empty, falls back to
                       unweighted mean.

        Example
        -------
        ensemble.set_model_weights({"lr": 0.5, "hgbc": 1.5, "et": 1.0, "rf": 1.0})
        """
        self._user_model_weights = dict(weights_dict) if weights_dict else {}

    def set_model_roles(self, roles_dict):
        """Set per-model roles controlling active/shadow/diagnostic classification.

        Parameters
        ----------
        roles_dict : dict[str, str] — model_id -> role string
                     Valid roles: "active", "shadow", "diagnostic", "disabled"

        Active models contribute to active_mean/active_std/active_n_agree and
        thus affect trading confidence.  Shadow/diagnostic models are predicted
        and logged but never affect trading decisions.

        Example
        -------
        ensemble.set_model_roles({
            "lr": "diagnostic", "hgbc": "active",
            "et": "active", "rf": "active",
        })
        """
        if roles_dict:
            self._model_roles.update(roles_dict)

    @property
    def model_ids(self):
        """Return list of active model IDs from the VotingClassifier estimators."""
        return [name for name, _ in self._estimators]

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
        # Store feature count and version stamps so load_state can detect stale
        # models after FEATURE_COUNT bumps or major architecture changes.
        self._feature_count   = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else FEATURE_COUNT
        self._model_version   = MODEL_VERSION
        self._feature_version = FEATURE_VERSION
        self._label_version   = LABEL_VERSION

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

        # ── Shadow model lab training ─────────────────────────────────────────
        if self._shadow_models:
            self._shadow_fitted = False
            n_ok = 0
            for name, est, _role in self._shadow_models:
                try:
                    if name == "iforest_veto":
                        est.fit(X)
                    elif name == "tabnet":
                        import numpy as _np
                        est.fit(_np.asarray(X, dtype=_np.float32),
                                _np.asarray(y_class, dtype=_np.int64),
                                eval_set=[(_np.asarray(X, dtype=_np.float32),
                                           _np.asarray(y_class, dtype=_np.int64))],
                                max_epochs=50, patience=10, batch_size=256)
                    elif name == "stack_meta" and self._v2_models:
                        pass  # trained separately after other models
                    else:
                        est.fit(X, y_class)
                    n_ok += 1
                except Exception as exc:
                    if self._logger:
                        self._logger(f"[shadow_lab] {name} fit failed: {exc}")
            # Train stack_meta on base model predictions
            if self._v2_models:
                _base_preds = {}
                for name, est, _role in self._shadow_models:
                    if name == "stack_meta" or name == "iforest_veto":
                        continue
                    try:
                        if name == "tabnet":
                            import numpy as _np
                            _base_preds[name] = est.predict_proba(
                                _np.asarray(X, dtype=_np.float32))[:, 1]
                        elif hasattr(est, "predict_proba"):
                            _base_preds[name] = est.predict_proba(X)[:, 1]
                    except Exception:
                        pass
                if len(_base_preds) >= 3:
                    _meta_X = np.column_stack(list(_base_preds.values()))
                    for name, est, _role in self._shadow_models:
                        if name == "stack_meta":
                            try:
                                est.fit(_meta_X, y_class)
                                n_ok += 1
                                if self._logger:
                                    self._logger(f"[ensemble_v2] stack_meta trained on {len(_base_preds)} models")
                            except Exception as exc:
                                if self._logger:
                                    self._logger(f"[ensemble_v2] stack_meta failed: {exc}")
                            break
            self._shadow_fitted = n_ok > 0
            if self._logger:
                self._logger(
                    f"[shadow_lab] trained {n_ok}/{len(self._shadow_models)} models"
                )

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
            ``class_proba``        — float: active-role mean P(class=1) (backward-compat).
            ``mean_proba``         — same as ``class_proba``.
            ``std_proba``          — float: std-dev of active-role probabilities.
            ``n_agree``            — int: active-role models with P >= agree_threshold.
            ``active_mean``        — float: mean of active-role models.
            ``active_std``         — float: std-dev of active-role models.
            ``active_n_agree``     — int: active-role models passing agree threshold.
            ``active_votes``       — dict[str, float]: active-role model probas.
            ``shadow_votes``       — dict[str, float]: shadow-role model probas.
            ``diagnostic_votes``   — dict[str, float]: diagnostic-role model probas.
            ``excluded_models``    — dict[str, str]: non-active model_id -> reason.
            ``weighted_mean``      — float: user-weight-adjusted active mean.
            ``vote_threshold``     — float: threshold used for agreement counting.
            ``pred_return``        — float: weighted-average predicted net return.
            ``return_dispersion``  — float: std-dev of regressor predictions.
            ``per_model``          — dict[str, float]: all models (active+shadow+diag).
            ``votes``              — same as ``per_model`` (alias for clarity).

        Raises
        ------
        RuntimeError  if called before fit().
        """
        if not self._fitted:
            raise RuntimeError("VoxEnsemble.fit() must be called before inference.")

        X_arr  = np.atleast_2d(X)
        roles  = self._model_roles

        # ── Classifier probabilities (VotingClassifier models) ────────────────
        probas = {}
        for name, est in self._model.named_estimators_.items():
            try:
                p = float(est.predict_proba(X_arr)[0, 1])
            except Exception:
                p = 0.5
            probas[name] = p

        # ── Shadow / diagnostic model probabilities ───────────────────────────
        shadow_probas = {}
        if self._shadow_fitted:
            for name, est, role in self._shadow_models:
                try:
                    if name == "iforest_veto":
                        sc = est.score_samples(X_arr)[0]
                        p = float(1.0 / (1.0 + np.exp(-sc)))
                    elif name == "tabnet":
                        p = float(est.predict_proba(
                            np.asarray(X_arr, dtype=np.float32))[0, 1])
                    elif name == "stack_meta" and self._v2_models:
                        continue  # handled after other predictions
                    else:
                        p = float(est.predict_proba(X_arr)[0, 1])
                except Exception:
                    p = 0.5
                shadow_probas[name] = (p, role)
            # stack_meta uses other model predictions as input
            if self._v2_models:
                _bp = {k: v for k, (v, r) in shadow_probas.items()
                       if k not in ("iforest_veto", "stack_meta")}
                if len(_bp) >= 3:
                    for name, est, role in self._shadow_models:
                        if name == "stack_meta":
                            try:
                                _mX = np.array(list(_bp.values())).reshape(1, -1)
                                p = float(est.predict_proba(_mX)[0, 1])
                                shadow_probas[name] = (p, role)
                            except Exception:
                                pass
                            break

        # ── Role-split per-model probas ───────────────────────────────────────
        active_votes     = {}
        diagnostic_votes = {}
        excluded_models  = {}

        for mid, p in probas.items():
            role = roles.get(mid, "active")
            if role == "active":
                active_votes[mid] = p
            elif role == "diagnostic":
                diagnostic_votes[mid] = p
                excluded_models[mid]  = "diagnostic_only"
            # shadow / disabled: should not appear in VotingClassifier probas

        shadow_votes = {}
        for mid, (p, role) in shadow_probas.items():
            if role == "shadow":
                shadow_votes[mid] = p
            elif role == "diagnostic":
                diagnostic_votes[mid] = p
                excluded_models[mid]  = "diagnostic_only"

        # ── Active-only statistics ────────────────────────────────────────────
        agree_thr = self._agree_threshold()
        if active_votes:
            active_vals  = list(active_votes.values())
            active_mean  = float(np.mean(active_vals))
            active_std   = float(np.std(active_vals))
            active_nagree = int(sum(1 for v in active_vals if v >= agree_thr))
        else:
            # Fallback: no active models — use only non-diagnostic VotingClassifier
            # models (shadow/unclassified) to avoid including degenerate diagnostics.
            fallback_vals = [
                p for mid, p in probas.items()
                if roles.get(mid, "active") not in ("diagnostic", "disabled")
            ] or list(probas.values()) or [0.5]
            active_mean  = float(np.mean(fallback_vals))
            active_std   = float(np.std(fallback_vals))
            active_nagree = int(sum(1 for v in fallback_vals if v >= agree_thr))

        # Optional user-configured weighted mean (over active models only)
        uw = self._user_model_weights
        if uw and active_votes:
            total_w = sum(uw.get(m, 1.0) for m in active_votes)
            if total_w > 0:
                weighted_mean = sum(uw.get(m, 1.0) * p for m, p in active_votes.items()) / total_w
            else:
                weighted_mean = active_mean
        else:
            weighted_mean = active_mean

        # ── Shadow aggregate statistics ───────────────────────────────────────
        if shadow_votes:
            sv = list(shadow_votes.values())
            shadow_mean = float(np.mean(sv))
            shadow_std  = float(np.std(sv))
        else:
            shadow_mean = 0.0
            shadow_std  = 0.0

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

        # ── All-model dict (for logging / backward compat) ────────────────────
        all_probas = dict(probas)
        all_probas.update({mid: p for mid, (p, _role) in shadow_probas.items()})

        _vs = _compute_vote_score(active_votes)
        return {
            # Active-only values — used for trading (backward-compat fields map here)
            "class_proba":       active_mean,    # backward compat → active mean
            "mean_proba":        active_mean,    # backward compat alias
            "std_proba":         active_std,     # backward compat → active std
            "n_agree":           active_nagree,  # backward compat → active n_agree
            # Explicit active role fields
            "active_mean":       active_mean,
            "active_std":        active_std,
            "active_n_agree":    active_nagree,
            "active_votes":      active_votes,
            # Profit-voting score fields
            "active_model_count": _vs["active_model_count"],
            "vote_yes_fraction":  _vs["vote_yes_fraction"],
            "top3_mean":          _vs["top3_mean"],
            "vote_score":         _vs["vote_score"],
            # Shadow/diagnostic role fields (never affect trading)
            "shadow_mean":       shadow_mean,
            "shadow_std":        shadow_std,
            "shadow_votes":      shadow_votes,
            "diagnostic_votes":  diagnostic_votes,
            "excluded_models":   excluded_models,
            # Weighted mean (over active models, user-weight-adjusted)
            "weighted_mean":     weighted_mean,
            "vote_threshold":    agree_thr,
            "pred_return":       pred_return,
            "return_dispersion": return_dispersion,
            # All-model dicts (for logging / trade journal)
            "per_model":         all_probas,
            "votes":             all_probas,    # alias for clarity in journal/logs
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

        # ── Shadow model probabilities (batch) ────────────────────────────────
        shadow_proba_per_model = {}  # id -> (array, role)
        if self._shadow_fitted:
            for name, est, role in self._shadow_models:
                try:
                    shadow_proba_per_model[name] = (
                        est.predict_proba(X)[:, 1].astype(float), role
                    )
                except Exception:
                    shadow_proba_per_model[name] = (np.full(N, 0.5, dtype=float), role)

        roles     = self._model_roles
        agree_thr = self._agree_threshold()

        # VotingClassifier weighted predictions (kept for internal reference)
        try:
            means = self._model.predict_proba(X)[:, 1].astype(float)
        except Exception:
            all_names = list(proba_per_model.keys())
            means = np.stack([proba_per_model[m] for m in all_names], axis=1).mean(axis=1)

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
            per_model_i = {m: float(proba_per_model[m][i]) for m in proba_per_model}

            # Role-split
            active_votes_i     = {}
            diagnostic_votes_i = {}
            excluded_models_i  = {}
            for mid, p in per_model_i.items():
                role = roles.get(mid, "active")
                if role == "active":
                    active_votes_i[mid] = p
                elif role == "diagnostic":
                    diagnostic_votes_i[mid] = p
                    excluded_models_i[mid]  = "diagnostic_only"

            shadow_votes_i = {}
            for mid, (arr_p, role) in shadow_proba_per_model.items():
                p = float(arr_p[i])
                if role == "shadow":
                    shadow_votes_i[mid] = p
                elif role == "diagnostic":
                    diagnostic_votes_i[mid] = p
                    excluded_models_i[mid]  = "diagnostic_only"
                per_model_i[mid] = p

            # Active statistics
            if active_votes_i:
                av = list(active_votes_i.values())
                am  = float(np.mean(av))
                ast = float(np.std(av))
                ana = int(sum(1 for v in av if v >= agree_thr))
            else:
                # Fallback: exclude diagnostics from the fallback pool too
                fb = [
                    p for mid, p in per_model_i.items()
                    if roles.get(mid, "active") not in ("diagnostic", "disabled")
                ] or list(per_model_i.values()) or [0.5]
                am  = float(np.mean(fb))
                ast = float(np.std(fb))
                ana = int(sum(1 for v in fb if v >= agree_thr))

            _vs_i = _compute_vote_score(active_votes_i)

            # Shadow statistics
            if shadow_votes_i:
                sv    = list(shadow_votes_i.values())
                sm    = float(np.mean(sv))
                sstd  = float(np.std(sv))
            else:
                sm    = 0.0
                sstd  = 0.0

            # Optional user-configured weighted mean (over active models only)
            uw = self._user_model_weights
            if uw and active_votes_i:
                total_w = sum(uw.get(m, 1.0) for m in active_votes_i)
                if total_w > 0:
                    wm = sum(uw.get(m, 1.0) * p for m, p in active_votes_i.items()) / total_w
                else:
                    wm = am
            else:
                wm = am

            out.append({
                # Backward-compat fields → active values
                "class_proba":       am,
                "mean_proba":        am,
                "std_proba":         ast,
                "n_agree":           ana,
                # Explicit active role fields
                "active_mean":       am,
                "active_std":        ast,
                "active_n_agree":    ana,
                "active_votes":      active_votes_i,
                # Profit-voting score fields
                "active_model_count": _vs_i["active_model_count"],
                "vote_yes_fraction":  _vs_i["vote_yes_fraction"],
                "top3_mean":          _vs_i["top3_mean"],
                "vote_score":         _vs_i["vote_score"],
                # Shadow/diagnostic
                "shadow_mean":       sm,
                "shadow_std":        sstd,
                "shadow_votes":      shadow_votes_i,
                "diagnostic_votes":  diagnostic_votes_i,
                "excluded_models":   excluded_models_i,
                # Other
                "weighted_mean":     wm,
                "vote_threshold":    agree_thr,
                "pred_return":       float(pred_returns[i]),
                "return_dispersion": float(ret_dispersions[i]),
                "per_model":         per_model_i,
                "votes":             per_model_i,
            })
        return out

    # ── State management ──────────────────────────────────────────────────────

    def load_state(self, saved):
        """Copy fitted state from a previously serialised VoxEnsemble.

        If the saved model was trained on a different feature count (i.e., before
        FEATURE_COUNT was bumped from 10 → 20), the fitted state is rejected and
        the caller will trigger a retrain on the next cycle.

        Version checks:
        - FEATURE_COUNT mismatch → discard (hard incompatibility).
        - MODEL_VERSION mismatch → warn and discard (architecture changed).
        - FEATURE_VERSION / LABEL_VERSION mismatch → warn but accept for now
          (soft incompatibility; retrain recommended).
        """
        saved_fc = getattr(saved, "_feature_count", None)
        if saved_fc is not None and saved_fc != FEATURE_COUNT:
            if self._logger:
                self._logger(
                    f"[VoxEnsemble] Stale model (trained on {saved_fc} features, "
                    f"current FEATURE_COUNT={FEATURE_COUNT}). Discarding saved state."
                )
            return   # leave self._fitted = False so caller retrains

        # Check model architecture version
        saved_mv = getattr(saved, "_model_version", None)
        if saved_mv is not None and saved_mv != MODEL_VERSION:
            if self._logger:
                self._logger(
                    f"[VoxEnsemble] MODEL_VERSION mismatch: "
                    f"saved={saved_mv}, current={MODEL_VERSION}. Discarding saved state."
                )
            return   # force retrain on architecture changes

        # Warn on feature/label version mismatches (soft — do not discard)
        saved_fv = getattr(saved, "_feature_version", None)
        if saved_fv is not None and saved_fv != FEATURE_VERSION and self._logger:
            self._logger(
                f"[VoxEnsemble] FEATURE_VERSION mismatch: "
                f"saved={saved_fv}, current={FEATURE_VERSION}. "
                "Retrain recommended."
            )
        saved_lv = getattr(saved, "_label_version", None)
        if saved_lv is not None and saved_lv != LABEL_VERSION and self._logger:
            self._logger(
                f"[VoxEnsemble] LABEL_VERSION mismatch: "
                f"saved={saved_lv}, current={LABEL_VERSION}. "
                "Retrain recommended."
            )

        self._model         = saved._model
        self._fitted        = saved._fitted
        self._positive_rate = getattr(saved, "_positive_rate", 0.0)
        # Load regression ensemble if it was saved (v2+).
        # Older pickles without regressors default to unfitted state.
        if getattr(saved, "_reg_fitted", False) and getattr(saved, "_regressors", None):
            self._regressors    = saved._regressors
            self._reg_fitted    = True
        # Preserve user model weights if saved (v5+); older pickles default to empty.
        saved_weights = getattr(saved, "_user_model_weights", {})
        if saved_weights and not self._user_model_weights:
            self._user_model_weights = saved_weights
        # Restore model roles if saved (v6+); older pickles keep defaults.
        saved_roles = getattr(saved, "_model_roles", {})
        if saved_roles:
            self._model_roles.update(saved_roles)
        # Restore shadow models if saved (v6+); older pickles keep defaults.
        saved_shadows = getattr(saved, "_shadow_models", [])
        saved_shadow_fitted = getattr(saved, "_shadow_fitted", False)
        if saved_shadows and saved_shadow_fitted:
            self._shadow_models  = saved_shadows
            self._shadow_fitted  = True

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
# TRAINING PIPELINE — moved to training.py for QuantConnect 63KB file limit
# Re-exported here to preserve backward-compatible imports.
# ═══════════════════════════════════════════════════════════════════════════════

from training import (  # noqa: E402,F401
    build_training_data,
    walk_forward_train,
    check_label_execution_alignment,
    derive_training_hour,
)
