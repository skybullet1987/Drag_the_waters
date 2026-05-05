# ── training.py: walk-forward training pipeline ──────────────────────────────
#
# Moved from models.py to keep models.py under the QuantConnect 63KB file limit.
# Re-exported from models.py for backward compatibility.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Import constants and helpers from models (safe: training.py is loaded after
# models.py's top-level definitions are already in place).
from models import (
    build_features,
    triple_barrier_outcome,
    LABEL_TP,
    LABEL_SL,
    LABEL_HORIZON_BARS,
    LABEL_COST_BPS,
    TRAIN_STRIDE,
    MAX_TRAIN_SAMPLES,
    VOX_ENABLE_CV,
    CV_SPLITS,
)



def derive_training_hour(bar_index, n_bars, decision_interval_min=15):
    """Approximate UTC hour from a training bar's position in the history deque.

    During training, actual timestamps are unavailable (only close/volume deques
    are stored).  This helper derives a plausible UTC hour using the assumption
    that bars are evenly spaced by ``decision_interval_min`` minutes ending at
    the current training call.  The most-recent bar is assigned hour 0 (UTC
    midnight proxy), and earlier bars are mapped backwards.

    Parameters
    ----------
    bar_index             : int — 0-based index of the bar (0 = oldest in window).
    n_bars                : int — total number of bars in the history window.
    decision_interval_min : int — bar cadence in minutes (default 15).

    Returns
    -------
    int — estimated UTC hour, in range [0, 23].
    """
    bars_from_end = (n_bars - 1) - bar_index
    minutes_back  = bars_from_end * decision_interval_min
    hours_back    = minutes_back // 60
    return int(hours_back % 24)


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
                hour       = derive_training_hour(i, n, decision_interval_min),
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


def check_label_execution_alignment(
    label_tp,
    label_sl,
    label_horizon_bars,
    exec_tp,
    exec_sl,
    exec_timeout_hours,
    decision_interval_min=15,
    logger=None,
):
    """Warn when training label params materially differ from execution params.

    Misalignment causes the classifier to optimise for targets it never sees
    in live/backtest execution, degrading precision.  Call at setup/retrain time.

    Parameters
    ----------
    label_tp / label_sl / label_horizon_bars : training label configuration.
    exec_tp / exec_sl / exec_timeout_hours   : live execution configuration.
    decision_interval_min : bar cadence (default 15 min).
    logger : callable(str) or None — QC algo.log or print.
    """
    warnings = []

    label_horizon_hours = (label_horizon_bars * decision_interval_min) / 60.0
    ratio_tp = label_tp / exec_tp if exec_tp > 0 else float("inf")
    ratio_sl = label_sl / exec_sl if exec_sl > 0 else float("inf")

    # TP mismatch: label TP more than 2× or less than 0.5× execution TP
    if ratio_tp > 2.0 or ratio_tp < 0.5:
        warnings.append(
            f"label_tp={label_tp:.3f} vs exec_tp={exec_tp:.3f} "
            f"(ratio={ratio_tp:.2f}; model optimises for a TP it rarely sees)"
        )

    # SL mismatch: label SL more than 2× or less than 0.5× execution SL
    if ratio_sl > 2.0 or ratio_sl < 0.5:
        warnings.append(
            f"label_sl={label_sl:.3f} vs exec_sl={exec_sl:.3f} "
            f"(ratio={ratio_sl:.2f}; model SL tolerance mismatches execution)"
        )

    # Horizon mismatch: label horizon more than 2× or less than 0.5× execution timeout
    if exec_timeout_hours > 0 and label_horizon_hours > 0:
        ratio_h = label_horizon_hours / exec_timeout_hours
        if ratio_h > 2.0 or ratio_h < 0.5:
            warnings.append(
                f"label_horizon={label_horizon_hours:.1f}h vs "
                f"exec_timeout={exec_timeout_hours:.1f}h "
                f"(ratio={ratio_h:.2f}; model horizon mismatches execution timeout)"
            )

    if warnings and logger:
        for w in warnings:
            logger(f"[label_align] WARNING: {w}")
    return warnings
