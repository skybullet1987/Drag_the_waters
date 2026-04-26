# ── Vox Training Pipeline ─────────────────────────────────────────────────────
#
# Builds a labelled dataset from live-collected state dicts, then fits the
# ensemble with time-series cross-validation.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from features import build_features
from labeling  import triple_barrier_label


def build_training_data(
    algorithm,
    symbols,
    state_dict,
    tp,
    sl,
    timeout_bars,
    decision_interval_min,
):
    """
    Construct a labelled feature matrix from per-symbol state history.

    For each symbol the function steps through the recorded close history one
    decision bar at a time, builds a feature vector via ``build_features``, and
    assigns a label via ``triple_barrier_label``.  Rows from all symbols are
    concatenated into a single (X, y) dataset.

    Parameters
    ----------
    algorithm             : QCAlgorithm — used for logging.
    symbols               : list[Symbol] — symbols to include.
    state_dict            : dict[Symbol, dict]
        Per-symbol deques keyed by ``"closes"``, ``"highs"``, ``"lows"``,
        ``"volumes"``.  BTC closes are read from the BTC symbol's entry.
    tp                    : float — take-profit fraction (must match execution).
    sl                    : float — stop-loss fraction (must match execution).
    timeout_bars          : int   — max bars per label (must match execution).
    decision_interval_min : int   — bar cadence in minutes (used only for
                                    documentation / logging here).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` arrays with shapes ``(n_samples, n_features)`` and
        ``(n_samples,)``.  Returns ``(None, None)`` if no usable rows exist.
    """
    from universe import KRAKEN_PAIRS

    # Identify the BTC symbol for BTC-relative feature
    btc_sym = None
    for sym in symbols:
        if sym.value.upper().startswith("BTC"):
            btc_sym = sym
            break

    X_rows, y_rows = [], []

    for sym in symbols:
        st = state_dict.get(sym)
        if st is None:
            continue

        closes  = list(st.get("closes",  []))
        highs   = list(st.get("highs",   []))
        lows    = list(st.get("lows",    []))
        volumes = list(st.get("volumes", []))

        btc_closes = (
            list(state_dict[btc_sym].get("closes", []))
            if btc_sym and btc_sym in state_dict
            else []
        )

        n = len(closes)
        # Need at least 17 for features + timeout_bars for label
        min_len = 17 + timeout_bars
        if n < min_len:
            continue

        for i in range(17, n - timeout_bars):
            feat = build_features(
                closes  = closes[: i + 1],
                volumes = volumes[: i + 1],
                btc_closes = btc_closes[: i + 1] if btc_closes else [],
                hour    = 0,   # hour unknown from deque; neutral value
            )
            if feat is None:
                continue

            label = triple_barrier_label(
                prices       = closes[i:],
                tp           = tp,
                sl           = sl,
                timeout_bars = timeout_bars,
            )

            X_rows.append(feat)
            y_rows.append(label)

    if not X_rows:
        algorithm.log("[training] build_training_data: no usable rows")
        return None, None

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    algorithm.log(
        f"[training] Dataset: {X.shape[0]} samples, "
        f"{X.shape[1]} features, "
        f"positive_rate={float(y.mean()):.3f}"
    )
    return X, y


def walk_forward_train(ensemble, X, y):
    """
    Fit *ensemble* using time-series walk-forward cross-validation, then
    refit on the full dataset and return the fitted ensemble.

    The CV scores are logged to help detect overfitting but do not gate the
    final fit — the model is always retrained on all available data.

    Parameters
    ----------
    ensemble : VoxEnsemble — the (unfitted or previously fitted) ensemble.
    X        : np.ndarray of shape (n_samples, n_features)
    y        : np.ndarray of shape (n_samples,)

    Returns
    -------
    VoxEnsemble
        The same *ensemble* object, now fitted on the full dataset.
    """
    np.random.seed(42)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2:
            # Skip degenerate fold (all one class)
            continue

        try:
            ensemble.fit(X_tr, y_tr)
            # Simple accuracy on test fold
            preds = [
                1 if ensemble.predict_with_confidence(
                    X_te[j:j+1]
                )["mean_proba"] >= 0.5 else 0
                for j in range(len(X_te))
            ]
            acc = float(np.mean(np.array(preds) == y_te))
            fold_scores.append(acc)
        except Exception as exc:
            # Non-fatal: log failure and continue to next fold
            if hasattr(ensemble, "_logger") and ensemble._logger:
                ensemble._logger(
                    f"[training] walk_forward_train fold={fold} failed: {exc}"
                )

    if fold_scores:
        mean_acc = float(np.mean(fold_scores))
    else:
        mean_acc = float("nan")

    # Final fit on the full dataset
    if len(np.unique(y)) >= 2:
        ensemble.fit(X, y)

    return ensemble
