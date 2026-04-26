# ── Vox Labeling ──────────────────────────────────────────────────────────────
#
# Triple-barrier labeling for supervised training.
#
# ALIGNMENT CONSTRAINT
# ────────────────────
# The tp, sl, and timeout_bars values used here at training time MUST exactly
# match the TP, SL, and TIMEOUT_HOURS / DECISION_INTERVAL_MIN values used in
# execution.py / main.py at live/backtest time.
#
# If the labels are generated with tp=0.020 but the live strategy exits at
# tp=0.012, the classifier will be optimising for a target it never sees in
# production and out-of-sample performance will be degraded.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np


def triple_barrier_label(prices, tp, sl, timeout_bars):
    """
    Assign a binary label to a trade starting at ``prices[0]``.

    The "triple barrier" method places three exit barriers around the entry:
      - Upper barrier: entry × (1 + tp)        → label = 1 (win)
      - Lower barrier: entry × (1 − sl)        → label = 0 (loss)
      - Vertical barrier: index == timeout_bars → label = 0 (timeout)

    The label is 1 if the upper barrier is hit *before* the lower barrier
    within ``timeout_bars`` steps; 0 otherwise.

    .. important::
        **Alignment constraint** — ``tp``, ``sl``, and ``timeout_bars`` MUST
        be identical to the values used in live execution (see config.py and
        execution.py).  Misalignment causes the model to learn a target that
        differs from what is actually traded in production.

    Parameters
    ----------
    prices       : array-like of float
        Price series starting at the entry bar (index 0).  Must contain at
        least ``timeout_bars + 1`` elements for a meaningful label; shorter
        series will be labelled 0 (timeout / insufficient data).
    tp           : float
        Take-profit fraction (e.g. 0.020 for +2 %).
    sl           : float
        Stop-loss fraction (e.g. 0.012 for −1.2 %).
    timeout_bars : int
        Maximum number of bars to hold the position.

    Returns
    -------
    int
        1 if the take-profit barrier is hit first; 0 otherwise.
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

    # Vertical barrier reached without hitting either horizontal barrier
    return 0
