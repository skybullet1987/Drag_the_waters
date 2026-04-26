# ── Vox Features ──────────────────────────────────────────────────────────────
#
# All feature-engineering lives here.  The output is a flat numpy array that
# can be fed directly to VoxEnsemble.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np


# ── ATR helper ────────────────────────────────────────────────────────────────

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
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows,  dtype=float)
    c = np.asarray(closes, dtype=float)

    if len(c) < period + 1:
        return 0.0

    # True-range for each bar (using prior close)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1]),
        ),
    )

    # Wilder smoothing: seed with simple mean, then EMA-style
    atr = float(np.mean(tr[-period:]))
    return atr


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(closes, volumes, btc_closes, hour):
    """
    Build the feature vector for one decision bar.

    Parameters
    ----------
    closes     : array-like of float, length >= 17  (symbol close prices)
    volumes    : array-like of float, length >= 16  (symbol volumes)
    btc_closes : array-like of float, length >= 5   (BTC close prices)
    hour       : int, 0–23  (UTC hour of current bar)

    Returns
    -------
    numpy.ndarray of shape (10,)
        Feature vector with elements:
          0  ret_1   — 1-bar return
          1  ret_4   — 4-bar return
          2  ret_8   — 8-bar return
          3  ret_16  — 16-bar return
          4  rsi_14  — RSI(14) in [0, 100]
          5  atr_n   — ATR(14) / close[-1]  (normalised)
          6  vol_r   — volume ratio: current / 15-bar mean
          7  btc_rel — 4-bar symbol return minus 4-bar BTC return
          8  hour_of_day — hour as float (0–23)
          9  (reserved / zero-padded for future use)

    Returns None when there is insufficient history.
    """
    c  = np.asarray(closes,    dtype=float)
    v  = np.asarray(volumes,   dtype=float)
    bc = np.asarray(btc_closes, dtype=float)

    # Minimum lengths
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
    deltas = np.diff(c[-15:])   # 14 differences from last 15 closes
    gains  = float(np.sum(deltas[deltas > 0]))
    losses = float(np.sum(-deltas[deltas < 0]))
    avg_g  = gains  / 14.0
    avg_l  = losses / 14.0
    rsi    = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    # ── ATR(14) normalised by close ───────────────────────────────────────────
    # We approximate using close-only ATR when high/low not passed
    # (caller can pass closes as highs/lows proxy; full OHLC version in main.py)
    if len(c) >= 16:
        tr_proxy = np.abs(np.diff(c[-15:]))
        atr_val  = float(np.mean(tr_proxy))
    else:
        atr_val = 0.0
    atr_n = atr_val / last if last != 0 else 0.0

    # ── Volume ratio ──────────────────────────────────────────────────────────
    prior_avg = float(np.mean(v[-16:-1]))   # 15 prior bars
    vol_r     = (v[-1] / prior_avg) if prior_avg > 0 else 1.0

    # ── BTC-relative return (4-bar) ───────────────────────────────────────────
    btc_ret_4 = (bc[-1] - bc[-5]) / bc[-5] if len(bc) >= 5 and bc[-5] != 0 else 0.0
    btc_rel   = ret_4 - btc_ret_4

    # ── Assemble vector ───────────────────────────────────────────────────────
    feats = np.array([
        ret_1,
        ret_4,
        ret_8,
        ret_16,
        rsi / 100.0,   # normalise to [0, 1]
        atr_n,
        vol_r,
        btc_rel,
        float(hour) / 23.0,   # normalise to [0, 1]
        0.0,                   # reserved
    ], dtype=float)

    return feats
