# momentum_entry.py — Trail-first momentum breakout entry system
#
# Architecture:
#   1. Momentum detector: price > N-bar high + volume spike → SIGNAL
#   2. Chronos forecast: confirms trend will continue → CONFIRM
#   3. Meta-label: XGBoost predicts trail vs SL outcome → TAKE/SKIP
#   4. Trailing stop: the actual profit engine (87-93% WR when fires)
#
# The key insight: don't predict direction. Detect momentum already
# happening and use a trailing stop to capture it.

import numpy as np


# ── Momentum Breakout Detection ──────────────────────────────────────────────

def detect_momentum_breakout(closes, volumes, highs=None,
                              lookback=20, vol_mult=1.3, min_ret=0.005):
    """Detect if current bar is a momentum breakout.

    A breakout occurs when:
      1. Price is above the highest close of the last N bars
      2. Volume is above vol_mult × average volume
      3. Short-term return is positive (momentum confirmation)

    Parameters
    ----------
    closes : array-like, length >= lookback+1
    volumes : array-like, length >= lookback
    highs : array-like or None — if provided, use high prices for breakout
    lookback : int — bars to look back for high
    vol_mult : float — volume must be this × average
    min_ret : float — minimum 1-bar return to confirm momentum

    Returns
    -------
    dict with:
      signal: bool — True if breakout detected
      strength: float — 0.0 to 1.0 indicating breakout strength
      breakout_dist: float — how far above prior high (as fraction)
      vol_ratio: float — current volume / average volume
      ret_1: float — 1-bar return
    """
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)

    if len(c) < lookback + 1 or len(v) < lookback:
        return {"signal": False, "strength": 0.0, "breakout_dist": 0.0,
                "vol_ratio": 0.0, "ret_1": 0.0}

    current = c[-1]
    prev_close = c[-2]

    # Use highs if available, else closes for breakout level
    if highs is not None and len(highs) >= lookback + 1:
        h = np.asarray(highs, dtype=float)
        prior_high = np.max(h[-(lookback+1):-1])
    else:
        prior_high = np.max(c[-(lookback+1):-1])

    # Breakout distance (how far above prior high)
    breakout_dist = (current - prior_high) / prior_high if prior_high > 0 else 0.0

    # Volume ratio
    vol_avg = np.mean(v[-lookback:-1]) if len(v) > lookback else np.mean(v[:-1])
    vol_ratio = v[-1] / max(vol_avg, 1e-9)

    # 1-bar return
    ret_1 = (current - prev_close) / prev_close if prev_close > 0 else 0.0

    # Signal: above prior high + volume spike + positive momentum
    above_high = current > prior_high
    vol_spike = vol_ratio >= vol_mult
    positive_mom = ret_1 >= min_ret

    signal = above_high and vol_spike and positive_mom

    # Strength: composite score
    strength = 0.0
    if signal:
        s1 = min(breakout_dist / 0.03, 1.0)     # normalize breakout distance
        s2 = min((vol_ratio - 1.0) / 2.0, 1.0)  # normalize volume spike
        s3 = min(ret_1 / 0.02, 1.0)              # normalize momentum
        strength = 0.4 * s1 + 0.3 * s2 + 0.3 * s3

    return {
        "signal": signal,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "breakout_dist": float(breakout_dist),
        "vol_ratio": float(vol_ratio),
        "ret_1": float(ret_1),
    }


def detect_pullback_entry(closes, rsi_values=None, lookback=20,
                           rsi_max=40, trend_bars=10):
    """Detect pullback entry in an established uptrend.

    Entry when:
      1. Price was trending up (close > close[N] bars ago)
      2. RSI has pulled back to oversold (< rsi_max)
      3. Price is above longer-term moving average

    Returns
    -------
    dict with signal, strength
    """
    c = np.asarray(closes, dtype=float)
    if len(c) < max(lookback, trend_bars) + 1:
        return {"signal": False, "strength": 0.0}

    current = c[-1]
    trend_start = c[-(trend_bars+1)]
    sma_long = np.mean(c[-lookback:])

    # Uptrend: price higher than N bars ago
    in_uptrend = current > trend_start

    # Above long SMA
    above_sma = current > sma_long

    # RSI pullback (compute simple RSI if not provided)
    if rsi_values is not None and len(rsi_values) > 0:
        rsi = float(rsi_values[-1])
    else:
        diffs = np.diff(c[-15:])
        gains = np.mean(np.where(diffs > 0, diffs, 0))
        losses = np.mean(np.where(diffs < 0, -diffs, 0))
        rs = gains / max(losses, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi_oversold = rsi <= rsi_max

    signal = in_uptrend and above_sma and rsi_oversold

    strength = 0.0
    if signal:
        trend_strength = min((current / trend_start - 1.0) / 0.05, 1.0)
        rsi_depth = min((rsi_max - rsi) / 20.0, 1.0)
        strength = 0.6 * trend_strength + 0.4 * rsi_depth

    return {"signal": signal, "strength": float(np.clip(strength, 0.0, 1.0))}


# ── Entry Decision Combiner ──────────────────────────────────────────────────

def momentum_entry_decision(closes, volumes, highs=None,
                             chronos_forecast=0.0, wavelet_forecast=0.0,
                             btc_ret_4=0.0,
                             breakout_lookback=20, vol_mult=1.3,
                             min_strength=0.3,
                             require_chronos_confirm=False):
    """Combined momentum entry decision.

    Checks breakout AND pullback signals, optionally confirms with
    Chronos forecast.

    Returns
    -------
    dict:
      enter: bool
      reason: str — "breakout", "pullback", or "none"
      strength: float
      chronos_confirmed: bool
    """
    result = {
        "enter": False, "reason": "none", "strength": 0.0,
        "chronos_confirmed": False, "details": {},
    }

    # Check breakout
    bo = detect_momentum_breakout(closes, volumes, highs,
                                   lookback=breakout_lookback, vol_mult=vol_mult)
    # Check pullback
    pb = detect_pullback_entry(closes, lookback=breakout_lookback)

    # Pick the stronger signal
    if bo["signal"] and bo["strength"] >= min_strength:
        result["reason"] = "breakout"
        result["strength"] = bo["strength"]
        result["details"] = bo
    elif pb["signal"] and pb["strength"] >= min_strength:
        result["reason"] = "pullback"
        result["strength"] = pb["strength"]
        result["details"] = pb

    if result["reason"] == "none":
        return result

    # Chronos confirmation
    chronos_bullish = chronos_forecast > 0.001
    result["chronos_confirmed"] = chronos_bullish

    if require_chronos_confirm and not chronos_bullish:
        result["enter"] = False
        result["reason"] = "chronos_rejected"
        return result

    # BTC context: don't enter if BTC is crashing
    if btc_ret_4 < -0.03:
        result["enter"] = False
        result["reason"] = "btc_crash"
        return result

    # Boost strength with forecasts
    if chronos_bullish:
        result["strength"] = min(result["strength"] + 0.15, 1.0)
    if wavelet_forecast > 0.005:
        result["strength"] = min(result["strength"] + 0.10, 1.0)

    result["enter"] = True
    return result
