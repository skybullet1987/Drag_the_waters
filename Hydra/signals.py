# signals.py — HYDRA multi-signal pump detection engine
#
# 4 independent signal types. ANY one firing = candidate.
# Multiple signals on same coin = higher strength.

import numpy as np


def detect_all_signals(closes, highs, lows, volumes,
                        chronos_forecast=0.0,
                        pump_ret_min=0.02, pump_vol_mult=2.0,
                        breakout_lookback=20, breakout_vol_mult=1.5,
                        squeeze_lookback=20, squeeze_threshold=0.3,
                        vol_anomaly_mult=3.0):
    """Scan for all signal types on one coin.

    Returns dict: enter, strength, reason, details
    """
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = len(c)

    if n < 50:
        return {"enter": False, "strength": 0.0, "reason": "insufficient_data", "details": {}}

    signals = []

    # ── SIGNAL 1: PUMP DETECTION ─────────────────────────────────────────
    # +2% in 1h (12 bars at 5min) with volume spike
    if n >= 13:
        ret_1h = (c[-1] - c[-13]) / c[-13]
        vol_avg = np.mean(v[-20:-1]) if n >= 21 else np.mean(v[:-1])
        vol_ratio = v[-1] / max(vol_avg, 1e-9)
        if ret_1h >= pump_ret_min and vol_ratio >= pump_vol_mult:
            strength = min(ret_1h / 0.05, 1.0) * 0.5 + min((vol_ratio - 1) / 3, 1.0) * 0.5
            signals.append(("pump", float(np.clip(strength, 0, 1)), {
                "ret_1h": ret_1h, "vol_ratio": vol_ratio}))

    # ── SIGNAL 2: BREAKOUT ───────────────────────────────────────────────
    # Price > N-bar high with volume confirmation
    if n >= breakout_lookback + 1:
        prior_high = np.max(h[-(breakout_lookback+1):-1])
        above = c[-1] > prior_high
        vol_avg = np.mean(v[-breakout_lookback:-1])
        vol_ratio = v[-1] / max(vol_avg, 1e-9)
        ret_1 = (c[-1] - c[-2]) / c[-2] if c[-2] > 0 else 0

        if above and vol_ratio >= breakout_vol_mult and ret_1 > 0:
            dist = (c[-1] - prior_high) / prior_high
            strength = min(dist / 0.02, 1.0) * 0.4 + min((vol_ratio-1)/2, 1.0) * 0.3 + min(ret_1/0.01, 1.0) * 0.3
            signals.append(("breakout", float(np.clip(strength, 0, 1)), {
                "breakout_dist": dist, "vol_ratio": vol_ratio}))

    # ── SIGNAL 3: BOLLINGER SQUEEZE ──────────────────────────────────────
    # Bands narrow then expand = volatility explosion
    if n >= squeeze_lookback + 5:
        sma = np.mean(c[-squeeze_lookback:])
        std = np.std(c[-squeeze_lookback:])
        upper = sma + 2 * std
        lower = sma - 2 * std
        bandwidth = (upper - lower) / sma if sma > 0 else 0

        # Historical bandwidths for percentile
        bws = []
        for i in range(max(5, n - 100), n - squeeze_lookback):
            _s = np.mean(c[i:i+squeeze_lookback])
            _std = np.std(c[i:i+squeeze_lookback])
            if _s > 0:
                bws.append((_s + 2*_std - (_s - 2*_std)) / _s)

        if bws:
            pct = np.mean([1 for bw in bws if bandwidth < bw]) / len(bws)
            # Squeeze: bandwidth in bottom 30% AND price breaking out
            if pct >= (1 - squeeze_threshold) and c[-1] > upper:
                strength = pct * 0.5 + min((c[-1] - upper) / (std + 1e-9), 1.0) * 0.5
                signals.append(("squeeze", float(np.clip(strength, 0, 1)), {
                    "bandwidth_pct": pct, "above_upper": True}))

    # ── SIGNAL 4: VOLUME ANOMALY ─────────────────────────────────────────
    # 3x+ volume with positive price = institutional accumulation
    if n >= 21:
        vol_avg = np.mean(v[-21:-1])
        vol_ratio = v[-1] / max(vol_avg, 1e-9)
        ret_3 = (c[-1] - c[-4]) / c[-4] if n >= 4 else 0

        if vol_ratio >= vol_anomaly_mult and ret_3 > 0:
            strength = min((vol_ratio - 2) / 3, 1.0) * 0.6 + min(ret_3 / 0.02, 1.0) * 0.4
            signals.append(("volume_anomaly", float(np.clip(strength, 0, 1)), {
                "vol_ratio": vol_ratio, "ret_3": ret_3}))

    # ── COMBINE SIGNALS ──────────────────────────────────────────────────
    if not signals:
        return {"enter": False, "strength": 0.0, "reason": "no_signal", "details": {}}

    # Multiple signals = higher conviction
    best_reason = max(signals, key=lambda x: x[1])[0]
    combined_strength = min(sum(s[1] for s in signals), 1.0)

    # Boost with Chronos if available
    if chronos_forecast > 0.005:
        combined_strength = min(combined_strength + 0.15, 1.0)
    elif chronos_forecast < -0.005:
        combined_strength *= 0.5  # dampen if Chronos says down

    # RSI overbought filter: don't enter if already overbought
    diffs = np.diff(c[-15:])
    gains = np.mean(np.where(diffs > 0, diffs, 0))
    losses_val = np.mean(np.where(diffs < 0, -diffs, 0))
    rs = gains / max(losses_val, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if rsi > 85:
        return {"enter": False, "strength": 0.0, "reason": "overbought", "details": {"rsi": rsi}}

    return {
        "enter": True,
        "strength": float(combined_strength),
        "reason": best_reason if len(signals) == 1 else f"multi({len(signals)})",
        "details": {
            "signals": [(s[0], round(s[1], 3)) for s in signals],
            "rsi": round(rsi, 1),
            "chronos": round(chronos_forecast, 4),
            "n_signals": len(signals),
        },
    }
