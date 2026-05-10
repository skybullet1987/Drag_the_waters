# signals.py — HYDRA v3: Regime-Aware Multi-Strategy Signal Engine
#
# THREE STRATEGIES that activate based on market regime:
#   BULL (BTC > SMA50): buy dips in uptrending coins (70%+ WR historically)
#   PUMP (BTC pumping): ride the wave, aggressive momentum (highest returns)
#   BEAR (BTC < SMA50): DON'T TRADE. Cash is a position.
#
# Plus: cross-coin momentum rotation — buy what's ALREADY winning

import numpy as np


def _ema(data, period):
    if len(data) < period:
        return float(np.mean(data))
    mult = 2.0 / (period + 1)
    ema = float(data[0])
    for val in data[1:]:
        ema = (float(val) - ema) * mult + ema
    return ema


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    diffs = np.diff(closes[-(period+1):])
    gains = np.mean(np.where(diffs > 0, diffs, 0))
    losses = np.mean(np.where(diffs < 0, -diffs, 0))
    rs = gains / max(losses, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _sma(data, period):
    if len(data) < period:
        return float(np.mean(data))
    return float(np.mean(data[-period:]))


def detect_regime(btc_closes):
    """Detect market regime from BTC price action.

    Returns: "bull", "pump", "bear", or "neutral"
    """
    if len(btc_closes) < 100:
        return "neutral"

    c = np.asarray(btc_closes, dtype=float)
    price = c[-1]
    sma50 = _sma(c, 50)
    sma20 = _sma(c, 20)
    ret_24h = (c[-1] - c[-min(289, len(c))]) / c[-min(289, len(c))]  # ~24h at 5min
    ret_4h = (c[-1] - c[-min(49, len(c))]) / c[-min(49, len(c))]

    # PUMP: BTC up >2% in 24h AND above both SMAs
    if ret_24h > 0.02 and price > sma20 and price > sma50:
        return "pump"

    # BULL: BTC above SMA50 and SMA20 > SMA50 (healthy uptrend)
    if price > sma50 and sma20 > sma50:
        return "bull"

    # BEAR: BTC below SMA50
    if price < sma50:
        return "bear"

    return "neutral"


def scan_all_coins(symbols, state, btc_closes, forecast_cache=None,
                    regime="neutral"):
    """Scan all coins and return ranked entry signals.

    Different strategies per regime:
      bull: buy dips in uptrending coins
      pump: momentum breakout + ride the wave
      bear: NO TRADES (returns empty list)
      neutral: conservative dip-buying only

    Returns: list of (sym, strength, reason, mode) sorted by strength
    """
    if regime == "bear":
        return []  # CASH IS A POSITION. Don't trade in bear markets.

    signals = []
    fc = forecast_cache or {}

    for sym in symbols:
        st = state.get(sym)
        if st is None:
            continue
        closes = list(st["closes"])
        volumes = list(st["volumes"])
        highs = list(st.get("highs", closes))

        if len(closes) < 100:
            continue

        c = np.asarray(closes, dtype=float)
        v = np.asarray(volumes, dtype=float)
        price = c[-1]
        if price <= 0:
            continue

        chronos = fc.get(sym, 0.0)

        if regime in ("bull", "neutral"):
            result = _strategy_dip_buy(c, v, highs, chronos, regime)
            if result:
                signals.append((sym, result["strength"], result["reason"], result["mode"]))

        if regime == "pump":
            result = _strategy_pump_ride(c, v, highs, chronos)
            if result:
                signals.append((sym, result["strength"], result["reason"], result["mode"]))

        # Always check: momentum rotation (strongest coin in the universe)
        result = _strategy_momentum_rotation(c, v, chronos, regime)
        if result:
            signals.append((sym, result["strength"], result["reason"], result["mode"]))

    # Sort by strength, deduplicate per symbol (keep strongest)
    signals.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    deduped = []
    for sig in signals:
        if sig[0] not in seen:
            seen.add(sig[0])
            deduped.append(sig)
    return deduped[:5]  # top 5 candidates


def _strategy_dip_buy(c, v, highs, chronos, regime):
    """BUY THE DIP in uptrending coins.

    Logic: coin pulled back 2-5% but is still in uptrend.
    In bull markets, dips recover 70-80% of the time.
    This is the HIGHEST WR strategy.
    """
    price = c[-1]
    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    rsi = _rsi(c)

    # Must be in uptrend: above SMA50 and SMA20 > SMA50
    if price < sma50 or sma20 < sma50:
        return None

    # Must be dipping: price below SMA20 (pulled back from trend)
    if price > sma20:
        return None

    # How much dip? 1-5% below recent high
    recent_high = np.max(c[-20:])
    dip_pct = (recent_high - price) / recent_high
    if dip_pct < 0.01 or dip_pct > 0.08:
        return None  # too small or too large

    # RSI should be pulling back (30-50 range = oversold in uptrend)
    if rsi > 55 or rsi < 20:
        return None

    # Volume should not be panicking (volume < 2x avg during dip = orderly)
    vol_avg = np.mean(v[-20:])
    vol_ratio = v[-1] / max(vol_avg, 1e-9)
    if vol_ratio > 3.0:
        return None  # panic selling, not a healthy dip

    # Strength: deeper dip + lower RSI = better entry
    strength = (
        0.35 * min(dip_pct / 0.05, 1.0) +       # dip depth
        0.30 * max(0, (50 - rsi) / 30.0) +        # RSI oversold
        0.20 * (1.0 if sma20 > sma50 else 0.0) +  # trend intact
        0.15 * max(0, chronos * 10)                 # Chronos bullish
    )

    # Require higher strength in neutral vs bull
    min_str = 0.35 if regime == "bull" else 0.50
    if strength < min_str:
        return None

    return {"strength": float(np.clip(strength, 0, 1)),
            "reason": "dip_buy", "mode": "runner"}  # dips in uptrend → RUNNER


def _strategy_pump_ride(c, v, highs, chronos):
    """RIDE THE PUMP — only in pump regime.

    Logic: coin is pumping RIGHT NOW (+2%+ in 1h with volume).
    Enter and ride with wide trail.
    """
    price = c[-1]
    if len(c) < 13:
        return None

    ret_1h = (price - c[-13]) / c[-13]
    ret_4h = (price - c[-min(49, len(c))]) / c[-min(49, len(c))]
    vol_avg = np.mean(v[-20:-1]) if len(v) > 20 else np.mean(v[:-1])
    vol_ratio = v[-1] / max(vol_avg, 1e-9)

    # Must be pumping: +1.5% in 1h with volume
    if ret_1h < 0.015 or vol_ratio < 1.5:
        return None

    # Not overbought yet
    rsi = _rsi(c)
    if rsi > 80:
        return None

    # EMA alignment: 8 > 21 (momentum intact)
    if len(c) >= 22:
        ema8 = _ema(c[-22:], 8)
        ema21 = _ema(c[-22:], 21)
        if ema8 < ema21:
            return None  # counter-trend pump, skip

    strength = (
        0.30 * min(ret_1h / 0.04, 1.0) +          # how strong is the pump
        0.25 * min((vol_ratio - 1) / 3, 1.0) +     # volume confirmation
        0.20 * min(ret_4h / 0.05, 1.0) +            # 4h trend supports
        0.15 * max(0, chronos * 10) +                # Chronos confirms
        0.10 * (1.0 if rsi > 50 and rsi < 75 else 0.5)  # sweet spot RSI
    )

    if strength < 0.30:
        return None

    return {"strength": float(np.clip(strength, 0, 1)),
            "reason": "pump_ride", "mode": "runner"}


def _strategy_momentum_rotation(c, v, chronos, regime):
    """MOMENTUM ROTATION: buy the strongest coin.

    Logic: rank coins by 4h return. Top performer with volume
    tends to continue. This is the "hot hand" effect in crypto.
    Only triggers for coins with strong multi-TF momentum.
    """
    if len(c) < 49:
        return None

    price = c[-1]
    ret_1h = (price - c[-13]) / c[-13]
    ret_4h = (price - c[-49]) / c[-49]
    sma50 = _sma(c, 50)

    # Must be above SMA50 (not in downtrend)
    if price < sma50:
        return None

    # Must have strong 4h momentum: top-tier move
    if ret_4h < 0.03:  # need +3% in 4h to qualify
        return None

    # 1h momentum should be positive too (not stalling)
    if ret_1h < 0.005:
        return None

    # Volume confirmation
    vol_avg = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
    vol_ratio = v[-1] / max(vol_avg, 1e-9)
    if vol_ratio < 1.2:
        return None

    rsi = _rsi(c)
    if rsi > 78:
        return None

    strength = (
        0.35 * min(ret_4h / 0.06, 1.0) +
        0.25 * min(ret_1h / 0.02, 1.0) +
        0.20 * min((vol_ratio - 1) / 2, 1.0) +
        0.10 * max(0, chronos * 10) +
        0.10 * (1.0 if regime == "pump" else 0.5)
    )

    min_str = 0.25 if regime == "pump" else 0.40
    if strength < min_str:
        return None

    return {"strength": float(np.clip(strength, 0, 1)),
            "reason": "momentum_rotation", "mode": "runner" if regime == "pump" else "scalp"}
