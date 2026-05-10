"""features — live-survivable microstructure features for Pulse.

These features replace the broken OBI signal (which read from QC's
minute-aggregated QuoteBars and was effectively noise in live).

Five primary live-survivable features:

1. ``cumulative_volume_delta()`` — CVD slope over N bars
   - Signed bar volume by close vs midrange position
   - Already in MG36 main.py:476-481; promoted here as primary signal

2. ``kyle_lambda()`` — price impact per unit volume
   - λ = avg(|Δprice|) / avg(volume) over rolling N bars
   - Low λ = elastic market (signals work); high λ = inelastic (skip)

3. ``yang_zhang_realized_variance()`` — bias-corrected OHLC vol
   - More accurate than close-to-close vol; uses overnight + intraday components
   - Used as a regime classifier and position-size scaler

4. ``trade_rate_burst()`` — bar volume / rolling mean
   - Replaces the ratio-style "volume_ignition" with an explicit z-score

5. ``vwap_band_position()`` — where the price sits relative to VWAP±σ
   - Returns -2/-1/0/+1/+2 for σ-zone (mean-reversion + breakout signal)

Bonus:

6. ``cross_symbol_momentum_spillover()`` — count of alts moving > X% in last N bars
   - Cross-sectional pump cascade detector

7. ``ema()`` and ``rsi()`` — kept around as small numerically-clean utilities

All functions are pure-Python (numpy-based when input is array-like).
No QC dependency. Returns 0.0 / None / sentinel on insufficient data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import math


# ─── Helpers ────────────────────────────────────────────────────────────────

def _to_list(x: Iterable[float]) -> list[float]:
    return [float(v) for v in x]


def _safe_mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


# ─── 1. Cumulative Volume Delta (CVD) ───────────────────────────────────────

def signed_bar_volume(open_p: float, high: float, low: float,
                      close: float, volume: float) -> float:
    """Return signed bar volume using close-vs-midrange technique.

    Positive when close is in the upper half of the bar's range (buying pressure),
    negative when close is in the lower half (selling pressure).

    From MG36 main.py:476-481:
        bar_delta = volume * ((close - low) - (high - close)) / (high - low)
                  = volume * (2*close - high - low) / (high - low)
    """
    if high <= low or volume <= 0:
        return 0.0
    return volume * (2.0 * close - high - low) / (high - low)


def cumulative_volume_delta(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float]:
    """Compute the running CVD series for a window of OHLCV bars."""
    n = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
    cvd = [0.0] * n
    running = 0.0
    for i in range(n):
        running += signed_bar_volume(
            opens[i], highs[i], lows[i], closes[i], volumes[i],
        )
        cvd[i] = running
    return cvd


def cvd_slope(cvd_series: Sequence[float], lookback: int = 12) -> float:
    """Slope of CVD over the last `lookback` bars (units of CVD per bar).

    Used as a buy/sell pressure trend signal. Positive = buying accumulating.
    """
    if len(cvd_series) < lookback or lookback < 2:
        return 0.0
    xs = list(range(lookback))
    ys = list(cvd_series[-lookback:])
    mx = _safe_mean(xs)
    my = _safe_mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def cvd_signal_score(cvd_series: Sequence[float], lookback: int = 12,
                     normalize_by_recent_max: bool = True) -> float:
    """Convert the CVD slope into a [-1, 1] signal score.

    When normalize_by_recent_max=True we divide by the largest recent
    absolute CVD level so the score is approximately scale-invariant.
    """
    slope = cvd_slope(cvd_series, lookback)
    if slope == 0.0:
        return 0.0
    if not normalize_by_recent_max:
        return max(-1.0, min(1.0, slope))
    # Normalize: slope per bar relative to recent CVD magnitude / lookback
    recent = cvd_series[-lookback:]
    if not recent:
        return 0.0
    scale = max(abs(v) for v in recent) or 1.0
    normalized = slope * lookback / scale
    return max(-1.0, min(1.0, normalized))


# ─── 2. Kyle's Lambda ───────────────────────────────────────────────────────

def kyle_lambda(
    closes: Sequence[float],
    volumes: Sequence[float],
    lookback: int = 20,
) -> float:
    """Price impact per unit volume = mean(|Δprice|) / mean(volume).

    Higher λ → less liquid market (a unit of volume moves price more).
    Lower λ → more elastic market.

    Returns 0.0 on insufficient data or zero-volume window.
    """
    n = min(len(closes), len(volumes))
    if n <= lookback:
        return 0.0

    abs_dp = [abs(closes[i] - closes[i - 1])
              for i in range(n - lookback, n) if i > 0]
    vols   = [volumes[i] for i in range(n - lookback, n)]
    mean_v = _safe_mean(vols)
    if mean_v <= 0:
        return 0.0
    return _safe_mean(abs_dp) / mean_v


def kyle_regime_multiplier(
    current_lambda: float,
    rolling_lambda_history: Sequence[float],
    high_pct: float = 0.75,
    low_pct: float = 0.25,
) -> float:
    """Return a [0.5, 1.5] position-size multiplier based on current λ vs history.

    - current λ above 75th percentile of recent → reduce size to 0.5×
    - current λ below 25th percentile → boost to 1.5×
    - in between → linear interpolation around 1.0
    """
    if not rolling_lambda_history:
        return 1.0
    sorted_hist = sorted(rolling_lambda_history)
    n = len(sorted_hist)
    if n < 4:
        return 1.0

    p_high = sorted_hist[int(high_pct * (n - 1))]
    p_low  = sorted_hist[int(low_pct  * (n - 1))]
    if p_high <= p_low:
        return 1.0

    if current_lambda >= p_high:
        return 0.5
    if current_lambda <= p_low:
        return 1.5
    # Linear interpolation: high λ → low mult, low λ → high mult
    frac = (current_lambda - p_low) / (p_high - p_low)
    return 1.5 - frac  # 1.5 → 0.5


# ─── 3. Yang-Zhang Realized Variance ─────────────────────────────────────────

def yang_zhang_realized_variance(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    lookback: int = 20,
    k: float | None = None,
) -> float:
    """Yang-Zhang realized variance over `lookback` bars.

    YZ combines:
        σ_overnight²  — close-to-open variance
        σ_open_close² — open-to-close variance
        σ_RS²         — Rogers-Satchell intra-bar variance (uses high/low)

    σ_YZ² = σ_overnight² + k × σ_open_close² + (1 - k) × σ_RS²

    Where k = 0.34 / (1.34 + (n+1)/(n-1)) is the optimal weight (Yang-Zhang 2000).

    Returns 0.0 on insufficient data.
    """
    n = min(len(opens), len(highs), len(lows), len(closes))
    if n <= lookback or lookback < 2:
        return 0.0

    o = opens[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-lookback:]

    # Defensive: if any non-positive prices, bail
    if any(p <= 0 for p in o + h + l + c):
        return 0.0

    # Overnight returns: log(O_t / C_{t-1})
    # We need closes from one bar prior; if the caller passed exactly
    # `lookback` bars we can't compute the first overnight. Use available.
    # For simplicity, use the lookback window as both prev-close and curr-open.
    # If callers pass longer arrays we can include the prior bar.
    closes_for_overnight = closes[-lookback - 1:] if len(closes) > lookback else c
    if len(closes_for_overnight) > lookback:
        on_returns = [
            math.log(o[i] / closes_for_overnight[i])
            for i in range(lookback)
        ]
    else:
        on_returns = [
            math.log(o[i] / c[i - 1])
            for i in range(1, lookback)
        ]

    if not on_returns:
        return 0.0

    var_overnight = _safe_std(on_returns) ** 2

    # Open-to-close returns
    oc_returns = [math.log(c[i] / o[i]) for i in range(lookback)]
    var_open_close = _safe_std(oc_returns) ** 2

    # Rogers-Satchell intra-bar variance
    rs_terms = []
    for i in range(lookback):
        if h[i] <= 0 or l[i] <= 0 or o[i] <= 0 or c[i] <= 0:
            continue
        u  = math.log(h[i] / o[i])
        d  = math.log(l[i] / o[i])
        rt = math.log(c[i] / o[i])
        rs_terms.append(u * (u - rt) + d * (d - rt))
    var_rs = _safe_mean(rs_terms) if rs_terms else 0.0

    if k is None:
        k = 0.34 / (1.34 + (lookback + 1) / max(lookback - 1, 1))

    return var_overnight + k * var_open_close + (1.0 - k) * var_rs


def realized_vol_bps(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    lookback: int = 20,
) -> float:
    """Convenience: YZ realized vol expressed in bps (per bar)."""
    var = yang_zhang_realized_variance(opens, highs, lows, closes, lookback)
    return math.sqrt(max(0.0, var)) * 10_000


def realized_vol_size_scaler(
    current_vol_bps: float,
    rolling_vol_bps_history: Sequence[float],
    high_pct: float = 0.80,
    low_pct: float = 0.20,
) -> float:
    """[0.5, 1.5] scaler: reduce size in high-vol regime, boost in low-vol.

    Same idea as kyle_regime_multiplier.
    """
    if not rolling_vol_bps_history:
        return 1.0
    sorted_hist = sorted(rolling_vol_bps_history)
    n = len(sorted_hist)
    if n < 4:
        return 1.0
    p_high = sorted_hist[int(high_pct * (n - 1))]
    p_low  = sorted_hist[int(low_pct  * (n - 1))]
    if p_high <= p_low:
        return 1.0
    if current_vol_bps >= p_high:
        return 0.5
    if current_vol_bps <= p_low:
        return 1.5
    frac = (current_vol_bps - p_low) / (p_high - p_low)
    return 1.5 - frac


# ─── 4. Trade-rate burst ─────────────────────────────────────────────────────

def trade_rate_burst_zscore(
    bar_volumes: Sequence[float],
    lookback: int = 60,
) -> float:
    """Z-score of the most recent bar's volume vs rolling mean+std.

    > +2 = very strong volume burst. Returns 0.0 if insufficient data.
    """
    n = len(bar_volumes)
    if n < lookback + 1 or lookback < 2:
        return 0.0
    history = list(bar_volumes[-lookback - 1:-1])  # exclude current bar
    current = float(bar_volumes[-1])
    mean = _safe_mean(history)
    std  = _safe_std(history)
    if std <= 0:
        return 0.0
    return (current - mean) / std


def volume_ignition_signal(
    bar_volumes: Sequence[float],
    lookback: int = 60,
    strong_z: float = 2.0,
    partial_z: float = 1.0,
) -> float:
    """Map z-score to a 0.0 / 0.10 / 0.20 signal contribution.

    Mirrors MG36's volume_ignition slot but uses z-score instead of ratio.
    """
    z = trade_rate_burst_zscore(bar_volumes, lookback)
    if z >= strong_z:
        return 0.20
    if z >= partial_z:
        return 0.10
    return 0.0


# ─── 5. VWAP ± σ band position ──────────────────────────────────────────────

@dataclass
class VWAPState:
    """Rolling VWAP + σ accumulator. Update once per bar."""
    pv_sum: float = 0.0
    v_sum:  float = 0.0
    bar_prices: deque = field(default_factory=lambda: deque(maxlen=120))

    def update(self, price: float, volume: float) -> None:
        if price <= 0 or volume <= 0:
            return
        self.pv_sum += price * volume
        self.v_sum  += volume
        self.bar_prices.append(price)

    def vwap(self) -> float:
        return self.pv_sum / self.v_sum if self.v_sum > 0 else 0.0

    def std(self) -> float:
        return _safe_std(list(self.bar_prices))

    def reset(self) -> None:
        self.pv_sum = 0.0
        self.v_sum  = 0.0
        self.bar_prices.clear()


def vwap_band_position(price: float, vwap: float, std: float) -> int:
    """Where does the price sit relative to VWAP ± σ bands?

    Returns:
        -2  price < VWAP - 2σ  (deep oversold)
        -1  price < VWAP - 1σ  (oversold)
         0  within ±1σ
        +1  price > VWAP + 1σ  (overbought)
        +2  price > VWAP + 2σ  (deep overbought / breakout)
    """
    if vwap <= 0 or std <= 0:
        return 0
    z = (price - vwap) / std
    if z >= 2.0:  return 2
    if z >= 1.0:  return 1
    if z <= -2.0: return -2
    if z <= -1.0: return -1
    return 0


# ─── 6. Cross-symbol momentum spillover ─────────────────────────────────────

def cross_symbol_momentum_spillover(
    symbol_returns: dict[str, float],
    threshold_pct: float = 0.02,
    min_count: int = 5,
) -> dict:
    """Detect when many alts are pumping simultaneously.

    Returns a dict with:
        n_pumping:        count of symbols with return > threshold
        spillover_active: True if n_pumping >= min_count
        pumpers:          list of symbol names that are pumping
    """
    pumpers = [s for s, r in symbol_returns.items() if r >= threshold_pct]
    return {
        "n_pumping":        len(pumpers),
        "spillover_active": len(pumpers) >= min_count,
        "pumpers":          pumpers,
    }


def cross_symbol_score_boost(
    candidate_symbol: str,
    symbol_returns: dict[str, float],
    threshold_pct: float = 0.02,
    min_count: int = 5,
    boost: float = 0.10,
) -> float:
    """Return a score boost for `candidate_symbol` when spillover is active.

    Boost only applies if:
    - The spillover condition is met (>= min_count alts pumping), AND
    - The candidate is NOT yet in the pumping cohort (i.e. it's the next-tier
      coin that often catches the spillover wave).
    """
    spill = cross_symbol_momentum_spillover(symbol_returns, threshold_pct, min_count)
    if not spill["spillover_active"]:
        return 0.0
    candidate_return = symbol_returns.get(candidate_symbol, 0.0)
    if candidate_return >= threshold_pct:
        return 0.0   # already pumping; spillover boost is for the laggards
    return boost


# ─── 7. EMA + RSI utilities ──────────────────────────────────────────────────

def ema(data: Sequence[float], period: int) -> float:
    """Standard exponential moving average; returns the latest value."""
    data = _to_list(data)
    if not data:
        return 0.0
    if len(data) < period:
        return _safe_mean(data)
    mult = 2.0 / (period + 1)
    e = data[0]
    for v in data[1:]:
        e = (v - e) * mult + e
    return e


def rsi(closes: Sequence[float], period: int = 14) -> float:
    """Wilder-style RSI; returns latest [0, 100] value or 50 on insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    gains  = []
    losses = []
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = _safe_mean(gains)
    avg_loss = _safe_mean(losses)
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gain = max(ch, 0.0)
        loss = max(-ch, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss <= 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)
