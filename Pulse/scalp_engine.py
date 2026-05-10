"""scalp_engine — MicroScalpEngine v8 (Pulse).

Evolution of the engine that powered MG36 / Macro Cannon / Hunting Rifle /
Sniper Rifle (best backtests in the user's portfolio). The v8 redesign
replaces the broken-in-live OBI signal with live-survivable features.

═══ Score composition (5 components, each 0.0 - 0.20, total 0.0 - 1.0) ═══

| # | Component | Replaces | Source |
|---|---|---|---|
| 1 | CVD slope                  | OBI                | features.cvd_signal_score |
| 2 | Volume ignition (z-score)  | volume ratio       | features.volume_ignition_signal |
| 3 | Micro-trend (EMA5/EMA20)   | (kept)             | features.ema |
| 4 | ADX/RSI hybrid             | (kept)             | local |
| 5 | VWAP±σ band confluence     | VWAP reclaim       | features.vwap_band_position |

═══ Multipliers applied AFTER score (gating + sizing) ═══

- Cross-symbol momentum spillover boost  (+0.10 to score, laggards only)
- Kyle's λ regime multiplier              (size scaler 0.5 - 1.5×)
- Yang-Zhang realized vol scaler          (size scaler 0.5 - 1.5×)
- Composed market regime multiplier       (selloff=0, pump+bull+alt=1.5)
- Fear & Greed size multiplier            (0.5 - 1.2×)
- Per-tier max-position-USD cap           (from Pulse.universe)

═══ Entry thresholds ═══
  score >= 0.55  → entry candidate (will be walk-forward tuned in Phase 3)
  score >= 0.70  → high-conviction entry (max position size)

The pure-Python `compute_scalp_score()` returns a `ScalpScore` dataclass with
the breakdown for logging and post-hoc debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from Pulse.features import (
    cvd_signal_score, cumulative_volume_delta,
    volume_ignition_signal,
    ema, rsi,
    vwap_band_position,
    cross_symbol_score_boost,
    kyle_lambda, kyle_regime_multiplier,
    yang_zhang_realized_variance, realized_vol_size_scaler, realized_vol_bps,
)
from Pulse.regime import (
    detect_market_mode, golden_cross_regime, btc_dominance_regime,
    compose_regime_size_multiplier,
)
from Pulse.alt_data import FGSignal


# ─── Default thresholds (will be walk-forward tuned in Phase 3) ─────────────

DEFAULT_ENTRY_THRESHOLD       = 0.55
DEFAULT_HIGH_CONVICTION_THRES = 0.70
DEFAULT_CVD_LOOKBACK          = 12
DEFAULT_VOL_LOOKBACK          = 60
DEFAULT_KYLE_LOOKBACK         = 20
DEFAULT_YZ_LOOKBACK           = 20
DEFAULT_EMA_FAST_PERIOD       = 5
DEFAULT_EMA_SLOW_PERIOD       = 20

# RSI/ADX hybrid defaults (port of MG36 logic, simplified)
DEFAULT_RSI_OVERSOLD  = 35
DEFAULT_RSI_OVERBOUGHT = 70


# ─── Output dataclasses ─────────────────────────────────────────────────────

@dataclass
class ScalpScore:
    """Per-symbol scalp score with full attribution."""
    symbol: str

    # Components (0 to 0.20 each)
    cvd_score:        float = 0.0
    vol_ignition:     float = 0.0
    micro_trend:      float = 0.0
    rsi_filter:       float = 0.0
    vwap_signal:      float = 0.0

    # Boosts
    spillover_boost:  float = 0.0

    # Final score
    raw_total:    float = 0.0   # sum of components + boosts (no caps applied)
    score:        float = 0.0   # raw_total clamped to [0, 1]
    enter:        bool  = False
    high_conviction: bool = False

    # Multipliers (applied to position size, not score)
    kyle_size_mult:    float = 1.0
    rv_size_mult:      float = 1.0
    regime_size_mult:  float = 1.0
    fg_size_mult:      float = 1.0
    composed_size_mult: float = 1.0   # product of all four

    # Diagnostics
    rsi:               float = 50.0
    cvd_slope_value:   float = 0.0
    volume_z_score:    float = 0.0
    vwap_position:     int   = 0
    realized_vol_bps:  float = 0.0
    market_mode:       str   = "chop"
    fg_regime:         str   = "neutral"

    def as_dict(self) -> dict:
        return {
            "symbol":            self.symbol,
            "score":             round(self.score, 4),
            "raw_total":         round(self.raw_total, 4),
            "enter":             self.enter,
            "high_conviction":   self.high_conviction,
            "components": {
                "cvd":          round(self.cvd_score, 4),
                "vol_ignition": round(self.vol_ignition, 4),
                "micro_trend":  round(self.micro_trend, 4),
                "rsi_filter":   round(self.rsi_filter, 4),
                "vwap_signal":  round(self.vwap_signal, 4),
                "spillover":    round(self.spillover_boost, 4),
            },
            "size_multipliers": {
                "kyle":    round(self.kyle_size_mult, 3),
                "rv":      round(self.rv_size_mult, 3),
                "regime":  round(self.regime_size_mult, 3),
                "fg":      round(self.fg_size_mult, 3),
                "composed": round(self.composed_size_mult, 3),
            },
            "diag": {
                "rsi":              round(self.rsi, 1),
                "cvd_slope":        round(self.cvd_slope_value, 4),
                "volume_z":         round(self.volume_z_score, 2),
                "vwap_position":    self.vwap_position,
                "realized_vol_bps": round(self.realized_vol_bps, 1),
                "market_mode":      self.market_mode,
                "fg_regime":        self.fg_regime,
            },
        }


@dataclass
class SymbolBars:
    """Per-symbol OHLCV history slice — what the engine needs to score one symbol.

    All sequences must be aligned (same length, indexed oldest → newest).
    """
    symbol: str
    opens:   Sequence[float]
    highs:   Sequence[float]
    lows:    Sequence[float]
    closes:  Sequence[float]
    volumes: Sequence[float]


@dataclass
class MarketContext:
    """Cross-symbol + macro context shared across the universe."""
    btc_4h_closes:         Sequence[float] = field(default_factory=list)
    btc_4h_volumes:        Sequence[float] = field(default_factory=list)
    btc_daily_closes:      Sequence[float] = field(default_factory=list)
    btc_30d_return:        float = 0.0
    alts_30d_returns:      Sequence[float] = field(default_factory=list)
    symbol_recent_returns: dict[str, float] = field(default_factory=dict)
    fg_value:              float | None = None
    kyle_lambda_history:   dict[str, list[float]] = field(default_factory=dict)
    rv_bps_history:        dict[str, list[float]] = field(default_factory=dict)


# ─── Component scorers ──────────────────────────────────────────────────────

def _cvd_component(bars: SymbolBars, lookback: int = DEFAULT_CVD_LOOKBACK) -> tuple[float, float]:
    """Return (component_score 0-0.20, raw_slope) from CVD."""
    cvd = cumulative_volume_delta(
        bars.opens, bars.highs, bars.lows, bars.closes, bars.volumes,
    )
    if len(cvd) < lookback:
        return 0.0, 0.0
    raw = cvd_signal_score(cvd, lookback=lookback)
    # Empirical thresholds for normalized CVD slope in typical alt regimes
    if raw >= 0.20: return 0.20, raw   # strong buy pressure trend
    if raw >= 0.08: return 0.10, raw   # moderate buy pressure
    return 0.0, raw


def _vol_ignition_component(bars: SymbolBars,
                            lookback: int = DEFAULT_VOL_LOOKBACK) -> float:
    return volume_ignition_signal(bars.volumes, lookback=lookback)


def _micro_trend_component(bars: SymbolBars,
                           fast: int = DEFAULT_EMA_FAST_PERIOD,
                           slow: int = DEFAULT_EMA_SLOW_PERIOD) -> float:
    """EMA fast > EMA slow = micro-uptrend → 0.20.

    Mid-range (within 0.5% of crossover) → 0.10.
    """
    if len(bars.closes) < slow:
        return 0.0
    e_fast = ema(bars.closes[-(slow * 2):], fast)
    e_slow = ema(bars.closes[-(slow * 2):], slow)
    if e_slow <= 0:
        return 0.0
    diff_pct = (e_fast - e_slow) / e_slow
    if diff_pct >= 0.005:    return 0.20    # > 50 bps above
    if diff_pct >= 0.0:      return 0.10
    return 0.0


def _rsi_component(bars: SymbolBars,
                   period: int = 14,
                   oversold: float = DEFAULT_RSI_OVERSOLD,
                   overbought: float = DEFAULT_RSI_OVERBOUGHT) -> tuple[float, float]:
    """RSI-based filter:
        > 70 (overbought) → 0.0 (skip; chasing top)
        45-70 (mid)       → 0.20
        35-45 (mild OS)   → 0.10
        < 35 (deep OS)    → 0.20 (mean-reversion buy zone)
    """
    r = rsi(bars.closes, period=period)
    if r > overbought:        score = 0.0
    elif 45 < r <= overbought: score = 0.20
    elif oversold < r <= 45:   score = 0.10
    else:                      score = 0.20    # very oversold = MR setup
    return score, r


def _vwap_component(bars: SymbolBars) -> tuple[float, int]:
    """VWAP±σ band position → score.

    Position −1 (below VWAP-1σ, mild OS) → 0.20  (bounce setup)
    Position −2 (capitulation)            → 0.20
    Position +1 (mild OB, breakout)       → 0.10
    Position +2 (deep OB, late chase)     → 0.0
    Position 0  (within VWAP)             → 0.10
    """
    closes = list(bars.closes)
    volumes = list(bars.volumes)
    if len(closes) < 5:
        return 0.0, 0
    # Compute VWAP from the visible window (caller decides window length)
    pv = sum(c * v for c, v in zip(closes, volumes))
    vsum = sum(volumes)
    if vsum <= 0:
        return 0.0, 0
    vwap = pv / vsum
    # Std of bar prices in window
    n = len(closes)
    mean = sum(closes) / n
    var = sum((c - mean) ** 2 for c in closes) / n
    std = var ** 0.5
    pos = vwap_band_position(closes[-1], vwap, std)
    if pos == -2 or pos == -1: return 0.20, pos
    if pos == 1:               return 0.10, pos
    if pos == 0:               return 0.10, pos
    return 0.0, pos             # +2 = deep overbought


# ─── Composer ────────────────────────────────────────────────────────────────

def compute_scalp_score(
    bars: SymbolBars,
    context: MarketContext | None = None,
    *,
    entry_threshold:        float = DEFAULT_ENTRY_THRESHOLD,
    high_conviction_thres:  float = DEFAULT_HIGH_CONVICTION_THRES,
    cvd_lookback:           int   = DEFAULT_CVD_LOOKBACK,
    vol_lookback:           int   = DEFAULT_VOL_LOOKBACK,
    kyle_lookback:          int   = DEFAULT_KYLE_LOOKBACK,
    yz_lookback:            int   = DEFAULT_YZ_LOOKBACK,
    spillover_threshold:    float = 0.02,
    spillover_min_count:    int   = 5,
    spillover_boost:        float = 0.10,
) -> ScalpScore:
    """Compute the 5-component scalp score + size multipliers for one symbol.

    Args:
      bars: per-symbol OHLCV
      context: cross-symbol context (use empty/default for solo scoring)

    Returns ScalpScore with full breakdown.
    """
    ctx = context or MarketContext()
    out = ScalpScore(symbol=bars.symbol)

    # 1. CVD
    cvd_score, raw_slope = _cvd_component(bars, cvd_lookback)
    out.cvd_score = cvd_score
    out.cvd_slope_value = raw_slope

    # 2. Volume ignition
    out.vol_ignition = _vol_ignition_component(bars, vol_lookback)

    # 3. Micro-trend
    out.micro_trend = _micro_trend_component(bars)

    # 4. RSI filter
    out.rsi_filter, out.rsi = _rsi_component(bars)

    # 5. VWAP band
    out.vwap_signal, out.vwap_position = _vwap_component(bars)

    # 6. Spillover boost (cross-symbol momentum)
    if ctx.symbol_recent_returns:
        out.spillover_boost = cross_symbol_score_boost(
            bars.symbol, ctx.symbol_recent_returns,
            threshold_pct=spillover_threshold,
            min_count=spillover_min_count,
            boost=spillover_boost,
        )

    # Total
    out.raw_total = (
        out.cvd_score + out.vol_ignition + out.micro_trend
        + out.rsi_filter + out.vwap_signal + out.spillover_boost
    )
    out.score = max(0.0, min(1.0, out.raw_total))
    out.enter           = out.score >= entry_threshold
    out.high_conviction = out.score >= high_conviction_thres

    # ── Size multipliers ───────────────────────────────────────────────────
    # Kyle's λ
    cur_kyle = kyle_lambda(bars.closes, bars.volumes, lookback=kyle_lookback)
    history = ctx.kyle_lambda_history.get(bars.symbol, [])
    out.kyle_size_mult = kyle_regime_multiplier(cur_kyle, history)

    # Realized vol regime
    rv = realized_vol_bps(bars.opens, bars.highs, bars.lows, bars.closes,
                         lookback=yz_lookback)
    out.realized_vol_bps = rv
    rv_hist = ctx.rv_bps_history.get(bars.symbol, [])
    out.rv_size_mult = realized_vol_size_scaler(rv, rv_hist)

    # Regime composition
    market_mode = detect_market_mode(ctx.btc_4h_closes, ctx.btc_4h_volumes)
    out.market_mode = market_mode
    gc_dec = (golden_cross_regime(ctx.btc_daily_closes)
              if len(ctx.btc_daily_closes) >= 200 else None)
    btcd_dec = (btc_dominance_regime(ctx.btc_30d_return, ctx.alts_30d_returns)
                if ctx.alts_30d_returns else None)
    is_alt = "BTC" not in bars.symbol.upper()
    composed = compose_regime_size_multiplier(
        market_mode, gc_dec, btcd_dec, is_alt=is_alt,
    )
    out.regime_size_mult = composed["size_mult"]

    # Fear & Greed
    fg = FGSignal.from_value(ctx.fg_value)
    out.fg_size_mult = fg.size_multiplier
    out.fg_regime = fg.regime
    if fg.block_new_entries:
        out.enter = False
        out.high_conviction = False

    out.composed_size_mult = (
        out.kyle_size_mult * out.rv_size_mult
        * out.regime_size_mult * out.fg_size_mult
    )

    # Diagnostics: volume z-score (always compute for logging)
    from Pulse.features import trade_rate_burst_zscore
    out.volume_z_score = trade_rate_burst_zscore(bars.volumes, vol_lookback)

    return out


# ─── Batch scorer ────────────────────────────────────────────────────────────

def rank_candidates(
    candidates: list[SymbolBars],
    context: MarketContext | None = None,
    **kwargs,
) -> list[ScalpScore]:
    """Score every candidate; return them ranked best-first (by score).

    Only returns candidates whose `enter == True`.
    """
    scored = [compute_scalp_score(c, context, **kwargs) for c in candidates]
    valid = [s for s in scored if s.enter]
    valid.sort(key=lambda s: s.score, reverse=True)
    return valid
