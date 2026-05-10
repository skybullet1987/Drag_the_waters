"""mr_engine — mean-reversion sub-strategy (sub-strategy 3 of the portfolio).

Designed to harvest chop regimes the way scalp/trend engines can't:
  - Scalp engine wants directional momentum
  - Trend engine wants sustained bull
  - MR engine wants oversold + capitulation volume + bounce confirmation

Entry conditions (all must hold) per PLAN.md §6.A.2:
  - RSI < 30 (oversold)
  - Price below VWAP - 2σ (capitulation)
  - Last bar GREEN with volume z-score >= 1.0 (bounce starting)
  - Optional: F&G in fear territory boosts conviction

Exit:
  - Take profit at first VWAP touch OR +2% (whichever first)
  - Stop loss tight: -1%
  - Time stop: 6 hours

Position sizing: 10-30% of capital per PLAN.md §6.A.3 (allocator decides).

This module is pure-Python; the QC integration goes through PulseAlgorithm's
sub-strategy plumbing (Phase 4.A.3 portfolio.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from Pulse.features import (
    rsi, VWAPState, vwap_band_position,
    trade_rate_burst_zscore,
)


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_RSI_OVERSOLD               = 30.0
DEFAULT_VWAP_BAND_REQUIRED         = -1     # below VWAP - 1σ (capitulation-ish)
                                            # Set to -2 in config if you want
                                            # only deep-capitulation entries.
DEFAULT_VOL_BURST_MIN_Z            = 1.0    # z-score threshold for volume confirm
DEFAULT_BOUNCE_BAR_REQUIRED        = True   # last bar must be green
DEFAULT_TAKE_PROFIT_PCT            = 0.02
DEFAULT_STOP_LOSS_PCT              = 0.01   # very tight: MR loses fast or wins fast
DEFAULT_TIME_STOP_HOURS            = 6.0
DEFAULT_FG_FEAR_BOOST_BELOW        = 30     # F&G < 30 boosts entry conviction
DEFAULT_FG_BOOST_AMOUNT            = 0.10   # added to score when F&G low


# ─── Output dataclasses ──────────────────────────────────────────────────────

@dataclass
class MREntrySignal:
    """Mean-reversion entry decision + diagnostics."""
    symbol:     str
    enter:      bool = False
    score:      float = 0.0    # 0.0 - 1.0; for ranking when slots are limited
    reason:     str = ""       # primary trigger reason
    rejection_reason: str = "" # why we didn't enter (if not enter)

    # Diagnostics
    rsi:                float = 50.0
    vwap_band_position: int = 0
    volume_z_score:     float = 0.0
    last_bar_green:     bool = False
    fg_fear_boost:      float = 0.0


@dataclass
class MRExitDecision:
    """Per-tick exit decision for an open MR position."""
    should_exit: bool = False
    reason:      str = ""
    pnl_pct:     float = 0.0


# ─── Helpers ────────────────────────────────────────────────────────────────

def _bar_is_green(open_p: float, close: float) -> bool:
    return close > open_p


# ─── Entry signal ────────────────────────────────────────────────────────────

def evaluate_mr_entry(
    *,
    symbol: str,
    opens:   Sequence[float],
    highs:   Sequence[float],
    lows:    Sequence[float],
    closes:  Sequence[float],
    volumes: Sequence[float],
    fg_value: float | None = None,
    rsi_period:               int   = 14,
    rsi_oversold:             float = DEFAULT_RSI_OVERSOLD,
    vwap_band_required:       int   = DEFAULT_VWAP_BAND_REQUIRED,
    vol_burst_min_z:          float = DEFAULT_VOL_BURST_MIN_Z,
    require_green_bar:        bool  = DEFAULT_BOUNCE_BAR_REQUIRED,
    fg_fear_boost_below:      float = DEFAULT_FG_FEAR_BOOST_BELOW,
    fg_boost_amount:          float = DEFAULT_FG_BOOST_AMOUNT,
    vol_lookback:             int   = 60,
) -> MREntrySignal:
    """Evaluate MR entry for one symbol.

    Returns MREntrySignal with `enter=True` only if all required conditions hold.
    """
    sig = MREntrySignal(symbol=symbol)

    if (len(closes) < max(rsi_period + 5, vol_lookback + 1)
            or len(opens) != len(closes)):
        sig.rejection_reason = "insufficient_data"
        return sig

    # 1. RSI
    sig.rsi = rsi(closes, period=rsi_period)
    if sig.rsi >= rsi_oversold:
        sig.rejection_reason = f"rsi_not_oversold({sig.rsi:.1f}>={rsi_oversold})"
        return sig

    # 2. VWAP band position (compute over recent window)
    pv = sum(c * v for c, v in zip(closes, volumes))
    vsum = sum(volumes)
    if vsum <= 0:
        sig.rejection_reason = "no_volume"
        return sig
    vwap = pv / vsum
    n = len(closes)
    mean = sum(closes) / n
    var = sum((c - mean) ** 2 for c in closes) / n
    std = var ** 0.5
    sig.vwap_band_position = vwap_band_position(closes[-1], vwap, std)
    if sig.vwap_band_position > vwap_band_required:
        sig.rejection_reason = (
            f"vwap_band_pos_too_high({sig.vwap_band_position}>{vwap_band_required})"
        )
        return sig

    # 3. Volume burst
    sig.volume_z_score = trade_rate_burst_zscore(volumes, lookback=vol_lookback)
    if sig.volume_z_score < vol_burst_min_z:
        sig.rejection_reason = (
            f"vol_burst_too_small(z={sig.volume_z_score:.2f}"
            f"<{vol_burst_min_z})"
        )
        return sig

    # 4. Last bar green (bounce confirmation)
    sig.last_bar_green = _bar_is_green(opens[-1], closes[-1])
    if require_green_bar and not sig.last_bar_green:
        sig.rejection_reason = "last_bar_red"
        return sig

    # All conditions passed → entry
    sig.enter = True
    sig.reason = "rsi_oversold + vwap_capitulation + vol_burst + green_bounce"

    # Score components (used for ranking when slots are limited)
    score = 0.0
    score += min(0.25, (rsi_oversold - sig.rsi) / rsi_oversold * 0.5)
    score += 0.25 if sig.vwap_band_position == -2 else 0.15
    score += min(0.25, (sig.volume_z_score - vol_burst_min_z) / 5.0 * 0.5)
    score += 0.15 if sig.last_bar_green else 0.0

    # F&G boost
    if fg_value is not None and fg_value <= fg_fear_boost_below:
        sig.fg_fear_boost = fg_boost_amount
        score += fg_boost_amount

    sig.score = max(0.0, min(1.0, score))
    return sig


# ─── Exit decision ───────────────────────────────────────────────────────────

def evaluate_mr_exit(
    *,
    entry_price: float,
    current_price: float,
    held_hours: float,
    current_vwap: float,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    stop_loss_pct:   float = DEFAULT_STOP_LOSS_PCT,
    time_stop_hours: float = DEFAULT_TIME_STOP_HOURS,
) -> MRExitDecision:
    """Decide whether to exit an open MR position.

    Exit cascade (priority order):
      1. STOP_LOSS at -1%
      2. TAKE_PROFIT at +2%
      3. VWAP_TOUCH (price reached VWAP from below)
      4. TIME_STOP after 6 hours
    """
    if entry_price <= 0 or current_price <= 0:
        return MRExitDecision(should_exit=False, reason="invalid_price")

    pnl = (current_price - entry_price) / entry_price

    if pnl <= -stop_loss_pct:
        return MRExitDecision(True, "STOP_LOSS", pnl)
    if pnl >= take_profit_pct:
        return MRExitDecision(True, "TAKE_PROFIT", pnl)
    # VWAP touch from below: entry was below VWAP; exit when current touches/crosses
    if entry_price < current_vwap and current_price >= current_vwap:
        return MRExitDecision(True, "VWAP_TOUCH", pnl)
    if held_hours >= time_stop_hours:
        return MRExitDecision(True, "TIME_STOP", pnl)
    return MRExitDecision(False, "hold", pnl)


# ─── Batch ranking ───────────────────────────────────────────────────────────

@dataclass
class MRCandidateInput:
    """One symbol's data feed for batch evaluation."""
    symbol:  str
    opens:   Sequence[float]
    highs:   Sequence[float]
    lows:    Sequence[float]
    closes:  Sequence[float]
    volumes: Sequence[float]


def rank_mr_candidates(
    candidates: list[MRCandidateInput],
    fg_value: float | None = None,
    **kwargs,
) -> list[MREntrySignal]:
    """Score every candidate; return enter==True signals sorted by score."""
    out: list[MREntrySignal] = []
    for c in candidates:
        sig = evaluate_mr_entry(
            symbol=c.symbol,
            opens=c.opens, highs=c.highs, lows=c.lows,
            closes=c.closes, volumes=c.volumes,
            fg_value=fg_value, **kwargs,
        )
        if sig.enter:
            out.append(sig)
    out.sort(key=lambda s: s.score, reverse=True)
    return out
