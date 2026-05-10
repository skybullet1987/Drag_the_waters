"""Tests for Pulse.mr_engine."""

from __future__ import annotations

import random

import pytest

from Pulse.mr_engine import (
    DEFAULT_RSI_OVERSOLD, DEFAULT_TAKE_PROFIT_PCT, DEFAULT_STOP_LOSS_PCT,
    MREntrySignal, MRExitDecision, MRCandidateInput,
    evaluate_mr_entry, evaluate_mr_exit, rank_mr_candidates,
)


# ───────────────────────────────────────────────────────────────────────────────
# Synthetic helpers — capitulation-then-bounce scenarios
# ───────────────────────────────────────────────────────────────────────────────

def _flat_market(n=100):
    """Dead-flat market — should produce no MR entry."""
    return dict(
        opens=[100.0] * n, highs=[100.05] * n, lows=[99.95] * n,
        closes=[100.0] * n, volumes=[100.0] * n,
    )


def _capitulation_then_bounce(n=100):
    """Steady decline + final-bar volume spike + green close = perfect MR setup."""
    # Decline from 100 to 90 over n-1 bars
    closes = [100 - 10 * (i / (n - 2)) for i in range(n - 1)]
    # Last bar: green close above prior, big volume
    last_open = closes[-1]
    last_close = last_open * 1.005   # green +0.5%
    closes.append(last_close)

    opens  = [c + 0.05 for c in closes[:-1]] + [last_open]
    highs  = [c + 0.1 for c in closes]
    lows   = [c - 0.1 for c in closes]

    rnd = random.Random(0)
    volumes = [100 + rnd.gauss(0, 10) for _ in range(n - 1)] + [600.0]
    return dict(opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes)


def _capitulation_red_bar(n=100):
    """Like capitulation but the last bar is RED (no bounce yet) — should reject."""
    base = _capitulation_then_bounce(n)
    # Make the last bar red instead of green
    base["closes"][-1] = base["opens"][-1] - 0.01
    return base


# ───────────────────────────────────────────────────────────────────────────────
# evaluate_mr_entry — happy path
# ───────────────────────────────────────────────────────────────────────────────

def test_capitulation_bounce_enters():
    sig = evaluate_mr_entry(symbol="SOLUSD", **_capitulation_then_bounce())
    assert sig.enter
    assert sig.score > 0
    assert "rsi_oversold" in sig.reason


# ───────────────────────────────────────────────────────────────────────────────
# Rejection paths — each individual condition
# ───────────────────────────────────────────────────────────────────────────────

def test_flat_market_rejects_rsi_not_oversold():
    sig = evaluate_mr_entry(symbol="X", **_flat_market())
    assert not sig.enter
    assert "rsi_not_oversold" in sig.rejection_reason


def test_red_bar_rejects():
    sig = evaluate_mr_entry(symbol="X", **_capitulation_red_bar())
    assert not sig.enter
    assert sig.rejection_reason == "last_bar_red"


def test_no_volume_burst_rejects():
    """Even a perfect technical setup needs vol confirm."""
    base = _capitulation_then_bounce()
    base["volumes"][-1] = 100   # no burst
    sig = evaluate_mr_entry(symbol="X", **base)
    assert not sig.enter
    assert "vol_burst_too_small" in sig.rejection_reason


def test_insufficient_data_rejects():
    sig = evaluate_mr_entry(
        symbol="X", opens=[100], highs=[101], lows=[99],
        closes=[100], volumes=[100],
    )
    assert not sig.enter
    assert sig.rejection_reason == "insufficient_data"


# ───────────────────────────────────────────────────────────────────────────────
# Diagnostics fields
# ───────────────────────────────────────────────────────────────────────────────

def test_diagnostics_populated_on_entry():
    sig = evaluate_mr_entry(symbol="SOLUSD", **_capitulation_then_bounce())
    assert sig.rsi < DEFAULT_RSI_OVERSOLD
    assert sig.vwap_band_position <= -1   # below VWAP-1σ at minimum
    assert sig.volume_z_score > 1.0
    assert sig.last_bar_green


# ───────────────────────────────────────────────────────────────────────────────
# Fear & Greed boost
# ───────────────────────────────────────────────────────────────────────────────

def test_fg_low_boosts_score():
    base = _capitulation_then_bounce()
    no_boost = evaluate_mr_entry(symbol="SOLUSD", fg_value=50, **base)
    with_boost = evaluate_mr_entry(symbol="SOLUSD", fg_value=15, **base)
    assert with_boost.fg_fear_boost > 0
    assert with_boost.score >= no_boost.score
    assert with_boost.score - no_boost.score == pytest.approx(0.10, rel=1e-3)


def test_fg_high_no_boost():
    base = _capitulation_then_bounce()
    sig = evaluate_mr_entry(symbol="SOLUSD", fg_value=80, **base)
    assert sig.fg_fear_boost == 0


# ───────────────────────────────────────────────────────────────────────────────
# evaluate_mr_exit — exit cascade
# ───────────────────────────────────────────────────────────────────────────────

def test_exit_stop_loss():
    d = evaluate_mr_exit(entry_price=100, current_price=98.9,
                         held_hours=0.1, current_vwap=101.0)
    assert d.should_exit
    assert d.reason == "STOP_LOSS"


def test_exit_take_profit():
    d = evaluate_mr_exit(entry_price=100, current_price=102.5,
                         held_hours=0.1, current_vwap=101.0)
    assert d.should_exit
    assert d.reason == "TAKE_PROFIT"


def test_exit_vwap_touch():
    """Entry below VWAP, current price reached VWAP → exit."""
    d = evaluate_mr_exit(entry_price=98.5, current_price=100.0,
                         held_hours=0.5, current_vwap=100.0)
    assert d.should_exit
    assert d.reason == "VWAP_TOUCH"


def test_exit_time_stop():
    d = evaluate_mr_exit(entry_price=100, current_price=100.5,
                         held_hours=7.0, current_vwap=102.0)
    assert d.should_exit
    assert d.reason == "TIME_STOP"


def test_exit_hold_when_no_trigger():
    d = evaluate_mr_exit(entry_price=100, current_price=100.3,
                         held_hours=2.0, current_vwap=102.0)
    assert not d.should_exit
    assert d.reason == "hold"


def test_exit_priority_sl_over_tp():
    """If both SL and TP could trigger, SL wins (rare; protects capital)."""
    # PnL = -1.5% triggers SL; can't trigger TP at the same time, but
    # check the order of evaluation
    d = evaluate_mr_exit(entry_price=100, current_price=98.5,
                         held_hours=0.1, current_vwap=100.0)
    assert d.reason == "STOP_LOSS"


def test_exit_invalid_price_holds():
    d = evaluate_mr_exit(entry_price=0, current_price=100,
                         held_hours=1, current_vwap=100)
    assert not d.should_exit


# ───────────────────────────────────────────────────────────────────────────────
# rank_mr_candidates batch
# ───────────────────────────────────────────────────────────────────────────────

def test_rank_filters_non_enters():
    cands = [
        MRCandidateInput("FLAT", **_flat_market()),
        MRCandidateInput("CAPI", **_capitulation_then_bounce()),
        MRCandidateInput("RED",  **_capitulation_red_bar()),
    ]
    ranked = rank_mr_candidates(cands)
    syms = [s.symbol for s in ranked]
    assert "CAPI" in syms
    assert "FLAT" not in syms
    assert "RED" not in syms


def test_rank_sorted_descending_by_score():
    """Multiple capitulation setups — best score first."""
    base = _capitulation_then_bounce()
    cands = [
        MRCandidateInput(f"S{i}", **base) for i in range(3)
    ]
    ranked = rank_mr_candidates(cands)
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)
