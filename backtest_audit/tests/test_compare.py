"""Tests for backtest_audit.compare."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from backtest_audit.log_parser import (
    parse_log_file, pair_trades, CompletedTrade,
)
from backtest_audit.compare import (
    build_report, pair_live_to_backtest, report_from_paired_log,
    TradePair, GapReport,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "mg36_paper_2026-03-16.txt"
)


# ───────────────────────────────────────────────────────────────────────────────
# Synthetic helpers
# ───────────────────────────────────────────────────────────────────────────────

def _trade(symbol, entry_ts, entry_px, exit_px,
           held_s=60, e_slip=0.0, x_slip=0.0, score=None):
    return CompletedTrade(
        symbol=symbol,
        entry_ts=entry_ts,
        entry_price=entry_px,
        entry_qty=1.0,
        entry_oid="x",
        exit_ts=entry_ts + timedelta(seconds=held_s),
        exit_price=exit_px,
        exit_oid="y",
        gross_pct=(exit_px - entry_px) / entry_px,
        held_seconds=held_s,
        entry_slippage_bps=e_slip,
        exit_slippage_bps=x_slip,
        score=score,
        score_components={},
    )


# ───────────────────────────────────────────────────────────────────────────────
# Pairing tests
# ───────────────────────────────────────────────────────────────────────────────

def test_perfect_match_one_trade():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 101.0)]
    bt   = [_trade("BTCUSD", ts, 100.0, 101.0)]
    pairs, lu, bu = pair_live_to_backtest(live, bt)
    assert len(pairs) == 1
    assert pairs[0].matched
    assert lu == [] and bu == []


def test_match_within_window():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 101.0)]
    bt   = [_trade("BTCUSD", ts + timedelta(minutes=5), 100.5, 101.5)]
    pairs, lu, bu = pair_live_to_backtest(live, bt, match_window_minutes=10)
    assert pairs[0].matched
    assert pairs[0].entry_price_gap_bps == pytest.approx(
        (100.0 - 100.5) / 100.5 * 10_000, rel=1e-3
    )


def test_no_match_outside_window():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 101.0)]
    bt   = [_trade("BTCUSD", ts + timedelta(hours=2), 100.0, 101.0)]
    pairs, lu, bu = pair_live_to_backtest(live, bt, match_window_minutes=30)
    assert len(pairs) == 2
    assert all(not p.matched for p in pairs)
    assert lu == [0] and bu == [0]


def test_different_symbols_dont_match():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 101.0)]
    bt   = [_trade("ETHUSD", ts, 100.0, 101.0)]
    pairs, lu, bu = pair_live_to_backtest(live, bt)
    assert len(pairs) == 2
    assert all(not p.matched for p in pairs)


def test_greedy_matches_closest_first():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [
        _trade("BTCUSD", ts, 100.0, 101.0),
        _trade("BTCUSD", ts + timedelta(minutes=15), 100.0, 101.0),
    ]
    bt = [
        _trade("BTCUSD", ts + timedelta(minutes=5),  100.0, 101.0),  # closer to live[0]
        _trade("BTCUSD", ts + timedelta(minutes=20), 100.0, 101.0),  # closer to live[1]
    ]
    pairs, lu, bu = pair_live_to_backtest(live, bt, match_window_minutes=30)
    assert sum(1 for p in pairs if p.matched) == 2
    assert lu == [] and bu == []


# ───────────────────────────────────────────────────────────────────────────────
# Gap report stats
# ───────────────────────────────────────────────────────────────────────────────

def test_report_no_gap_when_identical():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = bt = [_trade("BTCUSD", ts, 100.0, 101.0)]
    rep = build_report(live, bt)
    assert rep.n_matched == 1
    assert rep.mean_entry_gap_bps == 0.0
    assert rep.mean_exit_gap_bps == 0.0
    assert rep.mean_return_gap_pct == 0.0


def test_report_entry_gap_when_live_pays_more():
    """Live filled at 100.50, backtest at 100.00 — entry gap = +49.75bp."""
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.50, 102.0)]
    bt   = [_trade("BTCUSD", ts, 100.00, 102.0)]
    rep = build_report(live, bt)
    assert rep.n_matched == 1
    expected = (100.50 - 100.00) / 100.00 * 10_000   # ≈ 50.00 bp
    assert rep.mean_entry_gap_bps == pytest.approx(expected, rel=1e-3)


def test_report_return_gap_when_live_underperforms():
    """Backtest +1%, live +0.5% → return gap -0.5%."""
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 100.50)]   # +0.5%
    bt   = [_trade("BTCUSD", ts, 100.0, 101.00)]   # +1.0%
    rep = build_report(live, bt)
    assert rep.mean_return_gap_pct == pytest.approx(-0.5, rel=1e-3)
    assert rep.sum_return_gap_pct  == pytest.approx(-0.5, rel=1e-3)


def test_report_attribution_blames_slippage():
    """Live has 50bp entry+exit slippage, backtest has 20bp.
    Entry gap from slippage excess: +60bp round-trip = -0.6% return.
    """
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.50, 100.50, e_slip=50, x_slip=50)]
    bt   = [_trade("BTCUSD", ts, 100.00, 100.30, e_slip=20, x_slip=20)]
    rep = build_report(live, bt)
    # Slippage excess = -((50+50) - (20+20))/100 = -0.6%
    assert rep.attribution["slippage_excess"] == pytest.approx(-0.6, rel=1e-3)


def test_report_attribution_blames_missed_profit_bt_only():
    """Backtest took a +2% trade live missed → missed_profit_bt -2.0%."""
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = []
    bt   = [_trade("BTCUSD", ts, 100.0, 102.0)]
    rep = build_report(live, bt)
    assert rep.n_backtest_only == 1
    assert rep.attribution["missed_profit_bt"] == pytest.approx(-2.0, rel=1e-3)


def test_report_attribution_blames_extra_live_loser():
    """Live took a -3% trade backtest didn't → bad_extra_live -3.0%."""
    ts = datetime(2025, 1, 1, 12, 0, 0)
    live = [_trade("BTCUSD", ts, 100.0, 97.0)]
    bt   = []
    rep = build_report(live, bt)
    assert rep.n_live_only == 1
    assert rep.attribution["bad_extra_live"] == pytest.approx(-3.0, rel=1e-3)


def test_report_summary_text_contains_key_metrics():
    ts = datetime(2025, 1, 1, 12, 0, 0)
    rep = build_report(
        [_trade("BTCUSD", ts, 100.5, 100.5, e_slip=80, x_slip=80)],
        [_trade("BTCUSD", ts, 100.0, 101.0, e_slip=20, x_slip=20)],
    )
    text = rep.summary_text()
    assert "Backtest-vs-Live Gap Report" in text
    assert "Entry fill price gap" in text
    assert "slippage_excess" in text


# ───────────────────────────────────────────────────────────────────────────────
# MG36 fixture — real-world live-only smoke
# ───────────────────────────────────────────────────────────────────────────────

def test_report_from_mg36_live_only_log():
    """Run report against MG36 log with no backtest counterpart yet.
    All live trades should be unmatched and slippage stats should be high."""
    parsed = parse_log_file(FIXTURE)
    rep = report_from_paired_log(parsed)
    assert rep.n_live == 2          # KASUSD + AKTUSD round trips
    assert rep.n_backtest == 0
    assert rep.n_live_only == 2
    assert rep.n_matched == 0
    # Live slippage should be very high — this is the whole point of the audit
    assert rep.mean_live_entry_slip_bps >= 70
    assert rep.mean_live_exit_slip_bps  >= 100


def test_report_summary_text_renders_for_mg36():
    parsed = parse_log_file(FIXTURE)
    rep = report_from_paired_log(parsed)
    text = rep.summary_text()
    assert "live=2" in text
    assert "Live  mean slippage" in text
