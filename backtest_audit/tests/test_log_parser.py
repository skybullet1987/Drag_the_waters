"""Tests for backtest_audit.log_parser, anchored to the MG36 fixture."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from backtest_audit.log_parser import (
    parse_log,
    parse_log_file,
    pair_trades,
    ParsedLog,
    ScalpEntry,
    OrderEvent,
    ExitEvent,
    SlippageWarning,
    MakerLimit,
    Snapshot,
    CompletedTrade,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "mg36_paper_2026-03-16.txt"
)


# ───────────────────────────────────────────────────────────────────────────────
# Synthetic line tests — each pattern in isolation
# ───────────────────────────────────────────────────────────────────────────────

def test_parse_scalp_entry():
    log = "2026-03-18 03:53:00 SCALP ENTRY: KASUSD | score=0.95 | $29.88 | obi=0.20 vol=0.20 trend=0.20 adx=0.15 mean_rev=0.00 vwap=0.20"
    p = parse_log(log)
    assert len(p.entries) == 1
    e = p.entries[0]
    assert e.symbol == "KASUSD"
    assert e.score == 0.95
    assert e.price == 29.88
    assert e.components == {
        "obi": 0.20, "vol": 0.20, "trend": 0.20,
        "adx": 0.15, "mean_rev": 0.00, "vwap": 0.20,
    }
    assert e.ts == datetime(2026, 3, 18, 3, 53, 0)


def test_parse_order_filled():
    log = "2026-03-18 03:54:00 ORDER: KASUSD Filled Buy qty=736.38156 price=0.04057 id=3"
    p = parse_log(log)
    assert len(p.orders) == 1
    o = p.orders[0]
    assert o.symbol == "KASUSD"
    assert o.status == "Filled"
    assert o.side == "Buy"
    assert o.qty == pytest.approx(736.38156)
    assert o.price == pytest.approx(0.04057)
    assert o.order_id == "3"


def test_parse_order_canceled():
    log = "2026-03-18 02:02:00 ORDER: FARTCOINUSD Canceled Buy qty=110.32973 price=0.0 id=1"
    p = parse_log(log)
    assert len(p.orders) == 1
    assert p.orders[0].status == "Canceled"


def test_parse_exit_atr_trail():
    log = "2026-03-18 03:54:00 ATR Trail: KASUSD | PnL:-2.29% | Held:0.0h"
    p = parse_log(log)
    assert len(p.exits) == 1
    x = p.exits[0]
    assert x.symbol == "KASUSD"
    assert x.reason == "ATR Trail"
    assert x.pnl_pct == -2.29
    assert x.held_hours == 0.0


def test_parse_slippage_warning():
    log = "2026-03-18 03:54:00 ⚠️ HIGH SLIPPAGE: KASUSD | 1.0713% | dir=Buy"
    p = parse_log(log)
    assert len(p.slippage) == 1
    s = p.slippage[0]
    assert s.symbol == "KASUSD"
    assert s.slippage_pct == pytest.approx(0.010713)
    assert s.direction == "Buy"


def test_parse_maker_limit():
    log = "2026-03-18 03:53:00 MAKER LIMIT: KASUSD | qty=736.38156 | bid=$0.0406 | limit=$0.0406 | timeout=30s"
    p = parse_log(log)
    assert len(p.maker_limits) == 1
    m = p.maker_limits[0]
    assert m.symbol == "KASUSD"
    assert m.qty == pytest.approx(736.38156)
    assert m.bid == pytest.approx(0.0406)
    assert m.timeout_s == 30


def test_parse_snapshot_full():
    log = (
        "2026-03-18 22:17:46 Trades: 219 | WR: 38.1%\n"
        "2026-03-18 22:17:46 Final: $119.14\n"
        "2026-03-18 22:17:46 PnL: -5.24%"
    )
    p = parse_log(log)
    assert len(p.snapshots) == 1
    s = p.snapshots[0]
    assert s.trades_total == 219
    assert s.win_rate_pct == 38.1
    assert s.equity == pytest.approx(119.14)
    assert s.pnl_pct == pytest.approx(-5.24)


def test_parse_snapshot_with_avg():
    log = "2026-03-17 00:01:00 Trades: 153 | WR: 37.9% | Avg: -0.02%"
    p = parse_log(log)
    assert len(p.snapshots) == 1
    s = p.snapshots[0]
    assert s.trades_total == 153
    assert s.win_rate_pct == 37.9
    assert s.avg_pct == pytest.approx(-0.02)


def test_parse_unknown_line_ignored():
    log = "2026-03-18 21:53:00 REBALANCE: 8/72 above thresh=0.50 | cash=$119.14"
    p = parse_log(log)
    # No records produced; line counted as raw but no errors
    assert len(p.entries) == 0
    assert len(p.orders) == 0
    assert p.raw_line_count == 1
    assert p.parse_errors == []


# ───────────────────────────────────────────────────────────────────────────────
# MG36 fixture — known invariants
# ───────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mg36() -> ParsedLog:
    return parse_log_file(FIXTURE)


def test_fixture_loads_and_no_parse_errors(mg36):
    assert mg36.raw_line_count > 4000  # full log is ~4003 lines
    assert mg36.parse_errors == []


def test_fixture_scalp_entries_count_and_signature(mg36):
    """We saw exactly 4 SCALP ENTRY events in manual inspection."""
    assert len(mg36.entries) == 4
    syms = [e.symbol for e in mg36.entries]
    assert syms == ["FARTCOINUSD", "KASUSD", "KASUSD", "AKTUSD"]
    # Highest-conviction entry was KASUSD score=0.95
    max_entry = max(mg36.entries, key=lambda e: e.score)
    assert max_entry.symbol == "KASUSD"
    assert max_entry.score == 0.95


def test_fixture_order_event_counts(mg36):
    """Manual count: 14 ORDER lines total — 4 Filled, 6 Submitted,
    2 CancelPending, 2 Canceled."""
    assert len(mg36.orders) == 14
    by_status = {}
    for o in mg36.orders:
        by_status[o.status] = by_status.get(o.status, 0) + 1
    assert by_status.get("Filled", 0) == 4
    assert by_status.get("Submitted", 0) == 6
    assert by_status.get("CancelPending", 0) == 2
    assert by_status.get("Canceled", 0) == 2


def test_fixture_exit_events(mg36):
    """Two ATR Trail exits, both losses."""
    assert len(mg36.exits) == 2
    for x in mg36.exits:
        assert x.reason == "ATR Trail"
        assert x.pnl_pct < 0


def test_fixture_slippage_warnings(mg36):
    """Four ⚠️ HIGH SLIPPAGE events, all >65bp, two >100bp (the live finding)."""
    assert len(mg36.slippage) == 4
    bps = sorted([w.slippage_pct * 10_000 for w in mg36.slippage])
    # All warnings exceeded 65bp; max was 1.97% = 197bp
    assert bps[0] >= 60.0
    assert bps[-1] >= 190.0
    assert bps[-1] <= 200.0


def test_fixture_maker_limits_all_30s(mg36):
    """All maker limits used 30s timeout."""
    assert len(mg36.maker_limits) == 4
    for m in mg36.maker_limits:
        assert m.timeout_s == 30


def test_fixture_final_snapshot(mg36):
    """Final state at 2026-03-18 22:17:46: 219 trades, 38.1% WR, $119.14, -5.24%."""
    final = mg36.snapshots[-1]
    assert final.trades_total == 219
    assert final.win_rate_pct == pytest.approx(38.1)
    assert final.equity == pytest.approx(119.14)
    assert final.pnl_pct == pytest.approx(-5.24)


def test_fixture_summary_matches_evidence(mg36):
    """Top-level summary anchors the live evidence in PLAN.md §0.A."""
    s = mg36.summary()
    assert s["scalp_entries"] == 4
    assert s["orders_filled"] == 4
    assert s["orders_canceled"] == 2  # plus 2 CancelPending — see test_fixture_order_event_counts
    assert s["exits"] == 2
    assert s["slippage_warnings"] == 4
    assert s["mean_slippage_bps"] >= 60   # mean across the 4 fills
    assert s["max_slippage_bps"] >= 190
    assert s["final_trades"] == 219
    assert s["final_win_rate"] == pytest.approx(38.1)
    assert s["final_pnl_pct"] == pytest.approx(-5.24)


# ───────────────────────────────────────────────────────────────────────────────
# Round-trip trade reconstruction
# ───────────────────────────────────────────────────────────────────────────────

def test_pair_trades_finds_two_completed_trades(mg36):
    """KASUSD and AKTUSD both completed entry+exit cycles in the log."""
    trades = pair_trades(mg36)
    assert len(trades) == 2
    by_sym = {t.symbol: t for t in trades}
    assert "KASUSD" in by_sym
    assert "AKTUSD" in by_sym


def test_pair_trades_kasusd_economics(mg36):
    """KASUSD trade: entry 0.04057, exit 0.0406, 60s held, +0.07% gross
    but slippage 107bp+197bp = 304bp round-trip → catastrophic net.
    """
    trades = pair_trades(mg36)
    kas = next(t for t in trades if t.symbol == "KASUSD")

    assert kas.entry_price == pytest.approx(0.04057)
    assert kas.exit_price == pytest.approx(0.0406)
    assert kas.held_seconds == pytest.approx(60.0)
    # Gross is essentially flat
    assert -0.001 < kas.gross_pct < 0.001
    # Slippage attribution captured
    assert kas.entry_slippage_bps is not None
    assert kas.entry_slippage_bps >= 100
    assert kas.exit_slippage_bps is not None
    assert kas.exit_slippage_bps >= 190
    # Score=0.95 from preceding SCALP ENTRY
    assert kas.score == 0.95
    assert kas.score_components.get("obi", 0) == 0.20


def test_pair_trades_akt_held_longer_than_kas(mg36):
    """AKTUSD held ~12 minutes vs KASUSD's 1 minute."""
    trades = pair_trades(mg36)
    by_sym = {t.symbol: t for t in trades}
    assert by_sym["AKTUSD"].held_seconds > by_sym["KASUSD"].held_seconds
    assert by_sym["AKTUSD"].held_seconds > 600  # >10 min
