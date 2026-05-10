"""Tests for backtest_audit.qc_orders."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from backtest_audit.log_parser import CompletedTrade
from backtest_audit.qc_orders import (
    QCOrder,
    parse_qc_orders, pair_qc_trades, fetch_backtest_trades,
    _extract_symbol, _extract_direction, _extract_time, _extract_status,
)


# ───────────────────────────────────────────────────────────────────────────────
# Field extractors
# ───────────────────────────────────────────────────────────────────────────────

def test_extract_symbol_dict():
    assert _extract_symbol({"value": "BTCUSD", "id": "x"}) == "BTCUSD"
    assert _extract_symbol({"Value": "ETHUSD"}) == "ETHUSD"


def test_extract_symbol_string():
    assert _extract_symbol("SOLUSD") == "SOLUSD"


def test_extract_direction_int():
    assert _extract_direction(0) == "Buy"
    assert _extract_direction(1) == "Sell"


def test_extract_direction_string():
    assert _extract_direction("Buy") == "Buy"
    assert _extract_direction("OrderDirection.Sell") == "Sell"


def test_extract_status_int_to_filled():
    assert _extract_status(3) == "Filled"
    assert _extract_status(4) == "Canceled"


def test_extract_status_string_dotted():
    assert _extract_status("OrderStatus.Filled") == "Filled"
    assert _extract_status("PartiallyFilled") == "PartiallyFilled"


def test_extract_time_iso_z():
    t = _extract_time("2025-03-15T12:34:00Z")
    assert t == datetime(2025, 3, 15, 12, 34, 0)


def test_extract_time_iso_with_microseconds():
    t = _extract_time("2025-03-15T12:34:56.789Z")
    assert t.microsecond > 0


def test_extract_time_invalid_raises():
    with pytest.raises(ValueError):
        _extract_time("not a timestamp")


# ───────────────────────────────────────────────────────────────────────────────
# parse_qc_orders
# ───────────────────────────────────────────────────────────────────────────────

def test_parse_filters_non_fills():
    """Submitted/Canceled/Invalid orders should NOT appear in output."""
    raw = [
        {"id": 1, "symbol": "BTCUSD", "status": 1, "direction": 0,
         "quantity": 1, "price": 50000, "time": "2025-03-15T12:00:00Z"},
        {"id": 2, "symbol": "BTCUSD", "status": 3, "direction": 0,
         "quantity": 1, "price": 50000, "time": "2025-03-15T12:01:00Z"},
        {"id": 3, "symbol": "BTCUSD", "status": 4, "direction": 0,
         "quantity": 1, "price": 50000, "time": "2025-03-15T12:02:00Z"},
    ]
    parsed = parse_qc_orders(raw)
    assert len(parsed) == 1
    assert parsed[0].order_id == "2"
    assert parsed[0].status == "Filled"


def test_parse_handles_lower_and_upper_case_keys():
    raw = [{"Id": 99, "Symbol": "ETH", "Status": "Filled", "Direction": 1,
            "Quantity": 0.5, "Price": 3000, "Time": "2025-04-01T00:00:00Z"}]
    parsed = parse_qc_orders(raw)
    assert len(parsed) == 1
    assert parsed[0].order_id == "99"
    assert parsed[0].direction == "Sell"


def test_parse_skips_malformed_silently():
    """A bad row shouldn't kill the whole parse."""
    raw = [
        {"id": 1, "symbol": "BTC", "status": "Filled", "direction": "Buy",
         "quantity": 1, "price": 100, "time": "2025-01-01T00:00:00Z"},
        {"id": "bad", "time": "garbage", "status": "Filled"},     # malformed
        {"id": 2, "symbol": "ETH", "status": "Filled", "direction": "Sell",
         "quantity": 1, "price": 100, "time": "2025-01-01T00:01:00Z"},
    ]
    parsed = parse_qc_orders(raw)
    assert len(parsed) == 2
    assert {p.order_id for p in parsed} == {"1", "2"}


def test_parse_negative_quantity_normalized_to_abs():
    raw = [{"id": 1, "symbol": "BTC", "status": "Filled", "direction": "Sell",
            "quantity": -1.5, "price": 100, "time": "2025-01-01T00:00:00Z"}]
    parsed = parse_qc_orders(raw)
    assert parsed[0].quantity == 1.5


# ───────────────────────────────────────────────────────────────────────────────
# pair_qc_trades — round-trip pairing
# ───────────────────────────────────────────────────────────────────────────────

def _o(oid, symbol, direction, qty, price, t_minutes, status="Filled"):
    return QCOrder(
        order_id=str(oid), symbol=symbol, status=status, direction=direction,
        quantity=qty, price=price,
        time=datetime(2025, 3, 15, 12, 0, 0) + timedelta(minutes=t_minutes),
        tag="",
    )


def test_pair_simple_round_trip():
    orders = [
        _o(1, "BTC", "Buy",  0.1, 50000, t_minutes=0),
        _o(2, "BTC", "Sell", 0.1, 50500, t_minutes=10),
    ]
    trades = pair_qc_trades(orders)
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "BTC"
    assert t.entry_price == 50000
    assert t.exit_price == 50500
    assert t.held_seconds == 600
    assert t.gross_pct == pytest.approx(0.01)


def test_pair_aggregates_partial_buys():
    """Two partial buys at different prices → averaged entry."""
    orders = [
        _o(1, "BTC", "Buy", 0.5, 50000, t_minutes=0,  status="PartiallyFilled"),
        _o(2, "BTC", "Buy", 0.5, 51000, t_minutes=2,  status="PartiallyFilled"),
        _o(3, "BTC", "Sell", 1.0, 52000, t_minutes=10),
    ]
    trades = pair_qc_trades(orders)
    assert len(trades) == 1
    # Avg entry = (0.5*50000 + 0.5*51000)/1.0 = 50500
    assert trades[0].entry_price == pytest.approx(50500)
    assert trades[0].entry_qty == pytest.approx(1.0)
    assert trades[0].exit_price == 52000
    # Entry timestamp is the EARLIEST of the partials
    assert trades[0].entry_ts.minute == 0


def test_pair_skips_unpaired_sells():
    """Sell with no prior buy on that symbol should be skipped, not crash."""
    orders = [
        _o(1, "BTC", "Sell", 0.1, 50000, t_minutes=0),   # no prior buy
        _o(2, "ETH", "Buy",  1, 3000, t_minutes=1),
        _o(3, "ETH", "Sell", 1, 3050, t_minutes=2),
    ]
    trades = pair_qc_trades(orders)
    assert len(trades) == 1
    assert trades[0].symbol == "ETH"


def test_pair_invalid_price_skipped():
    orders = [
        _o(1, "BTC", "Buy",  0.1, 0,      t_minutes=0),    # invalid
        _o(2, "BTC", "Buy",  0.1, 50000,  t_minutes=1),
        _o(3, "BTC", "Sell", 0.1, 50500,  t_minutes=10),
    ]
    trades = pair_qc_trades(orders)
    # Only the second buy is paired
    assert len(trades) == 1
    assert trades[0].entry_price == 50000


def test_pair_multiple_symbols_independent():
    orders = [
        _o(1, "BTC", "Buy",  0.1, 50000, t_minutes=0),
        _o(2, "ETH", "Buy",  1,   3000,  t_minutes=1),
        _o(3, "BTC", "Sell", 0.1, 50500, t_minutes=10),
        _o(4, "ETH", "Sell", 1,   3050,  t_minutes=11),
    ]
    trades = pair_qc_trades(orders)
    assert len(trades) == 2
    syms = {t.symbol for t in trades}
    assert syms == {"BTC", "ETH"}


def test_pair_open_position_not_emitted():
    """A Buy with no matching Sell stays open — no CompletedTrade emitted."""
    orders = [_o(1, "BTC", "Buy", 0.1, 50000, t_minutes=0)]
    trades = pair_qc_trades(orders)
    assert trades == []


def test_pair_returns_completed_trade_compatible_for_compare():
    """Output must be drop-in for compare.build_report's `backtest=` arg."""
    from backtest_audit.compare import build_report
    orders = [
        _o(1, "BTC", "Buy",  0.1, 100, t_minutes=0),
        _o(2, "BTC", "Sell", 0.1, 101, t_minutes=10),
    ]
    bt_trades = pair_qc_trades(orders)
    # Construct a synthetic 'live' that matches with worse fills
    live_trades = [CompletedTrade(
        symbol="BTC",
        entry_ts=datetime(2025, 3, 15, 12, 0),
        entry_price=100.5,    # paid 50bp more
        entry_qty=0.1, entry_oid="L1",
        exit_ts=datetime(2025, 3, 15, 12, 10),
        exit_price=100.5,     # got 50bp less
        exit_oid="L2",
        gross_pct=0.0, held_seconds=600,
        entry_slippage_bps=50, exit_slippage_bps=50,
        score=None, score_components={},
    )]
    rep = build_report(live_trades, bt_trades)
    assert rep.n_matched == 1
    # Live entry was 100.5 vs backtest 100 → 50bp gap
    assert rep.mean_entry_gap_bps == pytest.approx(50.0, rel=0.05)


# ───────────────────────────────────────────────────────────────────────────────
# fetch_backtest_trades — pagination
# ───────────────────────────────────────────────────────────────────────────────

def test_fetch_backtest_trades_paginates_and_pairs():
    """Mock client returns one page of orders; verify trades are paired."""
    client = MagicMock()
    client.read_backtest_orders.return_value = [
        {"id": 1, "symbol": "BTC", "status": 3, "direction": 0,
         "quantity": 0.1, "price": 50000, "time": "2025-03-15T12:00:00Z"},
        {"id": 2, "symbol": "BTC", "status": 3, "direction": 1,
         "quantity": 0.1, "price": 50500, "time": "2025-03-15T12:10:00Z"},
    ]
    trades = fetch_backtest_trades(client, project_id=99, backtest_id="bt001",
                                    page=1000)
    assert len(trades) == 1
    assert trades[0].symbol == "BTC"


def test_fetch_backtest_trades_handles_empty_response():
    client = MagicMock()
    client.read_backtest_orders.return_value = []
    trades = fetch_backtest_trades(client, 99, "bt001")
    assert trades == []


def test_fetch_backtest_trades_paginates_multiple_chunks():
    """When page is full, fetch the next."""
    client = MagicMock()
    chunks = [
        # First page: 1000 orders
        [{"id": i, "symbol": "BTC", "status": 3,
          "direction": 0 if i % 2 == 0 else 1,
          "quantity": 0.1, "price": 50000 + i,
          "time": f"2025-03-15T{(12+i//60):02d}:{(i%60):02d}:00Z"}
         for i in range(1000)],
        # Second page: 5 orders (less than page → terminate)
        [{"id": 1001, "symbol": "BTC", "status": 3, "direction": 1,
          "quantity": 0.1, "price": 50500,
          "time": "2025-03-16T12:00:00Z"}],
    ]
    client.read_backtest_orders.side_effect = chunks
    trades = fetch_backtest_trades(client, 99, "bt001", page=1000)
    # Just verify we asked for both pages
    assert client.read_backtest_orders.call_count == 2
