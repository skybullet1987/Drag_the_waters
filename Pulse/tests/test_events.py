"""Tests for Pulse.events (pure-Python state machine)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.execution import OrderIntent
from Pulse.events import (
    OrderStatusName, OrderEventDTO, OrderAuditState, HandleResult,
    handle_order_event,
    rolling_win_rate, avg_win_size, avg_loss_size, expectancy,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _ev(status, side="Buy", qty=1.0, fill_qty=0.0, fill_price=0.0,
        symbol="BTCUSD", oid="ORDER_1",
        ts=datetime(2026, 5, 10, 12, 0, 0)):
    return OrderEventDTO(
        order_id=oid, symbol=symbol, status=status, direction=side,
        quantity=qty if side == "Buy" else -qty,
        fill_quantity=fill_qty if side == "Buy" else -fill_qty,
        fill_price=fill_price, timestamp=ts,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Submitted → bookkeeping only
# ───────────────────────────────────────────────────────────────────────────────

def test_submitted_records_pending():
    s = OrderAuditState()
    res = handle_order_event(s, _ev(OrderStatusName.SUBMITTED.value))
    assert res.action == "submitted"
    assert s.pending_orders["BTCUSD"] == 1.0
    assert "BTCUSD" in s.submitted_orders


# ───────────────────────────────────────────────────────────────────────────────
# Filled BUY → entry recorded
# ───────────────────────────────────────────────────────────────────────────────

def test_filled_buy_records_entry():
    s = OrderAuditState()
    handle_order_event(s, _ev(OrderStatusName.SUBMITTED.value))
    res = handle_order_event(
        s,
        _ev(OrderStatusName.FILLED.value, side="Buy", qty=1.0,
            fill_qty=1.0, fill_price=50_000.0),
    )
    assert res.action == "entry_recorded"
    assert s.entry_prices["BTCUSD"] == 50_000.0
    assert s.highest_prices["BTCUSD"] == 50_000.0
    assert s.daily_trade_count == 1
    assert "BTCUSD" not in s.pending_orders


def test_filled_buy_with_reference_price_logs_slippage():
    s = OrderAuditState()
    res = handle_order_event(
        s,
        _ev(OrderStatusName.FILLED.value, side="Buy", qty=1.0,
            fill_qty=1.0, fill_price=50_500.0),
        reference_price=50_000.0,
    )
    assert res.slippage_logged
    assert len(s.slippage_log) == 1
    assert s.slippage_log[0]["slippage_bps"] == pytest.approx(100.0, rel=0.01)


# ───────────────────────────────────────────────────────────────────────────────
# Filled SELL → exit recorded with PnL accounting
# ───────────────────────────────────────────────────────────────────────────────

def test_filled_sell_records_winner():
    s = OrderAuditState(estimated_round_trip_fee=0.0)  # ignore fees in this test
    handle_order_event(
        s, _ev(OrderStatusName.FILLED.value, side="Buy", qty=1, fill_qty=1,
               fill_price=100.0),
    )
    res = handle_order_event(
        s, _ev(OrderStatusName.FILLED.value, side="Sell", qty=1, fill_qty=1,
               fill_price=110.0),
    )
    assert res.action == "exit_recorded"
    assert res.is_winner
    assert res.pnl_pct == pytest.approx(0.10)
    assert s.winning_trades == 1
    assert s.losing_trades == 0
    assert s.consecutive_losses == 0
    assert s.total_pnl == pytest.approx(0.10)
    assert "BTCUSD" not in s.entry_prices    # state cleared


def test_filled_sell_records_loser():
    s = OrderAuditState(estimated_round_trip_fee=0.0)
    handle_order_event(
        s, _ev(OrderStatusName.FILLED.value, side="Buy", qty=1, fill_qty=1,
               fill_price=100.0),
    )
    res = handle_order_event(
        s, _ev(OrderStatusName.FILLED.value, side="Sell", qty=1, fill_qty=1,
               fill_price=98.0),
    )
    assert res.action == "exit_recorded"
    assert not res.is_winner
    assert s.consecutive_losses == 1
    assert s.losing_trades == 1


def test_consecutive_losses_resets_on_winner():
    s = OrderAuditState(estimated_round_trip_fee=0.0)
    # Three losers
    for _ in range(3):
        handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Buy",
                                  qty=1, fill_qty=1, fill_price=100.0))
        handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Sell",
                                  qty=1, fill_qty=1, fill_price=98.0))
    assert s.consecutive_losses == 3
    # One winner resets
    handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Buy",
                              qty=1, fill_qty=1, fill_price=100.0))
    handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Sell",
                              qty=1, fill_qty=1, fill_price=110.0))
    assert s.consecutive_losses == 0


def test_unpaired_sell_clears_state_safely():
    """Sell event with no prior entry — should clear state without crashing."""
    s = OrderAuditState()
    res = handle_order_event(
        s,
        _ev(OrderStatusName.FILLED.value, side="Sell", qty=1, fill_qty=1,
            fill_price=100.0),
    )
    assert res.action == "exit_unpaired"
    assert res.cleared_state


def test_fee_deduction_applied():
    """Default round-trip fee 80bp; +50bp gross win → -30bp net."""
    s = OrderAuditState(estimated_round_trip_fee=0.0080)
    handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Buy",
                              qty=1, fill_qty=1, fill_price=100.0))
    res = handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Sell",
                                    qty=1, fill_qty=1, fill_price=100.5))
    assert res.pnl_pct == pytest.approx(0.005 - 0.008)
    assert not res.is_winner   # net is negative after fees


# ───────────────────────────────────────────────────────────────────────────────
# Partially filled
# ───────────────────────────────────────────────────────────────────────────────

def test_partial_fill_records_entry_when_first_buy():
    """Partial-fill BUY without prior entry should record entry from first partial."""
    s = OrderAuditState()
    handle_order_event(s, _ev(OrderStatusName.SUBMITTED.value, qty=10))
    res = handle_order_event(
        s, _ev(OrderStatusName.PARTIALLY_FILLED.value, side="Buy",
               qty=10, fill_qty=4, fill_price=100.0),
    )
    assert res.action == "partial_fill"
    assert s.entry_prices["BTCUSD"] == 100.0
    # Pending should reflect remaining: 10 - 4 = 6
    assert s.pending_orders["BTCUSD"] == pytest.approx(6.0)


def test_partial_fill_completes_pending():
    s = OrderAuditState()
    handle_order_event(s, _ev(OrderStatusName.SUBMITTED.value, qty=10))
    handle_order_event(
        s, _ev(OrderStatusName.PARTIALLY_FILLED.value, side="Buy",
               qty=10, fill_qty=10, fill_price=100.0),
    )
    assert "BTCUSD" not in s.pending_orders


# ───────────────────────────────────────────────────────────────────────────────
# Canceled / Invalid
# ───────────────────────────────────────────────────────────────────────────────

def test_canceled_clears_pending():
    s = OrderAuditState()
    handle_order_event(s, _ev(OrderStatusName.SUBMITTED.value))
    res = handle_order_event(s, _ev(OrderStatusName.CANCELED.value))
    assert res.action == "canceled"
    assert "BTCUSD" not in s.pending_orders


def test_invalid_increments_failed_exit():
    s = OrderAuditState()
    s.entry_prices["BTCUSD"] = 50_000.0  # have an open position
    res = handle_order_event(s, _ev(OrderStatusName.INVALID.value, side="Sell"))
    assert res.action == "invalid"
    assert s.failed_exit_counts["BTCUSD"] == 1


def test_invalid_force_cleanup_after_10():
    """Threshold raised from 3 → 10 to avoid abandoning positions after
    a transient QC data-gap (e.g. one bad bar where Securities[sym].Price
    momentarily reads 0). 10 retries spans many bars; truly stuck position
    will fail past that, transient gaps will recover."""
    s = OrderAuditState()
    s.entry_prices["BTCUSD"] = 50_000.0
    s.highest_prices["BTCUSD"] = 50_000.0
    # Below 10: state preserved
    for _ in range(9):
        handle_order_event(s, _ev(OrderStatusName.INVALID.value, side="Sell"))
    assert "BTCUSD" in s.entry_prices
    # 10th: force cleanup
    handle_order_event(s, _ev(OrderStatusName.INVALID.value, side="Sell"))
    assert "BTCUSD" not in s.entry_prices


# ───────────────────────────────────────────────────────────────────────────────
# Stat helpers
# ───────────────────────────────────────────────────────────────────────────────

def test_rolling_win_rate_none_on_empty():
    assert rolling_win_rate(OrderAuditState()) is None


def test_rolling_win_rate_basic():
    s = OrderAuditState()
    s.recent_outcomes.extend([1, 1, 0, 1, 0])
    assert rolling_win_rate(s) == pytest.approx(0.6)


def test_avg_win_loss_zero_on_empty():
    s = OrderAuditState()
    assert avg_win_size(s) == 0.0
    assert avg_loss_size(s) == 0.0


def test_expectancy_basic():
    s = OrderAuditState()
    s.recent_outcomes.extend([1, 1, 0, 1, 0])  # WR 0.6
    s.rolling_win_sizes.extend([0.05, 0.04, 0.06])  # avg 0.05
    s.rolling_loss_sizes.extend([0.02, 0.03])       # avg 0.025
    # E = 0.6 * 0.05 - 0.4 * 0.025 = 0.03 - 0.01 = 0.02
    assert expectancy(s) == pytest.approx(0.02)


# ───────────────────────────────────────────────────────────────────────────────
# Cash-mode trigger hook
# ───────────────────────────────────────────────────────────────────────────────

def test_cash_mode_trigger_fires_on_low_wr():
    triggered = []

    def trigger(state, ts, wr):
        triggered.append((ts, wr))

    s = OrderAuditState(estimated_round_trip_fee=0.0)
    # Generate 16 trades, only 1 winner → WR ~6%
    for i in range(16):
        handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Buy",
                                  qty=1, fill_qty=1, fill_price=100.0,
                                  oid=f"OID_BUY_{i}"))
        # 15 losers, 1 winner
        sell_px = 95.0 if i < 15 else 110.0
        handle_order_event(
            s,
            _ev(OrderStatusName.FILLED.value, side="Sell", qty=1, fill_qty=1,
                fill_price=sell_px, oid=f"OID_SELL_{i}"),
            on_cash_mode_trigger=trigger,
            cash_mode_wr_threshold=0.15,
        )
    assert len(triggered) >= 1
    assert triggered[-1][1] < 0.15


def test_cash_mode_does_not_trigger_when_wr_healthy():
    triggered = []

    def trigger(state, ts, wr):
        triggered.append(wr)

    s = OrderAuditState(estimated_round_trip_fee=0.0)
    for i in range(16):
        handle_order_event(s, _ev(OrderStatusName.FILLED.value, side="Buy",
                                  qty=1, fill_qty=1, fill_price=100.0,
                                  oid=f"BUY_{i}"))
        # All winners
        handle_order_event(
            s,
            _ev(OrderStatusName.FILLED.value, side="Sell", qty=1, fill_qty=1,
                fill_price=110.0, oid=f"SELL_{i}"),
            on_cash_mode_trigger=trigger,
            cash_mode_wr_threshold=0.15,
        )
    assert triggered == []


# ───────────────────────────────────────────────────────────────────────────────
# Slippage attribution end-to-end (MG36 scenario)
# ───────────────────────────────────────────────────────────────────────────────

def test_kasusd_live_scenario_slip_logged():
    """Reproduce MG36 KASUSD round trip with slippage warnings logged."""
    s = OrderAuditState()
    handle_order_event(
        s,
        _ev(OrderStatusName.FILLED.value, side="Buy", qty=750, fill_qty=750,
            fill_price=0.04057, symbol="KASUSD", oid="ENTRY"),
        reference_price=0.04014,   # signal price was lower
    )
    handle_order_event(
        s,
        _ev(OrderStatusName.FILLED.value, side="Sell", qty=750, fill_qty=750,
            fill_price=0.04014, symbol="KASUSD", oid="EXIT"),
        reference_price=0.04094,   # signal price was higher
    )
    assert len(s.slippage_log) == 2
    # Buy slip ~107bp, sell slip ~196bp
    assert s.slippage_log[0]["slippage_bps"] >= 100
    assert s.slippage_log[1]["slippage_bps"] >= 190
