"""Tests for Pulse.execution (pure-Python core)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.execution import (
    OrderIntent,
    KRAKEN_MIN_QTY_FALLBACK, min_quantity_fallback,
    round_to_lot, validate_min_notional,
    safe_sell_quantity,
    compute_slippage_pct, slippage_log_entry, SlippageLogEntry,
    PendingOrder, stale_pending_orders,
)


# ───────────────────────────────────────────────────────────────────────────────
# OrderIntent enum
# ───────────────────────────────────────────────────────────────────────────────

def test_order_intent_values():
    """OrderIntent is a plain Enum (not str+Enum) because QC's Python.NET
    wrapper rejects multiple inheritance with managed classes. String
    comparison must go through .value."""
    assert OrderIntent.ENTRY.value == "entry"
    assert OrderIntent.HARD_KILL.value == "hard_kill"
    # Identity comparisons still work
    assert OrderIntent.ENTRY == OrderIntent.ENTRY
    assert OrderIntent.ENTRY != OrderIntent.EXIT


# ───────────────────────────────────────────────────────────────────────────────
# Min-quantity fallback
# ───────────────────────────────────────────────────────────────────────────────

def test_known_symbols_have_fallback():
    assert min_quantity_fallback("BTCUSD") == 0.0001
    assert min_quantity_fallback("KASUSD") == 1.0   # unknown → 1.0


def test_case_insensitive_lookup():
    assert min_quantity_fallback("btcusd") == min_quantity_fallback("BTCUSD")


def test_kraken_min_qty_table_at_least_15_symbols():
    assert len(KRAKEN_MIN_QTY_FALLBACK) >= 15


# ───────────────────────────────────────────────────────────────────────────────
# round_to_lot
# ───────────────────────────────────────────────────────────────────────────────

def test_round_to_lot_basic():
    assert round_to_lot(1.05, 0.1) == pytest.approx(1.0)
    assert round_to_lot(1.99, 0.1) == pytest.approx(1.9)


def test_round_to_lot_floor_not_ceil():
    """Always floors toward zero — never overshoots."""
    assert round_to_lot(0.099, 0.1) == 0.0


def test_round_to_lot_negative_quantity():
    assert round_to_lot(-1.05, 0.1) == pytest.approx(-1.0)


def test_round_to_lot_zero_lot_size_passthrough():
    assert round_to_lot(1.234, 0) == 1.234


def test_round_to_lot_zero_qty():
    assert round_to_lot(0, 0.1) == 0.0


# ───────────────────────────────────────────────────────────────────────────────
# validate_min_notional
# ───────────────────────────────────────────────────────────────────────────────

def test_min_notional_passes_when_above():
    """$10 notional with min $5 + 1.5x buffer = $7.5; passes."""
    ok, reason = validate_min_notional(quantity=1, price=10,
                                       min_notional_usd=5)
    assert ok
    assert reason == ""


def test_min_notional_fails_when_below():
    ok, reason = validate_min_notional(quantity=0.5, price=10,
                                       min_notional_usd=10)
    assert not ok
    assert "min_notional_violation" in reason


def test_min_notional_zero_buffer():
    """fee_buffer_mult=1.0 disables safety margin."""
    ok, _ = validate_min_notional(quantity=1, price=5, min_notional_usd=5,
                                  fee_buffer_mult=1.0)
    assert ok


# ───────────────────────────────────────────────────────────────────────────────
# safe_sell_quantity (the CashBook fix)
# ───────────────────────────────────────────────────────────────────────────────

def test_safe_sell_takes_min_of_portfolio_and_cashbook():
    """Portfolio shows 100, CashBook shows 99.5 → sell 99 (after lot+buffer)."""
    qty = safe_sell_quantity(
        portfolio_quantity=100, cashbook_quantity=99.5,
        lot_size=1.0, min_order_size=1.0,
    )
    # min(100, 99.5)=99.5 → floor to lot=99 → minus 1 buffer → 98
    assert qty == 98.0


def test_safe_sell_zero_when_portfolio_zero():
    qty = safe_sell_quantity(0, 0, lot_size=1.0, min_order_size=1.0)
    assert qty == 0.0


def test_safe_sell_returns_zero_when_below_min():
    qty = safe_sell_quantity(
        portfolio_quantity=2.0, cashbook_quantity=2.0,
        lot_size=1.0, min_order_size=5.0,
    )
    # min=2 → floor=2 → -1 buffer = 1 → < min_order=5 → 0
    assert qty == 0.0


def test_safe_sell_op_real_world_scenario():
    """The Vox README's exact scenario: portfolio.qty=202.509, CashBook=201.962,
    lot=1.0, min=2.0 → safe sell = floor(min(202.509, 201.962))=201 - 1 = 200.
    """
    qty = safe_sell_quantity(
        portfolio_quantity=202.509, cashbook_quantity=201.962,
        lot_size=1.0, min_order_size=2.0,
        exit_qty_buffer_lots=1,
    )
    assert qty == 200.0


def test_safe_sell_zero_buffer():
    qty = safe_sell_quantity(
        portfolio_quantity=100, cashbook_quantity=100,
        lot_size=1.0, min_order_size=1.0,
        exit_qty_buffer_lots=0,
    )
    assert qty == 100.0


# ───────────────────────────────────────────────────────────────────────────────
# Slippage helpers
# ───────────────────────────────────────────────────────────────────────────────

def test_slippage_buy_pays_more():
    s = compute_slippage_pct(fill_price=100.5, reference_price=100, direction="Buy")
    assert s == pytest.approx(0.005)   # 50bp paid


def test_slippage_sell_receives_less():
    s = compute_slippage_pct(fill_price=99.5, reference_price=100, direction="Sell")
    assert s == pytest.approx(0.005)


def test_slippage_clamped_to_non_negative():
    """A favorable fill (better than reference) reports zero slippage."""
    s = compute_slippage_pct(fill_price=99.5, reference_price=100, direction="Buy")
    assert s == 0.0


def test_slippage_zero_on_invalid_prices():
    assert compute_slippage_pct(0, 100, "Buy") == 0.0
    assert compute_slippage_pct(100, 0, "Buy") == 0.0


def test_slippage_log_entry_construction():
    e = slippage_log_entry(
        time=datetime(2026, 5, 10, 12, 0, 0),
        symbol="KASUSD", direction="Buy", quantity=750.0,
        fill_price=0.04057, reference_price=0.04014,
        intent=OrderIntent.ENTRY,
    )
    assert isinstance(e, SlippageLogEntry)
    # 0.04057 vs 0.04014 → ~107bp
    assert e.slippage_bps == pytest.approx(107.13, rel=0.01)
    assert e.intent == OrderIntent.ENTRY


# ───────────────────────────────────────────────────────────────────────────────
# PendingOrder + stale detection
# ───────────────────────────────────────────────────────────────────────────────

def test_pending_order_not_stale_immediately():
    p = PendingOrder("BTC", "Buy", 0.001, 50000,
                     submitted_at=datetime(2026, 1, 1, 12, 0, 0),
                     ttl_seconds=30, intent=OrderIntent.ENTRY)
    assert not p.is_stale(datetime(2026, 1, 1, 12, 0, 5))


def test_pending_order_stale_after_ttl():
    p = PendingOrder("BTC", "Buy", 0.001, 50000,
                     submitted_at=datetime(2026, 1, 1, 12, 0, 0),
                     ttl_seconds=30, intent=OrderIntent.ENTRY)
    assert p.is_stale(datetime(2026, 1, 1, 12, 0, 31))


def test_stale_pending_filter():
    now = datetime(2026, 1, 1, 12, 0, 30)
    fresh = PendingOrder("A", "Buy", 1, 100, submitted_at=now,
                         ttl_seconds=30, intent=OrderIntent.ENTRY)
    expired = PendingOrder("B", "Buy", 1, 100,
                           submitted_at=datetime(2026, 1, 1, 11, 59, 0),
                           ttl_seconds=30, intent=OrderIntent.ENTRY)
    out = stale_pending_orders([fresh, expired], now)
    assert out == [expired]
