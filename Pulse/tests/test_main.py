"""Smoke tests for Pulse.main — only the pure-Python helpers (the
PulseAlgorithm class itself requires QC AlgorithmImports)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.main import HAS_QC, SymbolBuffers, OpenPosition, PulseAlgorithm
from Pulse.execution import OrderIntent


def test_module_importable_without_qc():
    """Module should import cleanly outside QC environment."""
    assert HAS_QC is False or HAS_QC is True  # boolean exists
    if not HAS_QC:
        assert PulseAlgorithm is None


# ───────────────────────────────────────────────────────────────────────────────
# SymbolBuffers
# ───────────────────────────────────────────────────────────────────────────────

def test_symbol_buffers_initial_empty():
    b = SymbolBuffers()
    assert b.last_close == 0.0
    assert len(b.opens) == 0
    assert b.last_bid == 0.0


def test_symbol_buffers_update_bar_appends():
    b = SymbolBuffers()
    b.update_bar(100, 102, 99, 101, 1000)
    assert b.last_close == 101
    assert len(b.opens) == 1
    assert len(b.recent_dollar_volumes) == 1
    assert b.recent_dollar_volumes[-1] == 101 * 1000


def test_symbol_buffers_skips_non_positive_open():
    """An invalid bar (open<=0) should be ignored."""
    b = SymbolBuffers()
    b.update_bar(0, 102, 99, 101, 1000)
    assert b.last_close == 0.0


def test_symbol_buffers_quote_records_spread_bps():
    b = SymbolBuffers()
    b.update_quote(bid=99.95, ask=100.05)
    assert b.last_bid == 99.95
    assert b.last_ask == 100.05
    # spread = (100.05-99.95) / 100 = 10bp
    assert b.spread_bps_history[-1] == pytest.approx(10.0, rel=1e-3)


def test_symbol_buffers_skips_zero_quote():
    b = SymbolBuffers()
    b.update_quote(bid=0, ask=100)
    assert b.last_bid == 0.0


def test_stats_for_universe_constructs_correctly():
    b = SymbolBuffers()
    for _ in range(60):
        b.update_bar(100, 102, 99, 101, 1000)
        b.update_quote(99.95, 100.05)
    stats = b.stats_for_universe(symbol="BTCUSD", days_of_history=200)
    assert stats.symbol == "BTCUSD"
    # 60 bars × 1000 vol × $101 ≈ $6M
    assert stats.rolling_24h_dollar_vol_usd == pytest.approx(60 * 1000 * 101)
    assert stats.last_price_usd == 101
    assert stats.days_of_history == 200
    assert stats.rolling_60bar_mean_spread_bps == pytest.approx(10.0, rel=1e-2)


def test_to_symbol_bars_round_trip():
    b = SymbolBuffers()
    for i in range(10):
        b.update_bar(100 + i, 102 + i, 99 + i, 101 + i, 1000 + i)
    bars = b.to_symbol_bars("ETHUSD")
    assert bars.symbol == "ETHUSD"
    assert len(bars.closes) == 10
    assert bars.closes[-1] == 110


# ───────────────────────────────────────────────────────────────────────────────
# OpenPosition
# ───────────────────────────────────────────────────────────────────────────────

def test_open_position_initial_state():
    p = OpenPosition(
        symbol="BTC", entry_price=100.0,
        entry_time=datetime(2026, 5, 10, 12, 0, 0),
        quantity=1.0, intent=OrderIntent.ENTRY,
    )
    assert p.high_price == 100.0
    assert p.low_price == 100.0
    assert not p.trail_armed
    assert not p.partial_taken


def test_open_position_tracks_high_low():
    p = OpenPosition("BTC", 100.0, datetime.now(), 1.0, OrderIntent.ENTRY)
    p.update_extremes(110.0)
    assert p.high_price == 110.0
    assert p.low_price == 100.0
    p.update_extremes(95.0)
    assert p.low_price == 95.0
    assert p.high_price == 110.0   # didn't decrease


def test_open_position_held_hours():
    t0 = datetime(2026, 5, 10, 12, 0, 0)
    p = OpenPosition("BTC", 100.0, t0, 1.0, OrderIntent.ENTRY)
    later = t0 + timedelta(hours=2, minutes=30)
    assert p.held_hours(later) == pytest.approx(2.5)


# ───────────────────────────────────────────────────────────────────────────────
# Imports + cross-module wiring
# ───────────────────────────────────────────────────────────────────────────────

def test_imports_all_required_pulse_modules():
    """Smoke test: main.py must successfully import every Pulse module."""
    # If we got here, the import worked
    from Pulse import (
        config, universe, slippage, fees, circuit, features,
        regime, alt_data, scalp_engine, execution, events, main,
    )
    assert config is not None
    assert main is not None
