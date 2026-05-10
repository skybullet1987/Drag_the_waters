"""Tests for Pulse.circuit."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.circuit import (
    DrawdownCircuitBreaker,
    RollingMaxDrawdown,
    PerTradeKill,
    CircuitState,
)


# ───────────────────────────────────────────────────────────────────────────────
# DrawdownCircuitBreaker
# ───────────────────────────────────────────────────────────────────────────────

def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(trip_drawdown_pct=0.30, halt_drawdown_pct=0.20)
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(trip_drawdown_pct=-0.1, halt_drawdown_pct=0.2)


def test_initial_state_ok():
    cb = DrawdownCircuitBreaker()
    assert cb.can_enter_new_positions()
    assert not cb.should_liquidate_all()


def test_first_update_seeds_peak():
    cb = DrawdownCircuitBreaker()
    out = cb.update(1000.0, datetime(2026, 1, 1))
    assert out["action"] == "init"
    assert out["peak"] == 1000.0


def test_peak_tracks_new_highs():
    cb = DrawdownCircuitBreaker()
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    cb.update(1500.0, t + timedelta(hours=1))
    cb.update(1200.0, t + timedelta(hours=2))
    s = cb.state(1200.0)
    assert s.peak_equity == 1500.0
    assert s.drawdown_pct == pytest.approx((1500 - 1200) / 1500, rel=1e-9)


def test_trips_at_20pct():
    cb = DrawdownCircuitBreaker(trip_drawdown_pct=0.20, halt_drawdown_pct=0.25)
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    out = cb.update(799.99, t + timedelta(hours=1))   # 20.001% DD
    assert out["action"] == "tripped"
    assert not cb.can_enter_new_positions()
    assert not cb.should_liquidate_all()


def test_halts_at_25pct():
    cb = DrawdownCircuitBreaker(trip_drawdown_pct=0.20, halt_drawdown_pct=0.25)
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    out = cb.update(749.0, t + timedelta(hours=1))    # 25.1% DD
    assert out["action"] == "halted"
    assert not cb.can_enter_new_positions()
    assert cb.should_liquidate_all()


def test_no_double_trip():
    cb = DrawdownCircuitBreaker()
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    cb.update(750.0, t + timedelta(hours=1))   # tripped
    out = cb.update(740.0, t + timedelta(hours=2))
    # Still tripped; new trip-low recorded
    assert out["action"] == "new_trip_low"
    assert out["trip_low"] == 740.0


def test_auto_reset_after_recovery():
    cb = DrawdownCircuitBreaker(
        trip_drawdown_pct=0.20, halt_drawdown_pct=0.25, recovery_pct=0.05,
    )
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    cb.update(750.0, t + timedelta(hours=1))      # halted! oh wait, 25% exactly...

    # Re-test with cleaner numbers
    cb = DrawdownCircuitBreaker(
        trip_drawdown_pct=0.20, halt_drawdown_pct=0.30, recovery_pct=0.05,
    )
    cb.update(1000.0, t)
    cb.update(780.0, t + timedelta(hours=1))      # 22% → tripped, target = 780*1.05 = 819
    assert not cb.can_enter_new_positions()
    out = cb.update(820.0, t + timedelta(hours=2))   # recovered to target
    assert out["action"] == "reset"
    assert cb.can_enter_new_positions()


def test_halt_does_not_auto_reset():
    cb = DrawdownCircuitBreaker(
        trip_drawdown_pct=0.20, halt_drawdown_pct=0.25, recovery_pct=0.05,
    )
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    cb.update(700.0, t + timedelta(hours=1))     # 30% → halted
    assert cb.should_liquidate_all()
    # Even after huge recovery, halt requires manual reset
    out = cb.update(950.0, t + timedelta(days=7))
    assert out["action"] == "noop"
    assert cb.should_liquidate_all()


def test_halt_manual_reset():
    cb = DrawdownCircuitBreaker(
        trip_drawdown_pct=0.20, halt_drawdown_pct=0.25, recovery_pct=0.05,
    )
    t = datetime(2026, 1, 1)
    cb.update(1000.0, t)
    cb.update(700.0, t + timedelta(hours=1))     # halted
    out = cb.reset_halt(950.0)
    assert out["action"] == "manual_reset"
    assert cb.can_enter_new_positions()
    assert not cb.should_liquidate_all()


def test_state_snapshot_when_ok():
    cb = DrawdownCircuitBreaker()
    cb.update(1000.0, datetime(2026, 1, 1))
    s = cb.state(900.0)
    assert isinstance(s, CircuitState)
    assert s.tripped is False
    assert s.halted is False
    assert s.drawdown_pct == pytest.approx(0.10)


def test_zero_equity_handled():
    cb = DrawdownCircuitBreaker()
    out = cb.update(0.0, datetime(2026, 1, 1))
    assert out["action"] == "noop_zero_equity"
    assert cb.can_enter_new_positions()


def test_realistic_user_settings():
    """Anchor: PLAN.md user config — trip 20%, halt 25%."""
    cb = DrawdownCircuitBreaker(
        trip_drawdown_pct=0.20, halt_drawdown_pct=0.25, recovery_pct=0.05,
    )
    t = datetime(2026, 1, 1)
    # Compound to $9000 (MG36-style ramp)
    for eq in (100, 200, 500, 1000, 2000, 5000, 9000):
        cb.update(eq, t)
        t += timedelta(days=10)
    # Then a 22% drawdown
    cb.update(7020.0, t)   # 9000 → 7020 = 22% DD
    assert not cb.can_enter_new_positions()
    # Then recovers 5% above the trip-low (7020 * 1.05 = 7371)
    cb.update(7400.0, t + timedelta(days=2))
    assert cb.can_enter_new_positions()


# ───────────────────────────────────────────────────────────────────────────────
# RollingMaxDrawdown
# ───────────────────────────────────────────────────────────────────────────────

def test_rmd_invalid_lookback():
    with pytest.raises(ValueError):
        RollingMaxDrawdown(lookback_bars=1)


def test_rmd_empty_returns_zero():
    rmd = RollingMaxDrawdown()
    assert rmd.get_max_drawdown() == 0.0
    assert rmd.get_current_drawdown() == 0.0
    assert len(rmd) == 0


def test_rmd_simple_drawdown():
    rmd = RollingMaxDrawdown(lookback_bars=10)
    for eq in [100, 110, 120, 100, 105]:
        rmd.update(eq)
    # Peak 120, low 100 → 16.67% DD
    assert rmd.get_max_drawdown() == pytest.approx((120 - 100) / 120, rel=1e-9)


def test_rmd_window_drops_old_data():
    """Once outside the lookback window, old peaks shouldn't count."""
    rmd = RollingMaxDrawdown(lookback_bars=3)
    for eq in [200, 100, 110, 120, 130]:
        rmd.update(eq)
    # Window now contains [120, 130] effectively (last 3 = 110, 120, 130)
    # Peak in window = 130; min = 110 → DD relative to in-window peak
    # Actually: window peak is 130, but rmd computes peak that is reached
    # AFTER the window's first value, so [110, 120, 130] → no DD.
    assert rmd.get_max_drawdown() == 0.0


def test_rmd_current_drawdown():
    rmd = RollingMaxDrawdown(lookback_bars=10)
    for eq in [100, 200, 150]:
        rmd.update(eq)
    # Current 150, peak 200 → 25% DD
    assert rmd.get_current_drawdown() == pytest.approx(0.25)


# ───────────────────────────────────────────────────────────────────────────────
# PerTradeKill
# ───────────────────────────────────────────────────────────────────────────────

def test_per_trade_kill_invalid_threshold():
    with pytest.raises(ValueError):
        PerTradeKill(threshold_pct=0)
    with pytest.raises(ValueError):
        PerTradeKill(threshold_pct=-0.05)


def test_per_trade_kill_winner_no_kill():
    kill = PerTradeKill()
    d = kill.evaluate("BTCUSD", entry_price=100.0, current_price=110.0)
    assert d.return_pct == pytest.approx(0.10)
    assert not d.should_kill
    assert d.reason is None


def test_per_trade_kill_small_loss_no_kill():
    kill = PerTradeKill(threshold_pct=0.08)
    d = kill.evaluate("BTCUSD", entry_price=100.0, current_price=95.0)
    assert d.return_pct == pytest.approx(-0.05)
    assert not d.should_kill


def test_per_trade_kill_at_threshold_triggers():
    """-8% exactly should trigger (boundary inclusive)."""
    kill = PerTradeKill(threshold_pct=0.08)
    d = kill.evaluate("BTCUSD", entry_price=100.0, current_price=92.0)
    assert d.return_pct == pytest.approx(-0.08)
    assert d.should_kill
    assert "hard_kill" in d.reason


def test_per_trade_kill_below_threshold_triggers():
    kill = PerTradeKill(threshold_pct=0.08)
    d = kill.evaluate("KASUSD", entry_price=0.04057, current_price=0.037)
    # ~ -8.8%
    assert d.should_kill
    assert d.return_pct < -0.08


def test_per_trade_kill_zero_price_safe():
    kill = PerTradeKill()
    d = kill.evaluate("XYZ", entry_price=0, current_price=10)
    assert not d.should_kill
    d2 = kill.evaluate("XYZ", entry_price=10, current_price=0)
    assert not d2.should_kill


def test_kasusd_live_scenario_caught():
    """The live MG36 KASUSD trade lost 2.29% (under threshold).
    But if it had run to -10% (the worst case the strategy would allow),
    PerTradeKill at 8% would have caught it 2 percentage points earlier."""
    kill = PerTradeKill(threshold_pct=0.08)
    d_actual = kill.evaluate("KASUSD", 0.04057, 0.04057 * 0.9771)  # -2.29%
    assert not d_actual.should_kill
    d_worst = kill.evaluate("KASUSD", 0.04057, 0.04057 * 0.90)     # -10%
    assert d_worst.should_kill
