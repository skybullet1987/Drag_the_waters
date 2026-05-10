"""Tests for Pulse.online_learning."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.online_learning import (
    DEFAULT_STEP_SIZE, DEFAULT_LOOKBACK_TRADES, DEFAULT_MIN_TRADES_BEFORE_TUNE,
    ENTRY_THRESHOLD_MIN, ENTRY_THRESHOLD_MAX,
    HC_THRESHOLD_MIN, HC_THRESHOLD_MAX,
    TradeResult, ThresholdState, TuneAction,
    OnlineThresholdLearner,
    compute_edge, trades_per_day,
)


# ───────────────────────────────────────────────────────────────────────────────
# compute_edge / trades_per_day
# ───────────────────────────────────────────────────────────────────────────────

def _trade(pnl: float, score: float = 0.55, hc: bool = False,
           ts_offset_hours: float = 0):
    return TradeResult(
        timestamp=datetime(2026, 5, 10, 12) + timedelta(hours=ts_offset_hours),
        pnl_pct=pnl, score=score, high_conviction=hc,
    )


def test_compute_edge_empty():
    assert compute_edge([]) == (0.0, 0.0, 0)


def test_compute_edge_all_winners():
    trades = [_trade(0.01), _trade(0.02), _trade(0.015)]
    edge, wr, n = compute_edge(trades)
    assert wr == 1.0
    assert edge > 0
    assert n == 3


def test_compute_edge_all_losers():
    trades = [_trade(-0.01), _trade(-0.02)]
    edge, wr, n = compute_edge(trades)
    assert wr == 0.0
    assert edge < 0


def test_compute_edge_mixed_50_50():
    """50% WR with ±1% returns → edge = 0.5*0.01 + 0.5*(-0.01) = 0."""
    trades = [_trade(0.01), _trade(-0.01)]
    edge, wr, n = compute_edge(trades)
    assert wr == 0.5
    assert edge == pytest.approx(0.0)


def test_compute_edge_positive_50_50_asymmetric():
    """50% WR with avg_win=2% and avg_loss=-1% → edge = +0.5%."""
    trades = [_trade(0.02), _trade(-0.01)]
    edge, wr, n = compute_edge(trades)
    assert edge == pytest.approx(0.005)


def test_trades_per_day_single_trade():
    assert trades_per_day([_trade(0.01)]) == 0.0


def test_trades_per_day_one_per_hour():
    """24 trades over 23 hours → ~25 trades/day rate."""
    trades = [_trade(0.0, ts_offset_hours=i) for i in range(24)]
    tpd = trades_per_day(trades)
    assert 20 < tpd < 30


# ───────────────────────────────────────────────────────────────────────────────
# OnlineThresholdLearner — construction
# ───────────────────────────────────────────────────────────────────────────────

def test_invalid_initial_thresholds_raise():
    with pytest.raises(ValueError, match="initial_entry_threshold"):
        OnlineThresholdLearner(initial_entry_threshold=0.30)
    with pytest.raises(ValueError, match="initial_high_conviction"):
        OnlineThresholdLearner(initial_high_conviction_thres=0.95)


def test_default_state():
    L = OnlineThresholdLearner()
    assert L.entry_threshold == 0.55
    assert L.high_conviction_thres == 0.70
    assert L.trade_count == 0


# ───────────────────────────────────────────────────────────────────────────────
# Recording trades
# ───────────────────────────────────────────────────────────────────────────────

def test_record_increments_count():
    L = OnlineThresholdLearner()
    L.record(datetime(2026, 1, 1), 0.01, 0.6, False)
    L.record(datetime(2026, 1, 1, 1), -0.005, 0.55, False)
    assert L.trade_count == 2


# ───────────────────────────────────────────────────────────────────────────────
# Tune cycle
# ───────────────────────────────────────────────────────────────────────────────

def test_tune_warmup_when_too_few_trades():
    L = OnlineThresholdLearner(min_trades_before_tune=15)
    now = datetime(2026, 1, 1, 12)
    for i in range(5):
        L.record(now + timedelta(minutes=i), 0.01, 0.6, False)
    a = L.tune(now + timedelta(hours=24))
    assert a.action == "warmup"
    assert L.entry_threshold == 0.55  # unchanged


def test_tune_tightens_on_negative_edge():
    """All losers in window → tighten thresholds."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.005)
    base = datetime(2026, 1, 1)
    for i in range(15):
        L.record(base + timedelta(hours=i), -0.005, 0.6, False)
    a = L.tune(base + timedelta(days=1))
    assert a.action == "tightened"
    assert L.entry_threshold == pytest.approx(0.555)
    assert L.high_conviction_thres == pytest.approx(0.705)


def test_tune_loosens_on_positive_edge_under_target():
    """Positive edge but trade-rate below target → loosen thresholds."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.005,
                               target_trades_per_day=10.0,
                               loosen_edge_threshold=0.001)
    base = datetime(2026, 1, 1)
    # Few trades over many days to ensure tpd < target
    for i in range(15):
        L.record(base + timedelta(days=i * 0.5), 0.02, 0.6, False)
    a = L.tune(base + timedelta(days=10))
    assert a.action == "loosened"
    assert L.entry_threshold == pytest.approx(0.545)


def test_tune_noop_when_edge_positive_and_trade_rate_at_target():
    """Edge positive AND trades/day already at target → no change."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.005,
                               target_trades_per_day=2.0,
                               loosen_edge_threshold=0.001)
    base = datetime(2026, 1, 1)
    for i in range(15):
        L.record(base + timedelta(hours=i * 6), 0.02, 0.6, False)
    a = L.tune(base + timedelta(days=4))
    assert a.action == "noop"
    assert L.entry_threshold == 0.55


def test_tune_clamps_at_envelope_max():
    """Repeated tightening shouldn't push past entry_max."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.10,
                               tune_interval_hours=0)
    base = datetime(2026, 1, 1)
    for i in range(15):
        L.record(base + timedelta(hours=i), -0.01, 0.6, False)
    # Many tune cycles — each tightens by 0.10
    for k in range(5):
        L.tune(base + timedelta(days=k + 1))
    assert L.entry_threshold == ENTRY_THRESHOLD_MAX
    assert L.high_conviction_thres == HC_THRESHOLD_MAX


def test_tune_clamps_at_envelope_min():
    """Repeated loosening shouldn't push past entry_min."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.10,
                               tune_interval_hours=0,
                               target_trades_per_day=100.0,
                               loosen_edge_threshold=0.001)
    base = datetime(2026, 1, 1)
    for i in range(15):
        L.record(base + timedelta(days=i), 0.02, 0.6, False)
    for k in range(10):
        L.tune(base + timedelta(days=k * 5 + 20))
    assert L.entry_threshold == ENTRY_THRESHOLD_MIN
    assert L.high_conviction_thres == HC_THRESHOLD_MIN


def test_tune_throttled_within_interval():
    """Two tune calls within the interval → second is a noop-throttled."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.005,
                               tune_interval_hours=12)
    base = datetime(2026, 1, 1)
    for i in range(15):
        L.record(base + timedelta(hours=i), -0.005, 0.6, False)
    a1 = L.tune(base + timedelta(hours=20))
    a2 = L.tune(base + timedelta(hours=25))   # only 5h later
    assert a1.action == "tightened"
    assert a2.action == "noop"
    assert "throttled" in a2.reason


def test_tune_records_diagnostics():
    L = OnlineThresholdLearner(min_trades_before_tune=10)
    base = datetime(2026, 1, 1)
    for i in range(12):
        L.record(base + timedelta(hours=i), 0.01 if i % 2 == 0 else -0.01,
                 0.6, False)
    a = L.tune(base + timedelta(days=1))
    assert a.trades_observed == 12
    assert isinstance(a.edge_pct, float)
    assert isinstance(a.win_rate, float)


def test_reset_clears_state():
    L = OnlineThresholdLearner()
    base = datetime(2026, 1, 1)
    for i in range(5):
        L.record(base + timedelta(hours=i), 0.01, 0.6, False)
    L.reset()
    assert L.trade_count == 0


def test_realistic_flow_negative_then_positive():
    """Simulate: 20 losers → tighten → switches to 20 winners → loosen."""
    L = OnlineThresholdLearner(min_trades_before_tune=10, step_size=0.01,
                               tune_interval_hours=0,
                               target_trades_per_day=20.0,
                               loosen_edge_threshold=0.001)
    base = datetime(2026, 1, 1)
    # Phase 1: 15 losers
    for i in range(15):
        L.record(base + timedelta(hours=i), -0.01, 0.6, False)
    a1 = L.tune(base + timedelta(days=1))
    assert a1.action == "tightened"
    initial_after_tighten = L.entry_threshold

    # Phase 2: 15 winners (clears the rolling window)
    for i in range(15):
        L.record(base + timedelta(hours=24 + i), 0.02, 0.6, False)
    a2 = L.tune(base + timedelta(days=2))
    assert a2.action == "loosened"
    assert L.entry_threshold < initial_after_tighten
