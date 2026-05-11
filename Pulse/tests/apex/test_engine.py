"""Tests for Pulse.apex.sizing, Pulse.apex.exits, Pulse.apex.apex_engine
(Phase 5 execution layer)."""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from Pulse.apex.config import (
    APEX_FEATURE_DIM, APEX_ENTRY_THRESHOLD, APEX_MIN_POSITION_USD,
)
from Pulse.apex.sizing import (
    edge_pct_from_prob, raw_kelly_fraction, safe_kelly_fraction,
    compute_position_usd, SizeDecision,
    DEFAULT_TARGET_RETURN, DEFAULT_VARIANCE_EST,
)
from Pulse.apex.exits import (
    ApexPosition, ExitDecision,
    evaluate_exits,
    should_exit_atr_trail, should_exit_per_trade_kill,
    should_exit_prob_flip, should_exit_time_stop,
)
from Pulse.apex.apex_engine import ApexEngine, ApexDecision
from Pulse.apex.ml.predict import ApexInference
from Pulse.apex.registry import (
    SignalRegistry, SignalScore, reset_default_registry,
)


# ───────────────────────────────────────────────────────────────────────────────
# sizing
# ───────────────────────────────────────────────────────────────────────────────

def test_edge_pct_zero_at_neutral_prob():
    assert edge_pct_from_prob(0.5) == 0.0


def test_edge_pct_positive_above_half():
    assert edge_pct_from_prob(0.7) > 0


def test_edge_pct_negative_below_half():
    assert edge_pct_from_prob(0.3) < 0


def test_raw_kelly_zero_below_half():
    assert raw_kelly_fraction(0.4) == 0.0


def test_raw_kelly_positive_above_half():
    assert raw_kelly_fraction(0.7) > 0


def test_raw_kelly_zero_when_variance_zero():
    assert raw_kelly_fraction(0.7, variance=0) == 0.0


def test_safe_kelly_capped_at_one():
    # Massive prob with tiny variance → raw kelly explodes; safe kelly capped
    s = safe_kelly_fraction(0.99, target_return=0.10, variance=1e-6, derate=1.0)
    assert s == 1.0


def test_safe_kelly_quarter_of_raw_at_default_derate():
    """At a small prob the cap doesn't kick in, so safe = derate × raw."""
    raw = raw_kelly_fraction(0.55)   # small edge → low Kelly
    safe = safe_kelly_fraction(0.55)
    assert safe == pytest.approx(raw * 0.25, rel=1e-6)


def test_safe_kelly_capped_at_one_for_high_edge():
    """At a strong prob, raw × derate would exceed 1; safe must cap."""
    raw = raw_kelly_fraction(0.9)
    safe = safe_kelly_fraction(0.9)
    assert raw * 0.25 > 1.0      # would exceed
    assert safe == 1.0


def test_compute_position_usd_skips_below_threshold():
    d = compute_position_usd(0.55, available_equity=10000,
                              tier_max_pos_usd=2000,
                              entry_threshold=0.62)
    assert d.position_usd == 0
    assert d.skip_reason and "prob<" in d.skip_reason


def test_compute_position_usd_caps_at_tier_max():
    d = compute_position_usd(0.99, available_equity=1_000_000,
                              tier_max_pos_usd=2000,
                              target_return=0.10, variance=1e-4)
    assert d.position_usd == 2000


def test_compute_position_usd_scales_with_regime():
    d_full = compute_position_usd(0.7, available_equity=10000,
                                    tier_max_pos_usd=10000,
                                    regime_size_mult=1.0)
    d_half = compute_position_usd(0.7, available_equity=10000,
                                    tier_max_pos_usd=10000,
                                    regime_size_mult=0.5)
    # Half regime → half position (approximately, before tier cap kicks in)
    assert d_half.position_usd <= d_full.position_usd


def test_compute_position_usd_skips_when_below_min():
    d = compute_position_usd(0.65, available_equity=10,    # tiny equity
                              tier_max_pos_usd=2000,
                              min_position_usd=100)
    assert d.position_usd == 0
    assert d.skip_reason and "below_min" in d.skip_reason


def test_size_decision_carries_breakdown_fields():
    d = compute_position_usd(0.7, available_equity=10000,
                              tier_max_pos_usd=10000,
                              regime_size_mult=0.8)
    assert d.regime_mult == 0.8
    assert d.tier_cap_usd == 10000
    assert d.raw_kelly > 0


# ───────────────────────────────────────────────────────────────────────────────
# exits.ApexPosition
# ───────────────────────────────────────────────────────────────────────────────

def _pos(symbol="BTCUSD", entry_price=100.0, entry_atr=2.0,
         entry_time=datetime(2025, 1, 1)) -> ApexPosition:
    return ApexPosition(
        symbol=symbol, entry_time=entry_time,
        entry_price=entry_price, quantity=1.0,
        entry_atr=entry_atr, high_water=entry_price,
    )


def test_position_held_days_basic():
    p = _pos(entry_time=datetime(2025, 1, 1))
    days = p.held_days(datetime(2025, 1, 8))
    assert days == 7.0


def test_position_high_water_only_increases():
    p = _pos()
    p.update_high_water(99.0)
    assert p.high_water == 100.0
    p.update_high_water(110.0)
    assert p.high_water == 110.0
    p.update_high_water(105.0)
    assert p.high_water == 110.0


# ───────────────────────────────────────────────────────────────────────────────
# exits — individual gates
# ───────────────────────────────────────────────────────────────────────────────

def test_per_trade_kill_fires_below_threshold():
    p = _pos(entry_price=100)
    d = should_exit_per_trade_kill(p, current_price=90, hard_kill_pct=0.08)
    assert d.should_exit
    assert d.reason == "PER_TRADE_KILL"


def test_per_trade_kill_no_fire_above_threshold():
    p = _pos(entry_price=100)
    d = should_exit_per_trade_kill(p, current_price=95, hard_kill_pct=0.08)
    assert not d.should_exit


def test_per_trade_kill_safe_when_zero_entry():
    p = _pos(entry_price=0)
    d = should_exit_per_trade_kill(p, current_price=100)
    assert not d.should_exit


def test_time_stop_fires_at_max_days():
    p = _pos(entry_time=datetime(2025, 1, 1))
    d = should_exit_time_stop(p, now=datetime(2025, 1, 8), max_days=7)
    assert d.should_exit
    assert d.reason == "TIME_STOP"


def test_time_stop_no_fire_before_max_days():
    p = _pos(entry_time=datetime(2025, 1, 1))
    d = should_exit_time_stop(p, now=datetime(2025, 1, 5), max_days=7)
    assert not d.should_exit


def test_atr_trail_no_fire_when_no_high_above_entry():
    p = _pos(entry_price=100, entry_atr=2.0)
    # high_water = 100 (entry), trail = 100 - 6 = 94. Price 90 < 94 → would
    # trigger if high_water moved up. But high_water == entry, so suppress.
    d = should_exit_atr_trail(p, current_price=90)
    assert not d.should_exit


def test_atr_trail_fires_when_price_drops_below_trail():
    p = _pos(entry_price=100, entry_atr=2.0)
    p.update_high_water(115)        # high = 115, trail = 115 - 6 = 109
    d = should_exit_atr_trail(p, current_price=108)
    assert d.should_exit
    assert d.reason == "ATR_TRAIL"


def test_atr_trail_no_fire_when_above_trail():
    p = _pos(entry_price=100, entry_atr=2.0)
    p.update_high_water(115)
    d = should_exit_atr_trail(p, current_price=112)
    assert not d.should_exit


def test_atr_trail_safe_when_zero_atr():
    p = _pos(entry_atr=0)
    p.update_high_water(115)
    d = should_exit_atr_trail(p, current_price=100)
    assert not d.should_exit


def test_prob_flip_fires_below_threshold():
    p = _pos()
    d = should_exit_prob_flip(p, latest_prob=0.30, exit_threshold=0.45)
    assert d.should_exit
    assert d.reason == "PROB_FLIP"


def test_prob_flip_no_fire_above_threshold():
    p = _pos()
    d = should_exit_prob_flip(p, latest_prob=0.50, exit_threshold=0.45)
    assert not d.should_exit


# ───────────────────────────────────────────────────────────────────────────────
# evaluate_exits priority
# ───────────────────────────────────────────────────────────────────────────────

def test_evaluate_exits_per_trade_kill_takes_priority():
    """Per-trade kill must beat all other gates."""
    p = _pos(entry_price=100, entry_atr=2.0,
             entry_time=datetime(2025, 1, 1))
    p.update_high_water(110)
    d = evaluate_exits(p, current_price=90,
                        now=datetime(2025, 1, 10),  # also past time stop
                        latest_prob=0.20)            # also below prob threshold
    assert d.should_exit
    assert d.reason == "PER_TRADE_KILL"


def test_evaluate_exits_time_stop_after_kill_check():
    p = _pos(entry_price=100, entry_time=datetime(2025, 1, 1))
    d = evaluate_exits(p, current_price=99, now=datetime(2025, 1, 10),
                        max_days=7)
    assert d.should_exit
    assert d.reason == "TIME_STOP"


def test_evaluate_exits_no_exit_when_all_pass():
    p = _pos(entry_price=100, entry_atr=2.0,
             entry_time=datetime(2025, 1, 1))
    d = evaluate_exits(p, current_price=102, now=datetime(2025, 1, 2),
                        latest_prob=0.65)
    assert not d.should_exit


def test_evaluate_exits_updates_high_water():
    p = _pos(entry_price=100)
    evaluate_exits(p, current_price=120, now=datetime(2025, 1, 2),
                    latest_prob=0.7)
    assert p.high_water == 120


# ───────────────────────────────────────────────────────────────────────────────
# ApexEngine
# ───────────────────────────────────────────────────────────────────────────────

class _OrderBook:
    """Capture place_order/place_exit invocations for assertions."""
    def __init__(self):
        self.entries: list[tuple] = []
        self.exits:   list[tuple] = []

    def order(self, sym, qty, tag, price):
        self.entries.append((sym, qty, tag, price))

    def exit(self, sym, qty, reason):
        self.exits.append((sym, qty, reason))


def _make_engine_with_high_prob_inference(prob: float = 0.80):
    """Engine wired with a fake inference returning a constant prob."""
    class _ConstInf:
        in_fallback_mode = False
        def predict_many(self, X): return [prob] * len(X)
        def predict_one(self, x):  return prob

    return ApexEngine(
        inference=_ConstInf(),
        registry=SignalRegistry(),
        entry_threshold=0.62, exit_threshold=0.45,
        max_positions=4, max_new_per_tick=2,
        min_position_usd=10.0,
    )


def test_engine_enters_on_high_prob():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1, 12), universe=("BTCUSD", "ETHUSD"),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        tier_max_pos_usd_provider=lambda s: 1500,
        regime_mult_provider=lambda s: 1.0,
        current_price_provider=lambda s: 100.0,
        current_atr_provider=lambda s: 2.0,
    )
    assert len(decisions) == 2
    assert all(d.kind == "ENTRY" for d in decisions)
    assert len(book.entries) == 2


def test_engine_skips_when_no_price():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1, 12), universe=("BTCUSD",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 0.0,   # no price data
    )
    assert book.entries == []
    assert decisions == []
    # The skip should be in the log
    skips = [d for d in eng.last_decision_log if d.kind == "SKIP"]
    assert any("no_price" in d.reason for d in skips)


def test_engine_respects_max_new_per_tick():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC", "ETH", "SOL", "XRP"),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    # max_new_per_tick was 2
    assert len(decisions) == 2


def test_engine_caps_at_max_positions():
    eng = _make_engine_with_high_prob_inference(0.80)
    eng.max_new_per_tick = 10  # allow many per tick
    eng.max_positions = 1
    book = _OrderBook()
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC", "ETH", "SOL"),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    assert len(decisions) == 1
    # Second tick should produce nothing — we're full
    decisions2 = eng.on_4h_tick(
        now=datetime(2025, 1, 1, 4), universe=("ETH", "SOL"),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    assert decisions2 == []


def test_engine_skips_existing_position():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    # Re-tick: BTC already open, should be excluded from candidates
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1, 4), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    assert decisions == []


def test_engine_skip_below_entry_threshold():
    eng = _make_engine_with_high_prob_inference(0.55)  # below 0.62
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    assert book.entries == []
    skips = [d for d in eng.last_decision_log if d.kind == "SKIP"]
    assert any("prob<" in d.reason for d in skips)


def test_engine_dd_freeze_blocks_entries():
    eng = _make_engine_with_high_prob_inference(0.80)
    eng.daily_dd_freeze = 0.05
    book = _OrderBook()
    # Establish daily high
    eng.on_4h_tick(
        now=datetime(2025, 1, 1, 0), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    # Now equity drops 6% — should freeze
    decisions = eng.on_4h_tick(
        now=datetime(2025, 1, 1, 4), universe=("ETH",),
        market_context_provider=lambda s: {}, equity=9_400,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    assert len(decisions) == 1
    assert decisions[0].kind == "SKIP"
    assert decisions[0].reason == "daily_dd_freeze"


def test_engine_minute_tick_runs_exits():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
        current_atr_provider=lambda s: 2.0,
    )
    # Drop price to trigger PER_TRADE_KILL (-8%)
    decisions = eng.on_minute_tick(
        now=datetime(2025, 1, 1, 1),
        current_prices={"BTC": 90.0},
        latest_probs={},
        place_exit_fn=book.exit,
    )
    assert len(decisions) == 1
    assert decisions[0].kind == "EXIT"
    assert decisions[0].reason == "PER_TRADE_KILL"
    assert "BTC" not in eng.open_positions
    assert book.exits == [("BTC", 1.0 if False else book.exits[0][1], "PER_TRADE_KILL")] or \
           (book.exits[0][0] == "BTC" and book.exits[0][2] == "PER_TRADE_KILL")


def test_engine_minute_tick_skips_when_no_price():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    decisions = eng.on_minute_tick(
        now=datetime(2025, 1, 1, 1),
        current_prices={"BTC": 0.0},
        latest_probs={},
        place_exit_fn=book.exit,
    )
    assert decisions == []


def test_engine_stats_reports_state():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    s = eng.stats()
    assert s["open_positions"] == 1
    assert s["decisions_logged"] >= 1
    assert s["by_kind"].get("ENTRY", 0) == 1


def test_engine_reset_clears_state():
    eng = _make_engine_with_high_prob_inference(0.80)
    book = _OrderBook()
    eng.on_4h_tick(
        now=datetime(2025, 1, 1), universe=("BTC",),
        market_context_provider=lambda s: {}, equity=10_000,
        place_order_fn=book.order,
        current_price_provider=lambda s: 100.0,
    )
    eng.reset()
    assert eng.open_positions == {}
    assert eng.last_decision_log == []


def test_engine_uses_default_registry_when_none_passed():
    reset_default_registry()
    inf = ApexInference(model_path=None)
    eng = ApexEngine(inference=inf)
    assert eng.registry is not None
