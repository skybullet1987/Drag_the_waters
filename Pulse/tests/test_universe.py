"""Tests for Pulse.universe — explicitly proves FARTCOIN/PEAQ/MORPHO
(the symbols that polluted the live MG36 universe) get rejected by
the gate, while BTC/ETH/SOL/etc. pass.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Pulse.universe import (
    UniverseGate,
    SymbolTierClassifier,
    SymbolStats,
    GateDecision,
    screen,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers — construct realistic SymbolStats for known names
# ───────────────────────────────────────────────────────────────────────────────

def _good_btc() -> SymbolStats:
    """BTCUSD: $40B daily vol, 1bp spread, $50k price, years of history."""
    return SymbolStats(
        symbol="BTCUSD",
        rolling_24h_dollar_vol_usd=40_000_000_000.0,
        rolling_60bar_mean_spread_bps=1.0,
        last_price_usd=50_000.0,
        days_of_history=2000,
    )


def _good_sol() -> SymbolStats:
    return SymbolStats(
        symbol="SOLUSD",
        rolling_24h_dollar_vol_usd=2_000_000_000.0,
        rolling_60bar_mean_spread_bps=8.0,
        last_price_usd=170.0,
        days_of_history=1000,
    )


def _good_inj() -> SymbolStats:
    return SymbolStats(
        symbol="INJUSD",
        rolling_24h_dollar_vol_usd=80_000_000.0,
        rolling_60bar_mean_spread_bps=15.0,
        last_price_usd=20.0,
        days_of_history=600,
    )


# Symbols that polluted the live MG36 paper trade and need to be rejected:
def _bad_fartcoin() -> SymbolStats:
    """Live MG36 traded this; should never have been allowed."""
    return SymbolStats(
        symbol="FARTCOINUSD",
        rolling_24h_dollar_vol_usd=300_000.0,    # well below $5M floor
        rolling_60bar_mean_spread_bps=120.0,     # absurd spread
        last_price_usd=0.21,
        days_of_history=15,                      # very new listing
    )


def _bad_peaq() -> SymbolStats:
    return SymbolStats(
        symbol="PEAQUSD",
        rolling_24h_dollar_vol_usd=400_000.0,
        rolling_60bar_mean_spread_bps=80.0,
        last_price_usd=0.08,
        days_of_history=20,
    )


def _bad_morpho() -> SymbolStats:
    return SymbolStats(
        symbol="MORPHOUSD",
        rolling_24h_dollar_vol_usd=2_000_000.0,
        rolling_60bar_mean_spread_bps=45.0,
        last_price_usd=1.20,
        days_of_history=18,
    )


def _bad_toshi() -> SymbolStats:
    return SymbolStats(
        symbol="TOSHIUSD",
        rolling_24h_dollar_vol_usd=150_000.0,
        rolling_60bar_mean_spread_bps=200.0,
        last_price_usd=0.0008,             # below price floor
        days_of_history=40,
    )


def _bad_kasusd_micro() -> SymbolStats:
    """KAS — borderline. $4M vol, 25bp spread, $0.04 price.
    Should still be eligible but classified as micro tier."""
    return SymbolStats(
        symbol="KASUSD",
        rolling_24h_dollar_vol_usd=15_000_000.0,
        rolling_60bar_mean_spread_bps=25.0,
        last_price_usd=0.04,
        days_of_history=300,
    )


def _wrapped() -> SymbolStats:
    return SymbolStats(
        symbol="WBTCUSD",
        rolling_24h_dollar_vol_usd=200_000_000.0,
        rolling_60bar_mean_spread_bps=10.0,
        last_price_usd=50_000.0,
        days_of_history=500,
        is_wrapped_or_staked=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# UniverseGate — accepts good, rejects bad
# ───────────────────────────────────────────────────────────────────────────────

def test_btc_eth_sol_pass_gate():
    g = UniverseGate()
    assert g.is_eligible(_good_btc()).eligible
    assert g.is_eligible(_good_sol()).eligible
    assert g.is_eligible(_good_inj()).eligible


def test_fartcoin_rejected():
    g = UniverseGate()
    d = g.is_eligible(_bad_fartcoin())
    assert not d.eligible
    # Should fail multiple criteria
    assert any("dollar_vol_too_low" in r for r in d.reasons)
    assert any("spread_too_wide" in r for r in d.reasons)
    assert any("insufficient_history" in r for r in d.reasons)


def test_peaq_rejected():
    g = UniverseGate()
    d = g.is_eligible(_bad_peaq())
    assert not d.eligible
    assert any("dollar_vol_too_low" in r for r in d.reasons)
    assert any("insufficient_history" in r for r in d.reasons)


def test_morpho_rejected():
    g = UniverseGate()
    d = g.is_eligible(_bad_morpho())
    assert not d.eligible
    assert any("insufficient_history" in r for r in d.reasons)
    # Volume just under threshold and spread just over → should fail multi
    assert any("spread_too_wide" in r for r in d.reasons)


def test_toshi_rejected_for_price_floor():
    g = UniverseGate()
    d = g.is_eligible(_bad_toshi())
    assert not d.eligible
    assert any("price_below_floor" in r for r in d.reasons)


def test_wrapped_btc_rejected():
    g = UniverseGate()
    d = g.is_eligible(_wrapped())
    assert not d.eligible
    assert "wrapped_or_staked_variant" in d.reasons


def test_kas_passes_gate_but_micro_tier():
    """KAS has decent vol + history → eligible. Tier = micro."""
    g = UniverseGate()
    c = SymbolTierClassifier()
    s = _bad_kasusd_micro()
    assert g.is_eligible(s).eligible
    assert c.classify("KASUSD") == "micro"


def test_gate_truthy_decision():
    g = UniverseGate()
    d = g.is_eligible(_good_btc())
    assert bool(d) is True
    d2 = g.is_eligible(_bad_fartcoin())
    assert bool(d2) is False


def test_gate_filter_on_batch():
    g = UniverseGate()
    decisions = g.filter([
        _good_btc(), _good_sol(), _bad_fartcoin(), _bad_peaq(), _bad_toshi(),
    ])
    eligible = [d.symbol for d in decisions if d.eligible]
    rejected = [d.symbol for d in decisions if not d.eligible]
    assert sorted(eligible) == ["BTCUSD", "SOLUSD"]
    assert sorted(rejected) == ["FARTCOINUSD", "PEAQUSD", "TOSHIUSD"]


# ───────────────────────────────────────────────────────────────────────────────
# SymbolTierClassifier
# ───────────────────────────────────────────────────────────────────────────────

def test_btc_eth_classified_as_major():
    c = SymbolTierClassifier()
    assert c.classify("BTCUSD") == "major"
    assert c.classify("ETHUSD") == "major"


def test_sol_link_classified_as_large():
    c = SymbolTierClassifier()
    assert c.classify("SOLUSD") == "large"
    assert c.classify("LINKUSD") == "large"


def test_inj_op_classified_as_mid():
    c = SymbolTierClassifier()
    assert c.classify("INJUSD") == "mid"
    assert c.classify("OPUSD") == "mid"


def test_unknown_symbol_classified_as_micro():
    c = SymbolTierClassifier()
    assert c.classify("KASUSD") == "micro"
    assert c.classify("AKTUSD") == "micro"


def test_limits_per_tier():
    c = SymbolTierClassifier()
    # Major tier — generous
    assert c.limits_for("BTCUSD")["max_pos_usd"] == 5000.0
    assert c.limits_for("BTCUSD")["slip_budget_bps"] == 10.0
    # Large
    assert c.limits_for("SOLUSD")["max_pos_usd"] == 1500.0
    # Mid
    assert c.limits_for("INJUSD")["max_pos_usd"] == 500.0
    # Micro — strictest
    assert c.limits_for("KASUSD")["max_pos_usd"] == 100.0
    assert c.limits_for("KASUSD")["slip_budget_bps"] == 100.0


# ───────────────────────────────────────────────────────────────────────────────
# Auto-demote / eject (the dynamic behavior)
# ───────────────────────────────────────────────────────────────────────────────

def test_no_demote_under_warmup():
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    # 4 trades all bad → still warmup (need 5)
    for _ in range(4):
        out = c.record_trade_slippage("INJUSD", round_trip_bps=500.0, now=now)
    assert out["action"] == "noop_warmup"
    assert c.classify("INJUSD") == "mid"   # unchanged


def test_demote_when_avg_slip_exceeds_1_5x_budget():
    """Mid tier budget = 50bp. Demote at >75bp avg over 5 trades."""
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    # 5 trades at 90bp round-trip — exceeds 75 demote threshold but
    # not 150bp eject threshold
    for i in range(5):
        out = c.record_trade_slippage("INJUSD", 90.0, now)
    assert out["action"] == "demoted"
    assert out["from"] == "mid"
    assert out["to"] == "micro"
    assert c.classify("INJUSD") == "micro"


def test_eject_when_avg_slip_exceeds_3x_budget():
    """Mid tier budget = 50bp. Eject at >150bp avg over 5 trades."""
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    for i in range(5):
        out = c.record_trade_slippage("INJUSD", 200.0, now)
    assert out["action"] == "ejected"
    assert out["from_tier"] == "mid"
    # While in ejection window, classify returns 'ejected'
    assert c.classify("INJUSD", now=now) == "ejected"
    # And limits collapse to zero
    lim = c.limits_for("INJUSD", now=now)
    assert lim["max_pos_usd"] == 0.0
    assert lim["slip_budget_bps"] == 0.0
    assert lim["tier"] == "ejected"


def test_ejected_symbol_recovers_after_window():
    """After EJECT_DURATION_HOURS the symbol returns to its base tier."""
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    for i in range(5):
        c.record_trade_slippage("INJUSD", 200.0, now)
    assert c.classify("INJUSD", now=now) == "ejected"

    later = now + timedelta(days=8)
    # ejection window was 7 days; we're past it
    assert c.classify("INJUSD", now=later) == "mid"


def test_within_budget_does_nothing():
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    # Mid budget 50bp, demote at 75. 60bp = within demote window.
    for i in range(5):
        out = c.record_trade_slippage("INJUSD", 60.0, now)
    assert out["action"] == "noop_within_budget"
    assert c.classify("INJUSD") == "mid"


def test_recorded_slippage_ignored_when_ejected():
    c = SymbolTierClassifier()
    now = datetime(2026, 1, 1, 12)
    for i in range(5):
        c.record_trade_slippage("INJUSD", 200.0, now)
    out = c.record_trade_slippage("INJUSD", 50.0, now)
    assert out["action"] == "ignored_ejected"


# ───────────────────────────────────────────────────────────────────────────────
# screen() — end-to-end on a realistic mixed batch
# ───────────────────────────────────────────────────────────────────────────────

def test_screen_separates_eligible_from_rejected():
    candidates = [
        _good_btc(), _good_sol(), _good_inj(),
        _bad_fartcoin(), _bad_peaq(), _bad_morpho(), _bad_toshi(),
        _wrapped(),
        _bad_kasusd_micro(),  # eligible, micro
    ]
    out = screen(candidates)

    elig_syms = sorted(s.symbol for s in out["eligible"])
    rej_syms  = sorted(d.symbol for d in out["rejected"])

    assert elig_syms == ["BTCUSD", "INJUSD", "KASUSD", "SOLUSD"]
    assert "FARTCOINUSD" in rej_syms
    assert "PEAQUSD" in rej_syms
    assert "MORPHOUSD" in rej_syms
    assert "TOSHIUSD" in rej_syms
    assert "WBTCUSD" in rej_syms

    # Tier mapping for survivors
    tiers = out["tiers"]
    assert tiers["BTCUSD"]["tier"]  == "major"
    assert tiers["SOLUSD"]["tier"]  == "large"
    assert tiers["INJUSD"]["tier"]  == "mid"
    assert tiers["KASUSD"]["tier"]  == "micro"


def test_screen_with_custom_thresholds():
    """If we lower the dollar-vol floor, MORPHO becomes eligible."""
    g = UniverseGate(min_dollar_vol_24h_usd=1_000_000.0,
                     max_avg_spread_bps=100.0,
                     min_days_history=10)
    out = screen([_bad_morpho()], gate=g)
    assert any(s.symbol == "MORPHOUSD" for s in out["eligible"])
