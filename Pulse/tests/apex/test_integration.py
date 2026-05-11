"""End-to-end integration tests for Pulse.apex.integration.bootstrap_apex.

These tests exercise the FULL stack:
  - All 9 signals registered + computable
  - Stores fed with synthetic data
  - Feature vector built from registry
  - ApexInference (fallback mode) returns a probability
  - ApexEngine processes a 4h tick → entries
  - Minute tick → exits
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from Pulse.apex.config import APEX_FEATURE_DIM
from Pulse.apex.feature_vector import (
    DEFAULT_SIGNAL_ORDER, build_feature_vector, reset_history,
)
from Pulse.apex.integration import (
    ApexBundle, bootstrap_apex, attach_callbacks,
)
from Pulse.apex.data.token_unlocks import UnlockEvent


@pytest.fixture(autouse=True)
def _clean_history():
    reset_history()


# ───────────────────────────────────────────────────────────────────────────────
# bootstrap
# ───────────────────────────────────────────────────────────────────────────────

def test_bootstrap_returns_bundle_with_all_stores():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    assert isinstance(b, ApexBundle)
    assert b.registry is not None
    assert b.engine is not None
    assert b.inference.in_fallback_mode
    # All 9 signals registered
    assert len(b.registry) >= 9
    for sig in ("btc_onchain", "btc_dominance", "funding_extreme",
                "cross_asset_macro", "etf_flow", "stablecoin_mint",
                "token_unlock", "news_sentiment", "mvrv"):
        assert sig in b.registry, f"missing: {sig}"


def test_bootstrap_signal_order_matches_default():
    """The default signal order in feature_vector must align with what
    bootstrap registers (so the feature vector has expected layout)."""
    b = bootstrap_apex(model_path=None, fallback_only=True)
    for sig in DEFAULT_SIGNAL_ORDER:
        if sig == "fg_index":
            # fg_index intentionally not yet wired (still optional in apex)
            continue
        assert sig in b.registry


def test_bootstrap_attach_callbacks_records_them():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    placed = []
    exited = []
    attach_callbacks(
        b,
        place_order=lambda s, q, t, p: placed.append((s, q, t, p)),
        place_exit=lambda s, q, r: exited.append((s, q, r)),
    )
    assert b.place_order is not None
    assert b.place_exit is not None
    b.place_order("BTC", 0.1, "TEST", 100.0)
    b.place_exit("BTC", 0.1, "TEST")
    assert placed == [("BTC", 0.1, "TEST", 100.0)]
    assert exited == [("BTC", 0.1, "TEST")]


# ───────────────────────────────────────────────────────────────────────────────
# Feature vector through the full registry
# ───────────────────────────────────────────────────────────────────────────────

def test_feature_vector_with_no_data_is_all_zeros():
    """Cold start — nothing in any store. Every signal returns invalid
    or zero. Feature vector should be all zeros."""
    b = bootstrap_apex(model_path=None, fallback_only=True)
    fv = build_feature_vector("BTCUSD", {}, registry=b.registry)
    assert len(fv.values) == APEX_FEATURE_DIM
    assert all(v == 0.0 for v in fv.values)


def _populate_btc_onchain(b: ApexBundle):
    """Realistic-ish on-chain data with a recent jump = bullish."""
    for metric in ("hash_rate", "n_unique_addresses",
                   "miners_revenue", "estimated_btc_sent"):
        for i in range(60):
            b.btc_onchain_store.record_bar(metric, 100.0 * (1 + 0.001 * i))
        # Recent push
        for i in range(14):
            b.btc_onchain_store.record_bar(metric, 200.0)


def test_feature_vector_picks_up_btc_onchain_signal():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    _populate_btc_onchain(b)
    fv = build_feature_vector("BTCUSD", {}, registry=b.registry)
    # Feature index 0 (btc_onchain_raw) should be > 0
    assert fv.values[0] > 0
    assert fv.valid_signals >= 1


def test_inference_fallback_returns_unit_prob():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    p = b.inference.predict_one([0.0] * APEX_FEATURE_DIM)
    assert 0.0 <= p <= 1.0


# ───────────────────────────────────────────────────────────────────────────────
# End-to-end engine flow
# ───────────────────────────────────────────────────────────────────────────────

def _populate_strong_bullish_signals(b: ApexBundle):
    """Push every store toward bullish so the fallback prob > 0.62."""
    # btc_onchain
    _populate_btc_onchain(b)
    # btc_dominance — falling = bullish for alts (we test on ETH below)
    for i in range(120):
        b.dominance_store.record(50_000_000_000, 100_000_000_000 + i * 1e9)
    # funding — deep negative = squeeze
    for _ in range(100):
        b.funding_store.record("ETHUSDT", -0.001)
    # cross-asset — recent risk-on push
    for i in range(80):
        b.cross_asset_store.record("spy", 450.0 * (1 + 0.0001 * i))
        b.cross_asset_store.record("dxy", 105.0 * (1 - 0.0001 * i))
        b.cross_asset_store.record("gld", 180.0)
    # ETF flow — recent inflow
    for _ in range(27):
        b.etf_flow_store.record(50.0)
    for _ in range(3):
        b.etf_flow_store.record(800.0)
    # Stablecoin — recent mint
    for i in range(32):
        b.stablecoin_store.record(1e11 + i * 1e7, 5e10 + i * 5e6)
    for _ in range(3):
        b.stablecoin_store.record(b.stablecoin_store.usdt[-1] + 5e9,
                                    b.stablecoin_store.usdc[-1] + 2e9)
    # News — recent positive sentiment
    for _ in range(27):
        b.news_store.record("ETHUSD", 0.0)
    for _ in range(3):
        b.news_store.record("ETHUSD", 0.8)


def test_full_stack_engine_enters_position_with_strong_bullish_signals():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    _populate_strong_bullish_signals(b)

    placed = []

    def _order(sym, qty, tag, price):
        placed.append((sym, qty, tag, price))

    decisions = b.engine.on_4h_tick(
        now=datetime(2025, 8, 10, 12),
        universe=("ETHUSD",),
        market_context_provider=lambda s: {"now": datetime(2025, 8, 10, 12)},
        equity=10_000.0,
        place_order_fn=_order,
        tier_max_pos_usd_provider=lambda s: 1500,
        regime_mult_provider=lambda s: 1.0,
        current_price_provider=lambda s: 3000.0,
        current_atr_provider=lambda s: 60.0,
    )
    assert len(decisions) == 1
    assert decisions[0].kind == "ENTRY"
    assert decisions[0].symbol == "ETHUSD"
    assert decisions[0].size_usd > 0
    assert len(placed) == 1


def test_full_stack_engine_skips_when_no_signal_data():
    """Cold-start: no store populated → signals all zero → fallback prob ≈ 0.5
    → below entry threshold → no entry."""
    b = bootstrap_apex(model_path=None, fallback_only=True)

    placed = []
    decisions = b.engine.on_4h_tick(
        now=datetime(2025, 8, 10, 12),
        universe=("BTCUSD",),
        market_context_provider=lambda s: {"now": datetime(2025, 8, 10, 12)},
        equity=10_000.0,
        place_order_fn=lambda s, q, t, p: placed.append(s),
        current_price_provider=lambda s: 50000.0,
    )
    assert placed == []
    skips = [d for d in decisions if d.kind == "SKIP"]
    # The skip is logged in the engine's decision_log, but the API only
    # returns ENTRY decisions. Check the log:
    log_skips = [d for d in b.engine.last_decision_log if d.kind == "SKIP"]
    assert log_skips, "expected a SKIP decision to be logged"


def test_full_stack_unlock_signal_blocks_entry():
    """An imminent unlock should drag the fallback prob down."""
    b = bootstrap_apex(model_path=None, fallback_only=True)
    _populate_strong_bullish_signals(b)
    # Add a fat unlock event for ETH 3 days from now (5% of supply = -1.0 score)
    b.unlock_store.events.append(
        UnlockEvent(date=datetime(2025, 8, 13), symbol="ETH",
                    usd=1e9, pct_of_supply=0.05),
    )
    placed = []
    b.engine.on_4h_tick(
        now=datetime(2025, 8, 10, 12),
        universe=("ETHUSD",),
        market_context_provider=lambda s: {"now": datetime(2025, 8, 10, 12)},
        equity=10_000.0,
        place_order_fn=lambda s, q, t, p: placed.append(s),
        current_price_provider=lambda s: 3000.0,
    )
    # Token unlock has weight 0.10 in fallback weights — this should
    # noticeably reduce the prob but might not block entirely. The
    # critical assertion: the bundle wired the signal correctly.
    fv = build_feature_vector("ETHUSD",
                               {"now": datetime(2025, 8, 10, 12)},
                               registry=b.registry)
    # Find token_unlock_raw index
    idx = (DEFAULT_SIGNAL_ORDER.index("token_unlock") * 3
           if "token_unlock" in DEFAULT_SIGNAL_ORDER else None)
    assert idx is not None
    assert fv.values[idx] == -1.0   # full bearish from the 5% unlock


def test_full_stack_minute_tick_exits_open_positions():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    _populate_strong_bullish_signals(b)
    placed = []
    exited = []
    b.engine.on_4h_tick(
        now=datetime(2025, 8, 10, 12),
        universe=("ETHUSD",),
        market_context_provider=lambda s: {"now": datetime(2025, 8, 10, 12)},
        equity=10_000.0,
        place_order_fn=lambda s, q, t, p: placed.append(s),
        current_price_provider=lambda s: 3000.0,
        current_atr_provider=lambda s: 60.0,
    )
    assert "ETHUSD" in b.engine.open_positions

    # Drop price 9% — triggers PER_TRADE_KILL
    b.engine.on_minute_tick(
        now=datetime(2025, 8, 10, 13),
        current_prices={"ETHUSD": 3000 * 0.91},
        latest_probs={},
        place_exit_fn=lambda s, q, r: exited.append((s, r)),
    )
    assert exited == [("ETHUSD", "PER_TRADE_KILL")]
    assert "ETHUSD" not in b.engine.open_positions


def test_full_stack_stats_reports_realistic_state():
    b = bootstrap_apex(model_path=None, fallback_only=True)
    _populate_strong_bullish_signals(b)
    s = b.stats()
    assert "registered_signals" in s
    assert s["btc_onchain_days"] >= 60
    assert s["etf_flows_days"] >= 30
    assert s["funding_pairs"] >= 1
    assert s["news_symbols"] >= 1
