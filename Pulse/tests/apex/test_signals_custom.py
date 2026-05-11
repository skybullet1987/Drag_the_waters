"""Tests for Pulse.apex.data.* (Phase 3 custom-data signals)."""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from Pulse.apex.registry import SignalRegistry, SignalScore

from Pulse.apex.data.etf_flows import (
    ETFFlowStore,
    compute_etf_flow_score,
    make_etf_flow_signal_fn,
    parse_etf_csv,
    register_etf_flow,
)
from Pulse.apex.data.stablecoin_supply import (
    StablecoinSupplyStore,
    compute_stablecoin_score,
    make_stablecoin_signal_fn,
    register_stablecoin,
)
from Pulse.apex.data.token_unlocks import (
    UnlockCalendarStore,
    UnlockEvent,
    compute_token_unlock_score,
    make_token_unlock_signal_fn,
    parse_unlock_csv,
    register_token_unlock,
    upcoming_unlocks,
)
from Pulse.apex.data.news_sentiment import (
    NewsSentimentStore,
    compute_news_sentiment_score,
    make_news_sentiment_signal_fn,
    parse_sentiment_csv,
    register_news_sentiment,
)
from Pulse.apex.data.onchain_valuation import (
    OnchainValuationStore,
    compute_mvrv_series,
    compute_onchain_valuation_score,
    compute_realized_cap_proxy,
    compute_nupl_series,
    make_mvrv_signal_fn,
    register_mvrv,
)


# ───────────────────────────────────────────────────────────────────────────────
# etf_flows
# ───────────────────────────────────────────────────────────────────────────────

def _flow_series_with_recent_inflow(baseline: float, inflow: float,
                                     n: int = 30, recent: int = 3) -> list[float]:
    return [baseline] * (n - recent) + [inflow] * recent


def test_etf_score_zero_with_short_history():
    score, meta = compute_etf_flow_score([100.0] * 10, is_btc=True)
    assert score == 0.0
    assert "insufficient_history" in meta["error"]


def test_etf_score_bullish_on_recent_inflow():
    flows = _flow_series_with_recent_inflow(0, 500, n=30, recent=3)
    score, meta = compute_etf_flow_score(flows, is_btc=True)
    assert score > 0.0
    assert meta["z"] > 0


def test_etf_score_bearish_on_recent_outflow():
    flows = _flow_series_with_recent_inflow(0, -500, n=30, recent=3)
    score, _ = compute_etf_flow_score(flows, is_btc=True)
    assert score < 0.0


def test_etf_score_alts_attenuated_vs_btc():
    flows = _flow_series_with_recent_inflow(0, 500, n=30, recent=3)
    btc_score, _ = compute_etf_flow_score(flows, is_btc=True)
    alt_score, _ = compute_etf_flow_score(flows, is_btc=False)
    assert abs(alt_score) < abs(btc_score)


def test_etf_score_clamped():
    flows = _flow_series_with_recent_inflow(0, 100000, n=30, recent=3)
    score, _ = compute_etf_flow_score(flows, is_btc=True)
    assert -1.0 <= score <= 1.0


def test_parse_etf_csv_two_column():
    s = "Date,Inflow_USD_M\n2025-01-01,100\n2025-01-02,200\n"
    rows = parse_etf_csv(s)
    assert rows == [("2025-01-01", 100.0), ("2025-01-02", 200.0)]


def test_parse_etf_csv_consolidated():
    s = "Date,IBIT,FBTC,ARKB\n2025-01-01,100,50,25\n"
    rows = parse_etf_csv(s)
    assert rows == [("2025-01-01", 175.0)]


def test_parse_etf_csv_handles_dashes_and_dollar_prefix():
    """Farside CSVs use dashes for missing values and sometimes $ prefix
    (without comma thousands-separators)."""
    s = "Date,IBIT,FBTC\n2025-01-01,$1234.56,-\n2025-01-02,100,200\n"
    rows = parse_etf_csv(s)
    assert len(rows) == 2
    assert rows[0][1] == pytest.approx(1234.56)
    assert rows[1][1] == 300.0


def test_parse_etf_csv_skips_malformed_lines():
    s = "Date,Inflow\n2025-01-01,bad\n2025-01-02,200\n"
    rows = parse_etf_csv(s)
    assert rows == [("2025-01-02", 200.0)]


def test_etf_store_load_csv_and_get():
    store = ETFFlowStore(keep_days=10)
    store.load_csv("Date,Inflow\n2025-01-01,100\n2025-01-02,200\n")
    assert store.get() == [100.0, 200.0]


def test_etf_store_keep_days_window():
    store = ETFFlowStore(keep_days=2)
    for v in (1, 2, 3, 4, 5):
        store.record(v)
    assert store.get() == [4.0, 5.0]


def test_etf_signal_fn_via_provider():
    flows = _flow_series_with_recent_inflow(0, 500, n=30, recent=3)
    fn = make_etf_flow_signal_fn(lambda ctx: flows)
    sc = fn("BTCUSD", {})
    assert sc.name == "etf_flow"
    assert sc.score > 0
    assert sc.valid is True


def test_etf_signal_provider_exception_safe():
    fn = make_etf_flow_signal_fn(lambda ctx: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False


def test_etf_register_into_registry():
    r = SignalRegistry()
    register_etf_flow(r, lambda ctx: [0] * 30)
    assert "etf_flow" in r


# ───────────────────────────────────────────────────────────────────────────────
# stablecoin_supply
# ───────────────────────────────────────────────────────────────────────────────

def _supply_with_recent_change(base: float, recent_change: float,
                                n: int = 35) -> list[float]:
    """Build a stablecoin supply series whose final 3-day delta = recent_change.

    Earlier days have a small random-walk drift to give the baseline
    3-day-delta distribution non-zero variance so the z-score is well-defined.
    """
    seed = abs(int((base + recent_change) * 1e3)) or 12345
    out = [base]
    for _ in range(n - 4):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        u = (seed / 0x7FFFFFFF) - 0.5
        out.append(out[-1] + 1e7 * u)
    # Last 3 ticks: linear ramp adding `recent_change` to the prior level
    last = out[-1]
    for k in range(1, 4):
        out.append(last + recent_change * k / 3)
    return out


def test_stablecoin_score_zero_when_too_short():
    score, meta = compute_stablecoin_score([100] * 5, [100] * 5)
    assert score == 0.0
    assert "insufficient" in meta["error"]


def test_stablecoin_score_bullish_on_recent_mint():
    usdt = _supply_with_recent_change(100_000_000_000.0, +5_000_000_000.0)
    usdc = _supply_with_recent_change( 50_000_000_000.0, +2_000_000_000.0)
    score, meta = compute_stablecoin_score(usdt, usdc)
    assert score > 0.0, f"expected bullish, got {score} meta={meta}"


def test_stablecoin_score_bearish_on_recent_burn():
    usdt = _supply_with_recent_change(100_000_000_000.0, -5_000_000_000.0)
    usdc = _supply_with_recent_change( 50_000_000_000.0, -2_000_000_000.0)
    score, _ = compute_stablecoin_score(usdt, usdc)
    assert score < 0.0


def test_stablecoin_neutral_on_flat_supply():
    flat = [1e11] * 35
    score, meta = compute_stablecoin_score(flat, flat)
    assert score == 0.0


def test_stablecoin_score_clamped():
    usdt = _supply_with_recent_change(1e11, +1e15)
    usdc = _supply_with_recent_change(5e10, +1e15)
    score, _ = compute_stablecoin_score(usdt, usdc)
    assert -1.0 <= score <= 1.0


def test_stablecoin_signal_fn():
    usdt = _supply_with_recent_change(100_000_000_000.0, +5_000_000_000.0)
    usdc = _supply_with_recent_change( 50_000_000_000.0, +2_000_000_000.0)
    fn = make_stablecoin_signal_fn(lambda ctx: (usdt, usdc))
    sc = fn("BTCUSD", {})
    assert sc.name == "stablecoin_mint"
    assert sc.score > 0


def test_stablecoin_signal_provider_exception_safe():
    fn = make_stablecoin_signal_fn(lambda ctx: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False


def test_stablecoin_register_into_registry():
    r = SignalRegistry()
    register_stablecoin(r, lambda ctx: ([1e11] * 35, [5e10] * 35))
    assert "stablecoin_mint" in r


def test_stablecoin_store_load_csv():
    csv = "Date,USDT,USDC\n2025-01-01,1e11,5e10\n2025-01-02,1.1e11,5.5e10\n"
    s = StablecoinSupplyStore()
    s.load_csv(csv)
    ut, uc = s.get()
    assert ut == [1e11, 1.1e11]
    assert uc == [5e10, 5.5e10]


def test_stablecoin_store_keep_days():
    s = StablecoinSupplyStore(keep_days=2)
    for i in range(5):
        s.record(1e11 + i, 5e10 + i)
    ut, uc = s.get()
    assert len(ut) == 2 and len(uc) == 2


# ───────────────────────────────────────────────────────────────────────────────
# token_unlocks
# ───────────────────────────────────────────────────────────────────────────────

def test_parse_unlock_csv():
    s = "Date,Symbol,UnlockUSD,UnlockPctOfSupply\n" \
        "2025-08-15,ARB,500000000,0.025\n" \
        "2025-09-01,OP,200000000,0.012\n"
    events = parse_unlock_csv(s)
    assert len(events) == 2
    assert events[0].symbol == "ARB"
    assert events[0].pct_of_supply == 0.025


def test_parse_unlock_csv_skips_bad_rows():
    s = "Date,Symbol,UnlockUSD,UnlockPctOfSupply\nbad row\n2025-08-15,ARB,500,0.025\n"
    events = parse_unlock_csv(s)
    assert len(events) == 1


def test_upcoming_unlocks_filters_by_window():
    now = datetime(2025, 8, 10)
    events = [
        UnlockEvent(datetime(2025, 8, 15), "ARB", 500e6, 0.025),
        UnlockEvent(datetime(2025, 8, 25), "ARB", 100e6, 0.005),  # too far
        UnlockEvent(datetime(2025, 8, 12), "OP", 200e6, 0.012),   # wrong sym
    ]
    out = upcoming_unlocks(events, "ARBUSD", now, look_ahead_days=7)
    assert len(out) == 1
    assert out[0].symbol == "ARB"


def test_upcoming_unlocks_threshold_filters_small_unlocks():
    now = datetime(2025, 8, 10)
    events = [UnlockEvent(datetime(2025, 8, 15), "ARB", 100, 0.005)]
    out = upcoming_unlocks(events, "ARBUSD", now,
                            look_ahead_days=7, pct_threshold=0.01)
    assert out == []


def test_token_unlock_score_no_upcoming():
    score, meta = compute_token_unlock_score([], "BTCUSD", datetime(2025, 1, 1))
    assert score == 0.0
    assert meta["upcoming_count"] == 0


def test_token_unlock_score_negative_for_pending_unlock():
    now = datetime(2025, 8, 10)
    events = [UnlockEvent(datetime(2025, 8, 15), "ARB", 500e6, 0.025)]
    score, meta = compute_token_unlock_score(events, "ARBUSD", now)
    assert score < 0.0
    assert meta["upcoming_count"] == 1
    assert meta["biggest_pct"] == 0.025


def test_token_unlock_score_full_penalty_at_5pct():
    now = datetime(2025, 8, 10)
    events = [UnlockEvent(datetime(2025, 8, 15), "ARB", 1e9, 0.05)]
    score, _ = compute_token_unlock_score(events, "ARBUSD", now)
    assert score == -1.0


def test_token_unlock_score_clamps_above_threshold():
    now = datetime(2025, 8, 10)
    events = [UnlockEvent(datetime(2025, 8, 15), "ARB", 5e9, 0.50)]
    score, _ = compute_token_unlock_score(events, "ARBUSD", now)
    assert score == -1.0


def test_token_unlock_signal_fn_includes_now_provider():
    now = datetime(2025, 8, 10)
    events = [UnlockEvent(datetime(2025, 8, 15), "ARB", 500e6, 0.025)]
    fn = make_token_unlock_signal_fn(lambda c: events, lambda c: now)
    sc = fn("ARBUSD", {})
    assert sc.score < 0
    assert sc.valid is True


def test_token_unlock_signal_provider_exception_safe():
    fn = make_token_unlock_signal_fn(lambda c: 1/0, lambda c: datetime(2025, 1, 1))
    sc = fn("ARBUSD", {})
    assert sc.valid is False


def test_token_unlock_register_into_registry():
    r = SignalRegistry()
    register_token_unlock(r, lambda c: [], lambda c: datetime(2025, 1, 1))
    assert "token_unlock" in r


def test_unlock_calendar_store_load_csv():
    s = UnlockCalendarStore()
    s.load_csv("Date,Symbol,UnlockUSD,UnlockPctOfSupply\n"
               "2025-08-15,ARB,5e8,0.025\n")
    assert len(s.get()) == 1


# ───────────────────────────────────────────────────────────────────────────────
# news_sentiment
# ───────────────────────────────────────────────────────────────────────────────

def test_parse_sentiment_csv():
    s = "Date,Symbol,Articles,SentimentMean,SentimentStd\n" \
        "2025-01-01,BTCUSD,5,0.32,0.15\n" \
        "2025-01-02,BTCUSD,7,-0.15,0.20\n"
    out = parse_sentiment_csv(s)
    assert "BTCUSD" in out
    assert out["BTCUSD"] == [0.32, -0.15]


def test_parse_sentiment_csv_skips_bad_rows():
    s = "Date,Symbol,Articles,SentimentMean\nbad row\n"
    out = parse_sentiment_csv(s)
    assert out == {}


def test_news_sentiment_score_zero_with_short_history():
    score, _ = compute_news_sentiment_score([0.1] * 10)
    assert score == 0.0


def test_news_sentiment_score_bullish_on_recent_pump():
    series = [0.1] * 27 + [0.6, 0.7, 0.8]
    score, _ = compute_news_sentiment_score(series)
    assert score > 0.0


def test_news_sentiment_score_bearish_on_recent_dump():
    series = [0.1] * 27 + [-0.6, -0.7, -0.8]
    score, _ = compute_news_sentiment_score(series)
    assert score < 0.0


def test_news_sentiment_score_clamped():
    series = [0.0] * 27 + [10.0] * 3
    score, _ = compute_news_sentiment_score(series)
    assert -1.0 <= score <= 1.0


def test_news_sentiment_neutral_on_flat_series():
    """Flat series → no variance → 0 score."""
    score, _ = compute_news_sentiment_score([0.5] * 30)
    assert score == 0.0


def test_news_sentiment_signal_fn():
    series = [0.1] * 27 + [0.6, 0.7, 0.8]
    fn = make_news_sentiment_signal_fn(lambda sym, ctx: series)
    sc = fn("BTCUSD", {})
    assert sc.name == "news_sentiment"
    assert sc.score > 0


def test_news_sentiment_provider_exception_safe():
    fn = make_news_sentiment_signal_fn(lambda sym, ctx: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False


def test_news_sentiment_register_into_registry():
    r = SignalRegistry()
    register_news_sentiment(r, lambda sym, ctx: [0.0] * 30)
    assert "news_sentiment" in r


def test_news_sentiment_store_records_per_symbol():
    s = NewsSentimentStore()
    s.record("BTCUSD", 0.3)
    s.record("ETHUSD", -0.1)
    s.record("BTCUSD", 0.5)
    assert s.get("BTCUSD") == [0.3, 0.5]
    assert s.get("ETHUSD") == [-0.1]


def test_news_sentiment_store_load_csv():
    s = NewsSentimentStore()
    csv = ("Date,Symbol,Articles,SentimentMean\n"
           "2025-01-01,BTCUSD,5,0.32\n"
           "2025-01-02,BTCUSD,7,-0.15\n"
           "2025-01-01,ETHUSD,3,0.10\n")
    s.load_csv(csv)
    assert s.get("BTCUSD") == [0.32, -0.15]
    assert s.get("ETHUSD") == [0.10]


# ───────────────────────────────────────────────────────────────────────────────
# onchain_valuation (MVRV)
# ───────────────────────────────────────────────────────────────────────────────

def _trending_btc(start_price: float, daily_pct: float, n: int = 400,
                  noise: float = 0.02) -> tuple[list[float], list[float]]:
    """Returns (price, txn_volume) for BTC; deterministic noise."""
    seed = abs(int((start_price + n + daily_pct * 1000) * 1e3)) or 12345
    p = [start_price]
    v = [1e6]
    for _ in range(n - 1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        u = (seed / 0x7FFFFFFF) - 0.5
        p.append(p[-1] * (1 + daily_pct + noise * u))
        # Volume oscillates around 1M with some noise
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        u2 = (seed / 0x7FFFFFFF) - 0.5
        v.append(1e6 * (1 + 0.3 * u2))
    return p, v


def test_realized_cap_proxy_returns_same_length_as_input():
    p = [100, 101, 102, 103]
    v = [1, 1, 1, 1]
    rc = compute_realized_cap_proxy(p, v, lookback=2)
    assert len(rc) == len(p)


def test_realized_cap_proxy_equals_vwap_on_full_window():
    p = [100, 200, 300]
    v = [1, 2, 3]
    rc = compute_realized_cap_proxy(p, v, lookback=10)
    expected = (100 * 1 + 200 * 2 + 300 * 3) / (1 + 2 + 3)
    assert rc[-1] == pytest.approx(expected)


def test_mvrv_series_skips_invalid_realized():
    p = [100, 200, 300]
    rc = [50, 0, 100]
    out = compute_mvrv_series(p, rc)
    assert out == [2.0, 3.0]


def test_nupl_series_skips_invalid_price():
    p = [100, 0, 200]
    rc = [50, 25, 100]
    out = compute_nupl_series(p, rc)
    assert len(out) == 2
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)


def test_onchain_valuation_score_zero_when_too_short():
    score, meta = compute_onchain_valuation_score([1.0] * 10)
    assert score == 0.0
    assert "insufficient" in meta["error"]


def test_onchain_valuation_score_bearish_when_mvrv_high():
    """Recent MVRV well above its 1-year baseline → bearish (top of cycle)."""
    series = [1.0] * 360 + [3.0] * 5
    score, meta = compute_onchain_valuation_score(series)
    assert score < 0
    assert meta["z"] > 0


def test_onchain_valuation_score_bullish_when_mvrv_low():
    """MVRV << baseline → bottom of cycle → bullish."""
    series = [3.0] * 360 + [0.5] * 5
    score, _ = compute_onchain_valuation_score(series)
    assert score > 0


def test_onchain_valuation_score_clamped():
    series = [1.0] * 360 + [100.0] * 5
    score, _ = compute_onchain_valuation_score(series)
    assert -1.0 <= score <= 1.0


def test_mvrv_signal_fn_via_provider():
    series = [1.0] * 360 + [3.0] * 5
    fn = make_mvrv_signal_fn(lambda ctx: series)
    sc = fn("BTCUSD", {})
    assert sc.name == "mvrv"
    assert sc.score < 0


def test_mvrv_provider_exception_safe():
    fn = make_mvrv_signal_fn(lambda ctx: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False


def test_mvrv_register_into_registry():
    r = SignalRegistry()
    register_mvrv(r, lambda ctx: [1.0] * 365)
    assert "mvrv" in r


def test_onchain_valuation_store_record_and_mvrv():
    s = OnchainValuationStore(keep_days=400)
    p, v = _trending_btc(50000, 0.001, n=400)
    for price, vol in zip(p, v):
        s.record(price, vol)
    series = s.mvrv_series()
    assert len(series) == 400
    assert all(isinstance(x, float) for x in series)


def test_onchain_valuation_store_keep_days_window():
    s = OnchainValuationStore(keep_days=3)
    for i in range(5):
        s.record(50000 + i, 1e6)
    assert len(s.prices) == 3
