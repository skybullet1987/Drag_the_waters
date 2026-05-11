"""Tests for Pulse.apex.signals.* (Phase 2 native QC alt-data signals)."""

from __future__ import annotations

import math
import pytest

from Pulse.apex.registry import SignalRegistry, SignalScore

from Pulse.apex.signals.btc_onchain import (
    BitcoinMetadataHistoryStore,
    compute_btc_onchain_score,
    make_btc_onchain_signal_fn,
    register_btc_onchain,
)
from Pulse.apex.signals.btc_dominance import (
    CoinGeckoDominanceStore,
    compute_btc_dominance_series,
    compute_btc_dominance_score,
    make_btc_dominance_signal_fn,
    register_btc_dominance,
)
from Pulse.apex.signals.funding_native import (
    BinanceFundingRateStore,
    compute_funding_score,
    compute_funding_zscore,
    kraken_to_perp,
    make_funding_signal_fn,
    register_funding_native,
    KRAKEN_TO_BINANCE_PERP,
    _piecewise_score,
)
from Pulse.apex.signals.cross_asset import (
    CrossAssetSeriesStore,
    compute_cross_asset_score,
    make_cross_asset_signal_fn,
    register_cross_asset,
)


# ───────────────────────────────────────────────────────────────────────────────
# btc_onchain
# ───────────────────────────────────────────────────────────────────────────────

def _series_with_jump(baseline: float, jump_to: float, n: int = 60,
                      jump_at: int = 50) -> list[float]:
    """Flat at baseline for `jump_at` days then steady at jump_to."""
    out = [baseline] * jump_at
    out += [jump_to] * (n - jump_at)
    return out


def test_onchain_returns_zero_when_history_too_short():
    score, meta = compute_btc_onchain_score(
        {"hash_rate": [1.0] * 10},   # need 60
    )
    assert score == 0.0
    assert meta["valid_metrics"] == 0


def test_onchain_bullish_when_metrics_jump_up():
    history = {
        m: _series_with_jump(100.0, 200.0)
        for m in ("hash_rate", "n_unique_addresses", "miners_revenue",
                  "estimated_btc_sent")
    }
    score, meta = compute_btc_onchain_score(history)
    assert score > 0.4
    assert meta["valid_metrics"] == 4


def test_onchain_bearish_when_metrics_drop():
    history = {
        m: _series_with_jump(200.0, 100.0)
        for m in ("hash_rate", "n_unique_addresses", "miners_revenue",
                  "estimated_btc_sent")
    }
    score, meta = compute_btc_onchain_score(history)
    assert score < -0.4
    assert meta["valid_metrics"] == 4


def test_onchain_neutral_when_flat():
    history = {m: [100.0] * 60 for m in (
        "hash_rate", "n_unique_addresses", "miners_revenue",
        "estimated_btc_sent",
    )}
    score, _ = compute_btc_onchain_score(history)
    # Flat series → variance = 0 → z_score returns None → contribution 0
    assert score == 0.0


def test_onchain_partial_data_uses_only_valid_metrics():
    history = {
        "hash_rate":          _series_with_jump(100, 200),
        "n_unique_addresses": [1.0] * 10,    # too short
    }
    score, meta = compute_btc_onchain_score(history)
    assert score > 0.0    # valid metric is bullish
    assert meta["valid_metrics"] == 1


def test_onchain_clamped_to_unit():
    """Even extreme z-scores should never exceed |1|."""
    history = {
        m: _series_with_jump(1.0, 100.0)
        for m in ("hash_rate", "n_unique_addresses", "miners_revenue",
                  "estimated_btc_sent")
    }
    score, _ = compute_btc_onchain_score(history)
    assert -1.0 <= score <= 1.0


def test_onchain_score_via_registry_callable():
    history = {
        m: _series_with_jump(100, 200)
        for m in ("hash_rate", "n_unique_addresses", "miners_revenue",
                  "estimated_btc_sent")
    }
    fn = make_btc_onchain_signal_fn(lambda sym, ctx: history)
    sc = fn("BTCUSD", {})
    assert isinstance(sc, SignalScore)
    assert sc.name == "btc_onchain"
    assert sc.valid is True
    assert sc.score > 0.4


def test_onchain_signal_handles_provider_exception():
    fn = make_btc_onchain_signal_fn(lambda s, c: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False
    assert "error" in sc.meta


def test_onchain_register_into_registry():
    r = SignalRegistry()
    register_btc_onchain(r, lambda s, c: {"hash_rate": _series_with_jump(100, 200)})
    assert "btc_onchain" in r
    sc = r.get("btc_onchain")("BTCUSD", {})
    assert sc.score > 0.0


# ───────────────────────────────────────────────────────────────────────────────
# BitcoinMetadataHistoryStore
# ───────────────────────────────────────────────────────────────────────────────

def test_history_store_records_and_returns():
    s = BitcoinMetadataHistoryStore(keep_days=5)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        s.record_bar("hash_rate", v)
    h = s.history()
    assert h["hash_rate"] == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_history_store_ignores_unknown_metric():
    s = BitcoinMetadataHistoryStore()
    s.record_bar("not_a_metric", 99.0)
    assert "not_a_metric" not in s.history()


# ───────────────────────────────────────────────────────────────────────────────
# btc_dominance
# ───────────────────────────────────────────────────────────────────────────────

def test_dominance_series_skips_invalid_pairs():
    out = compute_btc_dominance_series(
        [100, None, 200, 300],
        [1000, 2000, 0, 1500],
    )
    # Skip indices where btc=None or total<=0
    assert out == [100/1000, 300/1500]


def test_dominance_score_insufficient_history():
    score, meta = compute_btc_dominance_score([0.5] * 10, is_btc=True)
    assert score == 0.0
    assert meta["error"] == "insufficient_history"


def test_dominance_score_neutral_when_flat():
    score, meta = compute_btc_dominance_score([0.5] * 90, is_btc=True)
    # flat series → variance 0 → z undefined
    assert score == 0.0


def test_dominance_falling_bullish_for_alts():
    # 90 days: dominance steadily falls from 0.55 to 0.45
    series = [0.55 - 0.001 * i for i in range(90)]
    score_alt, meta_alt = compute_btc_dominance_score(series, is_btc=False)
    score_btc, meta_btc = compute_btc_dominance_score(series, is_btc=True)
    assert score_alt > 0.0
    assert score_btc < 0.0
    assert score_alt == -score_btc


def test_dominance_rising_bullish_for_btc():
    series = [0.45 + 0.001 * i for i in range(90)]
    score_alt, _ = compute_btc_dominance_score(series, is_btc=False)
    score_btc, _ = compute_btc_dominance_score(series, is_btc=True)
    assert score_alt < 0.0
    assert score_btc > 0.0


def test_dominance_clamped():
    # Massive spike at the end of a flat baseline
    series = [0.5] * 80 + [0.99] * 10
    score, _ = compute_btc_dominance_score(series, is_btc=True)
    assert score == 1.0 or score == pytest.approx(1.0, abs=1e-6)


def test_dominance_via_registry_callable():
    series = [0.55 - 0.001 * i for i in range(90)]
    fn = make_btc_dominance_signal_fn(lambda ctx: series)
    sc = fn("ETHUSD", {})
    assert sc.name == "btc_dominance"
    assert sc.valid is True
    assert sc.score > 0.0   # alts get a boost on falling dominance


def test_dominance_btc_symbol_detection():
    series = [0.55 - 0.001 * i for i in range(90)]
    fn = make_btc_dominance_signal_fn(lambda ctx: series)
    btc_sc = fn("BTCUSD", {})
    eth_sc = fn("ETHUSD", {})
    assert btc_sc.score < 0.0
    assert eth_sc.score > 0.0


def test_dominance_handles_xbt_alias():
    series = [0.55 - 0.001 * i for i in range(90)]
    fn = make_btc_dominance_signal_fn(lambda ctx: series)
    sc = fn("XBTUSD", {})
    assert sc.score < 0.0   # treated as BTC


def test_dominance_provider_exception_safe():
    fn = make_btc_dominance_signal_fn(lambda ctx: 1/0)
    sc = fn("ETHUSD", {})
    assert sc.valid is False


def test_dominance_store_drops_invalid_records():
    s = CoinGeckoDominanceStore()
    s.record(None, 1000)         # None btc
    s.record(100, None)          # None total
    s.record(100, -50)           # negative total
    assert s.dominance_series() == []


def test_dominance_store_keeps_window():
    s = CoinGeckoDominanceStore(keep_days=3)
    for i in range(1, 6):
        s.record(i * 100, i * 1000)
    dom = s.dominance_series()
    # Only last 3 ratios stored
    assert len(dom) == 3
    assert all(abs(d - 0.1) < 1e-9 for d in dom)


def test_dominance_register_into_registry():
    r = SignalRegistry()
    register_btc_dominance(
        r,
        lambda ctx: [0.55 - 0.001 * i for i in range(90)],
    )
    assert "btc_dominance" in r


# ───────────────────────────────────────────────────────────────────────────────
# funding_native
# ───────────────────────────────────────────────────────────────────────────────

def test_piecewise_extreme_negative_returns_plus_one():
    assert _piecewise_score(-0.001) == 1.0


def test_piecewise_extreme_positive_returns_minus_one():
    assert _piecewise_score(0.001) == -1.0


def test_piecewise_neutral_zero():
    assert _piecewise_score(0.0) == 0.0


def test_piecewise_monotonic_in_rate():
    """Score must monotonically decrease as rate increases."""
    rates = [-0.001, -0.0005, -0.0002, 0, 0.0002, 0.0005, 0.001]
    scores = [_piecewise_score(r) for r in rates]
    for a, b in zip(scores, scores[1:]):
        assert a >= b


def test_funding_zscore_returns_none_on_short_history():
    z = compute_funding_zscore(0.0001, [0.0001] * 3)
    assert z is None


def test_funding_zscore_zero_on_flat_history():
    z = compute_funding_zscore(0.0001, [0.0001] * 30)
    assert z is None    # var=0 → returns None


def test_funding_zscore_positive_when_rate_above_mean():
    history = [0.0001] * 30
    z = compute_funding_zscore(0.0010, history)
    assert z is None    # var=0
    history2 = [0.0001 + 0.00001 * i for i in range(30)]
    z2 = compute_funding_zscore(0.0010, history2)
    assert z2 is not None and z2 > 0


def test_funding_score_no_rate_invalid():
    score, meta = compute_funding_score(None)
    assert score == 0.0
    assert meta["error"] == "no_rate"


def test_funding_score_zscore_mode_uses_history():
    history = [0.0001 + 0.00001 * i for i in range(30)]
    score, meta = compute_funding_score(0.0010, history=history,
                                         use_zscore=True)
    assert meta["mode"] == "zscore"
    assert score < 0   # rate well above mean → bearish


def test_funding_score_falls_back_to_piecewise_with_short_history():
    score, meta = compute_funding_score(-0.001, history=[0.0001],
                                         use_zscore=True)
    assert meta["mode"] == "piecewise"
    assert score == 1.0


def test_kraken_to_perp_known():
    assert kraken_to_perp("BTCUSD") == "BTCUSDT"
    assert kraken_to_perp("ethusd") == "ETHUSDT"


def test_kraken_to_perp_unknown_returns_none():
    assert kraken_to_perp("ZZZUSD") is None


def test_kraken_to_perp_covers_pulse_universe():
    # Spot-check: at minimum 20 mappings exist (sanity for our tier list)
    assert len(KRAKEN_TO_BINANCE_PERP) >= 20


def test_funding_signal_unsupported_symbol_invalid():
    fn = make_funding_signal_fn(lambda perp, ctx: (0.0, []))
    sc = fn("ZZZUSD", {})
    assert sc.valid is False
    assert "no_perp_mapping" in sc.meta.get("error", "")


def test_funding_signal_via_registry_callable():
    history = [0.0001 + 0.00001 * i for i in range(30)]
    fn = make_funding_signal_fn(lambda perp, ctx: (0.0010, history))
    sc = fn("BTCUSD", {})
    assert sc.valid is True
    assert sc.score < 0   # crowded longs → bearish


def test_funding_signal_provider_exception_safe():
    fn = make_funding_signal_fn(lambda perp, ctx: 1/0)
    sc = fn("BTCUSD", {})
    assert sc.valid is False


def test_funding_register_into_registry():
    r = SignalRegistry()
    register_funding_native(r, lambda perp, ctx: (0.0001, []))
    assert "funding_extreme" in r


def test_funding_store_records_and_returns():
    s = BinanceFundingRateStore(keep=3)
    for v in (0.0001, 0.0002, 0.0003, 0.0004, 0.0005):
        s.record("BTCUSDT", v)
    cur, hist = s.get("BTCUSDT")
    assert cur == 0.0005
    assert hist == [0.0003, 0.0004]


def test_funding_store_unknown_symbol_returns_none():
    s = BinanceFundingRateStore()
    cur, hist = s.get("UNKNOWN")
    assert cur is None
    assert hist == []


def test_funding_store_ignores_none_rate():
    s = BinanceFundingRateStore()
    s.record("BTCUSDT", None)
    cur, _ = s.get("BTCUSDT")
    assert cur is None


# ───────────────────────────────────────────────────────────────────────────────
# cross_asset
# ───────────────────────────────────────────────────────────────────────────────

def _trending_series(start: float, daily_pct: float, n: int = 90,
                     noise: float = 0.005) -> list[float]:
    """Pseudo-random walk with drift. `noise` is per-day std (decimal).

    Deterministic — uses a simple LCG so tests are reproducible.
    """
    seed = abs(int((start * 1000 + daily_pct * 100000) * 1e6)) or 12345
    s = [start]
    for i in range(n - 1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        u = (seed / 0x7FFFFFFF) - 0.5      # uniform [-0.5, 0.5]
        s.append(s[-1] * (1 + daily_pct + noise * u))
    return s


def _trending_with_recent_push(start: float, baseline_drift: float,
                                push_pct_per_day: float,
                                push_days: int = 5,
                                n: int = 90,
                                noise: float = 0.005) -> list[float]:
    """Noisy baseline drift, then a sharp recent push for the last `push_days`."""
    base = _trending_series(start, baseline_drift, n=n - push_days, noise=noise)
    last = base[-1]
    push = []
    for _ in range(push_days):
        last = last * (1 + push_pct_per_day)
        push.append(last)
    return base + push


def test_cross_asset_score_returns_zero_with_no_data():
    score, meta = compute_cross_asset_score({}, is_alt=True)
    assert score == 0.0
    assert "error" in meta


def test_cross_asset_score_bullish_on_risk_on():
    """Recent push up in SPY + push down in DXY = risk-on regime."""
    series = {
        "spy": _trending_with_recent_push(450, 0.0, +0.015, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, -0.005, push_days=5),
        "gld": _trending_series(180, 0.0, noise=0.005),
    }
    score, meta = compute_cross_asset_score(series, is_alt=True)
    assert score > 0.0, f"expected bullish, got {score} meta={meta}"


def test_cross_asset_score_bearish_on_risk_off():
    series = {
        "spy": _trending_with_recent_push(450, 0.0, -0.015, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, +0.008, push_days=5),
        "gld": _trending_with_recent_push(180, 0.0, +0.010, push_days=5),
    }
    score, meta = compute_cross_asset_score(series, is_alt=True)
    assert score < 0.0, f"expected bearish, got {score} meta={meta}"


def test_cross_asset_alt_beta_amplifies():
    series = {
        "spy": _trending_with_recent_push(450, 0.0, +0.015, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, -0.005, push_days=5),
        "gld": _trending_series(180, 0.0, noise=0.005),
    }
    btc_score, _ = compute_cross_asset_score(series, is_alt=False)
    alt_score, _ = compute_cross_asset_score(series, is_alt=True)
    # Alt should be ≥ BTC (positive regime) due to alt_beta multiplier
    assert alt_score >= btc_score


def test_cross_asset_partial_data():
    """Missing GLD shouldn't kill the signal; it just reduces weight."""
    series = {
        "spy": _trending_with_recent_push(450, 0.0, +0.015, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, -0.005, push_days=5),
        # gld omitted
    }
    score, meta = compute_cross_asset_score(series, is_alt=True)
    assert score > 0.0, f"expected bullish, got {score} meta={meta}"
    assert meta["raw_z"]["gld"] is None
    assert meta["raw_z"]["spy"] is not None
    assert meta["raw_z"]["dxy"] is not None


def test_cross_asset_clamped():
    """Extreme moves should not exceed |1|."""
    series = {
        "spy": _trending_with_recent_push(450, 0.0, +0.20, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, -0.05, push_days=5),
        "gld": _trending_series(180, 0.0, noise=0.005),
    }
    score, _ = compute_cross_asset_score(series, is_alt=True)
    assert -1.0 <= score <= 1.0


def test_cross_asset_via_registry_callable():
    series = {
        "spy": _trending_with_recent_push(450, 0.0, +0.015, push_days=5),
        "dxy": _trending_with_recent_push(105, 0.0, -0.005, push_days=5),
        "gld": _trending_series(180, 0.0, noise=0.005),
    }
    fn = make_cross_asset_signal_fn(lambda ctx: series)
    sc = fn("BTCUSD", {})
    assert sc.name == "cross_asset_macro"
    assert sc.valid is True


def test_cross_asset_provider_exception_safe():
    fn = make_cross_asset_signal_fn(lambda ctx: 1/0)
    sc = fn("ETHUSD", {})
    assert sc.valid is False


def test_cross_asset_register_into_registry():
    r = SignalRegistry()
    series = {"spy": _trending_series(450, +0.005),
              "dxy": _trending_series(105, -0.001),
              "gld": _trending_series(180,  0.000)}
    register_cross_asset(r, lambda ctx: series)
    assert "cross_asset_macro" in r


def test_cross_asset_store_records_and_returns():
    s = CrossAssetSeriesStore(keep_days=3)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        s.record("spy", v)
    series = s.series()
    assert series["spy"] == [3.0, 4.0, 5.0]


def test_cross_asset_store_ignores_unknown_asset():
    s = CrossAssetSeriesStore()
    s.record("unknown", 99.0)
    assert s.series() == {"spy": [], "dxy": [], "gld": []}


def test_cross_asset_store_ignores_none_close():
    s = CrossAssetSeriesStore()
    s.record("spy", None)
    assert s.series()["spy"] == []
