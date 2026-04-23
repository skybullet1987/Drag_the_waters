from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from kraken_client import TokenBucket, fetch_ohlcv
from svm_wavelet import SvmSignalModel, SvmWaveletStrategy, WalkForwardConfig, causal_wavelet_features, make_cost_aware_labels


class MockResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


class DummyModel:
    def predict(self, X):
        return pd.DataFrame({"signal": [1], "confidence": [0.9], "edge": [0.001]}, index=X.index)


class DummySlippage:
    def GetSlippageApproximation(self, asset, order):
        return asset.Price * 0.002


def test_wavelet_causality():
    idx = pd.date_range("2025-01-01", periods=300, freq="h")
    s1 = pd.Series(np.linspace(100, 130, 300), index=idx)
    s2 = s1.copy()
    s2.iloc[-1] += 1000.0

    f1 = causal_wavelet_features(s1, window=64, level=2)
    f2 = causal_wavelet_features(s2, window=64, level=2)

    pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1], check_dtype=False)


def test_svm_signal_smoke():
    rng = np.random.default_rng(7)
    n = 600
    r = np.zeros(n)
    noise = rng.normal(0, 0.001, n)
    for t in range(1, n):
        r[t] = 0.85 * r[t - 1] + noise[t]
    close = 100 * np.exp(np.cumsum(r))

    close = pd.Series(close)
    X = pd.DataFrame(
        {
            "ret1": close.pct_change(1),
            "ret2": close.pct_change(2),
            "ret3": close.pct_change(3),
        }
    )
    y = make_cost_aware_labels(close, horizon=1, k=0.0, round_trip_cost=0.0)

    model = SvmSignalModel(C=1.0, gamma="scale", neutral_threshold=0.52)
    wf = WalkForwardConfig(mode="expanding", min_train_size=120, retrain_every=10, drop_flat=True)
    preds = model.walk_forward_predict(X, y, config=wf).dropna()

    y_true = y.loc[preds.index]
    y_pred = preds["signal"].astype(int)

    active = y_pred != 0
    hit_rate = (y_true[active] == y_pred[active]).mean()

    assert hit_rate > 0.5
    assert set(y_pred.unique()).issubset({-1, 0, 1})


def test_strategy_respects_costs():
    strat = SvmWaveletStrategy(
        model=DummyModel(),
        config={"window": 16, "level": 2, "round_trip_cost": 0.0052},
        slippage_model=DummySlippage(),
        circuit_breaker=None,
    )

    for i in range(20):
        bar = {
            "timestamp": f"2025-01-01T{i:02d}:00:00Z",
            "open": 100 + i,
            "high": 101 + i,
            "low": 99 + i,
            "close": 100 + i,
            "volume": 1000,
        }
        out = strat.on_bar(bar, context={"timestamp": bar["timestamp"]})

    assert out["target_position"] == 0


def test_kraken_client_mocked():
    session = Mock()
    session.get.return_value = MockResponse(
        {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    [1700000000, "100", "110", "90", "105", "103", "12.3", "42"],
                    [1700003600, "105", "112", "101", "108", "107", "10.1", "35"],
                ],
                "last": "1700003600",
            },
        }
    )

    df = fetch_ohlcv("XBT/USD", interval=60, session=session)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "vwap", "volume", "count"]
    assert float(df.iloc[-1]["close"]) == 108.0

    bucket = TokenBucket(rate_per_sec=1.0, capacity=1.0)
    with patch("kraken_client.time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.2, 1.2, 1.2]), patch(
        "kraken_client.time.sleep"
    ) as sleep:
        bucket.last_refill = 0.0
        bucket.tokens = 1.0
        bucket.acquire()
        bucket.acquire()
        assert sleep.called
