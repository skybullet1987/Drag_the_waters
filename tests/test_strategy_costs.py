from strategy.svm_wavelet import SvmWaveletStrategy


class DummyModel:
    def predict(self, X):
        import pandas as pd
        return pd.DataFrame({"signal": [1], "confidence": [0.9], "edge": [0.001]}, index=X.index)


class DummySlippage:
    def GetSlippageApproximation(self, asset, order):
        return asset.Price * 0.002


def test_strategy_blocks_low_edge_signal():
    strat = SvmWaveletStrategy(
        model=DummyModel(),
        config={"window": 16, "level": 2, "round_trip_cost": 0.0052},
        slippage_model=DummySlippage(),
        circuit_breaker=None,
    )

    for i in range(20):
        bar = {"timestamp": f"2025-01-01T{i:02d}:00:00Z", "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 1000}
        out = strat.on_bar(bar, context={"timestamp": bar["timestamp"]})

    assert out["target_position"] == 0
