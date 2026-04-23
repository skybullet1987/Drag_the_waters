import sys
import types

_algo_imports = sys.modules.setdefault("AlgorithmImports", types.ModuleType("AlgorithmImports"))
if not hasattr(_algo_imports, "PythonData"):
    _algo_imports.PythonData = type("PythonData", (), {})

from svm_wavelet_qc import SvmWaveletForecaster, svm_allows_entry


class DummyForecaster:
    def __init__(self, forecast):
        self.forecast = forecast

    def update_and_forecast(self, symbol, prices):
        return self.forecast


class DummyAlgo:
    def __init__(self, enabled=True, forecast=100.0):
        self._svm_filter_enabled = enabled
        prices = [100.0] * SvmWaveletForecaster.LOOKBACK
        self.crypto_data = {"BTCUSD": {"svm_prices": prices}}
        self._svm_forecaster = DummyForecaster(forecast)
        self.expected_round_trip_fees = 0.006
        self.fee_slippage_buffer = 0.002
        self.min_expected_profit_pct = 0.020
        self._last_skip_reason = None


def test_partition_shapes():
    X, y = SvmWaveletForecaster._partition([1, 2, 3, 4, 5], 3)
    assert X.shape == (2, 3)
    assert y.shape == (2,)


def test_svm_allows_entry_fail_open_when_disabled():
    algo = DummyAlgo(enabled=False, forecast=90.0)
    assert svm_allows_entry(algo, "BTCUSD", 1) is True


def test_svm_allows_entry_blocks_on_low_edge():
    algo = DummyAlgo(enabled=True, forecast=101.0)
    assert svm_allows_entry(algo, "BTCUSD", 1) is False
    assert algo._last_skip_reason.startswith("svm_block long")


def test_svm_allows_entry_allows_high_edge():
    algo = DummyAlgo(enabled=True, forecast=104.0)
    assert svm_allows_entry(algo, "BTCUSD", 1) is True
