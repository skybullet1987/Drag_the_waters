# region imports
from AlgorithmImports import *
import numpy as np
import pywt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# endregion


class SvmWaveletForecaster:
    """
    Per-symbol SVR-on-wavelet-coefficients forecaster, adapted from the QC
    HandsOnAITradingBook FX SVM Wavelet sample.

    Differences vs the textbook sample (chosen for survival in a
    minute-resolution multi-symbol crypto backtest):
      - Single shared GridSearchCV is cheap (2x2 grid, not 6x5).
      - Model is REFIT every `refit_every_bars` bars per symbol, not every bar.
        Forecasts in between use the cached model.
      - Lookback fixed at 152 (minimum required for sym10 + 3 decomposition levels).
      - Returns NaN until the rolling window is full.
    """

    WAVELET = 'sym10'
    LOOKBACK = 152
    THRESHOLD = 0.5
    SVR_GRID = {'C': [0.5, 5.0], 'epsilon': [0.005, 0.05]}

    def __init__(self, refit_every_bars=60, sample_size=10):
        self.refit_every_bars = refit_every_bars
        self.sample_size = sample_size
        self._models = {}         # symbol -> fitted SVR
        self._bars_since_fit = {} # symbol -> int

    def update_and_forecast(self, symbol, price_window):
        """
        price_window: 1-D iterable of recent prices, oldest first, length >= LOOKBACK.
        Returns predicted next-bar price, or None if window too short.
        """
        if price_window is None or len(price_window) < self.LOOKBACK:
            return None
        data = np.asarray(price_window[-self.LOOKBACK:], dtype=float)
        if not np.all(np.isfinite(data)) or data.min() <= 0:
            return None

        bars = self._bars_since_fit.get(symbol, self.refit_every_bars)
        need_refit = bars >= self.refit_every_bars or symbol not in self._models

        try:
            w = pywt.Wavelet(self.WAVELET)
            coeffs = pywt.wavedec(data, w)
            for i in range(len(coeffs)):
                if i > 0:
                    coeffs[i] = pywt.threshold(coeffs[i], self.THRESHOLD * np.max(np.abs(coeffs[i])))
                forecast = self._svr_forecast(symbol, coeffs[i], allow_refit=need_refit and i == 0)
                if forecast is None:
                    return None
                coeffs[i] = np.roll(coeffs[i], -1)
                coeffs[i][-1] = forecast
            recon = pywt.waverec(coeffs, w)
            self._bars_since_fit[symbol] = 0 if need_refit else bars + 1
            return float(recon[-1])
        except Exception:
            return None

    def _svr_forecast(self, symbol, series, allow_refit):
        if len(series) <= self.sample_size + 1:
            return float(series[-1])
        X, y = self._partition(series, self.sample_size)
        if X.shape[0] < 5:
            return float(series[-1])
        if allow_refit:
            try:
                gsc = GridSearchCV(SVR(), self.SVR_GRID, scoring='neg_mean_squared_error', cv=3)
                self._models[symbol] = gsc.fit(X, y).best_estimator_
            except Exception:
                return float(series[-1])
        model = self._models.get(symbol)
        if model is None:
            return float(series[-1])
        return float(model.predict(series[np.newaxis, -self.sample_size:])[0])

    @staticmethod
    def _partition(arr, size):
        n = len(arr) - size
        if n <= 0:
            return np.empty((0, size)), np.empty((0,))
        X = np.array([arr[i:i + size] for i in range(n)])
        y = np.array([arr[i + size] for i in range(n)])
        return X, y


def svm_allows_entry(algo, symbol, side):
    if not getattr(algo, "_svm_filter_enabled", False):
        return True
    crypto = algo.crypto_data.get(symbol)
    if not crypto:
        return True
    prices = list(crypto.get('svm_prices', crypto.get('prices', [])))
    if len(prices) < SvmWaveletForecaster.LOOKBACK:
        return True
    forecast = algo._svm_forecaster.update_and_forecast(symbol, prices)
    if forecast is None:
        return True
    current = prices[-1]
    if current <= 0:
        return True
    edge = (forecast - current) / current
    gate = algo.expected_round_trip_fees + algo.fee_slippage_buffer + algo.min_expected_profit_pct
    if side > 0 and edge < gate:
        algo._last_skip_reason = f"svm_block long edge={edge:.4f} gate={gate:.4f}"
        return False
    if side < 0 and -edge < gate:
        algo._last_skip_reason = f"svm_block short edge={-edge:.4f} gate={gate:.4f}"
        return False
    return True
