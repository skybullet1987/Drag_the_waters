# forecast_features.py — Chronos + Wavelet forecast feature generators
#
# Provides forward-looking features from:
#   1. Amazon Chronos T5-tiny (pretrained transformer, no training needed)
#   2. Wavelet+SVR decomposition forecast (denoised price prediction)
#
# Both produce a single float: forecasted return relative to current price.
# These are used as FEATURES for the shallow tree classifiers, creating
# the "Transformer → XGBoost" hybrid architecture used by top crypto firms.
#
# QC packages: chronos-forecasting==2.2.2, torch==2.8.0, PyWavelets==1.9.0

import numpy as np

# ── Chronos (lazy load — heavy import) ───────────────────────────────────────
_chronos_pipeline = None
_chronos_available = None


def _load_chronos():
    """Lazy-load Chronos pipeline (only on first call)."""
    global _chronos_pipeline, _chronos_available
    if _chronos_available is not None:
        return _chronos_available
    try:
        import torch
        from chronos import ChronosPipeline
        _chronos_pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        _chronos_available = True
    except Exception:
        _chronos_available = False
    return _chronos_available


def chronos_forecast_return(closes, prediction_steps=4):
    """Generate forecast return using pretrained Chronos T5-tiny.

    Parameters
    ----------
    closes : array-like of float, length >= 32
        Recent closing prices.
    prediction_steps : int
        Number of bars to forecast ahead.

    Returns
    -------
    float
        Forecasted return: (median_forecast[-1] / closes[-1]) - 1.
        Returns 0.0 on any failure.
    """
    if len(closes) < 32:
        return 0.0
    if not _load_chronos():
        return 0.0
    try:
        import torch
        context = torch.tensor(closes[-64:], dtype=torch.float32)
        forecast = _chronos_pipeline.predict(
            context.unsqueeze(0), prediction_steps
        )
        median = np.quantile(forecast[0].numpy(), 0.5, axis=0)
        last_price = float(closes[-1])
        if last_price <= 0:
            return 0.0
        return float(median[-1] / last_price - 1.0)
    except Exception:
        return 0.0


# ── Wavelet + SVR (no heavy imports) ─────────────────────────────────────────

_pywt_available = None


def _check_pywt():
    global _pywt_available
    if _pywt_available is not None:
        return _pywt_available
    try:
        import pywt
        _pywt_available = True
    except Exception:
        _pywt_available = False
    return _pywt_available


def wavelet_forecast_return(closes, wavelet='db4', threshold_frac=0.5):
    """Generate forecast return using Wavelet decomposition + SVR.

    Decomposes price series into wavelet components, denoises via
    thresholding, forecasts each component 1 step ahead with SVR,
    then recombines for aggregate forecast.

    Based on QuantConnect HandsOnAI book implementation.

    Parameters
    ----------
    closes : array-like of float, length >= 64
        Recent closing prices.
    wavelet : str
        Wavelet family (default 'db4' — Daubechies 4, good for financial data).
    threshold_frac : float
        Fraction of max coefficient for soft thresholding.

    Returns
    -------
    float
        Forecasted return: (wavelet_forecast / closes[-1]) - 1.
        Returns 0.0 on any failure.
    """
    if len(closes) < 64 or not _check_pywt():
        return 0.0
    try:
        import pywt
        from sklearn.svm import SVR

        data = np.asarray(closes[-128:], dtype=float)
        if len(data) < 64:
            return 0.0

        coeffs = pywt.wavedec(data, wavelet)

        for i in range(len(coeffs)):
            if i > 0:
                thr = threshold_frac * np.max(np.abs(coeffs[i]))
                coeffs[i] = pywt.threshold(coeffs[i], thr, mode='soft')
            forecasted = _svr_forecast_component(coeffs[i])
            coeffs[i] = np.roll(coeffs[i], -1)
            coeffs[i][-1] = forecasted

        reconstructed = pywt.waverec(coeffs, wavelet)
        forecast_price = float(reconstructed[-1])
        last_price = float(closes[-1])
        if last_price <= 0:
            return 0.0
        return float(forecast_price / last_price - 1.0)
    except Exception:
        return 0.0


def _svr_forecast_component(component, sample_size=8):
    """Forecast 1 step ahead for a single wavelet component using SVR.

    Uses fixed hyperparameters (no GridSearchCV) for speed.
    """
    from sklearn.svm import SVR

    data = np.asarray(component, dtype=float)
    if len(data) <= sample_size + 1:
        return float(data[-1]) if len(data) > 0 else 0.0

    X = []
    y = []
    for i in range(len(data) - sample_size):
        X.append(data[i:i + sample_size])
        y.append(data[i + sample_size])
    X = np.array(X)
    y = np.array(y)

    model = SVR(C=1.0, epsilon=0.01, kernel='rbf')
    model.fit(X, y)
    return float(model.predict(data[np.newaxis, -sample_size:])[0])
