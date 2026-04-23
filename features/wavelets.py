import numpy as np
import pandas as pd
import pywt


def _flatten_coefficients(coeffs):
    flat = []
    for arr in coeffs:
        flat.extend(np.asarray(arr, dtype=float).ravel())
    return np.asarray(flat, dtype=float)


def causal_wavelet_features(series, wavelet="db4", level=3, window=256, verify_causality=True):
    """
    Build causal wavelet features by applying DWT over a trailing window ending at each timestamp.
    For each row t, only data with index <= t is used.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if window <= 0:
        raise ValueError("window must be positive")

    n = len(series)
    if n == 0:
        return pd.DataFrame(index=series.index)

    first_valid = max(window - 1, 0)
    probe_start = max(0, first_valid)
    probe = series.iloc[probe_start:probe_start + window]
    if len(probe) < window:
        probe = series.iloc[max(0, n - window):n]
    probe_values = np.array(probe.values, dtype=float, copy=True)
    probe_coeffs = pywt.wavedec(probe_values, wavelet=wavelet, level=level, mode="periodization")
    feature_len = len(_flatten_coefficients(probe_coeffs))

    features = np.full((n, feature_len), np.nan, dtype=float)

    for i in range(first_valid, n):
        trailing = series.iloc[i - window + 1:i + 1]
        if verify_causality:
            assert trailing.index.max() <= series.index[i], "Non-causal access detected"
        trailing_values = np.array(trailing.values, dtype=float, copy=True)
        coeffs = pywt.wavedec(trailing_values, wavelet=wavelet, level=level, mode="periodization")
        features[i, :] = _flatten_coefficients(coeffs)

    columns = [f"wcoef_{j}" for j in range(feature_len)]
    return pd.DataFrame(features, index=series.index, columns=columns)
