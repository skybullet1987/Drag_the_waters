import numpy as np
import pandas as pd

from features.wavelets import causal_wavelet_features


def test_wavelets_are_causal():
    idx = pd.date_range("2025-01-01", periods=300, freq="h")
    s1 = pd.Series(np.linspace(100, 130, 300), index=idx)
    s2 = s1.copy()
    s2.iloc[-1] += 1000.0  # perturb future value

    f1 = causal_wavelet_features(s1, window=64, level=2)
    f2 = causal_wavelet_features(s2, window=64, level=2)

    pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1], check_dtype=False)
