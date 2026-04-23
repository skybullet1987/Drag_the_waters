import numpy as np
import pandas as pd


def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period, min_periods=period).mean()
    avg_loss = down.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_ta_features(
    df,
    price_col="close",
    vol_windows=(8, 24, 72),
    momentum_windows=(3, 12, 24),
    z_windows=(24, 72),
    rsi_period=14,
):
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}'")

    out = pd.DataFrame(index=df.index)
    close = df[price_col].astype(float)
    log_ret = np.log(close).diff()
    out["log_return_1"] = log_ret

    for w in vol_windows:
        out[f"rolling_vol_{w}"] = log_ret.rolling(w, min_periods=w).std()

    for w in momentum_windows:
        out[f"momentum_{w}"] = close.pct_change(w)

    out[f"rsi_{rsi_period}"] = _rsi(close, period=rsi_period)

    if {"high", "low"}.issubset(df.columns):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        out["spread_proxy"] = (high - low) / close.replace(0, np.nan)
    else:
        out["spread_proxy"] = log_ret.abs()

    for w in z_windows:
        mean = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std()
        out[f"zscore_{w}"] = (close - mean) / std.replace(0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)
