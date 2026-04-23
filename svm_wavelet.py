from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import joblib
import numpy as np
import pandas as pd
import pywt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --- Features: wavelets ---

def _flatten_coefficients(coeffs):
    flat = []
    for arr in coeffs:
        flat.extend(np.asarray(arr, dtype=float).ravel())
    return np.asarray(flat, dtype=float)


def causal_wavelet_features(series, wavelet="db4", level=3, window=256, verify_causality=True):
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


# --- Features: TA ---

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


def ta_features(*args, **kwargs):
    return build_ta_features(*args, **kwargs)


# --- Model: SVM signal ---

LONG_SIGNAL = 1
SHORT_SIGNAL = -1
NEUTRAL_SIGNAL = 0


def _safe_round_trip_cost(default=0.0052):
    try:
        from execution import ESTIMATED_ROUND_TRIP_FEE  # type: ignore

        return float(ESTIMATED_ROUND_TRIP_FEE)
    except (ImportError, ModuleNotFoundError):
        return float(default)


def make_cost_aware_labels(close, horizon=1, k=2.0, round_trip_cost=None):
    round_trip_cost = _safe_round_trip_cost() if round_trip_cost is None else float(round_trip_cost)
    close = pd.Series(close).astype(float)
    fwd_ret = close.shift(-horizon) / close - 1.0
    upper = k * round_trip_cost
    lower = -k * round_trip_cost
    y = pd.Series(NEUTRAL_SIGNAL, index=close.index, dtype=int)
    y[fwd_ret > upper] = LONG_SIGNAL
    y[fwd_ret < lower] = SHORT_SIGNAL
    return y


@dataclass
class WalkForwardConfig:
    mode: str = "expanding"
    min_train_size: int = 500
    train_window: int = 2000
    retrain_every: int = 24
    drop_flat: bool = True


class SvmSignalModel:
    def __init__(self, C=1.0, gamma="scale", probability=True, neutral_threshold=0.55, max_train_samples=50000):
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.neutral_threshold = neutral_threshold
        self.max_train_samples = int(max_train_samples)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=C, gamma=gamma, probability=probability)),
            ]
        )

    def _prepare_training_data(self, X, y, drop_flat=True):
        X_df = pd.DataFrame(X).copy()
        y_sr = pd.Series(y).copy()
        valid = (~X_df.isna().any(axis=1)) & (~y_sr.isna())
        if drop_flat:
            valid &= y_sr != 0
        X_clean = X_df.loc[valid]
        y_clean = y_sr.loc[valid]
        if len(X_clean) > self.max_train_samples:
            step = int(np.ceil(len(X_clean) / self.max_train_samples))
            X_clean = X_clean.iloc[::step]
            y_clean = y_clean.iloc[::step]
        return X_clean, y_clean

    def fit(self, X, y, drop_flat=True):
        X_clean, y_clean = self._prepare_training_data(X, y, drop_flat=drop_flat)
        if y_clean.nunique() < 2:
            raise ValueError("Need at least two classes to train SVM")
        self.pipeline.fit(X_clean, y_clean)
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X)
        probs = self.pipeline.predict_proba(X_df)
        classes = list(self.pipeline.named_steps["svc"].classes_)
        out = pd.DataFrame(index=X_df.index)
        out["prob_up"] = probs[:, classes.index(LONG_SIGNAL)] if LONG_SIGNAL in classes else 0.0
        out["prob_down"] = probs[:, classes.index(SHORT_SIGNAL)] if SHORT_SIGNAL in classes else 0.0
        out["prob_flat"] = probs[:, classes.index(NEUTRAL_SIGNAL)] if NEUTRAL_SIGNAL in classes else 0.0
        return out

    def predict(self, X):
        prob = self.predict_proba(X)
        confidence = prob[["prob_up", "prob_down"]].max(axis=1)
        edge = (prob["prob_up"] - prob["prob_down"]).abs()
        signal = np.where(
            confidence >= self.neutral_threshold,
            np.where(prob["prob_up"] >= prob["prob_down"], LONG_SIGNAL, SHORT_SIGNAL),
            NEUTRAL_SIGNAL,
        )
        return pd.DataFrame(
            {
                "signal": signal.astype(int),
                "confidence": confidence.astype(float),
                "edge": edge.astype(float),
            },
            index=prob.index,
        )

    def walk_forward_predict(self, X, y, config=None):
        X_df = pd.DataFrame(X).copy()
        y_sr = pd.Series(y).copy()
        cfg = config or WalkForwardConfig()

        n = len(X_df)
        preds = pd.DataFrame(index=X_df.index, columns=["signal", "confidence", "edge"], dtype=float)

        start = max(cfg.min_train_size, 2)
        for i in range(start, n, cfg.retrain_every):
            train_end = i
            if cfg.mode == "rolling":
                train_start = max(0, train_end - cfg.train_window)
            else:
                train_start = 0

            X_train = X_df.iloc[train_start:train_end]
            y_train = y_sr.iloc[train_start:train_end]
            X_test = X_df.iloc[i : min(i + cfg.retrain_every, n)]
            if X_test.empty:
                continue

            try:
                self.fit(X_train, y_train, drop_flat=cfg.drop_flat)
                p = self.predict(X_test)
                preds.loc[p.index, :] = p[["signal", "confidence", "edge"]].values
            except ValueError:
                preds.loc[X_test.index, :] = [0, 0.0, 0.0]

        return preds

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)


# --- Model: registry ---

class ModelRegistry:
    def __init__(self, root="artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _config_hash(config):
        payload = json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def save(self, model, symbol, config=None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sym = symbol.replace("/", "_")
        cfg_hash = self._config_hash(config)
        path = self.root / f"{sym}_{ts}_{cfg_hash}.joblib"
        model.save(path)
        meta = {
            "symbol": symbol,
            "timestamp": ts,
            "config_hash": cfg_hash,
            "path": str(path),
        }
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def load_latest(self, symbol):
        sym = symbol.replace("/", "_")
        candidates = sorted(self.root.glob(f"{sym}_*.joblib"))
        if not candidates:
            raise FileNotFoundError(f"No model found for {symbol}")
        return candidates[-1]


# --- Strategy ---

INITIAL_LAST_CHANGE_BAR = float("-inf")


@dataclass
class StrategyConfig:
    wavelet: str = "db4"
    level: int = 3
    window: int = 256
    round_trip_cost: float = 0.0052
    risk_per_trade: float = 0.02
    max_daily_turnover: float = 3.0
    min_holding_period_bars: int = 3


class SvmWaveletStrategy:
    def __init__(self, model, config=None, slippage_model=None, circuit_breaker=None):
        self.model = model
        cfg = config or {}
        known = StrategyConfig.__dataclass_fields__
        unknown_keys = sorted(set(cfg.keys()) - set(known.keys()))
        if unknown_keys:
            warnings.warn(f"Unknown strategy config keys ignored: {unknown_keys}")
        self.config = StrategyConfig(**{k: v for k, v in cfg.items() if k in known})
        self._bars = deque(maxlen=max(self.config.window * 2, 512))
        self._last_position = 0
        self._last_change_bar = INITIAL_LAST_CHANGE_BAR
        self._bar_count = 0
        self._daily_turnover = defaultdict(float)
        self.slippage_model = slippage_model
        self.circuit_breaker = circuit_breaker

    def _estimate_slippage_pct(self, bar, signal):
        if signal == 0:
            return 0.0

        close = float(bar.get("close", 0.0) or 0.0)
        if close <= 0:
            return 0.0

        if self.slippage_model is None:
            return 0.001

        asset = SimpleNamespace(
            Price=close,
            BidPrice=float(bar.get("bid", 0.0) or 0.0),
            AskPrice=float(bar.get("ask", 0.0) or 0.0),
            Volume=float(bar.get("volume", 0.0) or 0.0),
        )
        order = SimpleNamespace(Quantity=signal)
        slip_abs = float(self.slippage_model.GetSlippageApproximation(asset, order))
        return max(0.0, slip_abs / close)

    def _breaker_tripped(self, context):
        if self.circuit_breaker is None:
            return False
        if hasattr(self.circuit_breaker, "update") and context.get("algorithm") is not None:
            self.circuit_breaker.update(context["algorithm"])
        if hasattr(self.circuit_breaker, "is_triggered"):
            return bool(self.circuit_breaker.is_triggered())
        return False

    def _build_features(self):
        frame = pd.DataFrame(list(self._bars))
        wave = causal_wavelet_features(
            frame["close"],
            wavelet=self.config.wavelet,
            level=self.config.level,
            window=self.config.window,
            verify_causality=True,
        )
        ta = build_ta_features(frame)
        features = pd.concat([wave, ta], axis=1)
        return features

    def on_bar(self, bar, context=None):
        context = context or {}
        self._bar_count += 1
        self._bars.append(bar)

        now = context.get("timestamp") or bar.get("timestamp") or datetime.now(timezone.utc)
        day_key = pd.Timestamp(now).date().isoformat()

        if self._breaker_tripped(context):
            self._last_position = 0
            return {"target_position": 0, "size": 0.0, "reason": "circuit_breaker"}

        if len(self._bars) < self.config.window:
            return {"target_position": 0, "size": 0.0, "reason": "warmup"}

        feats = self._build_features().dropna()
        if feats.empty:
            return {"target_position": 0, "size": 0.0, "reason": "no_features"}

        pred = self.model.predict(feats.tail(1)).iloc[-1]
        signal = int(pred["signal"])
        confidence = float(pred["confidence"])
        expected_edge = float(pred.get("edge", 0.0))

        slippage = self._estimate_slippage_pct(bar, signal)
        cost_gate = float(self.config.round_trip_cost) + slippage
        if signal != 0 and expected_edge <= cost_gate:
            signal = 0

        if signal != self._last_position:
            bars_since_change = self._bar_count - self._last_change_bar
            if bars_since_change < self.config.min_holding_period_bars:
                signal = self._last_position

        turnover_if_trade = self._daily_turnover[day_key] + abs(signal - self._last_position)
        if turnover_if_trade > self.config.max_daily_turnover:
            signal = self._last_position

        if signal != self._last_position:
            self._daily_turnover[day_key] += abs(signal - self._last_position)
            self._last_change_bar = self._bar_count
            self._last_position = signal

        size = min(1.0, max(0.0, confidence * float(self.config.risk_per_trade))) if signal != 0 else 0.0
        return {
            "target_position": int(signal),
            "size": size,
            "confidence": confidence,
            "expected_edge": expected_edge,
            "round_trip_cost": float(self.config.round_trip_cost),
            "slippage_estimate": slippage,
        }


__all__ = [
    "LONG_SIGNAL",
    "SHORT_SIGNAL",
    "NEUTRAL_SIGNAL",
    "WalkForwardConfig",
    "StrategyConfig",
    "SvmSignalModel",
    "ModelRegistry",
    "SvmWaveletStrategy",
    "causal_wavelet_features",
    "build_ta_features",
    "ta_features",
    "make_cost_aware_labels",
]
