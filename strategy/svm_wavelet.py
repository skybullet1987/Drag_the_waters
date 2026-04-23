from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
import pandas as pd

from features.ta import build_ta_features
from features.wavelets import causal_wavelet_features


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
        self.config = StrategyConfig(**{k: v for k, v in cfg.items() if hasattr(StrategyConfig, k)})
        self._bars = deque(maxlen=max(self.config.window * 2, 512))
        self._last_position = 0
        self._last_change_bar = -10**9
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

        now = context.get("timestamp") or bar.get("timestamp") or datetime.utcnow()
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
