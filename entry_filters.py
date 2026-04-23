from collections import defaultdict, deque
from types import SimpleNamespace

import pandas as pd

from realistic_slippage import RealisticCryptoSlippage
from svm_wavelet import (
    LONG_SIGNAL,
    SHORT_SIGNAL,
    ModelRegistry,
    SvmSignalModel,
    build_ta_features,
    causal_wavelet_features,
)

DEFAULT_HISTORY = 256
WAVELET_LEVEL = 2
MIN_BARS_REQUIRED = 64


class BarHistory:
    def __init__(self, maxlen=DEFAULT_HISTORY):
        self.maxlen = int(maxlen)
        self._buffers = defaultdict(lambda: deque(maxlen=self.maxlen))

    def update(self, symbol, bar):
        self._buffers[str(symbol)].append(dict(bar))

    def frame(self, symbol):
        rows = list(self._buffers.get(str(symbol), ()))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "bid", "ask"])
        return pd.DataFrame(rows)


def _normalize_pair(symbol):
    s = str(symbol).replace("/", "").upper()
    if s.endswith("USD") and len(s) > 3:
        base = s[:-3]
        if base == "BTC":
            base = "XBT"
        return f"{base}/USD"
    return str(symbol)


class SvmWaveletEntryFilter:
    def __init__(self, model_dir="./artifacts", confidence=0.55, history_size=DEFAULT_HISTORY):
        self.confidence = float(confidence)
        self.registry = ModelRegistry(root=model_dir)
        self.history = BarHistory(maxlen=history_size)
        self._models = {}
        self.slippage_model = RealisticCryptoSlippage()

    def update(self, symbol, bar):
        self.history.update(symbol, bar)

    def _load_model(self, symbol):
        key = str(symbol)
        if key in self._models:
            return self._models[key]
        candidates = [key, _normalize_pair(key), key.replace("/", "_"), key.replace("_", "/")]
        tried = set()
        for cand in candidates:
            if cand in tried:
                continue
            tried.add(cand)
            try:
                path = self.registry.load_latest(cand)
                model = SvmSignalModel.load(path)
                self._models[key] = model
                return model
            except FileNotFoundError:
                continue
        return None

    @staticmethod
    def _signal_for_side(side):
        text = str(side).lower()
        if text in {"long", "buy", "1", "+1"}:
            return LONG_SIGNAL
        if text in {"short", "sell", "-1"}:
            return SHORT_SIGNAL
        return LONG_SIGNAL

    def _estimate_slippage_pct(self, latest_bar, signal, default_slippage_bps):
        close = float(latest_bar.get("close", 0) or 0)
        if close <= 0:
            return float(default_slippage_bps) / 10000.0
        try:
            asset = SimpleNamespace(
                Price=close,
                BidPrice=float(latest_bar.get("bid", close) or close),
                AskPrice=float(latest_bar.get("ask", close) or close),
                Volume=float(latest_bar.get("volume", 0.0) or 0.0),
            )
            order = SimpleNamespace(Quantity=float(signal))
            slip_abs = float(self.slippage_model.GetSlippageApproximation(asset, order))
            if slip_abs >= 0:
                return slip_abs / close
        except Exception:
            pass
        return float(default_slippage_bps) / 10000.0

    def allows(self, symbol, side, *, fee_bps, slippage_bps):
        model = self._load_model(symbol)
        if model is None:
            return False, {"reason": "no_model", "proba": 0.0}

        bars = self.history.frame(symbol)
        if len(bars) < MIN_BARS_REQUIRED:
            return False, {"reason": "insufficient_history", "proba": 0.0}

        frame = bars.copy()
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
        wave = causal_wavelet_features(frame["close"], window=min(DEFAULT_HISTORY, len(frame)), level=WAVELET_LEVEL)
        feats = pd.concat([wave, build_ta_features(frame)], axis=1).dropna()
        if feats.empty:
            return False, {"reason": "no_features", "proba": 0.0}

        last = feats.tail(1)
        proba_df = model.predict_proba(last)
        pred = model.predict(last).iloc[-1]
        target_signal = self._signal_for_side(side)
        signal = int(pred.get("signal", 0))
        confidence = float(pred.get("confidence", 0.0))
        expected_edge = float(pred.get("edge", 0.0))
        proba = float(proba_df["prob_up"].iloc[-1] if target_signal == LONG_SIGNAL else proba_df["prob_down"].iloc[-1])
        if signal != target_signal:
            return False, {"reason": "direction_mismatch", "proba": proba, "confidence": confidence}
        if confidence < self.confidence:
            return False, {"reason": "low_confidence", "proba": proba, "confidence": confidence}
        slip_pct = self._estimate_slippage_pct(frame.iloc[-1].to_dict(), target_signal, slippage_bps)
        fee_pct = float(fee_bps) / 10000.0
        min_edge = 2.0 * (fee_pct + slip_pct)
        if expected_edge < min_edge:
            return False, {"reason": "edge_below_cost", "proba": proba, "confidence": confidence, "expected_edge": expected_edge}
        return True, {"reason": "accepted", "proba": proba, "confidence": confidence, "expected_edge": expected_edge}


def build_default_filter(model_dir="./artifacts", confidence=0.55):
    return SvmWaveletEntryFilter(model_dir=model_dir, confidence=confidence)


_DEFAULT_FILTER = None


def svm_wavelet_allows(symbol, bar_history, side, *, model_dir="./artifacts", confidence=0.55, fee_bps=26.0, slippage_bps=10.0):
    global _DEFAULT_FILTER
    if _DEFAULT_FILTER is None:
        _DEFAULT_FILTER = build_default_filter(model_dir=model_dir, confidence=confidence)
    if isinstance(bar_history, BarHistory):
        _DEFAULT_FILTER.history = bar_history
    allowed, _ = _DEFAULT_FILTER.allows(symbol, side, fee_bps=fee_bps, slippage_bps=slippage_bps)
    return allowed
