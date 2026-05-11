"""Pulse.apex.feature_vector — compose registered signals into a fixed-length
feature vector for the ML ensemble.

The vector layout is *order-stable*: each signal name slot in the schema
maps to a fixed feature index. Adding/removing signals requires bumping
the model version (Phase 4 trainer enforces this).

Each signal contributes 3 features:
  - raw_score   ∈ [-1, 1]                     (current value)
  - lag1_score  ∈ [-1, 1] or 0 if missing     (last tick's value)
  - z_3         z-score over last 3 ticks      (momentum of the signal)

This trio dramatically improves the model's ability to detect *changes*
in signals rather than just absolute levels.

If fewer than APEX_FEATURE_DIM features end up generated (e.g. signal
disabled), the trailing slots are zero-filled. The dim is *fixed* so
the trained joblib doesn't need re-shaping at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from Pulse.apex.config import APEX_FEATURE_DIM
from Pulse.apex.registry import SignalRegistry, SignalScore


# ─── The canonical signal order (matches the registry on first build) ───────
# Phase 4 trainer locks this. Adding a signal here means retraining.

DEFAULT_SIGNAL_ORDER = (
    "btc_onchain",
    "btc_dominance",
    "funding_extreme",
    "etf_flow",
    "stablecoin_mint",
    "token_unlock",
    "news_sentiment",
    "mvrv",
    "cross_asset_macro",
    "fg_index",
)

FEATURES_PER_SIGNAL = 3   # raw, lag1, z3


# ─── Output dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureVector:
    symbol:        str
    values:        list[float]    # length == APEX_FEATURE_DIM
    feature_names: list[str]
    valid_signals: int            # how many signals contributed real data
    meta:          dict

    def as_list(self) -> list[float]:
        return list(self.values)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "values": list(self.values),
            "feature_names": list(self.feature_names),
            "valid_signals": self.valid_signals,
            "meta": dict(self.meta),
        }


# ─── Feature naming helpers ──────────────────────────────────────────────────


def feature_names(signal_order: Iterable[str] = DEFAULT_SIGNAL_ORDER,
                  feature_dim: int = APEX_FEATURE_DIM) -> list[str]:
    """Return the canonical feature-name list of length feature_dim.

    Names follow the pattern <signal>_<suffix> where suffix ∈ {raw,lag1,z3}.
    Trailing slots beyond the last available signal are named "_pad_<i>".
    """
    names: list[str] = []
    for sig in signal_order:
        names.append(f"{sig}_raw")
        names.append(f"{sig}_lag1")
        names.append(f"{sig}_z3")
    if len(names) > feature_dim:
        return names[:feature_dim]
    while len(names) < feature_dim:
        names.append(f"_pad_{len(names)}")
    return names


DEFAULT_FEATURE_NAMES = feature_names()


# ─── Per-signal rolling history (lag1, z3) ──────────────────────────────────


class _SignalHistory:
    """Tracks the last 3 score values per (symbol, signal) for lag/z calc.

    Pure-Python, no numpy needed. Bounded size = O(N_signals × N_symbols)
    which is fine for our 25-symbol universe + ~10 signals = ~750 entries.
    """

    def __init__(self) -> None:
        self._buf: dict[tuple[str, str], list[float]] = {}

    def update(self, symbol: str, signal_name: str, value: float) -> None:
        key = (symbol, signal_name)
        hist = self._buf.setdefault(key, [])
        hist.append(float(value))
        if len(hist) > 3:
            hist.pop(0)

    def lag1(self, symbol: str, signal_name: str) -> float:
        """Last value appended to the buffer (the previous tick's value).

        Read BEFORE the current value is appended, so on the second call
        this returns the first call's value.
        """
        hist = self._buf.get((symbol, signal_name), [])
        return hist[-1] if hist else 0.0

    def z3(self, symbol: str, signal_name: str,
           current_value: float | None = None) -> float:
        """Z-score of `current_value` against the existing buffer.

        We include the current value as the most-recent sample to give
        a 3-point rolling window starting from the second tick. Returns
        0 when fewer than 2 historical points exist (no spread defined).
        """
        hist = self._buf.get((symbol, signal_name), [])
        sample = list(hist)
        if current_value is not None:
            sample.append(float(current_value))
        if len(sample) < 3:
            return 0.0
        # Use the last 3 to keep the window bounded
        window = sample[-3:]
        m = sum(window) / 3.0
        var = sum((x - m) ** 2 for x in window) / 3.0
        if var <= 1e-9:
            return 0.0
        sd = var ** 0.5
        return (window[-1] - m) / sd

    def reset(self) -> None:
        self._buf.clear()


# Module-level singleton — shared across feature-vector builds within
# one algorithm process. Tests should call reset() between cases.
_HISTORY = _SignalHistory()


def reset_history() -> None:
    _HISTORY.reset()


# ─── Vector builder ──────────────────────────────────────────────────────────


def build_feature_vector(
    symbol: str,
    context: dict,
    *,
    registry: SignalRegistry,
    signal_order: Iterable[str] = DEFAULT_SIGNAL_ORDER,
    feature_dim: int = APEX_FEATURE_DIM,
    update_history: bool = True,
) -> FeatureVector:
    """Compute every signal in `signal_order`, expand to (raw, lag1, z3)
    triplets, pad/truncate to `feature_dim`. Returns a FeatureVector.

    Signals not registered → all-zero contribution but feature names
    still emitted (so the ML model gets a fixed shape).
    """
    sig_order = list(signal_order)
    name_list = feature_names(sig_order, feature_dim)

    # Compute current values per signal (in registered order)
    current: dict[str, SignalScore] = {}
    for sig in sig_order:
        fn = registry.get(sig)
        if fn is None:
            current[sig] = SignalScore(
                name=sig, symbol=symbol, score=0.0, valid=False,
                meta={"error": "not_registered"},
            )
            continue
        try:
            sc = fn(symbol, context)
        except Exception as exc:   # noqa: BLE001
            sc = SignalScore(
                name=sig, symbol=symbol, score=0.0, valid=False,
                meta={"error": str(exc)[:120]},
            )
        if not isinstance(sc, SignalScore):
            sc = SignalScore(
                name=sig, symbol=symbol, score=0.0, valid=False,
                meta={"error": "non_SignalScore_return"},
            )
        current[sig] = sc.clamped()

    # Build the (raw, lag1, z3) triplets in insertion order
    values: list[float] = []
    valid_count = 0
    for sig in sig_order:
        sc = current[sig]
        raw = sc.as_feature_value()
        lag = _HISTORY.lag1(symbol, sig)
        # Pass the current raw value into z3 so it forms part of the
        # rolling sample (otherwise z3 would always be one tick stale).
        z3  = _HISTORY.z3(symbol, sig, current_value=raw if sc.valid else None)
        values.extend([raw, lag, z3])
        if sc.valid:
            valid_count += 1
        if update_history and sc.valid:
            _HISTORY.update(symbol, sig, sc.score)

    # Pad / truncate to fixed dim
    if len(values) > feature_dim:
        values = values[:feature_dim]
    while len(values) < feature_dim:
        values.append(0.0)

    return FeatureVector(
        symbol=symbol,
        values=values,
        feature_names=name_list,
        valid_signals=valid_count,
        meta={
            "signal_order": sig_order,
            "features_per_signal": FEATURES_PER_SIGNAL,
            "feature_dim": feature_dim,
        },
    )


def build_feature_matrix(
    symbols: Iterable[str],
    context_per_symbol: dict[str, dict],
    *,
    registry: SignalRegistry,
    signal_order: Iterable[str] = DEFAULT_SIGNAL_ORDER,
    feature_dim: int = APEX_FEATURE_DIM,
) -> tuple[list[str], list[list[float]], list[FeatureVector]]:
    """Convenience: build a (N_symbols × feature_dim) matrix in one call.

    Returns (row_symbols, matrix, full_vectors).
    """
    syms = list(symbols)
    rows: list[list[float]] = []
    fvs:  list[FeatureVector] = []
    for sym in syms:
        ctx = context_per_symbol.get(sym, {})
        fv = build_feature_vector(
            sym, ctx, registry=registry,
            signal_order=signal_order, feature_dim=feature_dim,
        )
        rows.append(fv.as_list())
        fvs.append(fv)
    return syms, rows, fvs
