"""Pulse.apex.registry — pluggable signal registry.

Each Apex signal is a small pure-Python function with the contract:

    def compute(symbol: str, context: dict) -> SignalScore

Modules register themselves via @register_signal("name"). Callers iterate
the registry to build a feature vector for the ML model (or fallback).

The registry is intentionally trivial — no async, no priority, no
dependencies. Determinism > flexibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


# ─── Output dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalScore:
    """Normalized output of one signal for one symbol at one point in time.

    Attributes
    ----------
    name      : registry key (e.g. "btc_onchain")
    symbol    : symbol the score applies to (or "GLOBAL" if symbol-agnostic)
    score     : float in [-1.0, +1.0]; convention: + = bullish, - = bearish
    weight    : optional override of the fallback weight (default 1.0)
    valid     : False when the signal could not be computed (missing data,
                stale, etc.); the feature vector treats invalid signals as 0
    meta      : freeform diagnostics, never used by the ML model
    """

    name:    str
    symbol:  str
    score:   float
    weight:  float = 1.0
    valid:   bool  = True
    meta:    dict  = field(default_factory=dict)

    def clamped(self) -> "SignalScore":
        """Return a copy with score clamped to [-1, +1]."""
        s = max(-1.0, min(1.0, float(self.score)))
        if s == self.score:
            return self
        return SignalScore(
            name=self.name, symbol=self.symbol, score=s,
            weight=self.weight, valid=self.valid, meta=self.meta,
        )

    def as_feature_value(self) -> float:
        """Value to put into the ML feature vector (0 if invalid)."""
        if not self.valid:
            return 0.0
        return max(-1.0, min(1.0, float(self.score)))


# ─── Registry ────────────────────────────────────────────────────────────────


SignalCallable = Callable[[str, dict], SignalScore]


class SignalRegistry:
    """Mapping name → callable. Stable iteration order = insertion order."""

    def __init__(self) -> None:
        self._signals: dict[str, SignalCallable] = {}

    def register(self, name: str, fn: SignalCallable) -> None:
        if not name:
            raise ValueError("signal name must be non-empty")
        if not callable(fn):
            raise TypeError(f"{fn!r} is not callable")
        if name in self._signals:
            raise ValueError(f"signal {name!r} already registered")
        self._signals[name] = fn

    def unregister(self, name: str) -> None:
        self._signals.pop(name, None)

    def get(self, name: str) -> SignalCallable | None:
        return self._signals.get(name)

    def names(self) -> list[str]:
        """Stable insertion-ordered list."""
        return list(self._signals.keys())

    def items(self) -> Iterable[tuple[str, SignalCallable]]:
        return self._signals.items()

    def compute_all(self, symbol: str, context: dict) -> list[SignalScore]:
        """Run every registered signal; return list in insertion order."""
        out: list[SignalScore] = []
        for name, fn in self._signals.items():
            try:
                score = fn(symbol, context)
            except Exception as exc:  # noqa: BLE001
                score = SignalScore(
                    name=name, symbol=symbol, score=0.0, valid=False,
                    meta={"error": str(exc)[:120]},
                )
            # Allow callers to register functions that don't bother to
            # set the name field.
            if not score.name:
                score = SignalScore(
                    name=name, symbol=score.symbol or symbol,
                    score=score.score, weight=score.weight,
                    valid=score.valid, meta=score.meta,
                )
            out.append(score.clamped())
        return out

    def __len__(self) -> int:
        return len(self._signals)

    def __contains__(self, name: object) -> bool:
        return name in self._signals


# ─── Process-level default registry ──────────────────────────────────────────
# Apex modules use the @register_signal decorator on import to populate
# this. Tests can build their own SignalRegistry instances for isolation.

_DEFAULT_REGISTRY = SignalRegistry()


def register_signal(name: str) -> Callable[[SignalCallable], SignalCallable]:
    """Decorator: registers fn under `name` in the default registry."""

    def deco(fn: SignalCallable) -> SignalCallable:
        _DEFAULT_REGISTRY.register(name, fn)
        return fn

    return deco


def list_registered_signals() -> list[str]:
    return _DEFAULT_REGISTRY.names()


def get_default_registry() -> SignalRegistry:
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Test helper — wipes the default registry."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = SignalRegistry()
