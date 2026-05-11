"""Tests for Pulse.apex.registry."""

from __future__ import annotations

import pytest

from Pulse.apex.registry import (
    SignalScore,
    SignalRegistry,
    register_signal,
    list_registered_signals,
    reset_default_registry,
    get_default_registry,
)


# ───────────────────────────────────────────────────────────────────────────────
# SignalScore
# ───────────────────────────────────────────────────────────────────────────────

def test_signal_score_basic():
    s = SignalScore(name="x", symbol="BTCUSD", score=0.5)
    assert s.score == 0.5
    assert s.valid is True
    assert s.weight == 1.0
    assert s.as_feature_value() == 0.5


def test_signal_score_invalid_yields_zero_feature():
    s = SignalScore(name="x", symbol="BTCUSD", score=0.7, valid=False)
    assert s.as_feature_value() == 0.0


def test_signal_score_clamped_to_unit_range():
    s = SignalScore(name="x", symbol="BTCUSD", score=2.5).clamped()
    assert s.score == 1.0
    s = SignalScore(name="x", symbol="BTCUSD", score=-3.0).clamped()
    assert s.score == -1.0


def test_signal_score_clamped_returns_self_when_in_range():
    s = SignalScore(name="x", symbol="BTCUSD", score=0.3)
    assert s.clamped() is s


def test_signal_score_meta_is_freeform():
    s = SignalScore(name="x", symbol="BTCUSD", score=0.5,
                    meta={"raw_value": 12.3, "regime": "bull"})
    assert s.meta["raw_value"] == 12.3
    assert s.meta["regime"] == "bull"


# ───────────────────────────────────────────────────────────────────────────────
# SignalRegistry — basics
# ───────────────────────────────────────────────────────────────────────────────

def test_registry_register_and_get():
    r = SignalRegistry()
    fn = lambda sym, ctx: SignalScore("a", sym, 0.5)
    r.register("a", fn)
    assert r.get("a") is fn
    assert "a" in r


def test_registry_register_rejects_empty_name():
    r = SignalRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        r.register("", lambda s, c: SignalScore("a", s, 0.0))


def test_registry_register_rejects_non_callable():
    r = SignalRegistry()
    with pytest.raises(TypeError, match="not callable"):
        r.register("a", "not a function")  # type: ignore[arg-type]


def test_registry_register_rejects_duplicate():
    r = SignalRegistry()
    r.register("a", lambda s, c: SignalScore("a", s, 0.0))
    with pytest.raises(ValueError, match="already registered"):
        r.register("a", lambda s, c: SignalScore("a", s, 0.0))


def test_registry_unregister_idempotent():
    r = SignalRegistry()
    r.register("a", lambda s, c: SignalScore("a", s, 0.0))
    r.unregister("a")
    r.unregister("a")  # double-unregister doesn't crash
    assert r.get("a") is None


def test_registry_names_preserves_insertion_order():
    r = SignalRegistry()
    r.register("c", lambda s, ctx: SignalScore("c", s, 0))
    r.register("a", lambda s, ctx: SignalScore("a", s, 0))
    r.register("b", lambda s, ctx: SignalScore("b", s, 0))
    assert r.names() == ["c", "a", "b"]


def test_registry_len():
    r = SignalRegistry()
    assert len(r) == 0
    r.register("a", lambda s, c: SignalScore("a", s, 0))
    assert len(r) == 1


# ───────────────────────────────────────────────────────────────────────────────
# SignalRegistry — compute_all
# ───────────────────────────────────────────────────────────────────────────────

def _good(sym, ctx):
    return SignalScore("g", sym, 0.7)


def _bad(sym, ctx):
    raise RuntimeError("boom")


def _too_high(sym, ctx):
    return SignalScore("h", sym, 5.0)   # un-clamped


def test_compute_all_collects_in_order():
    r = SignalRegistry()
    r.register("g", _good)
    r.register("h", _too_high)
    out = r.compute_all("BTCUSD", {})
    assert [s.name for s in out] == ["g", "h"]


def test_compute_all_clamps_values():
    r = SignalRegistry()
    r.register("h", _too_high)
    [score] = r.compute_all("BTCUSD", {})
    assert score.score == 1.0


def test_compute_all_catches_signal_exceptions():
    r = SignalRegistry()
    r.register("g", _good)
    r.register("b", _bad)
    out = r.compute_all("BTCUSD", {})
    assert len(out) == 2
    bad = next(s for s in out if s.name == "b")
    assert bad.valid is False
    assert "boom" in bad.meta.get("error", "")


def test_compute_all_repairs_missing_name_field():
    r = SignalRegistry()
    # Function returns a SignalScore with no name; registry should fill it
    r.register("renamed", lambda s, ctx: SignalScore(name="", symbol=s, score=0.3))
    [score] = r.compute_all("BTCUSD", {})
    assert score.name == "renamed"
    assert score.score == 0.3


# ───────────────────────────────────────────────────────────────────────────────
# Default registry + decorator
# ───────────────────────────────────────────────────────────────────────────────

def test_register_signal_decorator_populates_default():
    reset_default_registry()
    @register_signal("decorated_sig")
    def _fn(sym, ctx):
        return SignalScore("decorated_sig", sym, 0.9)
    assert "decorated_sig" in list_registered_signals()
    sc = get_default_registry().get("decorated_sig")("BTCUSD", {})
    assert sc.score == 0.9


def test_reset_default_registry_clears_state():
    @register_signal("temp_sig")
    def _fn(sym, ctx):
        return SignalScore("temp_sig", sym, 0.0)
    assert "temp_sig" in list_registered_signals()
    reset_default_registry()
    assert "temp_sig" not in list_registered_signals()


def test_default_registry_isolation_from_local_registries():
    """Using @register_signal must not leak into a fresh local registry."""
    reset_default_registry()
    @register_signal("global_sig")
    def _fn(sym, ctx):
        return SignalScore("global_sig", sym, 0.5)
    local = SignalRegistry()
    assert "global_sig" not in local
    assert "global_sig" in get_default_registry()
