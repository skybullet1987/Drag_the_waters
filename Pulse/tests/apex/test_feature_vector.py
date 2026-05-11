"""Tests for Pulse.apex.feature_vector."""

from __future__ import annotations

import pytest

from Pulse.apex.config import APEX_FEATURE_DIM
from Pulse.apex.feature_vector import (
    FeatureVector,
    FEATURES_PER_SIGNAL,
    DEFAULT_SIGNAL_ORDER,
    feature_names,
    build_feature_vector,
    build_feature_matrix,
    reset_history,
    DEFAULT_FEATURE_NAMES,
)
from Pulse.apex.registry import SignalRegistry, SignalScore


# ───────────────────────────────────────────────────────────────────────────────
# feature_names
# ───────────────────────────────────────────────────────────────────────────────

def test_feature_names_default_length_matches_dim():
    names = feature_names()
    assert len(names) == APEX_FEATURE_DIM


def test_feature_names_emits_raw_lag1_z3_per_signal():
    names = feature_names(("foo", "bar"), feature_dim=6)
    assert names == ["foo_raw", "foo_lag1", "foo_z3",
                     "bar_raw", "bar_lag1", "bar_z3"]


def test_feature_names_pads_when_under_dim():
    names = feature_names(("foo",), feature_dim=5)
    assert names[:3] == ["foo_raw", "foo_lag1", "foo_z3"]
    assert names[3:] == ["_pad_3", "_pad_4"]


def test_feature_names_truncates_when_over_dim():
    names = feature_names(("a", "b", "c"), feature_dim=4)
    assert len(names) == 4
    assert names[0] == "a_raw"


def test_default_feature_names_constant_matches_signal_order():
    """Sanity: the module-level constant matches what feature_names()
    returns with defaults."""
    assert DEFAULT_FEATURE_NAMES == feature_names()


# ───────────────────────────────────────────────────────────────────────────────
# build_feature_vector — happy path
# ───────────────────────────────────────────────────────────────────────────────

def _const_signal(score: float):
    def _fn(sym, ctx):
        return SignalScore("x", sym, score)
    return _fn


@pytest.fixture(autouse=True)
def _clean_history():
    reset_history()


def test_build_vector_correct_length_when_short_signal_order():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=10)
    assert len(fv.values) == 10
    assert fv.values[0] == 0.5     # raw
    assert fv.values[1] == 0.0     # lag1 (no history yet)
    assert fv.values[2] == 0.0     # z3 (no history yet)
    assert all(v == 0.0 for v in fv.values[3:])   # padding


def test_build_vector_returns_feature_names_aligned():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=6)
    assert fv.feature_names[:3] == ["a_raw", "a_lag1", "a_z3"]
    assert len(fv.feature_names) == 6


def test_build_vector_marks_valid_signal_count():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    r.register("b", _const_signal(-0.3))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a", "b"), feature_dim=6)
    assert fv.valid_signals == 2


def test_build_vector_handles_missing_signal_in_registry():
    """Signal in `signal_order` but not registered → all-zero contribution
    + valid_signals does not include it."""
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a", "missing"), feature_dim=6)
    assert fv.values[0] == 0.5
    assert fv.values[3:6] == [0.0, 0.0, 0.0]   # missing → zeros
    assert fv.valid_signals == 1


def test_build_vector_handles_signal_raising_exception():
    r = SignalRegistry()
    def _boom(sym, ctx):
        raise RuntimeError("kaboom")
    r.register("bad", _boom)
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("bad",), feature_dim=6)
    assert fv.values[:3] == [0.0, 0.0, 0.0]
    assert fv.valid_signals == 0


def test_build_vector_handles_signal_returning_non_score():
    r = SignalRegistry()
    r.register("weird", lambda s, c: "not a score")  # type: ignore[arg-type]
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("weird",), feature_dim=6)
    assert fv.values[:3] == [0.0, 0.0, 0.0]
    assert fv.valid_signals == 0


# ───────────────────────────────────────────────────────────────────────────────
# build_feature_vector — history (lag1, z3)
# ───────────────────────────────────────────────────────────────────────────────

def test_lag1_picks_up_after_two_calls():
    r = SignalRegistry()
    state = {"score": 0.2}
    def _dyn(sym, ctx):
        return SignalScore("a", sym, state["score"])
    r.register("a", _dyn)

    fv1 = build_feature_vector("BTCUSD", {}, registry=r,
                               signal_order=("a",), feature_dim=6)
    assert fv1.values[1] == 0.0   # no lag yet

    state["score"] = 0.7
    fv2 = build_feature_vector("BTCUSD", {}, registry=r,
                               signal_order=("a",), feature_dim=6)
    assert fv2.values[0] == 0.7
    assert fv2.values[1] == 0.2   # previous tick's value


def test_z3_zero_when_history_too_short():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=6)
    assert fv.values[2] == 0.0


def test_z3_zero_when_all_three_values_equal():
    """No variance → z-score must be 0, not NaN."""
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    for _ in range(3):
        fv = build_feature_vector("BTCUSD", {}, registry=r,
                                  signal_order=("a",), feature_dim=6)
    assert fv.values[2] == 0.0


def test_z3_nonzero_when_signal_changes():
    r = SignalRegistry()
    state = {"v": 0.0}
    def _dyn(sym, ctx):
        return SignalScore("a", sym, state["v"])
    r.register("a", _dyn)
    state["v"] = 0.1; build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    state["v"] = 0.2; build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    state["v"] = 0.9
    fv = build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    assert fv.values[2] > 0.5    # latest value is much higher than mean


def test_history_per_symbol_isolation():
    """A signal's history for BTCUSD must NOT leak into ETHUSD's lag1."""
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    fv_eth = build_feature_vector("ETHUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    assert fv_eth.values[1] == 0.0   # ETHUSD has no history yet


def test_invalid_signal_does_not_pollute_history():
    """An invalid SignalScore must not be added to the rolling buffer."""
    r = SignalRegistry()
    state = {"valid": True}
    def _dyn(sym, ctx):
        return SignalScore("a", sym, 0.5, valid=state["valid"])
    r.register("a", _dyn)

    state["valid"] = True
    build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    state["valid"] = False
    build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    state["valid"] = True
    fv = build_feature_vector("BTCUSD", {}, registry=r, signal_order=("a",), feature_dim=6)
    # lag1 should be the previous *valid* call (0.5), not the invalid one
    assert fv.values[1] == 0.5


def test_update_history_false_skips_buffer_update():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    build_feature_vector("BTCUSD", {}, registry=r,
                         signal_order=("a",), feature_dim=6,
                         update_history=False)
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=6,
                              update_history=False)
    assert fv.values[1] == 0.0   # buffer was never written


# ───────────────────────────────────────────────────────────────────────────────
# build_feature_vector — clamping & padding
# ───────────────────────────────────────────────────────────────────────────────

def test_build_vector_clamps_extreme_signal_outputs():
    r = SignalRegistry()
    r.register("a", _const_signal(99.0))   # way out of [-1, 1]
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=6)
    assert fv.values[0] == 1.0


def test_build_vector_truncates_when_too_many_signals():
    r = SignalRegistry()
    for n in ("a", "b", "c", "d"):
        r.register(n, _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a", "b", "c", "d"),
                              feature_dim=6)
    assert len(fv.values) == 6


def test_build_vector_pads_when_too_few_signals():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    fv = build_feature_vector("BTCUSD", {}, registry=r,
                              signal_order=("a",), feature_dim=APEX_FEATURE_DIM)
    assert len(fv.values) == APEX_FEATURE_DIM
    assert fv.values[3:] == [0.0] * (APEX_FEATURE_DIM - 3)


# ───────────────────────────────────────────────────────────────────────────────
# build_feature_matrix
# ───────────────────────────────────────────────────────────────────────────────

def test_matrix_correct_shape():
    r = SignalRegistry()
    r.register("a", _const_signal(0.5))
    syms = ("BTCUSD", "ETHUSD", "SOLUSD")
    rows, mat, fvs = build_feature_matrix(
        syms, {s: {} for s in syms},
        registry=r, signal_order=("a",), feature_dim=6,
    )
    assert rows == list(syms)
    assert len(mat) == 3
    assert all(len(row) == 6 for row in mat)
    assert len(fvs) == 3


def test_matrix_each_row_independent():
    r = SignalRegistry()
    state = {"v": {"BTCUSD": 0.5, "ETHUSD": -0.5}}
    def _dyn(sym, ctx):
        return SignalScore("a", sym, state["v"][sym])
    r.register("a", _dyn)
    rows, mat, _ = build_feature_matrix(
        ("BTCUSD", "ETHUSD"), {"BTCUSD": {}, "ETHUSD": {}},
        registry=r, signal_order=("a",), feature_dim=6,
    )
    assert mat[0][0] == 0.5
    assert mat[1][0] == -0.5


# ───────────────────────────────────────────────────────────────────────────────
# Apex DEFAULT_SIGNAL_ORDER sanity checks
# ───────────────────────────────────────────────────────────────────────────────

def test_default_signal_order_is_unique():
    assert len(DEFAULT_SIGNAL_ORDER) == len(set(DEFAULT_SIGNAL_ORDER))


def test_default_signal_order_fits_feature_dim():
    """Default order × 3 features must not exceed APEX_FEATURE_DIM."""
    needed = len(DEFAULT_SIGNAL_ORDER) * FEATURES_PER_SIGNAL
    assert needed <= APEX_FEATURE_DIM


# ───────────────────────────────────────────────────────────────────────────────
# FeatureVector dataclass
# ───────────────────────────────────────────────────────────────────────────────

def test_feature_vector_to_dict_roundtrip_keys():
    fv = FeatureVector(
        symbol="BTCUSD",
        values=[0.0] * APEX_FEATURE_DIM,
        feature_names=DEFAULT_FEATURE_NAMES,
        valid_signals=0,
        meta={"x": 1},
    )
    d = fv.to_dict()
    assert set(d.keys()) == {"symbol", "values", "feature_names",
                             "valid_signals", "meta"}
    assert d["symbol"] == "BTCUSD"
    assert d["valid_signals"] == 0


def test_feature_vector_as_list_returns_copy():
    fv = FeatureVector(
        symbol="BTCUSD",
        values=[0.5, 0.5, 0.5],
        feature_names=["a", "b", "c"],
        valid_signals=1,
        meta={},
    )
    out = fv.as_list()
    out[0] = 99.0
    assert fv.values[0] == 0.5     # internal state untouched
