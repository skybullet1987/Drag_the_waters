"""Tests for backtest_audit.regime_runner."""

from __future__ import annotations

from datetime import date

import pytest

from backtest_audit.regime_runner import (
    RegimeWindow, REGIME_WINDOWS,
    WindowResult, SurvivabilityResult, SweepResult,
    score_param_set, sweep,
)


# ───────────────────────────────────────────────────────────────────────────────
# REGIME_WINDOWS — coverage and ordering
# ───────────────────────────────────────────────────────────────────────────────

def test_six_regime_windows():
    assert len(REGIME_WINDOWS) == 6


def test_windows_cover_2022_through_2026():
    starts = [w.start for w in REGIME_WINDOWS]
    assert starts == sorted(starts)
    assert REGIME_WINDOWS[0].start == date(2022, 1, 1)
    assert REGIME_WINDOWS[-1].end == date(2026, 5, 10)


def test_window_names_unique():
    names = [w.name for w in REGIME_WINDOWS]
    assert len(set(names)) == len(names)


# ───────────────────────────────────────────────────────────────────────────────
# score_param_set — bless / catastrophic / scoring
# ───────────────────────────────────────────────────────────────────────────────

def _wr(name, ret, dd, sharpe=1.0, wr=50.0, trades=100):
    return WindowResult(window=name, net_return_pct=ret, drawdown_pct=dd,
                        sharpe=sharpe, win_rate_pct=wr, trades=trades)


def test_blessed_when_consistent_positive_no_catastrophe():
    """6 modest positive windows with low DD → blessed."""
    wrs = [
        _wr("w1", 8.0, 5.0),
        _wr("w2", 12.0, 8.0),
        _wr("w3", 10.0, 6.0),
        _wr("w4", 6.0, 9.0),
        _wr("w5", 15.0, 10.0),
        _wr("w6", 7.0, 4.0),
    ]
    s = score_param_set("blessed_set", wrs)
    assert s.blessed
    assert s.n_positive_windows == 6
    assert s.n_catastrophic_windows == 0
    assert s.survivability_score > 0


def test_not_blessed_when_one_catastrophic_window():
    """One window with -25% return → catastrophic → not blessed even if score>0."""
    wrs = [
        _wr("w1", 50.0, 5.0),
        _wr("w2", 60.0, 5.0),
        _wr("w3", 60.0, 5.0),
        _wr("w4", 60.0, 5.0),
        _wr("w5", 60.0, 5.0),
        _wr("w6", -25.0, 30.0),     # catastrophic by both criteria
    ]
    s = score_param_set("one_blowup", wrs)
    assert s.n_catastrophic_windows == 1
    assert not s.blessed


def test_not_blessed_when_only_3_positive_windows():
    """Default min_positive_windows=4."""
    wrs = [
        _wr("w1", 10.0, 8.0),
        _wr("w2", 10.0, 8.0),
        _wr("w3", 10.0, 8.0),
        _wr("w4", -2.0, 5.0),
        _wr("w5", -2.0, 5.0),
        _wr("w6", -2.0, 5.0),
    ]
    s = score_param_set("only3pos", wrs)
    assert s.n_positive_windows == 3
    assert not s.blessed


def test_score_penalizes_high_volatility():
    """Two parameter sets with same mean: lower std wins."""
    smooth = [_wr(f"w{i}", 10.0, 5.0) for i in range(6)]
    spiky  = [_wr("w1", 30.0, 5.0), _wr("w2", -10.0, 5.0),
              _wr("w3", 30.0, 5.0), _wr("w4", -10.0, 5.0),
              _wr("w5", 30.0, 5.0), _wr("w6", -10.0, 5.0)]
    s_smooth = score_param_set("smooth", smooth)
    s_spiky  = score_param_set("spiky", spiky)
    assert s_smooth.mean_return_pct == pytest.approx(s_spiky.mean_return_pct)
    assert s_smooth.survivability_score > s_spiky.survivability_score


def test_score_penalizes_worst_window_drawdown():
    """Two sets with same returns but different DDs: lower DD wins.
    With default worst_case_penalty=0.5, score = mean - 0*std - 0.5*max_dd:
      safe:  20 - 0 - 0.5*5  = +17.5
      risky: 20 - 0 - 0.5*15 = +12.5
    Both blessed (no catastrophe, 6/6 positive), safe scores higher.
    """
    safe  = [_wr(f"w{i}", 20.0, 5.0)  for i in range(6)]
    risky = [_wr(f"w{i}", 20.0, 15.0) for i in range(6)]
    s_safe  = score_param_set("safe",  safe)
    s_risky = score_param_set("risky", risky)
    assert s_safe.blessed and s_risky.blessed
    assert s_safe.survivability_score > s_risky.survivability_score


def test_summary_text_renders():
    wrs = [_wr(f"w{i}", 10.0, 5.0) for i in range(6)]
    s = score_param_set("renderme", wrs)
    text = s.summary_text()
    assert "renderme" in text
    assert "blessed=True" in text


# ───────────────────────────────────────────────────────────────────────────────
# sweep() — orchestration of multiple param sets
# ───────────────────────────────────────────────────────────────────────────────

def test_sweep_orders_results_by_score():
    """Three param sets with very different quality."""
    def fake_backtest(window, params):
        # `params['quality']` is the per-window return for this set
        # (constant across windows for test simplicity)
        return _wr(window.name, params["quality"], params["dd"])

    param_sets = [
        ("loser",   {"quality": -5.0, "dd":  8.0}),
        ("winner",  {"quality": 12.0, "dd":  6.0}),
        ("middle",  {"quality":  3.0, "dd": 10.0}),
    ]
    sw = sweep(param_sets, fake_backtest)
    assert len(sw.results) == 3
    ranked = sw.ranked
    assert ranked[0].param_id == "winner"
    assert ranked[-1].param_id == "loser"


def test_sweep_blessed_filter():
    def fake(w, p):
        return _wr(w.name, p["q"], p["d"])
    sw = sweep([
        ("good", {"q": 10.0, "d": 8.0}),
        ("bad",  {"q": -3.0, "d": 12.0}),
    ], fake)
    blessed = sw.blessed
    assert len(blessed) == 1
    assert blessed[0].param_id == "good"


def test_sweep_handles_backtest_exception():
    """A param set whose backtest throws should be marked catastrophic, not crash."""
    def fake(w, p):
        if p["q"] == "boom":
            raise RuntimeError("simulated backtest failure")
        return _wr(w.name, p["q"], 5.0)

    sw = sweep([
        ("ok",    {"q": 10.0}),
        ("boom",  {"q": "boom"}),
    ], fake)
    assert len(sw.results) == 2
    boom = next(r for r in sw.results if r.param_id == "boom")
    assert boom.n_catastrophic_windows == 6
    assert not boom.blessed


def test_sweep_summary_text():
    def fake(w, p):
        return _wr(w.name, p["q"], 5.0)
    sw = sweep([("a", {"q": 8}), ("b", {"q": 12})], fake)
    text = sw.summary_text()
    assert "param sets" in text
    assert "Top 10" in text


# ───────────────────────────────────────────────────────────────────────────────
# End-to-end: run sweep on REGIME_WINDOWS
# ───────────────────────────────────────────────────────────────────────────────

def test_sweep_runs_each_window_once_per_param_set():
    """Ensure every (param, window) pair is exercised."""
    seen: list[tuple[str, str]] = []

    def fake(w, p):
        seen.append((p["pid"], w.name))
        return _wr(w.name, 5.0, 5.0)

    sweep([
        ("alpha", {"pid": "alpha"}),
        ("beta",  {"pid": "beta"}),
    ], fake)

    # 2 param sets × 6 windows = 12 invocations
    assert len(seen) == 12
    # Each pair appears exactly once
    assert len(set(seen)) == 12
