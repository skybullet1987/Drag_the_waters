"""Tests for backtest_audit.qc_sweep_runner."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from backtest_audit.qc_sweep_runner import (
    PARAM_TO_CONFIG,
    render_runtime_overrides, render_window_overrides,
    remap_params_to_config_names, push_runtime_overrides,
    stats_to_window_result, _safe_float,
    make_qc_backtest_fn, run_sweep, run_sweep_dry,
    _dry_run_backtest_fn,
)
from backtest_audit.regime_runner import (
    REGIME_WINDOWS, RegimeWindow, WindowResult, SweepResult,
)


# ───────────────────────────────────────────────────────────────────────────────
# Param remapping
# ───────────────────────────────────────────────────────────────────────────────

def test_remap_known_params_to_config_names():
    out = remap_params_to_config_names({
        "entry_threshold": 0.55,
        "quick_take_profit": 0.12,
        "max_positions": 6,
    })
    assert "SCALP_ENTRY_THRESHOLD" in out
    assert out["SCALP_ENTRY_THRESHOLD"] == 0.55
    assert "QUICK_TAKE_PROFIT_PCT" in out
    assert "MAX_POSITIONS" in out


def test_remap_unknown_params_passthrough():
    out = remap_params_to_config_names({"some_custom_key": 42})
    assert out["some_custom_key"] == 42


def test_param_to_config_covers_sweep_params():
    """Every param in PARAM_TO_CONFIG should be a known sweep parameter."""
    from backtest_audit.param_sweep import DEFAULT_PARAM_RANGES
    sweep_params = set(DEFAULT_PARAM_RANGES.keys())
    config_keys = set(PARAM_TO_CONFIG.keys())
    # All mapped keys are sweep params
    assert config_keys.issubset(sweep_params)


# ───────────────────────────────────────────────────────────────────────────────
# render_runtime_overrides
# ───────────────────────────────────────────────────────────────────────────────

def test_render_runtime_overrides_valid_python():
    """Generated file must be importable Python."""
    src = render_runtime_overrides({"X": 1, "Y": "abc", "Z": 2.5})
    assert "OVERRIDES = {" in src
    # Verify it actually parses + executes
    g: dict = {}
    exec(src, g)
    assert g["OVERRIDES"] == {"X": 1, "Y": "abc", "Z": 2.5}


def test_render_runtime_overrides_empty():
    src = render_runtime_overrides({})
    g: dict = {}
    exec(src, g)
    assert g["OVERRIDES"] == {}


def test_render_runtime_overrides_handles_strings_and_floats():
    src = render_runtime_overrides({"SCALP_ENTRY_THRESHOLD": 0.55,
                                     "TAG": "harsh-sim"})
    g: dict = {}
    exec(src, g)
    assert g["OVERRIDES"]["SCALP_ENTRY_THRESHOLD"] == 0.55
    assert g["OVERRIDES"]["TAG"] == "harsh-sim"


def test_render_window_overrides_includes_dates():
    w = REGIME_WINDOWS[0]
    src = render_window_overrides(w, {"SCALP_ENTRY_THRESHOLD": 0.55})
    g: dict = {}
    exec(src, g)
    assert g["OVERRIDES"]["PULSE_OVERRIDE_START_YEAR"] == w.start.year
    assert g["OVERRIDES"]["PULSE_OVERRIDE_END_YEAR"] == w.end.year
    assert g["OVERRIDES"]["SCALP_ENTRY_THRESHOLD"] == 0.55


# ───────────────────────────────────────────────────────────────────────────────
# push_runtime_overrides
# ───────────────────────────────────────────────────────────────────────────────

def test_push_runtime_overrides_calls_update_file_with_remapped():
    client = MagicMock()
    push_runtime_overrides(client, 99, {"entry_threshold": 0.55})
    client.update_file.assert_called_once()
    args, kwargs = client.update_file.call_args
    pid, fname, contents = args
    assert pid == 99
    assert fname == "runtime_overrides.py"
    # Contents should contain the remapped key
    assert "SCALP_ENTRY_THRESHOLD" in contents
    assert "entry_threshold" not in contents.split("OVERRIDES")[1]


# ───────────────────────────────────────────────────────────────────────────────
# stats_to_window_result
# ───────────────────────────────────────────────────────────────────────────────

def test_safe_float_handles_pct_string():
    assert _safe_float("12.5%") == 12.5
    assert _safe_float("-47.5%") == -47.5


def test_safe_float_handles_none_default():
    assert _safe_float(None) == 0.0
    assert _safe_float(None, default=42) == 42


def test_safe_float_handles_garbage_default():
    assert _safe_float("not a number") == 0.0


def test_stats_to_window_result_full_payload():
    stats = {
        "Net Profit": "+15.0%",
        "Drawdown": "8.5%",
        "Sharpe Ratio": "1.8",
        "Win Rate": "55%",
        "Total Orders": "1234",
    }
    wr = stats_to_window_result("test_window", stats)
    assert wr.window == "test_window"
    assert wr.net_return_pct == 15.0
    assert wr.drawdown_pct == 8.5
    assert wr.sharpe == 1.8
    assert wr.win_rate_pct == 55.0
    assert wr.trades == 1234


def test_stats_to_window_result_missing_fields_default_zero():
    wr = stats_to_window_result("empty_window", {})
    assert wr.net_return_pct == 0.0
    assert wr.drawdown_pct == 0.0
    assert wr.trades == 0


def test_stats_to_window_result_negative_drawdown_normalized():
    """QC sometimes returns drawdown as negative; we want positive number."""
    stats = {"Drawdown": "-25.0%"}
    wr = stats_to_window_result("dd_neg", stats)
    assert wr.drawdown_pct == 25.0


# ───────────────────────────────────────────────────────────────────────────────
# make_qc_backtest_fn (mocked QCClient)
# ───────────────────────────────────────────────────────────────────────────────

def _mock_client_with_bt_stats(stats: dict):
    c = MagicMock()
    c.create_compile.return_value = {"compileId": "c001"}
    c.wait_for_compile.return_value = {"state": "BuildSuccess"}
    c.create_backtest.return_value = {"backtest": {"backtestId": "bt001"}}
    c.wait_for_backtest.return_value = {
        "completed": True, "name": "x",
        "statistics": stats,
    }
    return c


def test_qc_backtest_fn_pushes_overrides_then_runs():
    """The closure must push runtime_overrides AND invoke run_backtest."""
    c = _mock_client_with_bt_stats({"Net Profit": "+10%"})
    bt_fn = make_qc_backtest_fn(c, project_id=99, push_pulse_first=False)
    wr = bt_fn(REGIME_WINDOWS[0], {"entry_threshold": 0.55})
    # Must have pushed the overrides file
    assert c.update_file.called
    # Must have compiled + run a backtest
    assert c.create_compile.called
    assert c.create_backtest.called
    # Result reflects the stats payload
    assert wr.net_return_pct == 10.0


def test_qc_backtest_fn_deploys_first_only_once(monkeypatch):
    """push_pulse_first=True → deploy_pulse called exactly once on first call."""
    deploy_mock = MagicMock()
    monkeypatch.setattr("backtest_audit.qc_sweep_runner.deploy_pulse",
                        deploy_mock)
    c = _mock_client_with_bt_stats({"Net Profit": "+5%"})
    bt_fn = make_qc_backtest_fn(c, project_id=99, push_pulse_first=True)
    bt_fn(REGIME_WINDOWS[0], {})
    bt_fn(REGIME_WINDOWS[1], {})
    bt_fn(REGIME_WINDOWS[2], {})
    deploy_mock.assert_called_once()


def test_qc_backtest_fn_use_harsh_sim_pushed_in_overrides():
    """When use_harsh_sim=True, the overrides file should include that flag."""
    c = _mock_client_with_bt_stats({})
    bt_fn = make_qc_backtest_fn(c, project_id=99, push_pulse_first=False,
                                use_harsh_sim=True)
    bt_fn(REGIME_WINDOWS[0], {})
    # Inspect what got pushed
    args, _ = c.update_file.call_args
    contents = args[2]
    assert "use_harsh_sim" in contents
    assert "True" in contents


# ───────────────────────────────────────────────────────────────────────────────
# run_sweep + run_sweep_dry
# ───────────────────────────────────────────────────────────────────────────────

def test_dry_run_backtest_fn_returns_window_result():
    fn = _dry_run_backtest_fn(seed=42)
    wr = fn(REGIME_WINDOWS[0], {"entry_threshold": 0.55})
    assert isinstance(wr, WindowResult)
    assert wr.window == REGIME_WINDOWS[0].name


def test_dry_run_reproducible():
    a = _dry_run_backtest_fn(seed=42)(REGIME_WINDOWS[0], {})
    b = _dry_run_backtest_fn(seed=42)(REGIME_WINDOWS[0], {})
    assert a.net_return_pct == b.net_return_pct


def test_run_sweep_dry_returns_sweep_result():
    """End-to-end smoke: dry-run sweep produces a SweepResult."""
    result = run_sweep_dry(n_samples=5, seed=42)
    assert isinstance(result, SweepResult)
    assert len(result.results) == 5
    # Each result has 6 windows
    for r in result.results:
        assert len(r.windows) == 6


def test_run_sweep_dry_subset_windows():
    result = run_sweep_dry(n_samples=3, windows=REGIME_WINDOWS[:2], seed=42)
    for r in result.results:
        assert len(r.windows) == 2


def test_run_sweep_dry_ranks_by_score():
    """ranked property orders by survivability_score descending."""
    result = run_sweep_dry(n_samples=10, seed=42)
    scores = [r.survivability_score for r in result.ranked]
    assert scores == sorted(scores, reverse=True)


# ───────────────────────────────────────────────────────────────────────────────
# CLI smoke
# ───────────────────────────────────────────────────────────────────────────────

def test_cli_dry_run_writes_html(tmp_path, monkeypatch):
    """--dry-run with --out should write a valid HTML report."""
    out = tmp_path / "sweep.html"
    import sys as _sys
    monkeypatch.setattr(_sys, "argv", [
        "qc_sweep_runner.py",
        "--dry-run", "--n-samples", "3",
        "--max-windows", "2",
        "--out", str(out),
    ])
    from backtest_audit import qc_sweep_runner
    qc_sweep_runner.main()
    assert out.exists()
    content = out.read_text()
    assert "Walk-Forward Sweep" in content
    assert "<html" in content
