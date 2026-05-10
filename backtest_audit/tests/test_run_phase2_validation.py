"""Tests for backtest_audit.run_phase2_validation."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from backtest_audit.run_phase2_validation import (
    _net_profit_pct, _gap_ratio, _classify_gap_ratio, run_phase2,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def test_net_profit_extracts_pct_string():
    assert _net_profit_pct({"Net Profit": "+12.34%"}) == 12.34
    assert _net_profit_pct({"Net Profit": "-47.5%"}) == -47.5


def test_net_profit_handles_alternative_keys():
    assert _net_profit_pct({"net_profit": "5.0%"}) == 5.0
    assert _net_profit_pct({"netProfit": "10.5"}) == 10.5


def test_net_profit_returns_none_on_missing():
    assert _net_profit_pct({}) is None
    assert _net_profit_pct({"unrelated": "x"}) is None


def test_net_profit_returns_none_on_garbage():
    assert _net_profit_pct({"Net Profit": "not a number"}) is None


# ───────────────────────────────────────────────────────────────────────────────
# Gap ratio classification
# ───────────────────────────────────────────────────────────────────────────────

def test_gap_ratio_basic():
    """Harsh +5%, standard +10% → ratio 0.5 (well within band)."""
    assert _gap_ratio(harsh_return=5.0, standard_return=10.0) == 0.5


def test_gap_ratio_negative_harsh():
    """Harsh negative against positive standard."""
    assert _gap_ratio(harsh_return=-5.0, standard_return=10.0) == -0.5


def test_gap_ratio_indeterminate_when_standard_tiny():
    """Standard return < 0.5% in absolute → can't compute meaningful ratio."""
    assert _gap_ratio(harsh_return=10.0, standard_return=0.1) is None
    assert _gap_ratio(harsh_return=10.0, standard_return=0.0) is None


def test_classify_gap_ratio_in_band_passes():
    assert "PASS" in _classify_gap_ratio(0.50)
    assert "PASS" in _classify_gap_ratio(0.30)
    assert "PASS" in _classify_gap_ratio(0.70)


def test_classify_gap_ratio_below_band_fails():
    assert "FAIL" in _classify_gap_ratio(0.10)
    assert "edge erodes" in _classify_gap_ratio(0.20)


def test_classify_gap_ratio_above_band_warns():
    assert "WARN" in _classify_gap_ratio(0.85)
    assert "look-ahead" in _classify_gap_ratio(0.95)


def test_classify_gap_ratio_indeterminate():
    assert "indeterminate" in _classify_gap_ratio(None)


# ───────────────────────────────────────────────────────────────────────────────
# run_phase2 with mocked dependencies
# ───────────────────────────────────────────────────────────────────────────────

def _make_mock_client(standard_return: float = 50.0,
                       harsh_return: float = 25.0):
    """Build a mock QCClient that returns plausible Phase 2 data."""
    c = MagicMock()
    c.create_compile.return_value = {"compileId": "c001"}
    c.wait_for_compile.return_value = {"state": "BuildSuccess"}

    # Each create_backtest call returns a different backtest id
    bt_ids = ["bt_standard", "bt_harsh"]
    def create_bt(*args, **kwargs):
        return {"backtest": {"backtestId": bt_ids.pop(0)}}
    c.create_backtest.side_effect = create_bt

    def wait_for_bt(pid, bid, **kwargs):
        ret_pct = standard_return if "standard" in bid else harsh_return
        return {
            "completed": True,
            "name": bid,
            "backtestId": bid,
            "statistics": {"Net Profit": f"{ret_pct}%"},
        }
    c.wait_for_backtest.side_effect = wait_for_bt

    c.read_backtest_orders.return_value = []   # empty for simplicity
    return c


def test_run_phase2_pass_band(tmp_path, monkeypatch):
    """Ratio 0.5 → PASS classification."""
    c = _make_mock_client(standard_return=50.0, harsh_return=25.0)
    # Mock deploy_pulse so we don't hit any filesystem
    with patch("backtest_audit.run_phase2_validation.deploy_pulse") as mock_dep:
        result = run_phase2(
            project_id=99, client=c, deploy_first=True,
            out_dir=str(tmp_path),
        )
    mock_dep.assert_called_once()
    assert result["gap_ratio"] == pytest.approx(0.5)
    assert "PASS" in result["classification"]
    # Report file exists
    assert os.path.exists(result["report_path"])


def test_run_phase2_fail_band(tmp_path):
    c = _make_mock_client(standard_return=50.0, harsh_return=5.0)   # ratio 0.10
    with patch("backtest_audit.run_phase2_validation.deploy_pulse"):
        result = run_phase2(
            project_id=99, client=c, deploy_first=False,
            out_dir=str(tmp_path),
        )
    assert "FAIL" in result["classification"]


def test_run_phase2_warn_band(tmp_path):
    c = _make_mock_client(standard_return=10.0, harsh_return=9.0)   # ratio 0.90
    with patch("backtest_audit.run_phase2_validation.deploy_pulse"):
        result = run_phase2(
            project_id=99, client=c, deploy_first=False,
            out_dir=str(tmp_path),
        )
    assert "WARN" in result["classification"]


def test_run_phase2_skips_deploy_when_no_deploy(tmp_path):
    c = _make_mock_client()
    with patch("backtest_audit.run_phase2_validation.deploy_pulse") as mock_dep:
        run_phase2(
            project_id=99, client=c, deploy_first=False,
            out_dir=str(tmp_path),
        )
    mock_dep.assert_not_called()


def test_run_phase2_returns_complete_payload(tmp_path):
    c = _make_mock_client()
    with patch("backtest_audit.run_phase2_validation.deploy_pulse"):
        result = run_phase2(
            project_id=99, client=c, deploy_first=False,
            out_dir=str(tmp_path),
        )
    expected_keys = {
        "standard_stats", "harsh_stats", "standard_bt_id", "harsh_bt_id",
        "standard_return", "harsh_return", "gap_ratio", "classification",
        "report_path",
    }
    assert set(result.keys()) == expected_keys


def test_run_phase2_cli_argparse_help(monkeypatch, capsys):
    """CLI should at least handle --help cleanly."""
    from backtest_audit import run_phase2_validation as mod
    monkeypatch.setattr(sys, "argv", ["run_phase2_validation.py", "--help"])
    with pytest.raises(SystemExit):
        mod.main()
