"""Tests for backtest_audit.report — verify HTML renders cleanly."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from backtest_audit.log_parser import parse_log_file, pair_trades
from backtest_audit.compare import build_report, report_from_paired_log
from backtest_audit.regime_runner import (
    REGIME_WINDOWS, WindowResult, score_param_set, sweep,
)
from backtest_audit.report import (
    render_audit_page, write_audit_page,
    render_gap_report, render_survivability, render_sweep,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "mg36_paper_2026-03-16.txt"
)


def _wr(name, ret, dd, sharpe=1.0, wr=50.0, trades=100):
    return WindowResult(window=name, net_return_pct=ret, drawdown_pct=dd,
                        sharpe=sharpe, win_rate_pct=wr, trades=trades)


# ───────────────────────────────────────────────────────────────────────────────
# render_audit_page on MG36 fixture
# ───────────────────────────────────────────────────────────────────────────────

def test_renders_mg36_live_only_page(tmp_path):
    parsed = parse_log_file(FIXTURE)
    rep = report_from_paired_log(parsed)
    page = render_audit_page(
        title="MG36 paper log — live-only audit",
        parsed_log=parsed,
        gap_report=rep,
    )
    assert "<!DOCTYPE html>" in page
    assert "MG36" in page
    assert "Live log summary" in page
    assert "Backtest-vs-Live Gap" in page
    # Slippage stats from the live log should appear
    assert "111" in page or "91" in page  # mean live entry slip ~91, mean overall ~111

    out = tmp_path / "mg36.html"
    write_audit_page(str(out),
                     title="MG36 paper log — live-only audit",
                     parsed_log=parsed,
                     gap_report=rep)
    assert out.exists()
    content = out.read_text()
    # Reasonable size threshold for a populated report
    assert len(content) > 4000


def test_render_with_blessed_param():
    wrs = [_wr(f"w{i}", 12.0, 5.0) for i in range(6)]
    s = score_param_set("good_set", wrs)
    page = render_audit_page(title="Blessed test", survivability=s)
    assert "BLESSED" in page
    assert "good_set" in page


def test_render_with_rejected_param():
    """Param with one catastrophic window is rejected."""
    wrs = [
        _wr("w1", 50.0, 5.0), _wr("w2", 50.0, 5.0),
        _wr("w3", 50.0, 5.0), _wr("w4", 50.0, 5.0),
        _wr("w5", 50.0, 5.0), _wr("w6", -25.0, 30.0),
    ]
    s = score_param_set("bad_set", wrs)
    page = render_audit_page(title="Rejected test", survivability=s)
    assert "NOT BLESSED" in page
    assert "CATASTROPHIC" in page


def test_render_sweep_page():
    def fake(w, p):
        return _wr(w.name, p["q"], 5.0)
    sw = sweep([("a", {"q": 8}), ("b", {"q": -3}), ("c", {"q": 12})], fake)
    page = render_audit_page(title="Sweep test", sweep_result=sw)
    assert "blessed" in page.lower()
    assert "Walk-Forward Sweep" in page


def test_render_with_notes():
    page = render_audit_page(title="Notes test", notes="Important: read me!")
    assert "Important: read me!" in page
    assert "Notes" in page


def test_render_html_escapes_user_data():
    """Make sure the report doesn't allow XSS via the title."""
    page = render_audit_page(title="<script>alert('xss')</script>")
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
