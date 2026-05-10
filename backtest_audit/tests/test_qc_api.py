"""Unit tests for backtest_audit.qc_api.

Most tests don't make network calls — they verify auth-header construction
and request shaping. A small set of tests use the live QC API; they only
run when QC_USER_ID + QC_API_TOKEN are present in env (i.e. opt-in).
"""

from __future__ import annotations

import os
import pytest
import urllib.request
from unittest.mock import patch, MagicMock

from backtest_audit.qc_api import QCClient, QCError


# ───────────────────────────────────────────────────────────────────────────────
# Auth header construction
# ───────────────────────────────────────────────────────────────────────────────

def test_from_env_missing_raises():
    """Without env vars, from_env() raises QCError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(QCError, match="QC_USER_ID"):
            QCClient.from_env()


def test_from_env_legacy_aliases_work():
    """QC_UID / QC_TOKEN aliases supported for backward compat."""
    with patch.dict(os.environ, {"QC_UID": "42", "QC_TOKEN": "abc"}, clear=True):
        c = QCClient.from_env()
        assert c.user_id == "42"
        assert c.token == "abc"


def test_headers_have_basic_auth_and_timestamp():
    c = QCClient(user_id="123", token="secret")
    h = c._headers()
    assert h["Authorization"].startswith("Basic ")
    assert "Timestamp" in h
    assert h["Content-Type"] == "application/json"
    # Timestamp is unix-seconds-string
    assert h["Timestamp"].isdigit()


def test_headers_change_per_call():
    """Each call signs with a fresh timestamp; digest differs."""
    import time
    c = QCClient(user_id="123", token="secret")
    h1 = c._headers()
    time.sleep(1.1)
    h2 = c._headers()
    assert h1["Timestamp"] != h2["Timestamp"]
    assert h1["Authorization"] != h2["Authorization"]


# ───────────────────────────────────────────────────────────────────────────────
# Request construction (mocked urlopen)
# ───────────────────────────────────────────────────────────────────────────────

def _mock_response(payload: dict):
    mock = MagicMock()
    mock.read.return_value = (
        b'{"success": true, ' + str(payload)[1:-1].replace("'", '"').encode() + b"}"
        if payload else b'{"success": true}'
    )
    mock.__enter__ = lambda self: mock
    mock.__exit__  = lambda self, *a: None
    return mock


def test_authenticate_call_get_request(monkeypatch):
    """authenticate() sends a GET request, no body."""
    seen = {}
    def fake_urlopen(req, timeout=None):
        seen["method"] = req.get_method()
        seen["url"]    = req.full_url
        seen["data"]   = req.data
        return _mock_response({})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    c = QCClient(user_id="123", token="abc")
    out = c.authenticate()
    assert out == {"success": True}
    assert seen["method"] == "GET"
    assert seen["url"].endswith("/authenticate")
    assert seen["data"] is None


def test_call_raises_on_qc_error(monkeypatch):
    """When QC payload has success=false, _call raises QCError."""
    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        m.read.return_value = b'{"success": false, "errors": ["bad token"]}'
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    c = QCClient(user_id="x", token="y")
    with pytest.raises(QCError, match="bad token"):
        c.authenticate()


# ───────────────────────────────────────────────────────────────────────────────
# Live integration — opt-in via env vars
# ───────────────────────────────────────────────────────────────────────────────

NEEDS_LIVE = pytest.mark.skipif(
    not (os.environ.get("QC_USER_ID") or os.environ.get("QC_UID")),
    reason="Set QC_USER_ID + QC_API_TOKEN (or QC_UID/QC_TOKEN) for live API tests",
)


@NEEDS_LIVE
def test_live_authenticate():
    c = QCClient.from_env()
    out = c.authenticate()
    assert out.get("success") is True


@NEEDS_LIVE
def test_live_list_projects_returns_some():
    c = QCClient.from_env()
    projects = c.list_projects()
    assert isinstance(projects, list)
    assert len(projects) > 0


@NEEDS_LIVE
def test_live_hydra_project_visible():
    """HYDRA-100x (pid=31410009) should be in the user's project list."""
    c = QCClient.from_env()
    projects = c.list_projects()
    names = {p.get("name") for p in projects}
    assert "HYDRA-100x" in names


@NEEDS_LIVE
def test_live_read_hydra_backtest():
    """Read the latest HYDRA backtest's stats."""
    c = QCClient.from_env()
    bts = c.list_backtests(project_id=31410009)
    assert len(bts) >= 1
    # The most recent backtest from the analysis was "Retrospective Violet Bull"
    bt = c.read_backtest(31410009, bts[0]["backtestId"])
    assert "statistics" in bt or "Statistics" in bt
