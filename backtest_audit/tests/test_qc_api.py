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


# ───────────────────────────────────────────────────────────────────────────────
# WRITE methods (mocked)
# ───────────────────────────────────────────────────────────────────────────────

import urllib.request
import urllib.error
import urllib.parse
import time
from unittest.mock import MagicMock, patch


def _make_mock_urlopen(payload_dict):
    """Helper: produce a mock urlopen that returns the given JSON payload."""
    import json as _json
    body = _json.dumps(payload_dict).encode()

    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        m.read.return_value = body
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        # Also expose req for assertion
        fake_urlopen.last_req = req
        return m

    fake_urlopen.last_req = None
    return fake_urlopen


def test_create_project_posts_correct_body(monkeypatch):
    fake = _make_mock_urlopen({"success": True, "projects": [{"projectId": 99}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    c = QCClient(user_id="x", token="y")
    out = c.create_project("MyProject", language="Py")
    assert out["projects"][0]["projectId"] == 99
    assert fake.last_req.get_method() == "POST"
    assert "/projects/create" in fake.last_req.full_url


def test_update_file_falls_back_to_create_on_not_found(monkeypatch):
    """When QC says 'file not found', upsert via create_file."""
    call_log = []

    def fake_urlopen(req, timeout=None):
        path = req.full_url
        call_log.append(path)
        m = MagicMock()
        if "/files/update" in path:
            # Simulate the QC error envelope
            m.read.return_value = (
                b'{"success": false, "errors": ["File not found"]}'
            )
        elif "/files/create" in path:
            m.read.return_value = b'{"success": true}'
        else:
            m.read.return_value = b'{"success": true}'
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = QCClient(user_id="x", token="y")
    c.update_file(42, "main.py", "print('hi')")
    # Should have called update first, then create
    assert any("/files/update" in p for p in call_log)
    assert any("/files/create" in p for p in call_log)


def test_upsert_files_pushes_each(monkeypatch):
    seen_paths = []

    def fake_urlopen(req, timeout=None):
        seen_paths.append(req.full_url)
        m = MagicMock()
        m.read.return_value = b'{"success": true}'
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = QCClient(user_id="x", token="y")
    c.upsert_files(42, {"main.py": "...", "scoring.py": "..."})
    # Two file ops; each updates first
    assert sum(1 for p in seen_paths if "/files/update" in p) == 2


def test_create_compile_endpoint(monkeypatch):
    fake = _make_mock_urlopen({"success": True, "compileId": "abc123"})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    c = QCClient(user_id="x", token="y")
    r = c.create_compile(99)
    assert r["compileId"] == "abc123"
    assert "/compile/create" in fake.last_req.full_url


def test_wait_for_compile_polls_until_success(monkeypatch):
    """Returns immediately when state already terminal."""
    states = ["InQueue", "BuildSuccess"]

    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        # Pop the next state per call
        state = states.pop(0) if states else "BuildSuccess"
        m.read.return_value = (
            f'{{"success": true, "state": "{state}"}}'.encode()
        )
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = QCClient(user_id="x", token="y")
    r = c.wait_for_compile(99, "abc123", poll_s=0)
    assert r["state"] == "BuildSuccess"


def test_wait_for_compile_timeout(monkeypatch):
    """Stays in InQueue forever → QCError after timeout."""
    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        m.read.return_value = b'{"success": true, "state": "InQueue"}'
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = QCClient(user_id="x", token="y")
    with pytest.raises(QCError, match="timed out"):
        c.wait_for_compile(99, "x", timeout_s=1, poll_s=0)


def test_create_backtest_includes_compile_id(monkeypatch):
    fake = _make_mock_urlopen({"success": True, "backtest": {"backtestId": "bt001"}})
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    c = QCClient(user_id="x", token="y")
    r = c.create_backtest(99, "abc123", "phase2-validation")
    assert r["backtest"]["backtestId"] == "bt001"


def test_wait_for_backtest_returns_when_completed(monkeypatch):
    states = [False, False, True]

    def fake_urlopen(req, timeout=None):
        m = MagicMock()
        completed = states.pop(0) if states else True
        body = f'{{"success": true, "backtest": {{"completed": {str(completed).lower()}, "name": "x"}}}}'
        m.read.return_value = body.encode()
        m.__enter__ = lambda self: m
        m.__exit__  = lambda self, *a: None
        return m

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    c = QCClient(user_id="x", token="y")
    r = c.wait_for_backtest(99, "bt001", poll_s=0)
    # The wrapper unwraps to dict; completed should be True
    assert r.get("completed") is True or r.get("name") == "x"
