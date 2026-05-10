"""qc_api — thin wrapper over QuantConnect REST v2 API.

Auth: HMAC-SHA256(token + ":" + timestamp), Basic Auth(uid:digest).

Credentials are read from the QC_USER_ID and QC_API_TOKEN environment
variables. NEVER hard-code credentials in source.

This wrapper is intentionally minimal: only the endpoints we actually use
for the backtest-vs-live audit. Read-only operations only.

Usage:
    client = QCClient.from_env()
    projects = client.list_projects()
    backtests = client.list_backtests(project_id=31410009)
    bt = client.read_backtest(project_id=31410009, backtest_id="b18bc60...")
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


QC_BASE_URL = "https://www.quantconnect.com/api/v2"


class QCError(Exception):
    """Raised when a QC API call fails."""


@dataclass
class QCClient:
    user_id: str
    token: str
    base_url: str = QC_BASE_URL
    timeout_s: int = 60

    @classmethod
    def from_env(cls) -> "QCClient":
        uid = os.environ.get("QC_USER_ID") or os.environ.get("QC_UID")
        tok = os.environ.get("QC_API_TOKEN") or os.environ.get("QC_TOKEN")
        if not uid or not tok:
            raise QCError(
                "QC_USER_ID and QC_API_TOKEN environment variables must be set "
                "(legacy aliases QC_UID / QC_TOKEN also accepted)."
            )
        return cls(user_id=str(uid), token=str(tok))

    # ── Auth helpers ─────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        ts = str(int(time.time()))
        digest = hashlib.sha256(f"{self.token}:{ts}".encode()).hexdigest()
        auth = base64.b64encode(f"{self.user_id}:{digest}".encode()).decode()
        return {
            "Authorization": "Basic " + auth,
            "Timestamp": ts,
            "Content-Type": "application/json",
        }

    def _call(self, path: str, data: dict | None = None) -> dict[str, Any]:
        body = json.dumps(data).encode() if data else None
        method = "POST" if data else "GET"
        req = urllib.request.Request(
            self.base_url + path,
            headers=self._headers(),
            data=body,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise QCError(f"HTTP {e.code} on {method} {path}: {e.read()[:300]}") from e
        except urllib.error.URLError as e:
            raise QCError(f"Network error on {method} {path}: {e.reason}") from e
        if isinstance(payload, dict) and not payload.get("success", True):
            errs = payload.get("errors") or payload.get("error") or payload
            raise QCError(f"QC API error on {method} {path}: {errs}")
        return payload

    # ── Endpoints ────────────────────────────────────────────────────────────

    def authenticate(self) -> dict[str, Any]:
        """Sanity check — does the API accept our credentials?"""
        return self._call("/authenticate")

    def list_projects(self) -> list[dict[str, Any]]:
        return self._call("/projects/read").get("projects", [])

    def read_project(self, project_id: int) -> dict[str, Any]:
        return self._call(f"/projects/read?projectId={project_id}")

    def list_files(self, project_id: int) -> list[dict[str, Any]]:
        r = self._call(f"/files/read?projectId={project_id}")
        return r.get("files", [])

    def read_file(self, project_id: int, name: str) -> str:
        encoded = urllib.parse.quote(name, safe="")
        r = self._call(f"/files/read?projectId={project_id}&name={encoded}")
        files = r.get("files", [])
        if not files:
            raise QCError(f"No file {name!r} in project {project_id}")
        return files[0].get("content", "")

    def list_backtests(self, project_id: int) -> list[dict[str, Any]]:
        return self._call(f"/backtests/list?projectId={project_id}").get(
            "backtests", []
        )

    def read_backtest(self, project_id: int, backtest_id: str) -> dict[str, Any]:
        r = self._call(
            f"/backtests/read?projectId={project_id}&backtestId={backtest_id}"
        )
        return r.get("backtest", r)

    def read_backtest_orders(self, project_id: int, backtest_id: str,
                             start: int = 0, end: int = 1000) -> list[dict[str, Any]]:
        """Fetch orders for a backtest (paginated by QC).

        QC's /backtests/orders/read returns up to 1000 orders per call.
        Caller is responsible for paginating beyond that.
        """
        path = (
            f"/backtests/orders/read?projectId={project_id}"
            f"&backtestId={backtest_id}&start={start}&end={end}"
        )
        return self._call(path).get("orders", [])

    def list_live_algorithms(self, project_id: int) -> list[dict[str, Any]]:
        """List live (and paper-live) algorithms for a project."""
        return self._call(f"/live/list?projectId={project_id}").get("live", [])

    def read_live(self, project_id: int, deploy_id: str) -> dict[str, Any]:
        r = self._call(
            f"/live/read?projectId={project_id}&deployId={deploy_id}"
        )
        return r.get("live", r)

    def read_live_orders(self, project_id: int, deploy_id: str,
                         start: int = 0, end: int = 1000) -> list[dict[str, Any]]:
        path = (
            f"/live/orders/read?projectId={project_id}"
            f"&deployId={deploy_id}&start={start}&end={end}"
        )
        return self._call(path).get("orders", [])

    def read_live_logs(self, project_id: int, algorithm_id: str,
                       start_line: int = 0, end_line: int = 100_000) -> list[str]:
        """Fetch the log lines for a live deployment.

        QC's /live/logs/read returns logs in segments. For a full pull,
        repeatedly call with increasing offsets.
        """
        path = (
            f"/live/logs/read?projectId={project_id}"
            f"&algorithmId={algorithm_id}"
            f"&startLine={start_line}&endLine={end_line}"
        )
        return self._call(path).get("LiveLogs", []) or []
