"""Tests for backtest_audit.qc_runner."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from backtest_audit import qc_runner
from backtest_audit.qc_runner import (
    PULSE_FILES, _read_pulse_dir, deploy_pulse, run_backtest,
)
from backtest_audit.qc_api import QCError


# ───────────────────────────────────────────────────────────────────────────────
# PULSE_FILES manifest sanity
# ───────────────────────────────────────────────────────────────────────────────

def test_pulse_files_includes_essential_modules():
    """The manifest must cover every Pulse module needed for QC to compile."""
    essentials = ("config.py", "main.py", "scalp_engine.py", "circuit.py",
                  "universe.py", "execution.py", "events.py")
    for e in essentials:
        assert e in PULSE_FILES


def test_pulse_files_omits_init_py():
    """QC projects are flat — __init__.py is meaningless and would conflict."""
    assert "__init__.py" not in PULSE_FILES


# ───────────────────────────────────────────────────────────────────────────────
# _read_pulse_dir
# ───────────────────────────────────────────────────────────────────────────────

def test_read_pulse_dir_returns_dict_of_contents(tmp_path):
    """Synthetic Pulse dir — verify all files are read."""
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    for name in PULSE_FILES:
        (pulse_dir / name).write_text(f"# stub for {name}\nx = 1\n")
    files = _read_pulse_dir(str(pulse_dir))
    assert set(files.keys()) == set(PULSE_FILES)
    for name, content in files.items():
        assert f"stub for {name}" in content


def test_read_pulse_dir_rewrites_pulse_imports(tmp_path):
    """`from Pulse.X import Y` should be flattened to `from X import Y`."""
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    src = "from Pulse.config import X\nimport Pulse.universe\n"
    for name in PULSE_FILES:
        (pulse_dir / name).write_text(src)
    files = _read_pulse_dir(str(pulse_dir))
    for content in files.values():
        assert "from Pulse." not in content
        assert "import Pulse." not in content
        assert "from config import X" in content
        assert "import universe" in content


def test_read_pulse_dir_rewrites_from_pulse_import_form(tmp_path):
    """Regression for diag-edge-discovery runtime error:
    `from Pulse import config as _pulse_config` must be rewritten too,
    not only the `from Pulse.X import Y` form."""
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    src = (
        "from Pulse import config as _pulse_config\n"
        "from Pulse import alpha, beta\n"
        "x = _pulse_config.SOMETHING\n"
    )
    for name in PULSE_FILES:
        (pulse_dir / name).write_text(src)
    files = _read_pulse_dir(str(pulse_dir))
    for content in files.values():
        # No bare-Pulse references must survive — that's what crashed QC.
        for line in content.split("\n"):
            stripped = line.strip()
            assert not stripped.startswith("from Pulse"), \
                f"unrewritten line: {stripped!r}"
            assert not stripped.startswith("import Pulse "), \
                f"unrewritten line: {stripped!r}"
        # And the rewritten target should be present
        assert "import config as _pulse_config" in content
        assert "import alpha" in content
        assert "import beta" in content


def test_read_pulse_dir_no_bare_pulse_references_in_real_files():
    """Hard regression: scan the actual workspace Pulse/ files and ensure
    NO line begins with `from Pulse` or `import Pulse` after rewrite."""
    workspace_pulse = os.path.join(
        os.path.dirname(__file__), "..", "..", "Pulse",
    )
    files = _read_pulse_dir(workspace_pulse)
    bad: list[str] = []
    for name, content in files.items():
        for ln_num, line in enumerate(content.split("\n"), 1):
            s = line.strip()
            if s.startswith("from Pulse") or s.startswith("import Pulse "):
                bad.append(f"{name}:{ln_num}: {s}")
    assert not bad, "Unrewritten Pulse imports leaked to QC payload:\n" + \
                    "\n".join(bad)


def test_read_pulse_dir_missing_file_raises(tmp_path):
    """Missing file should be a clean FileNotFoundError, not silent success."""
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    # Only create some of the manifest
    (pulse_dir / "config.py").write_text("x = 1")
    with pytest.raises(FileNotFoundError, match="Missing Pulse file"):
        _read_pulse_dir(str(pulse_dir))


def test_read_pulse_dir_handles_real_workspace_pulse():
    """The actual Pulse/ directory in this repo should load cleanly."""
    workspace_pulse = os.path.join(
        os.path.dirname(__file__), "..", "..", "Pulse",
    )
    files = _read_pulse_dir(workspace_pulse)
    assert set(files.keys()) == set(PULSE_FILES)
    # Pulse main.py imports `from Pulse.universe import ...` — verify it
    # got rewritten to `from universe import ...`
    main_content = files["main.py"]
    assert "from Pulse." not in main_content
    # And the rewritten import should be present somewhere
    assert "from universe import" in main_content or "from universe " in main_content


# ───────────────────────────────────────────────────────────────────────────────
# deploy_pulse with mocked QCClient
# ───────────────────────────────────────────────────────────────────────────────

def _mock_client(create_project_pid=42):
    c = MagicMock()
    c.create_project.return_value = {
        "projects": [{"projectId": create_project_pid}]
    }
    c.update_file.return_value = {"success": True}
    c.create_compile.return_value = {"compileId": "comp001"}
    c.wait_for_compile.return_value = {"state": "BuildSuccess"}
    c.create_backtest.return_value = {"backtest": {"backtestId": "bt001"}}
    c.wait_for_backtest.return_value = {
        "completed": True, "name": "test-bt",
        "statistics": {"Net Profit": "12.34%"},
    }
    return c


def test_deploy_pulse_uses_existing_project_id(tmp_path):
    """When project_id is given, no create_project should be called."""
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    for n in PULSE_FILES:
        (pulse_dir / n).write_text("# stub\n")
    c = _mock_client()
    out = deploy_pulse(client=c, project_id=99, pulse_dir=str(pulse_dir))
    assert out == 99
    c.create_project.assert_not_called()
    assert c.update_file.call_count == len(PULSE_FILES)


def test_deploy_pulse_creates_project_when_no_id(tmp_path):
    pulse_dir = tmp_path / "Pulse"
    pulse_dir.mkdir()
    for n in PULSE_FILES:
        (pulse_dir / n).write_text("# stub\n")
    c = _mock_client(create_project_pid=77)
    out = deploy_pulse(client=c, project_name="MyProject", pulse_dir=str(pulse_dir))
    assert out == 77
    c.create_project.assert_called_once_with(name="MyProject", language="Py")


def test_deploy_pulse_requires_id_or_name():
    c = _mock_client()
    with pytest.raises(ValueError, match="project_id or project_name"):
        deploy_pulse(client=c, project_id=None, project_name=None)


# ───────────────────────────────────────────────────────────────────────────────
# run_backtest with mocked QCClient
# ───────────────────────────────────────────────────────────────────────────────

def test_run_backtest_compiles_then_runs():
    c = _mock_client()
    bt = run_backtest(99, "phase2-test", client=c)
    c.create_compile.assert_called_once_with(99)
    c.wait_for_compile.assert_called_once()
    c.create_backtest.assert_called_once_with(99, "comp001", "phase2-test")
    c.wait_for_backtest.assert_called_once()
    assert bt["statistics"]["Net Profit"] == "12.34%"


def test_run_backtest_raises_on_compile_failure():
    c = _mock_client()
    c.wait_for_compile.return_value = {"state": "BuildError",
                                       "errors": ["syntax error in main.py"]}
    with pytest.raises(QCError, match="compile failed"):
        run_backtest(99, "x", client=c)


def test_run_backtest_raises_when_compile_id_missing():
    c = _mock_client()
    c.create_compile.return_value = {}   # no compileId
    with pytest.raises(QCError, match="compile_id missing"):
        run_backtest(99, "x", client=c)


def test_run_backtest_raises_when_backtest_id_missing():
    c = _mock_client()
    c.create_backtest.return_value = {}    # no backtestId
    with pytest.raises(QCError, match="backtest_id missing"):
        run_backtest(99, "x", client=c)


# ───────────────────────────────────────────────────────────────────────────────
# CLI smoke
# ───────────────────────────────────────────────────────────────────────────────

def test_cli_no_args_prints_help_or_errors(capsys, monkeypatch):
    """Running with no args + no env should fail cleanly."""
    monkeypatch.delenv("QC_USER_ID", raising=False)
    monkeypatch.delenv("QC_API_TOKEN", raising=False)
    monkeypatch.delenv("QC_UID", raising=False)
    monkeypatch.delenv("QC_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["qc_runner.py"])
    with pytest.raises((SystemExit, QCError)):
        qc_runner.main()


# ───────────────────────────────────────────────────────────────────────────────
# Regression: PULSE_FILES manifest must include every Pulse/*.py that is
# imported by main.py or by any other Pulse module. Catches the bug where
# online_learning.py / optimal_execution.py were added to Pulse but never
# to PULSE_FILES, causing QC compile to fail with "No module named X".
# ───────────────────────────────────────────────────────────────────────────────

def test_pulse_files_manifest_complete_for_real_pulse_dir():
    """Every .py file in /workspace/Pulse (except __init__, sanity_check,
    tests/) must appear in PULSE_FILES."""
    import glob
    import os as _os
    pulse_dir = _os.path.join(
        _os.path.dirname(__file__), "..", "..", "Pulse",
    )
    on_disk = set()
    for path in glob.glob(_os.path.join(pulse_dir, "*.py")):
        name = _os.path.basename(path)
        if name in ("__init__.py", "sanity_check.py"):
            continue
        on_disk.add(name)
    in_manifest = set(PULSE_FILES)
    missing = on_disk - in_manifest
    assert not missing, (
        f"PULSE_FILES manifest is missing these Pulse modules: {sorted(missing)}\n"
        f"Add them to qc_runner.PULSE_FILES or QC compile will fail with "
        f"'No module named X'."
    )


def test_pulse_files_manifest_imports_resolve():
    """Cross-check: every name imported in main.py via 'from X import' or
    'import X' must either be in PULSE_FILES or be a known stdlib/third-party.
    Detects accidentally renamed modules.
    """
    import re
    import os as _os
    main_py = _os.path.join(
        _os.path.dirname(__file__), "..", "..", "Pulse", "main.py",
    )
    with open(main_py) as f:
        src = f.read()
    # Find 'from Pulse.X import' and 'from X import' (after rewrite)
    pulse_imports = re.findall(r"from\s+Pulse\.(\w+)\s+import", src)
    bare_imports  = re.findall(r"^from\s+(\w+)\s+import", src, flags=re.M)
    # Pulse imports become 'from X import' after qc_runner rewrites them
    expected_from_main = set(pulse_imports)
    for name in expected_from_main:
        assert f"{name}.py" in PULSE_FILES, (
            f"main.py imports 'from Pulse.{name}' but '{name}.py' "
            f"is not in qc_runner.PULSE_FILES manifest"
        )
