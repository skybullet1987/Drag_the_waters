#!/usr/bin/env python3
"""
qc_backtest_runner.py — Run a Vox gatling backtest on QuantConnect Cloud
and analyze per-model results.

Usage:
  export QC_USER_ID="your_user_id"
  export QC_API_TOKEN="your_api_token"
  python3 Vox/qc_backtest_runner.py [--project-id PROJECT_ID]

If --project-id is not provided, the script creates a new project, uploads
all Vox/*.py files, compiles, runs the backtest, waits for completion,
downloads the trade log, and runs model assessment.

Requires: QC_USER_ID and QC_API_TOKEN environment variables.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import requests
from pathlib import Path

# ── QC API config ────────────────────────────────────────────────────────────
QC_BASE_URL = "https://www.quantconnect.com/api/v2"
BACKTEST_POLL_INTERVAL = 15  # seconds between status checks
BACKTEST_TIMEOUT = 1800      # 30 min max wait


def _auth():
    uid = os.environ.get("QC_USER_ID", "")
    token = os.environ.get("QC_API_TOKEN", "")
    if not uid or not token:
        print("ERROR: Set QC_USER_ID and QC_API_TOKEN environment variables.")
        print("  Get them from: QuantConnect > My Account > Security")
        sys.exit(1)
    timestamp = str(int(time.time()))
    hash_bytes = hashlib.sha256(f"{token}{timestamp}".encode()).hexdigest()
    return {"Timestamp": timestamp, "Authorization": f"Basic {uid}:{hash_bytes}"}


def _api(method, endpoint, **kwargs):
    url = f"{QC_BASE_URL}/{endpoint}"
    headers = _auth()
    r = getattr(requests, method)(url, headers=headers, **kwargs)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True):
        print(f"API error: {data}")
        sys.exit(1)
    return data


# ── Project management ───────────────────────────────────────────────────────

def create_project(name="Vox-Gatling-Backtest"):
    """Create a new QC project."""
    data = _api("post", "projects/create", json={
        "name": name,
        "language": "Py",
    })
    pid = data["projects"][0]["projectId"]
    print(f"Created project '{name}' (ID: {pid})")
    return pid


def upload_files(project_id, vox_dir="Vox"):
    """Upload all Vox/*.py files to a QC project."""
    vox_path = Path(vox_dir)
    files_uploaded = 0
    for py_file in sorted(vox_path.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        if py_file.name == "qc_backtest_runner.py":
            continue
        if py_file.name == "research_model_vote_outcomes.py":
            continue
        content = py_file.read_text()
        _api("post", "files/create", json={
            "projectId": project_id,
            "name": py_file.name,
            "content": content,
        })
        files_uploaded += 1
        print(f"  Uploaded: {py_file.name} ({len(content):,} bytes)")
    print(f"Uploaded {files_uploaded} files to project {project_id}")
    return files_uploaded


def set_parameters(project_id, params):
    """Set QC project parameters."""
    _api("post", "projects/update", json={
        "projectId": project_id,
        "parameters": params,
    })
    print(f"Set parameters: {params}")


# ── Compile & Backtest ───────────────────────────────────────────────────────

def compile_project(project_id):
    """Compile the project and return compile ID."""
    data = _api("post", "compile/create", json={"projectId": project_id})
    compile_id = data["compileId"]
    state = data["state"]
    print(f"Compilation {compile_id}: {state}")
    if state == "BuildError":
        print("Build errors:")
        for err in data.get("logs", []):
            print(f"  {err}")
        sys.exit(1)
    return compile_id


def create_backtest(project_id, compile_id, name="Gatling-Backtest"):
    """Launch a backtest and return backtest ID."""
    data = _api("post", "backtests/create", json={
        "projectId": project_id,
        "compileId": compile_id,
        "backtestName": name,
    })
    bt_id = data["backtest"]["backtestId"]
    print(f"Backtest started: {bt_id}")
    return bt_id


def wait_for_backtest(project_id, backtest_id):
    """Poll until backtest completes. Returns backtest result dict."""
    start = time.time()
    while time.time() - start < BACKTEST_TIMEOUT:
        data = _api("get", "backtests/read", params={
            "projectId": project_id,
            "backtestId": backtest_id,
        })
        bt = data["backtest"]
        progress = bt.get("progress", 0)
        completed = bt.get("completed", False)
        error = bt.get("error")

        if error:
            print(f"Backtest ERROR: {error}")
            return bt

        if completed:
            print(f"Backtest completed! Progress: 100%")
            return bt

        elapsed = int(time.time() - start)
        print(f"  Progress: {progress:.0%} (elapsed: {elapsed}s)")
        time.sleep(BACKTEST_POLL_INTERVAL)

    print("Backtest timed out!")
    return None


# ── Results analysis ─────────────────────────────────────────────────────────

def analyze_results(bt):
    """Print key backtest statistics."""
    if not bt:
        print("No backtest results to analyze.")
        return

    stats = bt.get("statistics", {})
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    key_stats = [
        "Total Return", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown", "Win Rate", "Profit-Loss Ratio",
        "Total Trades", "Average Win", "Average Loss",
        "Net Profit", "Compounding Annual Return",
    ]
    for k in key_stats:
        v = stats.get(k, "N/A")
        print(f"  {k:<30} {v}")

    total_fees = stats.get("Total Fees", "N/A")
    print(f"  {'Total Fees':<30} {total_fees}")

    print("\n" + "-" * 60)
    print("TRADE COUNT ASSESSMENT")
    print("-" * 60)
    total_trades = stats.get("Total Trades", 0)
    if isinstance(total_trades, str):
        total_trades = int(total_trades.replace(",", "")) if total_trades != "N/A" else 0
    print(f"  Total trades: {total_trades}")
    if total_trades >= 100:
        print("  ✓ High trade count — good for model assessment")
    elif total_trades >= 30:
        print("  ~ Moderate trade count — usable for model assessment")
    else:
        print("  ✗ Low trade count — need more trades for reliable assessment")

    return stats


def fetch_trade_log(project_id):
    """Try to read the trade log from QC ObjectStore."""
    try:
        data = _api("get", "object/get", params={
            "organizationId": "",
            "key": "vox/trade_log.jsonl",
        })
        raw = data.get("data", "")
        if raw:
            trades = [json.loads(line) for line in raw.splitlines() if line.strip()]
            print(f"Fetched {len(trades)} trade log entries from ObjectStore")
            return trades
    except Exception as e:
        print(f"Could not fetch trade log: {e}")
    return []


def run_model_assessment(trades):
    """Run per-model accuracy assessment on trade log entries."""
    sys.path.insert(0, str(Path(__file__).parent))
    from model_assessment import compute_model_accuracy, rank_models, format_assessment_report

    entry_trades = [t for t in trades if t.get("event") == "exit"]
    if not entry_trades:
        entry_trades = [t for t in trades if "active_votes" in t and "realized_return" in t]

    if not entry_trades:
        print("No trades with model votes found in log.")
        return

    formatted = []
    for t in entry_trades:
        formatted.append({
            "active_votes": t.get("active_votes", t.get("model_votes", {})),
            "realized_return": t.get("realized_return", t.get("ret", 0.0)),
            "winner": t.get("winner", t.get("realized_return", 0) > 0),
        })

    accuracy = compute_model_accuracy(formatted, vote_threshold=0.50)
    print("\n" + format_assessment_report(accuracy, min_trades=3))

    print("\n=== Model Ranking (by Profit Factor) ===")
    for model_id, stats in rank_models(accuracy, by="profit_factor"):
        if stats["yes_count"] >= 3:
            pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] < 100 else "inf"
            print(f"  {model_id:<15} PF={pf:>8}  WR={stats['win_rate_when_yes']*100:>5.1f}%"
                  f"  Yes={stats['yes_count']:>3}  Edge={stats['edge_vs_no']*100:>+6.2f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Vox gatling backtest on QC Cloud")
    parser.add_argument("--project-id", type=int, help="Existing QC project ID")
    parser.add_argument("--project-name", default="Vox-Gatling-Backtest",
                        help="Name for new project")
    parser.add_argument("--backtest-name", default="Gatling-Run",
                        help="Name for the backtest")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip file upload (use existing project files)")
    args = parser.parse_args()

    print("=" * 60)
    print("Vox Gatling Backtest Runner")
    print("=" * 60)

    # 1. Create or use project
    if args.project_id:
        pid = args.project_id
        print(f"Using existing project: {pid}")
    else:
        pid = create_project(args.project_name)

    # 2. Upload files
    if not args.skip_upload:
        upload_files(pid)

    # 3. Set gatling parameters
    set_parameters(pid, {
        "risk_profile": "gatling",
    })

    # 4. Compile
    compile_id = compile_project(pid)

    # 5. Run backtest
    bt_id = create_backtest(pid, compile_id, args.backtest_name)

    # 6. Wait for completion
    bt = wait_for_backtest(pid, bt_id)

    # 7. Analyze results
    stats = analyze_results(bt)

    # 8. Fetch and analyze trade log for model assessment
    trades = fetch_trade_log(pid)
    if trades:
        run_model_assessment(trades)
    else:
        print("\nTo run model assessment, export the trade log from QC Research:")
        print('  qb.object_store.read("vox/trade_log.jsonl")')

    print("\n" + "=" * 60)
    print("Done! Review results above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
