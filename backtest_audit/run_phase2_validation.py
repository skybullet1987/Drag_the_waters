"""run_phase2_validation — end-to-end Phase 2 harsh-sim validation runner.

PLAN.md §2.1 — measures the standard-vs-harsh backtest gap for Pulse on a
chosen window. Phase 2's exit criterion is **gap_ratio in [0.30, 0.70]**:

    gap_ratio = harsh_return / standard_return

  - < 0.30 → too much edge erodes under live-realistic assumptions →
            iterate on signal quality
  - 0.30 to 0.70 → acceptable; the strategy survives live realism
  - > 0.70 → suspiciously close → check for residual look-ahead bias

Workflow:
  1. Push Pulse to two QC projects (or to the same project with two backtest names)
  2. Run "standard" backtest (default QC slippage/fees)
  3. Run "harsh" backtest (with HarshSimulator overrides)
  4. Fetch both backtest payloads + orders
  5. Pair into CompletedTrade lists
  6. Build the gap report comparing standard (treated as 'live' baseline)
     vs harsh (treated as 'backtest' baseline)
  7. Render HTML audit report

Usage:
    python3 backtest_audit/run_phase2_validation.py \\
        --project-id 31410009 \\
        --start-year 2025 --end-year 2025

Output: prints a summary + writes:
    backtest_audit/reports/phase2_validation_<timestamp>.html

Note: this script uploads Pulse code via the QC API. To run the harsh-sim
variant, the user must currently set the harsh_simulator flag inside main.py
manually (auto-flag plumbing is a follow-up).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest_audit.qc_api import QCClient, QCError
from backtest_audit.qc_runner import deploy_pulse, run_backtest
from backtest_audit.qc_orders import fetch_backtest_trades
from backtest_audit.compare import build_report
from backtest_audit.report import write_audit_page
from backtest_audit.qc_sweep_runner import render_runtime_overrides


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "reports")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _net_profit_pct(stats: dict) -> float | None:
    """Extract Net Profit % from the QC statistics dict."""
    raw = stats.get("Net Profit") or stats.get("net_profit") or stats.get("netProfit")
    if raw is None:
        return None
    try:
        return float(str(raw).rstrip("%"))
    except (ValueError, TypeError):
        return None


def _gap_ratio(harsh_return: float, standard_return: float) -> float | None:
    """harsh / standard, with sign convention preserved.

    Returns None if standard_return is ~zero (would divide by zero meaningfully).
    """
    if abs(standard_return) < 0.5:    # less than 0.5% — too small to be meaningful
        return None
    return harsh_return / standard_return


def _classify_gap_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "indeterminate (standard return too small)"
    if ratio < 0.30:
        return "FAIL: too much edge erodes under live realism — iterate on signal"
    if ratio > 0.70:
        return "WARN: suspiciously close to standard — check for residual look-ahead"
    return "PASS: gap ratio within Phase 2 target band [0.30, 0.70]"


def run_phase2(
    project_id: int,
    *,
    start_year: int = 2025,
    end_year:   int = 2025,
    standard_backtest_name: str | None = None,
    harsh_backtest_name:    str | None = None,
    client: QCClient | None = None,
    out_dir: str = DEFAULT_OUT_DIR,
    deploy_first: bool = True,
) -> dict:
    """Execute the full Phase 2 validation workflow.

    Returns a dict with: ``standard_stats``, ``harsh_stats``, ``gap_ratio``,
    ``classification``, ``report_path``, ``standard_bt_id``, ``harsh_bt_id``.
    """
    c = client or QCClient.from_env()
    tag = _now_tag()
    standard_name = standard_backtest_name or f"phase2-standard-{tag}"
    harsh_name    = harsh_backtest_name    or f"phase2-harsh-{tag}"

    if deploy_first:
        print(f"[phase2] Deploying Pulse to project {project_id}...")
        deploy_pulse(client=c, project_id=project_id)

    # ── STANDARD backtest: clear any existing runtime overrides ──────────
    print(f"[phase2] === STANDARD backtest: {standard_name} ===")
    print(f"[phase2] Clearing runtime_overrides.py (use_harsh_sim=False)")
    c.update_file(
        project_id, "runtime_overrides.py",
        render_runtime_overrides({"use_harsh_sim": False}),
    )
    standard_bt = run_backtest(project_id, standard_name, client=c)
    standard_id = (standard_bt.get("backtestId")
                   or (standard_bt.get("backtest", {}) or {}).get("backtestId"))

    # ── HARSH backtest: push runtime_overrides with use_harsh_sim=True ───
    print(f"[phase2] === HARSH backtest: {harsh_name} ===")
    print(f"[phase2] Pushing runtime_overrides.py with use_harsh_sim=True")
    c.update_file(
        project_id, "runtime_overrides.py",
        render_runtime_overrides({"use_harsh_sim": True}),
    )
    harsh_bt = run_backtest(project_id, harsh_name, client=c)
    harsh_id = (harsh_bt.get("backtestId")
                or (harsh_bt.get("backtest", {}) or {}).get("backtestId"))

    standard_stats = standard_bt.get("statistics") or {}
    harsh_stats    = harsh_bt.get("statistics")    or {}

    standard_return = _net_profit_pct(standard_stats)
    harsh_return    = _net_profit_pct(harsh_stats)
    ratio = _gap_ratio(harsh_return or 0.0, standard_return or 0.0)
    classification = _classify_gap_ratio(ratio)

    print()
    print(f"[phase2] standard return: {standard_return}%")
    print(f"[phase2] harsh    return: {harsh_return}%")
    print(f"[phase2] gap_ratio:       {ratio}")
    print(f"[phase2] classification:  {classification}")

    # Fetch orders for both, build a comparable trade-pair report
    print(f"[phase2] Fetching orders for trade-pair comparison...")
    standard_trades = fetch_backtest_trades(c, project_id, standard_id)
    harsh_trades    = fetch_backtest_trades(c, project_id, harsh_id)
    print(f"[phase2] standard: {len(standard_trades)} trades")
    print(f"[phase2] harsh:    {len(harsh_trades)} trades")
    rep = build_report(live=standard_trades, backtest=harsh_trades)
    print()
    print(rep.summary_text())

    out_path = os.path.join(out_dir, f"phase2_validation_{tag}.html")
    notes = (
        f"Phase 2 harsh-sim validation\n\n"
        f"Project: {project_id}\n"
        f"Standard backtest: {standard_name} ({standard_id})\n"
        f"Harsh backtest:    {harsh_name} ({harsh_id})\n\n"
        f"Standard return: {standard_return}%\n"
        f"Harsh return:    {harsh_return}%\n"
        f"Gap ratio:       {ratio}\n\n"
        f"Classification: {classification}\n"
    )
    write_audit_page(
        out_path,
        title=f"Phase 2 Validation — {tag}",
        gap_report=rep,
        notes=notes,
    )
    print(f"\n[phase2] ✓ Wrote audit report: {out_path}")

    return {
        "standard_stats":    standard_stats,
        "harsh_stats":       harsh_stats,
        "standard_bt_id":    standard_id,
        "harsh_bt_id":       harsh_id,
        "standard_return":   standard_return,
        "harsh_return":      harsh_return,
        "gap_ratio":         ratio,
        "classification":    classification,
        "report_path":       out_path,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-id", type=int, required=True,
                    help="QC project ID to deploy Pulse into")
    ap.add_argument("--standard-name",
                    help="Backtest name for standard run (default auto-tagged)")
    ap.add_argument("--harsh-name",
                    help="Backtest name for harsh-sim run (default auto-tagged)")
    ap.add_argument("--no-deploy", action="store_true",
                    help="Skip the file-push step (project already has Pulse)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    run_phase2(
        project_id=args.project_id,
        standard_backtest_name=args.standard_name,
        harsh_backtest_name=args.harsh_name,
        deploy_first=not args.no_deploy,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
