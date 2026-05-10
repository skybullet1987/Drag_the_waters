"""Run the Phase 0 audit against the MG36 paper-trade fixture.

This is the entry point that produces a tangible artifact — an HTML report
showing the MG36 live-vs-backtest gap measured by our harness.

Usage (from repo root):
    python3 backtest_audit/run_mg36_audit.py [--out PATH]

If no --out is given, writes to backtest_audit/reports/mg36_audit.html.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the parent directory importable when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtest_audit.log_parser import parse_log_file, pair_trades
from backtest_audit.compare import build_report, report_from_paired_log
from backtest_audit.report import write_audit_page


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(HERE, "fixtures", "mg36_paper_2026-03-16.txt")
DEFAULT_OUT = os.path.join(HERE, "reports", "mg36_audit.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=DEFAULT_FIXTURE,
                    help="Path to QC algorithm-log_*.txt to audit")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Output HTML path")
    args = ap.parse_args()

    print(f"Parsing log: {args.log}")
    parsed = parse_log_file(args.log)
    summary = parsed.summary()
    print("\n--- LIVE LOG SUMMARY ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nBuilding gap report (live-only — no backtest counterpart yet)...")
    rep = report_from_paired_log(parsed)
    print()
    print(rep.summary_text())

    notes = (
        "Phase 0 audit anchored to MG36 paper-trade evidence (see PLAN.md §0.A).\n\n"
        "Findings (verified):\n"
        f"  - Mean per-fill slippage (live): {summary['mean_slippage_bps']}bp\n"
        f"  - Max per-fill slippage (live):  {summary['max_slippage_bps']}bp\n"
        f"  - Live trades:    {summary['orders_filled']} fills / {summary['exits']} round trips\n"
        f"  - Maker limits:   {summary['maker_limits']} created, "
        f"{summary['orders_canceled']} canceled (timed out)\n"
        f"  - Final equity:   ${summary['final_equity']} (PnL {summary['final_pnl_pct']}%)\n"
        f"  - Win rate:       {summary['final_win_rate']}% (vs backtest 74%)\n\n"
        "Smoking-gun trade: KASUSD score=0.95 max-conviction entry held 60s,\n"
        "  +0.07% gross move, but 304bp round-trip slippage = -2.29% net loss.\n\n"
        "Next: backtest the same period in QC under the harsh simulator,\n"
        "fetch backtest orders via QC API, then re-run this audit with both\n"
        "sides populated to produce the full attribution table."
    )

    out = write_audit_page(
        args.out,
        title="MG36 Phase 0 Audit — Live-Only View",
        parsed_log=parsed,
        gap_report=rep,
        notes=notes,
    )
    print(f"\n✓ Wrote audit report: {out}")


if __name__ == "__main__":
    main()
