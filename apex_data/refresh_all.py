"""apex_data.refresh_all — one-shot script to refresh all Apex data CSVs.

Run before each QC redeploy:
    python3 apex_data/refresh_all.py [--out-dir Pulse]

Outputs:
    <out_dir>/apex_etf_flows.csv          ← network fetch
    <out_dir>/apex_token_unlocks.csv      ← projected from hardcoded schedules

Continues on error — if one fetcher fails, the others still run.
"""

from __future__ import annotations

import argparse
import os
import sys

from apex_data import fetch_etf_flows
from apex_data import fetch_token_unlocks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="Pulse",
                    help="directory where the CSVs are written")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    failed: list[str] = []

    # ETF flows
    try:
        rc = fetch_etf_flows.main([
            "--out", os.path.join(args.out_dir, "apex_etf_flows.csv"),
        ])
        if rc != 0:
            failed.append(f"fetch_etf_flows (rc={rc})")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"fetch_etf_flows ({exc})")

    # Token unlocks
    try:
        rc = fetch_token_unlocks.main([
            "--out", os.path.join(args.out_dir, "apex_token_unlocks.csv"),
        ])
        if rc != 0:
            failed.append(f"fetch_token_unlocks (rc={rc})")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"fetch_token_unlocks ({exc})")

    if failed:
        print("\n[refresh_all] FAILURES:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\n[refresh_all] all fetchers completed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
