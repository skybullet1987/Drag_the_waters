"""apex_data.refresh_all — one-shot driver for all Apex data refreshes.

Run before each QC redeploy:
    python3 apex_data/refresh_all.py [--out-dir Pulse]

Outputs (under <out_dir>):
    apex_etf_flows.csv             ← network fetch
    apex_etf_flows_data.py         ← .py wrapper (QC blocks .csv extension)
    apex_token_unlocks.csv         ← projected from hardcoded schedules
    apex_token_unlocks_data.py     ← .py wrapper

The .py wrappers each define `CONTENT = "<the csv text>"`. This is the
form Pulse/main.py loads at QC runtime (since QC's project-file API
rejects non-Python extensions).

Continues on error — if one fetcher fails, the others still run.
"""

from __future__ import annotations

import argparse
import os
import sys

from apex_data import fetch_etf_flows
from apex_data import fetch_token_unlocks


def csv_to_py_module(csv_path: str, out_py_path: str,
                      docstring: str = "Auto-generated CSV bundle.") -> None:
    """Wrap a CSV file's content into a Python module with `CONTENT = ...`."""
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    tmp = out_py_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f'"""{docstring}"""\n\n')
        f.write(f"CONTENT = {content!r}\n")
    os.replace(tmp, out_py_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="Pulse",
                    help="directory where outputs are written")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    failed: list[str] = []

    pairs: list[tuple[str, str, str]] = [
        # (fetcher_name, csv_filename, py_module_filename)
        ("fetch_etf_flows", "apex_etf_flows.csv",
         "apex_etf_flows_data.py"),
        ("fetch_token_unlocks", "apex_token_unlocks.csv",
         "apex_token_unlocks_data.py"),
    ]

    for fetcher_name, csv_name, py_name in pairs:
        csv_path = os.path.join(args.out_dir, csv_name)
        py_path = os.path.join(args.out_dir, py_name)
        try:
            mod = {"fetch_etf_flows": fetch_etf_flows,
                   "fetch_token_unlocks": fetch_token_unlocks}[fetcher_name]
            rc = mod.main(["--out", csv_path])
            if rc != 0:
                failed.append(f"{fetcher_name} (rc={rc})")
                continue
            # Wrap the CSV into a .py bundle for QC consumption
            csv_to_py_module(
                csv_path, py_path,
                docstring=f"Auto-generated bundle of {csv_name} as a "
                          f"Python module (QC blocks .csv extensions).",
            )
            print(f"[refresh_all] wrapped → {py_path} "
                  f"({os.path.getsize(py_path):,} bytes)")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{fetcher_name} ({exc})")

    if failed:
        print("\n[refresh_all] FAILURES:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\n[refresh_all] all fetchers completed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
