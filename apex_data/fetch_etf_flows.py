"""apex_data.fetch_etf_flows — produce apex_etf_flows.csv for QC.

Pulls daily Bitcoin ETF flow data (USD millions) from the
``boonchuan/btc-etf-price-impact`` GitHub mirror, which itself sources
from Farside Investors. The mirror is a flat CSV updated periodically
and is the most reliable FREE source we found (Farside's own site is
behind Cloudflare, so direct scraping fails).

Output format (consumed by Pulse.apex.data.etf_flows.ETFFlowStore):
    Date,Inflow_USD_M
    2024-01-11,655.3
    2024-01-12,203.0
    ...

Only the date + total flow is kept; per-ETF columns are summed.

Usage:
    python3 apex_data/fetch_etf_flows.py [--out PATH] [--source URL]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import urllib.request
import urllib.error


DEFAULT_SOURCE = (
    "https://raw.githubusercontent.com/boonchuan/btc-etf-price-impact/"
    "main/data/btc_etf_flows.csv"
)
DEFAULT_OUT = "apex_etf_flows.csv"


# ─── Pure functions (testable without network) ──────────────────────────────


def parse_source_csv(content: str) -> list[tuple[str, float]]:
    """Parse the mirror CSV and return [(date_str, total_inflow_usd_m), ...].

    The mirror's first column is `date` and the LAST column is `total`.
    Some rows can be malformed (empty cells, '-', etc.) — those are skipped.

    Returns rows in source order (oldest → newest).
    """
    out: list[tuple[str, float]] = []
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return out

    header = rows[0]
    if not header or header[0].strip().lower() not in ("date",):
        return out
    total_idx = None
    for i, col in enumerate(header):
        if col.strip().lower() == "total":
            total_idx = i
            break
    if total_idx is None:
        return out

    for r in rows[1:]:
        if len(r) <= total_idx:
            continue
        date = r[0].strip()
        cell = r[total_idx].strip().replace("$", "").replace(",", "")
        if not date or not cell or cell in ("-", "—", "n/a", "N/A"):
            continue
        try:
            total = float(cell)
        except ValueError:
            continue
        out.append((date, total))
    return out


def to_apex_csv(rows: list[tuple[str, float]]) -> str:
    """Render in the format Pulse.apex.data.etf_flows.ETFFlowStore loads."""
    lines = ["Date,Inflow_USD_M"]
    for date, total in rows:
        lines.append(f"{date},{total:.2f}")
    return "\n".join(lines) + "\n"


# ─── Fetcher ────────────────────────────────────────────────────────────────


def fetch_csv(source: str = DEFAULT_SOURCE, *, timeout_s: int = 30) -> str:
    """HTTP GET the source CSV; returns the body as text."""
    req = urllib.request.Request(
        source,
        headers={"User-Agent": "apex-data-fetcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    return body.decode("utf-8", errors="replace")


def write_apex_csv(rows: list[tuple[str, float]], out_path: str) -> None:
    """Write rendered CSV atomically (write to tmp, then rename)."""
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(to_apex_csv(rows))
    os.replace(tmp, out_path)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="URL of the source CSV mirror")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output path (default: apex_etf_flows.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write; just print the first 10 rows")
    args = ap.parse_args(argv)

    print(f"[fetch_etf_flows] GET {args.source}")
    try:
        body = fetch_csv(args.source)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"[fetch_etf_flows] FAILED: {exc}", file=sys.stderr)
        return 2
    rows = parse_source_csv(body)
    if not rows:
        print("[fetch_etf_flows] WARNING: parsed 0 rows; check source format",
              file=sys.stderr)
        return 1
    print(f"[fetch_etf_flows] parsed {len(rows)} daily rows "
          f"({rows[0][0]} → {rows[-1][0]})")
    if args.dry_run:
        for date, total in rows[:10]:
            print(f"  {date}  {total:+.1f}M")
        return 0
    write_apex_csv(rows, args.out)
    size = os.path.getsize(args.out)
    print(f"[fetch_etf_flows] wrote {args.out} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
