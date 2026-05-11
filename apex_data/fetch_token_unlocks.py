"""apex_data.fetch_token_unlocks — produce apex_token_unlocks.csv for QC.

Token unlock data is locked behind paid APIs (Tokenomist $99/mo,
CryptoRank $49/mo, DefiLlama Pro $300/mo). Free public scraping is
blocked by Cloudflare on every major source.

PRAGMATIC SOLUTION
==================
Hardcode the well-documented monthly cliff vesting schedules for the
two coins in our Pulse universe with ACTIONABLE unlock pressure:

  ARB (Arbitrum)   ~92.65M ARB on the 16th of every month
                   ≈ 1.85% of circulating supply per unlock
                   ≈ $25-50M depending on price
                   (DAO-published schedule, started 2024-03-16,
                    continues monthly through 2027)

  OP  (Optimism)   ~24.16M OP on the last day of every month
                   ≈ 1.50% of circulating supply per unlock
                   (Mix of investors + ecosystem cliff vesting,
                    started 2023-05-31, continues through 2026)

Both schedules are publicly documented and don't change. The
fetcher PROJECTS the schedule forward to a configurable end date
and emits a CSV the apex/data/token_unlocks.py loader consumes.

For other tokens (BTC, ETH, SOL, XRP, ADA, etc.) — there are NO
imminent material unlocks (BTC and ETH are fully circulating; SOL
unlock schedule has tapered to negligible amounts).

When new tokens with material unlocks are added to the universe, add
them to KNOWN_UNLOCK_SCHEDULES below and re-run.

Usage:
    python3 apex_data/fetch_token_unlocks.py [--out PATH] [--end-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from datetime import date, datetime, timedelta


DEFAULT_OUT = "apex_token_unlocks.csv"
DEFAULT_END_DATE = "2027-12-31"


# ─── Known schedules (publicly documented vesting cliffs) ───────────────────


KNOWN_UNLOCK_SCHEDULES: list[dict] = [
    {
        "symbol":          "ARB",
        "first_unlock":    date(2024, 3, 16),
        "cadence":         "monthly_day",
        "day_of_month":    16,
        "tokens_per_event": 92_652_008,
        "circulating_at_first_unlock": 5_000_000_000,
        "stop_after":      date(2027, 3, 16),    # ~48 monthly tranches
        "source":          "arbitrum-foundation/dao-treasury-spec",
    },
    {
        "symbol":          "OP",
        "first_unlock":    date(2023, 5, 31),
        "cadence":         "monthly_eom",
        "day_of_month":    None,
        "tokens_per_event": 24_160_000,
        "circulating_at_first_unlock": 1_100_000_000,
        "stop_after":      date(2026, 5, 31),    # 36 monthly tranches
        "source":          "optimism-foundation/tokenomics-v2",
    },
]


# ─── Pure schedule projector ────────────────────────────────────────────────


def _eom(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, d.day)
    # Clamp to month length (avoids "Feb 30" errors)
    last = calendar.monthrange(d.year, d.month + 1)[1]
    return date(d.year, d.month + 1, min(d.day, last))


def project_schedule(schedule: dict, end_date: date) -> list[tuple[date, str, float, float]]:
    """Project a single schedule into [(date, symbol, usd_estimate_mock,
    pct_of_supply), ...].

    USD estimate is a *mock* — we use $1 per token because the loader
    only cares about the percentage of supply, not the USD figure.
    The downstream signal scales by pct_of_supply alone.

    pct_of_supply uses a SIMPLE growing-supply model: assume 100%
    of the prior unlock has been absorbed into circulation.
    """
    out: list[tuple[date, str, float, float]] = []
    cadence = schedule["cadence"]
    cur = schedule["first_unlock"]
    stop = min(schedule["stop_after"], end_date)
    circ = float(schedule["circulating_at_first_unlock"])
    tokens = float(schedule["tokens_per_event"])
    while cur <= stop:
        if cadence == "monthly_eom":
            event_date = _eom(cur.year, cur.month)
        elif cadence == "monthly_day":
            d = schedule["day_of_month"]
            last = calendar.monthrange(cur.year, cur.month)[1]
            event_date = date(cur.year, cur.month, min(d, last))
        else:
            raise ValueError(f"unknown cadence {cadence!r}")
        pct = tokens / circ if circ > 0 else 0.0
        # Mock USD: tokens_per_event × $1 (loader doesn't use it)
        usd_estimate = tokens * 1.0
        out.append((event_date, schedule["symbol"], usd_estimate, pct))
        circ += tokens   # next unlock dilutes against the new supply
        cur = _next_month(cur)
    return out


def project_all(schedules: list[dict], end_date: date
                ) -> list[tuple[date, str, float, float]]:
    rows: list[tuple[date, str, float, float]] = []
    for s in schedules:
        rows.extend(project_schedule(s, end_date))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def to_csv(rows: list[tuple[date, str, float, float]]) -> str:
    lines = ["Date,Symbol,UnlockUSD,UnlockPctOfSupply"]
    for d, sym, usd, pct in rows:
        lines.append(f"{d.isoformat()},{sym},{usd:.2f},{pct:.6f}")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[tuple[date, str, float, float]], out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(to_csv(rows))
    os.replace(tmp, out_path)


# ─── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output path (default: apex_token_unlocks.csv)")
    ap.add_argument("--end-date", default=DEFAULT_END_DATE,
                    help="project unlocks up to this date (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write; just print the first 20 events")
    args = ap.parse_args(argv)

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    rows = project_all(KNOWN_UNLOCK_SCHEDULES, end_date)
    print(f"[fetch_token_unlocks] projected {len(rows)} unlock events "
          f"({rows[0][0]} → {rows[-1][0]})")
    if args.dry_run:
        for d, sym, usd, pct in rows[:20]:
            print(f"  {d}  {sym:5s}  pct={pct:.4%}  ~${usd/1e6:.0f}M tokens")
        return 0
    write_csv(rows, args.out)
    print(f"[fetch_token_unlocks] wrote {args.out} "
          f"({os.path.getsize(args.out):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
