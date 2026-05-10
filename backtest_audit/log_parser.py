"""log_parser — parse QuantConnect algorithm-log_*.txt files into structured records.

Supports both QC paper-trading logs and QC backtest logs. The MG36 paper-trade
log fixture (`fixtures/mg36_paper_2026-03-16.txt`) is the reference target
for the parser's test suite.

Records produced:
- ScalpEntry(ts, symbol, score, components, price)
- OrderEvent(ts, symbol, status, side, qty, price, order_id)
- ExitEvent(ts, symbol, reason, pnl_pct, held_hours)
- SlippageWarning(ts, symbol, slippage_pct, direction)
- Snapshot(ts, trades_total, win_rate, equity, pnl_pct, ...)
- MakerLimit(ts, symbol, qty, bid, limit, timeout_s)

The parser is line-oriented and tolerant of unrecognized lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Iterator, Iterable


# ─── Record types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScalpEntry:
    ts: datetime
    symbol: str
    score: float
    price: float
    components: dict   # {"obi": 0.20, "vol": 0.20, ...}


@dataclass(frozen=True)
class OrderEvent:
    ts: datetime
    symbol: str
    status: str         # Submitted | Filled | Canceled | CancelPending | Invalid | PartiallyFilled
    side: str           # Buy | Sell
    qty: float
    price: float        # 0.0 for non-fill events
    order_id: str


@dataclass(frozen=True)
class ExitEvent:
    ts: datetime
    symbol: str
    reason: str         # 'ATR Trail', 'Take Profit', 'Stop Loss', etc.
    pnl_pct: float
    held_hours: float


@dataclass(frozen=True)
class SlippageWarning:
    ts: datetime
    symbol: str
    slippage_pct: float   # decimal, e.g. 0.0107 for 1.07%
    direction: str        # Buy | Sell


@dataclass(frozen=True)
class MakerLimit:
    ts: datetime
    symbol: str
    qty: float
    bid: float
    limit: float
    timeout_s: int


@dataclass(frozen=True)
class Snapshot:
    ts: datetime
    trades_total: int
    win_rate_pct: float       # 0-100
    equity: float | None      # None if not in same line
    pnl_pct: float | None
    avg_pct: float | None     # avg per-trade %


@dataclass
class ParsedLog:
    """Full structured view of one algorithm-log file."""
    entries:       list[ScalpEntry]      = field(default_factory=list)
    orders:        list[OrderEvent]      = field(default_factory=list)
    exits:         list[ExitEvent]       = field(default_factory=list)
    slippage:      list[SlippageWarning] = field(default_factory=list)
    maker_limits:  list[MakerLimit]      = field(default_factory=list)
    snapshots:     list[Snapshot]        = field(default_factory=list)
    raw_line_count: int                  = 0
    parse_errors:  list[str]             = field(default_factory=list)

    def summary(self) -> dict:
        """High-level summary suitable for printing or comparison."""
        last_snap = self.snapshots[-1] if self.snapshots else None
        first_ts = (self.entries + self.orders + self.exits +
                    self.slippage + self.snapshots)
        first_ts = min((r.ts for r in first_ts), default=None)
        last_ts = max(
            (r.ts for r in self.entries + self.orders + self.exits +
                          self.slippage + self.snapshots),
            default=None,
        )
        # Per-fill mean slippage
        fill_slip = [s.slippage_pct for s in self.slippage]
        mean_slip_bps = (sum(fill_slip) / len(fill_slip) * 10_000) if fill_slip else 0.0
        max_slip_bps  = (max(fill_slip) * 10_000) if fill_slip else 0.0
        # Order outcomes
        filled = sum(1 for o in self.orders if o.status == "Filled")
        canceled = sum(1 for o in self.orders if o.status == "Canceled")
        invalid = sum(1 for o in self.orders if o.status == "Invalid")
        partial = sum(1 for o in self.orders if o.status == "PartiallyFilled")
        return {
            "first_ts": first_ts.isoformat() if first_ts else None,
            "last_ts":  last_ts.isoformat() if last_ts else None,
            "raw_line_count":    self.raw_line_count,
            "parse_error_count": len(self.parse_errors),
            "scalp_entries":     len(self.entries),
            "orders_total":      len(self.orders),
            "orders_filled":     filled,
            "orders_canceled":   canceled,
            "orders_invalid":    invalid,
            "orders_partial":    partial,
            "exits":             len(self.exits),
            "slippage_warnings": len(self.slippage),
            "mean_slippage_bps": round(mean_slip_bps, 2),
            "max_slippage_bps":  round(max_slip_bps, 2),
            "maker_limits":      len(self.maker_limits),
            "snapshots":         len(self.snapshots),
            "final_trades":   last_snap.trades_total if last_snap else None,
            "final_win_rate": last_snap.win_rate_pct if last_snap else None,
            "final_equity":   last_snap.equity if last_snap else None,
            "final_pnl_pct":  last_snap.pnl_pct if last_snap else None,
        }


# ─── Regex patterns (compiled once) ──────────────────────────────────────────

# QC log timestamp: YYYY-MM-DD HH:MM:SS at line start
_TS = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

# SCALP ENTRY: SYMBOLUSD | score=0.95 | $29.88 | obi=0.20 vol=0.20 trend=0.20 adx=0.15 mean_rev=0.00 vwap=0.20
RE_SCALP_ENTRY = re.compile(
    _TS + r" SCALP ENTRY: (?P<sym>\S+) \| score=(?P<score>[\d.]+) \| \$(?P<price>[\d.]+) "
    r"\| (?P<components>.+)$"
)

# ORDER: SYMBOL Status Side qty=N price=N id=N
RE_ORDER = re.compile(
    _TS + r" ORDER: (?P<sym>\S+) (?P<status>\w+) (?P<side>Buy|Sell) "
    r"qty=(?P<qty>-?[\d.]+) price=(?P<price>-?[\d.]+) id=(?P<oid>\S+)"
)

# ATR Trail: SYMBOL | PnL:-2.29% | Held:0.0h
# Also matches: 'Take Profit', 'Stop Loss', 'Time Stop', etc.
RE_EXIT = re.compile(
    _TS + r" (?P<reason>(?:ATR Trail|Take Profit|Stop Loss|Time Stop|Trail|EXIT_\w+|"
    r"Hard Kill|Trailing Stop)): (?P<sym>\S+) \| PnL:(?P<pnl>-?[\d.]+)% \| Held:(?P<held>[\d.]+)h"
)

# ⚠️ HIGH SLIPPAGE: SYMBOL | 1.0713% | dir=Buy
RE_SLIPPAGE = re.compile(
    _TS + r" ⚠️ HIGH SLIPPAGE: (?P<sym>\S+) \| (?P<slip>[\d.]+)% \| dir=(?P<dir>Buy|Sell)"
)

# MAKER LIMIT: SYMBOL | qty=N | bid=$N | limit=$N | timeout=30s
RE_MAKER_LIMIT = re.compile(
    _TS + r" MAKER LIMIT: (?P<sym>\S+) \| qty=(?P<qty>[\d.]+) "
    r"\| bid=\$(?P<bid>[\d.]+) \| limit=\$(?P<limit>[\d.]+) \| timeout=(?P<timeout>\d+)s"
)

# Trades: 219 | WR: 38.1%             (no avg, no pnl)
# Trades: 153 | WR: 37.9% | Avg: -0.02%
RE_TRADES = re.compile(
    _TS + r" Trades: (?P<n>\d+) \| WR: (?P<wr>[\d.]+)%"
    r"(?: \| Avg: (?P<avg>-?[\d.]+)%)?"
)

# Final: $119.14   AND   PnL: -5.24%   appear on separate lines
RE_FINAL_EQUITY = re.compile(_TS + r" Final: \$(?P<eq>[\d.]+)")
RE_PNL_LINE     = re.compile(_TS + r" PnL: (?P<pnl>-?[\d.]+)%")

# Components within SCALP ENTRY: "obi=0.20 vol=0.00 trend=0.00 adx=0.00 mean_rev=0.00 vwap=0.15"
RE_COMPONENT = re.compile(r"(\w+)=([\d.]+)")


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


# ─── Main parser ─────────────────────────────────────────────────────────────

def parse_log(text: str) -> ParsedLog:
    """Parse a QC algorithm log string into a ParsedLog."""
    out = ParsedLog()
    pending_snapshot: dict | None = None  # accumulates Trades + Final + PnL

    for line in text.splitlines():
        out.raw_line_count += 1

        # Try each pattern; first match wins. Order matters: most specific first.
        try:
            m = RE_SCALP_ENTRY.match(line)
            if m:
                comps = dict(
                    (k, float(v))
                    for k, v in RE_COMPONENT.findall(m.group("components"))
                )
                out.entries.append(ScalpEntry(
                    ts=_parse_ts(m.group("ts")),
                    symbol=m.group("sym"),
                    score=float(m.group("score")),
                    price=float(m.group("price")),
                    components=comps,
                ))
                continue

            m = RE_ORDER.match(line)
            if m:
                out.orders.append(OrderEvent(
                    ts=_parse_ts(m.group("ts")),
                    symbol=m.group("sym"),
                    status=m.group("status"),
                    side=m.group("side"),
                    qty=float(m.group("qty")),
                    price=float(m.group("price")),
                    order_id=m.group("oid"),
                ))
                continue

            m = RE_EXIT.match(line)
            if m:
                out.exits.append(ExitEvent(
                    ts=_parse_ts(m.group("ts")),
                    symbol=m.group("sym"),
                    reason=m.group("reason"),
                    pnl_pct=float(m.group("pnl")),
                    held_hours=float(m.group("held")),
                ))
                continue

            m = RE_SLIPPAGE.match(line)
            if m:
                out.slippage.append(SlippageWarning(
                    ts=_parse_ts(m.group("ts")),
                    symbol=m.group("sym"),
                    slippage_pct=float(m.group("slip")) / 100.0,
                    direction=m.group("dir"),
                ))
                continue

            m = RE_MAKER_LIMIT.match(line)
            if m:
                out.maker_limits.append(MakerLimit(
                    ts=_parse_ts(m.group("ts")),
                    symbol=m.group("sym"),
                    qty=float(m.group("qty")),
                    bid=float(m.group("bid")),
                    limit=float(m.group("limit")),
                    timeout_s=int(m.group("timeout")),
                ))
                continue

            m = RE_TRADES.match(line)
            if m:
                # Buffer; equity/pnl may arrive on the next 1-2 lines
                pending_snapshot = {
                    "ts": _parse_ts(m.group("ts")),
                    "trades_total": int(m.group("n")),
                    "win_rate_pct": float(m.group("wr")),
                    "avg_pct": float(m.group("avg")) if m.group("avg") else None,
                    "equity": None,
                    "pnl_pct": None,
                }
                continue

            m = RE_FINAL_EQUITY.match(line)
            if m and pending_snapshot is not None:
                pending_snapshot["equity"] = float(m.group("eq"))
                continue

            m = RE_PNL_LINE.match(line)
            if m and pending_snapshot is not None:
                pending_snapshot["pnl_pct"] = float(m.group("pnl"))
                # Now flush snapshot
                out.snapshots.append(Snapshot(
                    ts=pending_snapshot["ts"],
                    trades_total=pending_snapshot["trades_total"],
                    win_rate_pct=pending_snapshot["win_rate_pct"],
                    equity=pending_snapshot["equity"],
                    pnl_pct=pending_snapshot["pnl_pct"],
                    avg_pct=pending_snapshot["avg_pct"],
                ))
                pending_snapshot = None
                continue

        except Exception as exc:
            out.parse_errors.append(f"line {out.raw_line_count}: {exc}")

    # Flush a snapshot that didn't get a PnL line
    if pending_snapshot is not None:
        out.snapshots.append(Snapshot(
            ts=pending_snapshot["ts"],
            trades_total=pending_snapshot["trades_total"],
            win_rate_pct=pending_snapshot["win_rate_pct"],
            equity=pending_snapshot["equity"],
            pnl_pct=pending_snapshot["pnl_pct"],
            avg_pct=pending_snapshot["avg_pct"],
        ))

    return out


def parse_log_file(path: str) -> ParsedLog:
    """Convenience: read a file and parse its contents."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_log(f.read())


# ─── Trade pairing (entry → exit) ────────────────────────────────────────────

@dataclass(frozen=True)
class CompletedTrade:
    """Round-trip trade reconstructed from order events."""
    symbol: str
    entry_ts: datetime
    entry_price: float
    entry_qty: float
    entry_oid: str
    exit_ts: datetime
    exit_price: float
    exit_oid: str
    gross_pct: float                    # (exit - entry) / entry, no fees
    held_seconds: float
    entry_slippage_bps: float | None    # if a SlippageWarning matched within ±60s
    exit_slippage_bps: float | None
    score: float | None                 # from preceding ScalpEntry
    score_components: dict              # from preceding ScalpEntry


def pair_trades(parsed: ParsedLog) -> list[CompletedTrade]:
    """Reconstruct round-trip trades from filled orders.

    Strategy:
    - Walk filled orders in time order.
    - First Filled Buy on a symbol = entry; next Filled Sell on same symbol = exit.
    - Match SlippageWarning records by symbol + ts within ±60 seconds of fill.
    - Match the ScalpEntry preceding the entry buy by symbol within last 5 minutes.
    """
    fills = [o for o in parsed.orders if o.status == "Filled"]
    fills.sort(key=lambda o: o.ts)

    completed: list[CompletedTrade] = []
    open_buys: dict[str, OrderEvent] = {}

    for o in fills:
        if o.side == "Buy":
            open_buys[o.symbol] = o
        elif o.side == "Sell":
            entry = open_buys.pop(o.symbol, None)
            if not entry or entry.price <= 0 or o.price <= 0:
                continue

            held = (o.ts - entry.ts).total_seconds()
            gross = (o.price - entry.price) / entry.price

            # Match slippage warnings within ±60s
            entry_slip = _match_slippage(parsed.slippage, o.symbol, entry.ts, "Buy")
            exit_slip  = _match_slippage(parsed.slippage, o.symbol, o.ts, "Sell")

            # Match preceding ScalpEntry within 5 min
            score, comps = _match_scalp_entry(parsed.entries, o.symbol, entry.ts)

            completed.append(CompletedTrade(
                symbol=o.symbol,
                entry_ts=entry.ts,
                entry_price=entry.price,
                entry_qty=abs(entry.qty),
                entry_oid=entry.order_id,
                exit_ts=o.ts,
                exit_price=o.price,
                exit_oid=o.order_id,
                gross_pct=gross,
                held_seconds=held,
                entry_slippage_bps=entry_slip,
                exit_slippage_bps=exit_slip,
                score=score,
                score_components=comps,
            ))

    return completed


def _match_slippage(warnings, symbol, ts, direction, window_s=60):
    """Find a SlippageWarning for this symbol/direction within ±window_s of ts."""
    for w in warnings:
        if w.symbol != symbol or w.direction != direction:
            continue
        if abs((w.ts - ts).total_seconds()) <= window_s:
            return round(w.slippage_pct * 10_000, 2)
    return None


def _match_scalp_entry(entries, symbol, entry_ts, window_s=300):
    """Find the most recent ScalpEntry for this symbol within window before entry_ts."""
    candidates = [
        e for e in entries
        if e.symbol == symbol and 0 <= (entry_ts - e.ts).total_seconds() <= window_s
    ]
    if not candidates:
        return None, {}
    best = max(candidates, key=lambda e: e.ts)
    return best.score, best.components
