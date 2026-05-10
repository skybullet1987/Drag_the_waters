"""qc_orders — convert QC backtest order JSON into CompletedTrade records.

QC's `/backtests/orders/read` endpoint returns a list of order dicts with
fields like:
    {
        "id": 12345,
        "symbol": {"value": "BTCUSD", ...},
        "type": "Market",
        "status": "Filled",
        "direction": 0,        # 0=Buy, 1=Sell
        "quantity": 0.001,
        "price": 50000.0,
        "time": "2025-03-15T12:34:00Z",
        "tag": "ENTRY",
        ...
    }

This module:
  1. Parses raw QC order dicts into a typed `OrderEvent`-like form
  2. Pairs round-trip trades (buy → sell on same symbol)
  3. Returns a list of `CompletedTrade` records compatible with
     ``backtest_audit.compare.build_report``

So the full flow is:
    qc.read_backtest_orders(pid, bid)  →  parse_qc_orders()
        →  pair_qc_trades()  →  build_report(live=..., backtest=these_trades)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from backtest_audit.log_parser import CompletedTrade


# ─── Symbol extraction ──────────────────────────────────────────────────────

def _extract_symbol(raw_sym) -> str:
    """QC's symbol field is sometimes a dict, sometimes a string, sometimes
    has a nested ``{value: 'BTCUSD'}`` form."""
    if isinstance(raw_sym, dict):
        return (
            raw_sym.get("value")
            or raw_sym.get("Value")
            or raw_sym.get("ID", {}).get("Symbol")
            or str(raw_sym)
        )
    return str(raw_sym)


def _extract_direction(raw_dir) -> str:
    """QC direction can be int (0=Buy, 1=Sell) or string."""
    if isinstance(raw_dir, int):
        return "Buy" if raw_dir == 0 else "Sell"
    s = str(raw_dir).lower()
    if "buy" in s:
        return "Buy"
    if "sell" in s:
        return "Sell"
    return "Unknown"


def _extract_time(raw_ts) -> datetime:
    """QC time is ISO-8601 UTC string."""
    if isinstance(raw_ts, datetime):
        return raw_ts
    s = str(raw_ts).rstrip("Z")
    # Handle fractional seconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unparseable timestamp: {raw_ts!r}")


def _extract_status(raw_status) -> str:
    """QC status can be int enum or string."""
    if isinstance(raw_status, int):
        return {
            0: "New", 1: "Submitted", 2: "PartiallyFilled",
            3: "Filled", 4: "Canceled", 5: "Invalid",
            6: "None", 7: "CancelPending",
        }.get(raw_status, str(raw_status))
    return str(raw_status).split(".")[-1]


# ─── Parsed order DTO ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class QCOrder:
    """One QC backtest order, normalized."""
    order_id:   str
    symbol:     str
    status:     str
    direction:  str
    quantity:   float
    price:      float
    time:       datetime
    tag:        str = ""


def parse_qc_orders(raw_orders: Iterable[dict]) -> list[QCOrder]:
    """Convert QC's order JSON list into typed QCOrder records.

    Filters out non-fill events (only Filled / PartiallyFilled retained).
    """
    out: list[QCOrder] = []
    for o in raw_orders:
        try:
            status = _extract_status(o.get("status") or o.get("Status"))
            if status not in ("Filled", "PartiallyFilled"):
                continue   # Pairing only cares about fills
            out.append(QCOrder(
                order_id=str(o.get("id") or o.get("Id") or o.get("orderId")),
                symbol=_extract_symbol(o.get("symbol") or o.get("Symbol")),
                status=status,
                direction=_extract_direction(
                    o.get("direction", o.get("Direction"))
                ),
                quantity=abs(float(
                    o.get("quantity") or o.get("Quantity") or 0
                )),
                price=float(o.get("price") or o.get("Price") or 0),
                time=_extract_time(o.get("time") or o.get("Time")),
                tag=str(o.get("tag") or o.get("Tag") or ""),
            ))
        except Exception:
            # Skip malformed orders silently — audit prefers partial truth
            # over crashing on one bad row
            continue
    return out


# ─── Pair into round-trip trades ────────────────────────────────────────────

def pair_qc_trades(orders: Iterable[QCOrder]) -> list[CompletedTrade]:
    """Pair filled buys with the next filled sell of the same symbol.

    Walks orders in time order; for each Buy fill on a symbol, the next Sell
    fill on the same symbol becomes its exit. Partial fills are aggregated
    (multiple partial buys at the same time → averaged entry price).

    Returns CompletedTrade records compatible with compare.build_report().
    """
    sorted_orders = sorted(orders, key=lambda o: (o.time, o.order_id))

    open_buys: dict[str, list[QCOrder]] = {}
    completed: list[CompletedTrade] = []

    for o in sorted_orders:
        if o.price <= 0:
            continue   # invalid fill
        if o.direction == "Buy":
            open_buys.setdefault(o.symbol, []).append(o)
        elif o.direction == "Sell":
            buys = open_buys.get(o.symbol, [])
            if not buys:
                continue   # short / unpaired — skip
            # Aggregate buys into a single entry (qty-weighted price)
            total_qty = sum(b.quantity for b in buys)
            if total_qty <= 0:
                open_buys.pop(o.symbol, None)
                continue
            avg_entry = sum(b.price * b.quantity for b in buys) / total_qty
            entry_ts  = min(b.time for b in buys)
            entry_oid = ",".join(b.order_id for b in buys)
            held_s    = (o.time - entry_ts).total_seconds()
            gross     = (o.price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
            completed.append(CompletedTrade(
                symbol=o.symbol,
                entry_ts=entry_ts,
                entry_price=avg_entry,
                entry_qty=total_qty,
                entry_oid=entry_oid,
                exit_ts=o.time,
                exit_price=o.price,
                exit_oid=o.order_id,
                gross_pct=gross,
                held_seconds=held_s,
                entry_slippage_bps=None,    # not available from order JSON
                exit_slippage_bps=None,
                score=None,
                score_components={},
            ))
            open_buys.pop(o.symbol, None)
    return completed


# ─── End-to-end convenience ─────────────────────────────────────────────────

def fetch_backtest_trades(client, project_id: int, backtest_id: str,
                          start: int = 0, page: int = 1000,
                          max_pages: int = 20) -> list[CompletedTrade]:
    """Fetch all backtest orders via QC API, parse, pair into CompletedTrade.

    Auto-paginates: keeps calling read_backtest_orders with bumped offset
    until the API returns < `page` orders or `max_pages` is hit.
    """
    all_orders: list[dict] = []
    offset = start
    for _ in range(max_pages):
        chunk = client.read_backtest_orders(
            project_id, backtest_id, start=offset, end=offset + page,
        )
        if not chunk:
            break
        all_orders.extend(chunk)
        if len(chunk) < page:
            break
        offset += page
    qc_orders = parse_qc_orders(all_orders)
    return pair_qc_trades(qc_orders)
