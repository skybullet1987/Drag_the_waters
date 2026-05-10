"""compare — backtest-vs-live trade-by-trade gap attribution.

Given two streams of completed trades (one from a live deployment, one from a
backtest of the same period), produce:
- Per-trade match table (live ↔ backtest pairs by symbol + entry_ts proximity)
- Aggregate gap statistics: fill-price diff in bps, count diffs, etc.
- Gap attribution: how much of the live underperformance is explained by
  slippage, missed signals, fee model, latency, etc.

Inputs
------
Both `live_trades` and `backtest_trades` must be lists of CompletedTrade
records (or dict-shaped equivalents). The CompletedTrade dataclass from
log_parser.py is the canonical shape:

    symbol, entry_ts, entry_price, entry_qty, entry_oid,
    exit_ts, exit_price, exit_oid, gross_pct, held_seconds,
    entry_slippage_bps, exit_slippage_bps, score, score_components

Output: GapReport (dataclass with summary() and to_dict() methods).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Iterable, Sequence


# ─── Trade pair representation ───────────────────────────────────────────────

@dataclass(frozen=True)
class TradePair:
    """One live trade matched against one backtest trade (or unmatched)."""
    symbol: str
    live_entry_ts: datetime | None
    bt_entry_ts: datetime | None
    live_entry_price: float | None
    bt_entry_price: float | None
    live_exit_price: float | None
    bt_exit_price: float | None
    live_gross_pct: float | None
    bt_gross_pct: float | None
    live_entry_slip_bps: float | None
    bt_entry_slip_bps: float | None
    live_exit_slip_bps: float | None
    bt_exit_slip_bps: float | None
    entry_price_gap_bps: float | None   # (live - bt) / bt × 10_000
    exit_price_gap_bps: float | None
    return_gap_pct: float | None        # live_gross_pct - bt_gross_pct
    matched: bool = True

    @property
    def status(self) -> str:
        if self.matched:
            return "matched"
        if self.live_entry_ts is not None and self.bt_entry_ts is None:
            return "live_only"   # live took it, backtest didn't
        if self.bt_entry_ts is not None and self.live_entry_ts is None:
            return "backtest_only"  # backtest took it, live didn't
        return "unknown"


# ─── Aggregate report ────────────────────────────────────────────────────────

@dataclass
class GapReport:
    """Aggregate backtest-vs-live gap report."""
    n_live: int = 0
    n_backtest: int = 0
    n_matched: int = 0
    n_live_only: int = 0
    n_backtest_only: int = 0

    # Fill-price gaps (positive = live paid more on buy / received less on sell)
    mean_entry_gap_bps: float = 0.0
    median_entry_gap_bps: float = 0.0
    p90_entry_gap_bps: float = 0.0
    p99_entry_gap_bps: float = 0.0
    mean_exit_gap_bps: float = 0.0
    median_exit_gap_bps: float = 0.0
    p90_exit_gap_bps: float = 0.0
    p99_exit_gap_bps: float = 0.0

    # Round-trip return gap
    mean_return_gap_pct: float = 0.0
    sum_return_gap_pct: float = 0.0

    # Slippage attribution (live)
    mean_live_entry_slip_bps: float = 0.0
    mean_live_exit_slip_bps: float = 0.0

    # Slippage attribution (backtest)
    mean_bt_entry_slip_bps: float = 0.0
    mean_bt_exit_slip_bps: float = 0.0

    # Per-trade pairs (for downstream rendering)
    pairs: list[TradePair] = field(default_factory=list)

    # Gap attribution: how much of the underperformance comes from each source
    attribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # TradePair list is verbose; provide a count and drop the body
        d["pairs_count"] = len(self.pairs)
        d.pop("pairs", None)
        # Round numeric fields
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d

    def summary_text(self) -> str:
        lines = [
            "─── Backtest-vs-Live Gap Report ───",
            f"  trade counts: live={self.n_live}  backtest={self.n_backtest}  "
            f"matched={self.n_matched}  live_only={self.n_live_only}  "
            f"bt_only={self.n_backtest_only}",
            "",
            "  Entry fill price gap (bps, +ve = live paid worse):",
            f"    mean={self.mean_entry_gap_bps:+.2f}  median={self.median_entry_gap_bps:+.2f}  "
            f"p90={self.p90_entry_gap_bps:+.2f}  p99={self.p99_entry_gap_bps:+.2f}",
            "  Exit fill price gap (bps, +ve = live got worse):",
            f"    mean={self.mean_exit_gap_bps:+.2f}  median={self.median_exit_gap_bps:+.2f}  "
            f"p90={self.p90_exit_gap_bps:+.2f}  p99={self.p99_exit_gap_bps:+.2f}",
            "",
            f"  Round-trip return gap: mean={self.mean_return_gap_pct:+.4f}%  "
            f"sum={self.sum_return_gap_pct:+.3f}%",
            "",
            f"  Live  mean slippage: entry={self.mean_live_entry_slip_bps:.1f}bp  "
            f"exit={self.mean_live_exit_slip_bps:.1f}bp",
            f"  BT    mean slippage: entry={self.mean_bt_entry_slip_bps:.1f}bp  "
            f"exit={self.mean_bt_exit_slip_bps:.1f}bp",
        ]
        if self.attribution:
            lines.append("")
            lines.append("  Gap attribution:")
            for src, pct in sorted(self.attribution.items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"    {src:<28} {pct:+.3f}%")
        return "\n".join(lines)


# ─── Pairing logic ───────────────────────────────────────────────────────────

def _to_dict(t):
    """Accept either a dataclass or dict-shaped trade record."""
    if hasattr(t, "__dataclass_fields__"):
        return asdict(t)
    return dict(t)


def pair_live_to_backtest(
    live: Sequence,
    backtest: Sequence,
    match_window_minutes: int = 30,
) -> tuple[list[TradePair], list, list]:
    """Match live trades to backtest trades by (symbol, entry_ts within window).

    Greedy pairing: for each live trade, find the closest unmatched backtest
    trade on the same symbol within ±match_window_minutes. If none, mark as
    live-only. Any backtest trades left unmatched are backtest-only.

    Returns (pairs, live_unmatched, bt_unmatched).
    """
    live_d = [_to_dict(t) for t in live]
    bt_d   = [_to_dict(t) for t in backtest]

    # Sort by entry_ts for determinism
    live_d.sort(key=lambda t: t["entry_ts"])
    bt_d.sort(key=lambda t: t["entry_ts"])
    bt_used: set[int] = set()

    pairs: list[TradePair] = []
    live_unmatched_idx: list[int] = []

    for li, lt in enumerate(live_d):
        # Find best backtest match
        best_idx = -1
        best_dt = timedelta(minutes=match_window_minutes + 1)
        for bi, bt in enumerate(bt_d):
            if bi in bt_used or bt["symbol"] != lt["symbol"]:
                continue
            dt = abs(bt["entry_ts"] - lt["entry_ts"])
            if dt < best_dt:
                best_dt = dt
                best_idx = bi
        if best_idx >= 0 and best_dt <= timedelta(minutes=match_window_minutes):
            bt = bt_d[best_idx]
            bt_used.add(best_idx)
            pairs.append(_make_pair(lt, bt))
        else:
            pairs.append(_make_pair(lt, None))
            live_unmatched_idx.append(li)

    bt_unmatched_idx = [bi for bi in range(len(bt_d)) if bi not in bt_used]
    for bi in bt_unmatched_idx:
        pairs.append(_make_pair(None, bt_d[bi]))

    return (pairs, live_unmatched_idx, bt_unmatched_idx)


def _make_pair(live: dict | None, bt: dict | None) -> TradePair:
    """Construct a TradePair from one or both sides."""
    sym = (live or bt or {}).get("symbol", "?")

    def g(d, k):
        return d.get(k) if d else None

    le_px = g(live, "entry_price")
    bt_px = g(bt,   "entry_price")
    lx_px = g(live, "exit_price")
    bx_px = g(bt,   "exit_price")

    entry_gap = (
        (le_px - bt_px) / bt_px * 10_000
        if (le_px is not None and bt_px is not None and bt_px != 0)
        else None
    )
    exit_gap = (
        (lx_px - bx_px) / bx_px * 10_000
        if (lx_px is not None and bx_px is not None and bx_px != 0)
        else None
    )

    le_ret = g(live, "gross_pct")
    bt_ret = g(bt,   "gross_pct")
    return_gap = (
        (le_ret - bt_ret) * 100
        if (le_ret is not None and bt_ret is not None)
        else None
    )

    return TradePair(
        symbol=sym,
        live_entry_ts=g(live, "entry_ts"),
        bt_entry_ts=g(bt, "entry_ts"),
        live_entry_price=le_px,
        bt_entry_price=bt_px,
        live_exit_price=lx_px,
        bt_exit_price=bx_px,
        live_gross_pct=le_ret,
        bt_gross_pct=bt_ret,
        live_entry_slip_bps=g(live, "entry_slippage_bps"),
        bt_entry_slip_bps=g(bt, "entry_slippage_bps"),
        live_exit_slip_bps=g(live, "exit_slippage_bps"),
        bt_exit_slip_bps=g(bt, "exit_slippage_bps"),
        entry_price_gap_bps=entry_gap,
        exit_price_gap_bps=exit_gap,
        return_gap_pct=return_gap,
        matched=(live is not None and bt is not None),
    )


# ─── Statistics ──────────────────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def build_report(
    live: Sequence,
    backtest: Sequence,
    match_window_minutes: int = 30,
) -> GapReport:
    """Top-level: pair, compute stats, attribute the gap."""
    pairs, live_unmatched, bt_unmatched = pair_live_to_backtest(
        live, backtest, match_window_minutes=match_window_minutes,
    )

    rep = GapReport()
    rep.n_live = len(live)
    rep.n_backtest = len(backtest)
    rep.pairs = pairs

    matched_pairs = [p for p in pairs if p.matched]
    rep.n_matched = len(matched_pairs)
    rep.n_live_only = len(live_unmatched)
    rep.n_backtest_only = len(bt_unmatched)

    # Per-pair stats
    entry_gaps = [p.entry_price_gap_bps for p in matched_pairs
                  if p.entry_price_gap_bps is not None]
    exit_gaps  = [p.exit_price_gap_bps for p in matched_pairs
                  if p.exit_price_gap_bps is not None]
    return_gaps = [p.return_gap_pct for p in matched_pairs
                   if p.return_gap_pct is not None]

    rep.mean_entry_gap_bps   = _mean(entry_gaps)
    rep.median_entry_gap_bps = _percentile(entry_gaps, 0.50)
    rep.p90_entry_gap_bps    = _percentile(entry_gaps, 0.90)
    rep.p99_entry_gap_bps    = _percentile(entry_gaps, 0.99)
    rep.mean_exit_gap_bps    = _mean(exit_gaps)
    rep.median_exit_gap_bps  = _percentile(exit_gaps, 0.50)
    rep.p90_exit_gap_bps     = _percentile(exit_gaps, 0.90)
    rep.p99_exit_gap_bps     = _percentile(exit_gaps, 0.99)
    rep.mean_return_gap_pct  = _mean(return_gaps)
    rep.sum_return_gap_pct   = sum(return_gaps)

    # Slippage means (where data available)
    le_slips = [p.live_entry_slip_bps for p in pairs
                if p.live_entry_slip_bps is not None]
    lx_slips = [p.live_exit_slip_bps for p in pairs
                if p.live_exit_slip_bps is not None]
    be_slips = [p.bt_entry_slip_bps for p in pairs
                if p.bt_entry_slip_bps is not None]
    bx_slips = [p.bt_exit_slip_bps for p in pairs
                if p.bt_exit_slip_bps is not None]

    rep.mean_live_entry_slip_bps = _mean(le_slips)
    rep.mean_live_exit_slip_bps  = _mean(lx_slips)
    rep.mean_bt_entry_slip_bps   = _mean(be_slips)
    rep.mean_bt_exit_slip_bps    = _mean(bx_slips)

    # Gap attribution
    rep.attribution = _attribute_gap(rep, pairs)
    return rep


def _attribute_gap(rep: GapReport, pairs: list[TradePair]) -> dict[str, float]:
    """Decompose the live-minus-backtest return gap into named sources.

    Heuristic attribution: each source is expressed in % of NAV (per trade).
    Components currently modeled:
      - slippage_excess: live paid more in fill slippage than backtest modeled
      - missed_trades:   live signals never fired (live_only with neg returns)
                         OR backtest signals not fired in live (bt_only positive
                         returns are an opportunity cost on the live side)
      - residual:        unexplained gap = total - (slippage + missed)
    """
    matched = [p for p in pairs if p.matched]

    # Total return gap, %
    total_gap = sum(
        p.return_gap_pct for p in matched if p.return_gap_pct is not None
    )

    # Slippage excess (live - backtest), in % of trade notional, summed
    slip_excess = 0.0
    for p in matched:
        l_e = p.live_entry_slip_bps or 0.0
        l_x = p.live_exit_slip_bps  or 0.0
        b_e = p.bt_entry_slip_bps   or 0.0
        b_x = p.bt_exit_slip_bps    or 0.0
        # Convert bps to % per leg; round-trip = entry + exit
        # Negative because slippage hurts returns
        slip_excess -= ((l_e + l_x) - (b_e + b_x)) / 100.0

    # Opportunity cost from backtest-only trades that would have been profitable
    bt_only = [p for p in pairs if not p.matched and p.bt_entry_ts is not None]
    missed_profit = sum(
        p.bt_gross_pct * 100 for p in bt_only
        if p.bt_gross_pct is not None and p.bt_gross_pct > 0
    )

    # Bad live-only trades the backtest wouldn't have taken
    live_only = [p for p in pairs if not p.matched and p.live_entry_ts is not None]
    bad_live_only = sum(
        p.live_gross_pct * 100 for p in live_only
        if p.live_gross_pct is not None and p.live_gross_pct < 0
    )

    residual = total_gap - slip_excess + missed_profit + bad_live_only

    return {
        "slippage_excess":      round(slip_excess, 3),
        "missed_profit_bt":     round(-missed_profit, 3),
        "bad_extra_live":       round(bad_live_only, 3),
        "residual_unexplained": round(residual, 3),
        "total_observed_gap":   round(total_gap, 3),
    }


# ─── Convenience wrappers for log-driven flow ────────────────────────────────

def report_from_paired_log(parsed_log) -> GapReport:
    """Treat the paired live trades as both 'live' and 'backtest=empty'.

    Useful sanity check when no backtest counterpart exists yet — the report
    still surfaces live slippage stats.
    """
    from backtest_audit.log_parser import pair_trades
    live_trades = pair_trades(parsed_log)
    return build_report(live_trades, [])
