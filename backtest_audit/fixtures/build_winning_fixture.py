"""Generate a synthetic 'winning paper trade' QC log fixture.

The MG36 fixture covers the LOSING case (the live evidence that anchored
the audit harness). We also need a winning fixture to verify the harness
correctly handles positive-edge live data.

This script writes:
    backtest_audit/fixtures/synthetic_winner_paper.txt

It's a synthetic but realistic QC log: round-trips that close in profit,
a few small slippage warnings (to verify the parser still picks them up),
and a final equity that's > start.

Usage:
    python3 backtest_audit/fixtures/build_winning_fixture.py

This is a one-shot generator; the resulting file is committed alongside
the MG36 fixture as a permanent test asset.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "synthetic_winner_paper.txt")


def render_log() -> str:
    """Build a fully-realistic synthetic QC paper-trade log with winning trades."""
    rnd = random.Random(2026)
    lines: list[str] = []
    base_t = datetime(2026, 5, 1, 12, 0, 0)

    def L(t: datetime, msg: str) -> None:
        lines.append(f"{t.strftime('%Y-%m-%d %H:%M:%S')} {msg}")

    # ── Boilerplate header ────────────────────────────────────────────────
    lines.append("Algorithm Initialization: Paper Brokerage account base currency: USD")
    lines.append("Launching analysis for L-synthetic-winner-fixture with LEAN Engine v2.5")
    L(base_t, "Loaded persisted state: blacklist 0, trades W:0/L:0")
    L(base_t, "=== LIVE TRADING (PULSE) v8.0.0 ===")
    L(base_t, "Capital: $120.00 | Max pos: 6 | Size: 80%")
    L(base_t + timedelta(seconds=10), "Algorithm starting warm up...")
    L(base_t + timedelta(seconds=20), "Algorithm finished warming up.")

    # ── 8 round-trip trades, mostly winners ──────────────────────────────
    # Mix of 6 winners + 2 small losers to give WR ~75%, realistic
    plays = [
        ("BTCUSD",   50_000.0, 0.001, +0.025, 25, "TP",         5,  3),
        ("ETHUSD",   3_200.0,  0.012, +0.018, 12, "TP",         8,  5),
        ("SOLUSD",   175.0,    0.20,  -0.012, 30, "STOP_LOSS",  18, 22),
        ("LINKUSD",  18.0,     2.0,   +0.045, 60, "TRAIL_STOP", 9,  6),
        ("AVAXUSD",  35.0,     1.0,   +0.020, 18, "TP",         11, 4),
        ("DOTUSD",   7.0,      5.0,   +0.012, 14, "TP",         6,  3),
        ("ADAUSD",   0.55,     65.0,  -0.018, 35, "STOP_LOSS",  12, 8),
        ("INJUSD",   22.0,     1.5,   +0.030, 22, "TP",         15, 7),
    ]

    cur_t = base_t + timedelta(minutes=5)
    oid = 1
    final_pnl_pct = 0.0
    n_wins = n_losses = 0

    for sym, entry_px, qty, ret, hold_min, exit_reason, slip_buy_bps, slip_sell_bps in plays:
        # ── SCALP ENTRY log ────────────────────────────────────────────
        score = round(rnd.uniform(0.55, 0.95), 2)
        components = (
            f"obi=0.20 vol=0.20 trend=0.20 adx=0.15 mean_rev=0.00 vwap=0.20"
        )
        L(cur_t,
          f"SCALP ENTRY: {sym} | score={score} | ${entry_px} | {components}")

        # ── Submit + fill ─────────────────────────────────────────────
        L(cur_t,
          f"ORDER: {sym} Submitted Buy qty={qty} price=0.0 id={oid}")
        # Fill happens 1 minute later in the synthetic log
        fill_t = cur_t + timedelta(minutes=1)
        L(fill_t,
          f"ORDER: {sym} Filled Buy qty={qty} price={entry_px} id={oid}")
        if slip_buy_bps > 5:
            slip_pct = slip_buy_bps / 100.0
            L(fill_t,
              f"⚠️ HIGH SLIPPAGE: {sym} | {slip_pct}% | dir=Buy")

        # ── Hold + exit ───────────────────────────────────────────────
        exit_t = fill_t + timedelta(minutes=hold_min)
        exit_px = entry_px * (1.0 + ret)
        oid += 1
        L(exit_t,
          f"{exit_reason}: {sym} | PnL:{ret*100:+.2f}% | Held:{hold_min/60:.1f}h")
        L(exit_t,
          f"ORDER: {sym} Submitted Sell qty=-{qty} price=0.0 id={oid}")
        sell_fill_t = exit_t + timedelta(minutes=1)
        L(sell_fill_t,
          f"ORDER: {sym} Filled Sell qty=-{qty} price={exit_px:.4f} id={oid}")
        if slip_sell_bps > 5:
            slip_pct = slip_sell_bps / 100.0
            L(sell_fill_t,
              f"⚠️ HIGH SLIPPAGE: {sym} | {slip_pct}% | dir=Sell")

        oid += 1
        if ret > 0:
            n_wins += 1
        else:
            n_losses += 1
        final_pnl_pct += ret

        cur_t = sell_fill_t + timedelta(minutes=10)

    # ── Final stats snapshot ───────────────────────────────────────────────
    total = n_wins + n_losses
    wr = (n_wins / total) * 100 if total else 0.0
    final_equity = 120.0 * (1 + final_pnl_pct)
    L(cur_t, f"=== FINAL ===")
    L(cur_t, f"Trades: {total} | WR: {wr:.1f}%")
    L(cur_t, f"Final: ${final_equity:.2f}")
    L(cur_t, f"PnL: {final_pnl_pct*100:+.2f}%")
    lines.append("Algorithm Liquidated")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    log = render_log()
    with open(OUT, "w") as f:
        f.write(log)
    print(f"Wrote {OUT}: {len(log)} bytes, {log.count(chr(10))} lines")
