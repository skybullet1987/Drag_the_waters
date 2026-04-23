import argparse
from collections import deque

import pandas as pd


ENTRY_PREFIX = "entry"
EPSILON = 1e-12


def _is_entry_tag(tag):
    return str(tag or "").strip().lower().startswith(ENTRY_PREFIX)


def _fee_bps_for_row(row, fee_bps, maker_bps):
    tag = str(row.get("Tag", "")).lower()
    if "post" in tag or "maker" in tag:
        return maker_bps
    return fee_bps


def _calculate_gross_pnl(entry_price, exit_price, qty, lot_side):
    if lot_side > 0:
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def load_orders(path):
    df = pd.read_csv(path)
    required = ["Time", "Symbol", "Price", "Quantity", "Type", "Status", "Value", "Tag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=True)
    for col in ("Price", "Quantity", "Value"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def analyze_orders(df, *, starting_equity=5000.0, fee_bps=26.0, maker_bps=16.0):
    total_rows = len(df)
    status = df["Status"].astype(str).str.lower()
    filled = df[status.eq("filled")].copy()
    cancel_rate = float(status.str.contains("cancel").sum() / total_rows) if total_rows else 0.0
    invalid_rate = float(status.str.contains("invalid").sum() / total_rows) if total_rows else 0.0
    if filled.empty:
        return {
            "summary_rows": [("Total fills", 0), ("Cancel rate", 0.0), ("Invalid rate", 0.0)],
            "trades": pd.DataFrame(),
            "symbol_pnl": pd.Series(dtype=float),
            "exit_breakdown": pd.DataFrame(columns=["count", "net_pnl"]),
        }

    filled["abs_value"] = filled["Value"].abs()
    fallback = (filled["Price"].abs() * filled["Quantity"].abs()).fillna(0.0)
    filled.loc[filled["abs_value"] <= 0, "abs_value"] = fallback
    filled["fee_bps"] = filled.apply(lambda r: _fee_bps_for_row(r, fee_bps, maker_bps), axis=1)
    filled["fee"] = filled["abs_value"] * filled["fee_bps"] / 10000.0

    open_lots = {}
    trades = []
    for row in filled.itertuples(index=False):
        symbol = row.Symbol
        qty = float(row.Quantity or 0.0)
        if qty == 0:
            continue
        side = 1 if qty > 0 else -1
        lot_queue = open_lots.setdefault(symbol, deque())
        notional = float(abs(row.abs_value))
        per_unit_fee = float(row.fee) / max(abs(qty), EPSILON)
        is_entry = _is_entry_tag(row.Tag)
        if is_entry:
            lot_queue.append(
                {
                    "side": side,
                    "qty": abs(qty),
                    "price": float(row.Price),
                    "time": row.Time,
                    "fee_per_unit": per_unit_fee,
                    "entry_tag": row.Tag,
                }
            )
            continue

        remaining = abs(qty)
        while remaining > EPSILON and lot_queue:
            lot = lot_queue[0]
            if lot["side"] != -side:
                break
            matched = min(remaining, lot["qty"])
            entry_price = float(lot["price"])
            exit_price = float(row.Price)
            gross = _calculate_gross_pnl(entry_price, exit_price, matched, lot["side"])
            fee = matched * (lot["fee_per_unit"] + per_unit_fee)
            net = gross - fee
            base_notional = max(abs(entry_price * matched), EPSILON)
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": lot["time"],
                    "exit_time": row.Time,
                    "entry_tag": lot["entry_tag"],
                    "exit_tag": row.Tag,
                    "qty": matched,
                    "gross_pnl": gross,
                    "fees": fee,
                    "net_pnl": net,
                    "holding_hours": float((row.Time - lot["time"]).total_seconds() / 3600.0) if pd.notna(row.Time) and pd.notna(lot["time"]) else 0.0,
                    "return_pct": net / base_notional,
                }
            )
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= EPSILON:
                lot_queue.popleft()

    trades_df = pd.DataFrame(trades)
    total_fees = float(filled["fee"].sum())
    symbol_pnl = trades_df.groupby("symbol")["net_pnl"].sum() if not trades_df.empty else pd.Series(dtype=float)
    exit_breakdown = (
        trades_df.groupby("exit_tag")["net_pnl"].agg(["count", "sum"]).rename(columns={"sum": "net_pnl"}).sort_values("count", ascending=False)
        if not trades_df.empty
        else pd.DataFrame(columns=["count", "net_pnl"])
    )

    if trades_df.empty:
        wins = pd.Series(dtype=float)
        losses = pd.Series(dtype=float)
    else:
        wins = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"]
        losses = -trades_df.loc[trades_df["return_pct"] < 0, "return_pct"]
    win_rate = float(len(wins) / len(trades_df)) if len(trades_df) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    required_wr = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 0.0
    edge_gap = win_rate - required_wr
    turnover = float(filled["abs_value"].sum() / max(starting_equity, EPSILON))
    min_t, max_t = filled["Time"].min(), filled["Time"].max()
    days = max(((max_t - min_t).total_seconds() / 86400.0), 0.0) + 1.0 if pd.notna(min_t) and pd.notna(max_t) else 1.0
    trades_per_day = float(len(trades_df) / days)
    summary_rows = [
        ("Total fills", int(len(filled))),
        ("Cancel rate", cancel_rate),
        ("Invalid rate", invalid_rate),
        ("Total round-trip trades", int(len(trades_df))),
        ("Win rate", win_rate),
        ("Avg win %", avg_win),
        ("Avg loss %", avg_loss),
        ("Profit/Loss ratio", pl_ratio),
        ("Expectancy (R)", expectancy),
        ("Required win rate to break even", required_wr),
        ("Edge gap", edge_gap),
        ("Total fees $", total_fees),
        ("Fees as % of starting equity", total_fees / max(starting_equity, EPSILON)),
        ("Turnover", turnover),
        ("Trades/day", trades_per_day),
    ]
    return {
        "summary_rows": summary_rows,
        "trades": trades_df,
        "symbol_pnl": symbol_pnl,
        "exit_breakdown": exit_breakdown,
        "filled": filled,
    }


def _print_summary(report):
    table = pd.DataFrame(report["summary_rows"], columns=["Metric", "Value"])
    print(table.to_string(index=False))
    pnl = report["symbol_pnl"]
    print("\nTop 5 symbols by net loss:")
    print(pnl.sort_values().head(5).to_string() if len(pnl) else "None")
    print("\nTop 5 symbols by net profit:")
    print(pnl.sort_values(ascending=False).head(5).to_string() if len(pnl) else "None")
    print("\nExit-tag breakdown:")
    if len(report["exit_breakdown"]):
        print(report["exit_breakdown"].to_string())
    else:
        print("None")


def _print_group_by(report, group_by):
    trades = report["trades"]
    if trades.empty or group_by == "none":
        return
    if group_by == "symbol":
        grouped = trades.groupby("symbol").agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), win_rate=("net_pnl", lambda s: (s > 0).mean()))
        print("\nPer-symbol breakdown:")
        print(grouped.sort_values("net_pnl").to_string())
    elif group_by == "tag":
        grouped = trades.groupby("entry_tag").agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), win_rate=("net_pnl", lambda s: (s > 0).mean()))
        print("\nPer-entry-tag breakdown:")
        print(grouped.sort_values("net_pnl").to_string())


def main():
    parser = argparse.ArgumentParser(description="QuantConnect-style backtest order diagnostics")
    parser.add_argument("orders_csv")
    parser.add_argument("--starting-equity", type=float, default=5000.0)
    parser.add_argument("--fee-bps", type=float, default=26.0)
    parser.add_argument("--maker-bps", type=float, default=16.0)
    parser.add_argument("--group-by", choices=["symbol", "tag", "none"], default="none")
    args = parser.parse_args()
    report = analyze_orders(
        load_orders(args.orders_csv),
        starting_equity=args.starting_equity,
        fee_bps=args.fee_bps,
        maker_bps=args.maker_bps,
    )
    _print_summary(report)
    _print_group_by(report, args.group_by)


if __name__ == "__main__":
    main()
