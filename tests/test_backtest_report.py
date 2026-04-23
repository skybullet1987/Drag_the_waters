import pandas as pd

from backtest_report import analyze_orders, load_orders


def test_backtest_report_roundtrip_partial_and_fees(tmp_path):
    rows = [
        {"Time": "2025-01-01T00:00:00Z", "Symbol": "XBTUSD", "Price": 100, "Quantity": 2, "Type": "Limit", "Status": "Filled", "Value": 200, "Tag": "Entry"},
        {"Time": "2025-01-01T01:00:00Z", "Symbol": "XBTUSD", "Price": 110, "Quantity": -1, "Type": "Limit", "Status": "Filled", "Value": -110, "Tag": "ATR Trail"},
        {"Time": "2025-01-01T02:00:00Z", "Symbol": "XBTUSD", "Price": 105, "Quantity": -1, "Type": "Limit", "Status": "Filled", "Value": -105, "Tag": "Time Stop"},
        {"Time": "2025-01-01T03:00:00Z", "Symbol": "ETHUSD", "Price": 50, "Quantity": 1, "Type": "Limit", "Status": "Canceled", "Value": 50, "Tag": "Entry"},
        {"Time": "2025-01-01T04:00:00Z", "Symbol": "ETHUSD", "Price": 50, "Quantity": 1, "Type": "Limit", "Status": "Invalid", "Value": 50, "Tag": "Entry"},
    ]
    path = tmp_path / "orders.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = load_orders(path)
    report = analyze_orders(df, starting_equity=5000, fee_bps=10, maker_bps=10)

    trades = report["trades"]
    assert len(trades) == 2  # one entry, one full + one partial close
    expected_fees = (200 + 110 + 105) * 10 / 10000.0
    assert abs(report["filled"]["fee"].sum() - expected_fees) < 1e-9

    metrics = dict(report["summary_rows"])
    assert metrics["Cancel rate"] > 0
    assert metrics["Invalid rate"] > 0

    exit_sum = report["exit_breakdown"]["net_pnl"].sum()
    assert abs(exit_sum - trades["net_pnl"].sum()) < 1e-9
