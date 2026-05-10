"""backtest_audit — measure the backtest-vs-live gap on QC strategies.

Built to ground every Pulse improvement in *measured* live behavior rather
than backtest fantasy.

Modules:
- log_parser.py     — parse QC algorithm-log_*.txt → trade records
- compare.py        — trade-by-trade backtest-vs-live diff + report
- harsh_simulator.py — pessimistic QC simulator overrides (slippage, taker, T+1)
- regime_runner.py  — multi-window walk-forward + survivability score
- qc_api.py         — thin wrapper over QC REST (auth, list/read backtests)

Phase 0a + 0 of PLAN.md.
"""

__version__ = "0.0.1"
