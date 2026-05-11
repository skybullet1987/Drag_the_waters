"""runtime_overrides — DIAGNOSTIC CONFIG (Step 2 of edge-discovery plan).

Purpose: prove or disprove per-trade edge of the scalp engine.

Previous backtest (Pulse v1, $100, Jan-Jul 2025) produced 22 trades and
-0.81% net. 77% of exits were TIME_STOP — the signal had no follow-through
at the 3-hour horizon. Trade frequency was too low to measure raw edge.

This override loosens entry gates to FORCE a trade frequency of ~50-100/month.
We then look at one number: avg net % per trade after fees.

  Positive (even +0.05%) → signal has edge; turn gates back on, scale capital.
  Zero or negative      → kill scalp engine, pivot to trend/MR engines.

Knobs flipped vs default (Pulse/config.py):
  SCALP_ENTRY_THRESHOLD       0.55 → 0.35   (let weaker setups through)
  SCALP_HIGH_CONVICTION_THRES 0.70 → 0.55   (more conviction trades)
  TIME_STOP_HOURS             3.0  → 1.5    (faster turnover, tighter horizon)
  FG_BLOCK_ABOVE              90   → 999    (disable F&G kill switch)
  FUNDING_BLOCK_ABOVE         0.001 → 99.0  (disable funding kill switch)
  MAX_POSITIONS               6    → 8      (more concurrent slots, more samples)

Safety rails kept ON:
  PER_TRADE_HARD_KILL_PCT     0.08          (per-trade -8% absolute stop)
  MAX_DRAWDOWN_TRIP_PCT       0.20          (DD breaker still arms at -20%)
  MAX_DRAWDOWN_HALT_PCT       0.25          (full halt at -25%)
  Min-qty validation per trade

After this backtest finishes, restore the original defaults by overwriting
this file with `OVERRIDES: dict = {}`.
"""

OVERRIDES: dict = {
    # Loosen entry thresholds
    "SCALP_ENTRY_THRESHOLD":       0.35,
    "SCALP_HIGH_CONVICTION_THRES": 0.55,

    # Disable macro kill switches (testing raw signal alone)
    "FG_BLOCK_ABOVE":              999.0,
    "FUNDING_BLOCK_ABOVE":         99.0,

    # Tighter horizon — give more samples per unit time
    "TIME_STOP_HOURS":             1.5,

    # More concurrent positions for sample size
    "MAX_POSITIONS":               8,

    # Backtest window — full year
    "PULSE_OVERRIDE_START_YEAR":   2025,
    "PULSE_OVERRIDE_START_MONTH":  1,
    "PULSE_OVERRIDE_START_DAY":    1,
    "PULSE_OVERRIDE_END_YEAR":     2025,
    "PULSE_OVERRIDE_END_MONTH":    7,
    "PULSE_OVERRIDE_END_DAY":      31,

    # Keep $100 — testing edge per trade, not capital scaling
    "initial_cash":                100.0,
}
