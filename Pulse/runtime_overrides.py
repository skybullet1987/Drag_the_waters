"""runtime_overrides — per-backtest parameter overrides.

This default file ships with empty OVERRIDES so Pulse compiles cleanly
in QC even when no sweep is running.

The Phase 3 sweep runner (`backtest_audit/qc_sweep_runner.py`) replaces
this file in the QC project before each backtest with one containing the
sweep parameters + window dates, e.g.:

    OVERRIDES = {
        "SCALP_ENTRY_THRESHOLD": 0.55,
        "QUICK_TAKE_PROFIT_PCT": 0.12,
        "PULSE_OVERRIDE_START_YEAR": 2024,
        "use_harsh_sim": True,
    }

`Pulse/main.py._apply_runtime_overrides()` reads this dict and routes
each key either into the `config` module or into a special-keys store
that `_param()` consults at parameter-read time.

Empty default: no overrides applied → defaults from Pulse/config.py win.
"""

OVERRIDES: dict = {}
