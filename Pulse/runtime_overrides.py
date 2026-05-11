"""runtime_overrides — APEX SMOKE-TEST CONFIG.

This config:
  - Restores DEFAULT scalp thresholds (the diagnostic config is gone)
  - Enables the new Apex engine in fallback mode (no joblib yet)
  - Keeps the safety rails ON

Goal of this backtest: prove the Apex code compiles + runs in QC
without crashing, places at least a few trades, and the Pulse base
strategy continues to function correctly alongside it.

After this passes, the next steps are:
  Step A — Train apex_model_v1.joblib offline on real signal history
  Step B — Push the joblib to the QC project
  Step C — Re-run with apex_enabled=True + APEX_MODEL_REQUIRED=True
"""

OVERRIDES: dict = {
    # ── Apex engine ─────────────────────────────────────────────────
    "apex_enabled":            True,    # bootstrap & schedule the 4h tick
    # In smoke-test mode the joblib is absent → Apex runs in fallback
    # (rule-based) prob using APEX_FALLBACK_WEIGHTS.

    # ── Backtest window ─────────────────────────────────────────────
    "PULSE_OVERRIDE_START_YEAR":  2025,
    "PULSE_OVERRIDE_START_MONTH": 1,
    "PULSE_OVERRIDE_START_DAY":   1,
    "PULSE_OVERRIDE_END_YEAR":    2025,
    "PULSE_OVERRIDE_END_MONTH":   3,
    "PULSE_OVERRIDE_END_DAY":     31,    # 3-month window for fast iteration

    # ── Capital ─────────────────────────────────────────────────────
    "initial_cash":  100.0,
}
