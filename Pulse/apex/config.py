"""Pulse.apex.config — central tunables for the Apex edge engine.

All "magic numbers" live here so Phase 4 (ML) and Phase 7 (walk-forward
sweep) can override them via runtime_overrides without touching code.
"""

# ─── Versioning ──────────────────────────────────────────────────────────────
APEX_VERSION = "1.0.0"

# ─── Decision cadence ────────────────────────────────────────────────────────
# How often the Apex engine wakes up to score the universe + place trades.
# 4 hours is a deliberate retreat from the failed minute-bar scalping —
# medium-frequency edge sources don't need (and don't reward) high freq.
APEX_REBALANCE_HOURS = 4

# Minute-tick processing is still done for ATR-trail / per-trade kill;
# the 4h cadence is only for ENTRY decisions.
APEX_MINUTE_TICK_FOR_EXITS = True

# ─── Entry / exit thresholds ─────────────────────────────────────────────────
# ML model outputs a calibrated probability ∈ [0, 1] that the symbol's
# 24h forward return exceeds +1.5%. Above APEX_ENTRY_THRESHOLD we enter;
# once a position is open, we only liquidate (signal-flip exit) when the
# probability drops below APEX_EXIT_THRESHOLD. Hysteresis prevents churning.
APEX_ENTRY_THRESHOLD = 0.62      # require fairly strong conviction
APEX_EXIT_THRESHOLD  = 0.45      # only exit when the model has clearly cooled
APEX_HOLD_DAYS_MAX   = 7         # hard time-stop for any Apex position

# ─── Position sizing ─────────────────────────────────────────────────────────
# Final size = base_share × prob × kelly_fraction × regime_mult × tier_cap
# Kelly fraction here is a SAFETY DERATING (0.25 = quarter-Kelly).
APEX_KELLY_FRACTION = 0.25
APEX_MAX_POSITIONS  = 8          # concurrent Apex positions
APEX_MIN_POSITION_USD = 25.0     # below this we skip (fees too punishing)

# ─── ML feature vector spec ──────────────────────────────────────────────────
# When PHASE 4 finalizes the trained model this number is locked. The
# registry will refuse to serve a vector that doesn't match this length.
APEX_FEATURE_DIM = 30

# ─── Signal weights (used in legacy non-ML mode) ────────────────────────────
# When the ML model is unavailable (cold start, file missing, runtime
# error), Apex falls back to a simple signed weighted sum. These weights
# are intentionally rough — the ML layer is the primary decision system.
APEX_FALLBACK_WEIGHTS = {
    "btc_onchain":          0.15,
    "btc_dominance":        0.10,
    "funding_extreme":      0.15,
    "etf_flow":             0.20,
    "stablecoin_mint":      0.10,
    "token_unlock":         0.10,   # negative = bearish unlock pressure
    "news_sentiment":       0.10,
    "mvrv":                 0.05,
    "cross_asset_macro":    0.05,
}

# ─── Risk / exit tuning ──────────────────────────────────────────────────────
APEX_ATR_TRAIL_MULT          = 3.0    # 3× ATR trailing stop
APEX_PER_TRADE_HARD_KILL_PCT = 0.08   # safety net inherited from Pulse
APEX_DAILY_DD_FREEZE_PCT     = 0.05   # if today's DD > 5%, no new entries

# ─── ML model file ───────────────────────────────────────────────────────────
# Trained model artifact (joblib) shipped alongside Pulse files. Set to
# None to force fallback-mode for development/testing.
APEX_MODEL_FILENAME = "apex_model_v1.joblib"
APEX_MODEL_REQUIRED = False    # True in production once Phase 4 lands

# ─── Universe scoping for Apex ───────────────────────────────────────────────
# Apex only trades symbols with sufficient on-chain / data coverage.
# Most signals only resolve for major caps; trying to enter a fresh
# micro-cap with only price data wastes positions on low-confidence
# trades.
APEX_UNIVERSE_TIERS = ("major", "large", "mid")  # exclude micro

# ─── Diagnostic / logging ────────────────────────────────────────────────────
APEX_LOG_FEATURE_VECTOR     = False     # set True for verbose backtests
APEX_LOG_SIGNAL_BREAKDOWN   = True      # log which signals fired per entry
