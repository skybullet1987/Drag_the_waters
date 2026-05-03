# ── Ruthless V2 — configuration defaults ─────────────────────────────────────
#
# All module-level constants for the Ruthless V2 engine.
# Imported by other ruthless sub-modules and by ruthless_v2.py (shim).
# ─────────────────────────────────────────────────────────────────────────────

RUTHLESS_V2_MODE                       = False  # global default off; activated by profile/param

RUTHLESS_V2_MAX_CONCURRENT_POSITIONS   = 5
RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY    = 12
RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY = 3
RUTHLESS_V2_MAX_SYMBOL_ALLOCATION      = 0.35
RUTHLESS_V2_MIN_SYMBOL_ALLOCATION      = 0.08
RUTHLESS_V2_MAX_TOTAL_EXPOSURE         = 1.50
RUTHLESS_V2_DECISION_INTERVAL_MIN      = 15
RUTHLESS_V2_MIN_SCORE_TO_TRADE         = -0.25
RUTHLESS_V2_REENTRY_COOLDOWN_MIN       = 30

# ── V2 machine-gun mode ───────────────────────────────────────────────────────
RUTHLESS_V2_MACHINE_GUN_MODE          = True
RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES = 2
RUTHLESS_V2_REGIME_HARD_BLOCK         = False
RUTHLESS_V2_META_HARD_FILTER          = False
RUTHLESS_V2_META_AS_SCORE_PENALTY     = True
RUTHLESS_V2_META_SCORE_WEIGHT         = 0.20
RUTHLESS_V2_LOW_META_ALLOC_MULT       = 0.50
RUTHLESS_V2_ALLOW_CHOP_SCALPS         = True
RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC      = 0.10
RUTHLESS_V2_BASE_ALLOCATION           = 0.12
RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION = 0.30
RUTHLESS_V2_SOFT_REGIME_PENALTY          = 0.12
RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER   = 1.5
RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT     = 0.5
RUTHLESS_V2_HARD_BLOCK_MODES          = frozenset(["risk_off_crash", "dump", "emergency"])

# Partial TP / split-exit defaults
RUTHLESS_V2_PARTIAL_TP_ENABLED         = True
RUTHLESS_V2_PARTIAL_TP_FRACTION        = 0.50
RUTHLESS_V2_SCALP_TP                   = 0.015
RUTHLESS_V2_CONTINUATION_TP            = 0.04
RUTHLESS_V2_RUNNER_INITIAL_TP          = 0.06
RUTHLESS_V2_RUNNER_TRAIL_PCT           = 0.04
RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT      = 0.06

# V2 active voter pool
RUTHLESS_V2_ACTIVE_MODELS    = ["rf", "et", "hgbc_l2", "lgbm_bal", "gbc", "ada"]
RUTHLESS_V2_OPTIONAL_MODELS  = ["xgb_bal", "catboost_bal"]
RUTHLESS_V2_DIAGNOSTIC_MODELS = [
    "gnb", "lr", "lr_bal", "cal_et", "cal_rf",
    "et_shallow", "rf_shallow",
    "markov_regime", "hmm_regime", "kmeans_regime",
    "isoforest_risk",
]

# Base weights for V2 active models
RUTHLESS_V2_BASE_WEIGHTS = {
    "rf":           1.35,
    "hgbc_l2":      1.10,
    "lgbm_bal":     1.00,
    "et":           0.80,
    "gbc":          0.85,
    "ada":          0.70,
    "xgb_bal":      0.75,
    "catboost_bal": 0.75,
}

# Dynamic weight caps
RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER  = 2.0
RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER  = 0.25
RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST  = 5
RUTHLESS_V2_DECAY_FACTOR           = 0.85
