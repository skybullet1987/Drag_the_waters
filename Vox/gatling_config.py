# gatling_config.py — "Gatling V2" trend-following ensemble strategy
#
# Rebuilt based on 313-trade analysis that proved:
#   - 5-min scalping has NO learnable signal (24% WR, all models PF<1)
#   - Degenerate models (lgbm_bal=86% YES, hgbc_l2=84% YES) dominated
#   - Fee drag at high frequency was catastrophic (40% of capital)
#
# V2 Philosophy:
#   - TREND-FOLLOWING: catch 5-20% crypto moves, not 1% scalps
#   - 15-min decisions, 24h+ hold, trailing stops to let winners run
#   - Regime filtering: only trade in trend/pump, skip chop/selloff
#   - Survivable sizing: 20% per trade, survive 10+ consecutive losses
#   - Quarantine degenerate models, weight selective models higher
#   - Wider labels so models learn real trends, not noise
#
# Activate via:  risk_profile=gatling  (QC parameter panel)

# ── Entry gates (loose but not zero — require SOME signal) ───────────────────
GATLING_SCORE_MIN               = 0.18   # require minimal classifier signal
GATLING_MIN_EV                  = -0.002 # allow slightly negative EV
GATLING_PRED_RETURN_MIN         = -0.005 # loose regression veto
GATLING_MAX_DISPERSION          = 0.40   # some model agreement needed
GATLING_MIN_AGREE               = 0      # zero agreement gate (let voting decide)
GATLING_EV_GAP                  = 0.0    # no gap between candidates
GATLING_COST_BPS                = 30     # realistic Kraken fee estimate

# ── Position sizing (survivable — 20% per trade) ────────────────────────────
GATLING_ALLOCATION              = 0.80   # 80% per trade — aggressive test
GATLING_MAX_ALLOC               = 0.90   # max 90%
GATLING_MIN_ALLOC               = 0.50   # at least 50%
GATLING_USE_KELLY               = False  # flat 80% per trade
GATLING_KELLY_FRAC              = 1.00   # full-Kelly (unused when Kelly off)

# ── Exit parameters (trend-following: wide TP, trailing stop) ────────────────
GATLING_TAKE_PROFIT             = 0.06   # +6% TP target
GATLING_STOP_LOSS               = 0.025  # -2.5% SL — room to breathe
GATLING_TIMEOUT_HOURS           = 36.0   # 36h hold — let trends develop
GATLING_MIN_HOLD_MINUTES        = 30     # hold at least 30min (avoid noise chop)
GATLING_EMERGENCY_SL            = 0.05   # 5% emergency stop

# ── Cooldowns (short but present — avoid re-entering failed trades) ──────────
GATLING_COOLDOWN_MINS           = 5      # 5min global cooldown
GATLING_SL_COOLDOWN_MINS        = 30     # 30min per-coin after SL
GATLING_PENALTY_COOLDOWN_LOSSES = 5      # 5 consecutive SL → penalty
GATLING_PENALTY_COOLDOWN_HOURS  = 6      # 6h penalty block
GATLING_MAX_DAILY_SL            = 10     # 10 daily SL cap
GATLING_MAX_DD_PCT              = 0.30   # 30% drawdown circuit-breaker

# ── Decision frequency (15-min — standard, proven to have some signal) ───────
GATLING_DECISION_INTERVAL_MIN   = 15     # every 15-min bar

# ── Runner mode ON (trailing stop — let winners run) ─────────────────────────
GATLING_RUNNER_MODE             = True   # trailing stop instead of instant TP
GATLING_TRAIL_AFTER_TP          = 0.04   # arm trailing at +4%
GATLING_TRAIL_PCT               = 0.025  # trail 2.5% from high-water mark

# ── Anti-chop / loss-streak (active — protect from chop regimes) ─────────────
GATLING_LOSS_WINDOW_HOURS       = 12     # 12h window for SL counting
GATLING_LOSS_LIMIT              = 4      # 4 SLs in window → block
GATLING_LOSS_BLOCK_HOURS        = 3.0    # 3h block
GATLING_PORTFOLIO_LOSS_STREAK   = 6      # 6 consecutive losses → pause
GATLING_PORTFOLIO_PAUSE_HOURS   = 2.0    # 2h pause

# ── Confirmation gate (disabled — handled by gatling_bypass in strategy.py) ──
GATLING_CONFIRM_EV_MIN          = -1.0
GATLING_CONFIRM_PROBA_MIN       = 0.0
GATLING_CONFIRM_AGREE_MIN       = 0
GATLING_CONFIRM_RET4_MIN        = -1.0
GATLING_CONFIRM_RET16_MIN       = -1.0
GATLING_CONFIRM_VOLR_MIN        = 0.0

# ── Label parameters (WIDE labels — teach models to find real trends) ────────
GATLING_LABEL_TP                = 0.05   # +5% — real trend target
GATLING_LABEL_SL                = 0.02   # -2% — meaningful reversal
GATLING_LABEL_HORIZON_BARS      = 96     # 24h at 15-min bars

# ── Profit-voting (active with moderate thresholds) ──────────────────────────
GATLING_PROFIT_VOTING_MODE      = True
GATLING_VOTE_THRESHOLD          = 0.45   # moderate yes/no split
GATLING_VOTE_YES_FRACTION_MIN   = 0.15   # at least 15% of models agree
GATLING_TOP3_MEAN_MIN           = 0.35   # top-3 models mildly bullish
GATLING_VOTE_EV_FLOOR           = 0.0    # no EV floor

# ── Chop thresholds (stricter — avoid chop regime trades) ────────────────────
GATLING_CHOP_VOTE_YES_FRAC_MIN  = 0.25   # need more agreement in chop
GATLING_CHOP_TOP3_MEAN_MIN      = 0.45   # need stronger signal in chop
GATLING_CHOP_PRED_RETURN_MIN    = -0.005
GATLING_CHOP_EV_MIN             = -0.002

# ── Meta-filter (enabled — light filtering) ──────────────────────────────────
GATLING_META_FILTER_ENABLED     = True
GATLING_META_MIN_PROBA          = 0.30   # very loose meta-filter

# ── Market mode (ENABLED — only trade in favorable regimes) ──────────────────
GATLING_MARKET_MODE_ENABLED     = True
GATLING_ALLOWED_MODES           = ["risk_on_trend", "pump", "chop",
                                   "high_vol_reversal"]  # skip selloff only

# ── Breakeven (active — protect profitable trades) ───────────────────────────
GATLING_BREAKEVEN_AFTER         = 0.03   # arm breakeven at +3%
GATLING_BREAKEVEN_BUFFER        = 0.005  # stop at entry + 0.5%
GATLING_MOM_FAIL_ENABLED        = True   # cut momentum failures early
GATLING_MOM_FAIL_MIN_HOLD       = 60     # after 1h
GATLING_MOM_FAIL_LOSS           = -0.015 # if down 1.5% with broken momentum

# ── Timeout extension (active — let winning trends run) ──────────────────────
GATLING_TIMEOUT_MIN_PROFIT      = 0.02   # extend if +2% at timeout
GATLING_TIMEOUT_EXTEND_HOURS    = 12     # extend 12h
GATLING_MAX_TIMEOUT_HOURS       = 48     # max 48h total hold

# ── V2 model pool ────────────────────────────────────────────────────────────
GATLING_USE_ENSEMBLE_V2 = False  # legacy models for now

# Active: ONLY selective models (never-degenerate, <50% YES rate)
# ALL gradient boosters removed — they ALL become 100% YES on crypto data
GATLING_ACTIVE_MODELS = [
    "gbc",                  # PF=1.16, 9% fire rate — ONLY PF>1 model
    "rf_shallow",           # PF=1.47 (prior run), very selective
    "et_shallow",           # PF=2.50 (prior run), very selective
    "cal_et",               # PF=2.23 (prior run), selective
    "rf",                   # selective (3% fire rate)
    "et",                   # selective
    "ridge_cal",            # linear — different from trees
    "svc_cal",              # SVM — different geometry
    "ebm",                  # GAM — different paradigm
    "knn_cal",              # KNN — instance-based
    "mlp",                  # neural net — different
    "ada",                  # AdaBoost — weaker booster, less degenerate
    "ngboost",              # probabilistic — uncertainty-aware
]
GATLING_VETO_MODELS = []
GATLING_DIAGNOSTIC_MODELS = [
    # ALL gradient boosters are degenerate (100% YES rate on crypto):
    "hgbc", "hgbc_l2",            # HistGradientBoosting — always YES
    "lgbm_bal", "lgbm_dart",      # LightGBM — always YES
    "catboost_bal", "catboost_d3", # CatBoost — always YES
    "xgb_bal", "lgbm_goss",       # XGBoost/GOSS — always YES
    "bal_rf",                      # BalancedRF — always YES
    "gnb", "lr", "lr_bal",        # always-bull/bear
    "cal_rf",                      # anti-signal
]
GATLING_SHADOW_MODELS = []

# ── Model weights (winning models weighted 2x) ──────────────────────────────
GATLING_MODEL_WEIGHTS = {
    # Proven selective models (high weight)
    "gbc": 2.0,            # ONLY PF>1 model in latest run
    "et_shallow": 2.0,     # PF=2.50 in prior run, very selective
    "cal_et": 1.5,         # PF=2.23 in prior run
    "rf_shallow": 1.5,     # PF=1.47 in prior run
    # Selective tree models
    "rf": 1.0,             # very selective (3% fire rate)
    "et": 1.0,             # selective
    "ada": 0.75,           # weaker booster, less degenerate
    # Non-tree diversity
    "ridge_cal": 1.0,      # linear
    "svc_cal": 1.0,        # SVM
    "ebm": 1.0,            # GAM
    "knn_cal": 1.0,        # KNN
    "mlp": 1.0,            # neural net
    "ngboost": 1.0,        # probabilistic
}

# ── Regime-adaptive allocation ───────────────────────────────────────────────
GATLING_REGIME_SIZING = True
GATLING_REGIME_ALLOC = {
    "pump": 0.90,             # bullish pump — max aggression
    "risk_on_trend": 0.80,    # trending — full allocation
    "high_vol_reversal": 0.50, # volatile — moderate
    "chop": 0.25,             # choppy — minimal
    "selloff": 0.10,          # bearish — tiny or skip
}
GATLING_REGIME_DEFAULT_ALLOC = 0.50  # unknown regime

# ── Diag logging ─────────────────────────────────────────────────────────────
GATLING_DIAG_INTERVAL_HOURS     = 1
GATLING_SKIP_DIAG_INTERVAL_SECS = 3600

# ── Model assessment tracking ────────────────────────────────────────────────
GATLING_TRACK_MODEL_ACCURACY    = True
GATLING_MIN_TRADES_FOR_ASSESS   = 10

# ── Vote logging ─────────────────────────────────────────────────────────────
GATLING_LOG_MODEL_VOTES         = True
