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
GATLING_SCORE_MIN               = 0.25   # require SOME model confidence
GATLING_MIN_EV                  = -0.003 # slight negative EV OK
GATLING_PRED_RETURN_MIN         = -0.005 # loose regression veto
GATLING_MAX_DISPERSION          = 0.40   # some model agreement needed
GATLING_MIN_AGREE               = 0      # zero agreement gate (let voting decide)
GATLING_EV_GAP                  = 0.0    # no gap between candidates
GATLING_COST_BPS                = 30     # realistic Kraken fee estimate

# ── Position sizing (aggressive compounding for 100x target) ────────────────
GATLING_ALLOCATION              = 0.80   # 80% base — compound aggressively
GATLING_MAX_ALLOC               = 0.95   # near-full
GATLING_MIN_ALLOC               = 0.50   # minimum 50%
GATLING_USE_KELLY               = False  # flat 80% — maximize compounding
GATLING_KELLY_FRAC              = 1.00   # full-Kelly (unused when Kelly off)

# ── Exit parameters (trend-following: wide TP, trailing stop) ────────────────
GATLING_TAKE_PROFIT             = 0.12   # +12% TP — big trend, trail captures
GATLING_STOP_LOSS               = 0.025  # -2.5% SL — TIGHT (asymmetric: small loss, big win)
GATLING_TIMEOUT_HOURS           = 48.0   # 48h hold
GATLING_MIN_HOLD_MINUTES        = 45     # 45min min hold
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
GATLING_TRAIL_AFTER_TP          = 0.03   # arm trailing early at +3%
GATLING_TRAIL_PCT               = 0.02   # tight trail 2% — lock in profits fast

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
GATLING_LABEL_TP                = 0.06   # +6% — trend target for labels
GATLING_LABEL_SL                = 0.025  # -2.5% — aligned with tight execution SL
GATLING_LABEL_HORIZON_BARS      = 96     # 24h at 15-min bars

# ── Profit-voting (active with moderate thresholds) ──────────────────────────
GATLING_PROFIT_VOTING_MODE      = True
GATLING_VOTE_THRESHOLD          = 0.40   # threshold for model "yes" vote
GATLING_VOTE_YES_FRACTION_MIN   = 0.20   # at least 1 of 4 proven models must say yes
GATLING_TOP3_MEAN_MIN           = 0.25   # low bar for top-3
GATLING_VOTE_EV_FLOOR           = 0.0    # no EV floor

# ── Chop thresholds (stricter — avoid chop regime trades) ────────────────────
GATLING_CHOP_VOTE_YES_FRAC_MIN  = 0.25   # need more agreement in chop
GATLING_CHOP_TOP3_MEAN_MIN      = 0.45   # need stronger signal in chop
GATLING_CHOP_PRED_RETURN_MIN    = -0.005
GATLING_CHOP_EV_MIN             = -0.002

# ── Meta-filter (enabled — light filtering) ──────────────────────────────────
GATLING_META_FILTER_ENABLED     = False  # disabled — was blocking too many trades
GATLING_META_MIN_PROBA          = 0.0

# ── Market mode (ENABLED — only trade in favorable regimes) ──────────────────
GATLING_MARKET_MODE_ENABLED     = True   # enabled for SIZING only (not as gate)
GATLING_ALLOWED_MODES           = ["risk_on_trend", "pump", "chop",
                                   "high_vol_reversal", "selloff"]  # all allowed

# ── Breakeven (active — protect profitable trades) ───────────────────────────
GATLING_BREAKEVEN_AFTER         = 0.015  # arm breakeven early at +1.5%
GATLING_BREAKEVEN_BUFFER        = 0.005  # stop at entry + 0.5%
GATLING_MOM_FAIL_ENABLED        = False  # disabled — was causing premature exits
GATLING_MOM_FAIL_MIN_HOLD       = 999
GATLING_MOM_FAIL_LOSS           = -1.0

# ── Timeout extension (active — let winning trends run) ──────────────────────
GATLING_TIMEOUT_MIN_PROFIT      = 0.02   # extend if +2% at timeout
GATLING_TIMEOUT_EXTEND_HOURS    = 12     # extend 12h
GATLING_MAX_TIMEOUT_HOURS       = 48     # max 48h total hold

# ── V2 model pool ────────────────────────────────────────────────────────────
GATLING_USE_ENSEMBLE_V2 = False  # legacy models for now

# GBC-FOCUSED: gbc is the ONLY consistently profitable model across 7 backtests
# Other models as confirmers/shadow for data collection
GATLING_ACTIVE_MODELS = [
    "gbc",                  # ★ CORE: PF>1 in 3/4 backtests, 78% WR in latest
    "cal_et",               # confirmer — sometimes brilliant
    "et_shallow",           # confirmer — shallow tree
    "xgb_d2",               # industry standard depth=2
    "lgbm_d2",              # industry standard depth=2
    "logreg",               # generalization baseline
    "markov_vol",           # regime detector — P(low-vol) = safer to trade
]
GATLING_VETO_MODELS = []
GATLING_DIAGNOSTIC_MODELS = [
    # ALL gradient boosters are degenerate (100% YES rate on crypto):
    "hgbc", "hgbc_l2",            # HistGradientBoosting — always YES
    "lgbm_bal", "lgbm_dart",      # LightGBM — always YES
    "catboost_bal", "catboost_d3", # CatBoost — always YES
    "xgb_bal", "lgbm_goss",       # XGBoost/GOSS — always YES
    "bal_rf", "xgb_dart",         # BalancedRF, XGB DART — always YES
    "gnb", "lr", "lr_bal",        # always-bull/bear
    "cal_rf",                      # anti-signal WR=32%
    # PROVEN LOSERS (overfitting deep trees + anti-signal non-tree):
    "rf", "et",                    # depth=5 trees: PF=0.55/0.27, overfit
    "ridge_cal",                   # 0% WR — complete anti-signal
    "ebm",                         # 0% WR — anti-signal
    "knn_cal",                     # didn't vote at all across 4 backtests
    "rf_shallow",                  # collapsed to PF=0.04 in latest run
    # UNTESTED — never voted YES or insufficient data
    "svc_cal", "mlp", "ada", "ngboost", "sgd_cal", "qda_cal", "bag_dt2",
]
GATLING_SHADOW_MODELS = []

# ── Model weights (winning models weighted 2x) ──────────────────────────────
GATLING_MODEL_WEIGHTS = {
    # GBC is king — 3x weight (most consistent across all backtests)
    "gbc": 3.0,
    # Confirmers
    "cal_et": 1.5, "et_shallow": 1.0,
    # Industry standard
    "xgb_d2": 1.0, "lgbm_d2": 1.0, "logreg": 0.75,
    # Regime
    "markov_vol": 1.0,
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
