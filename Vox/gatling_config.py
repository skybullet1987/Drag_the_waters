# gatling_config.py — constants for the "gatling" risk profile
#
# GATLING MODE: extremely active, high-frequency ensemble-voting strategy.
# Designed to fire as many trades as possible ("gatling gun") using the
# ensemble vote to find edge, then assess individual model performance
# from backtest results.
#
# Philosophy:
#   - Near-zero gates: let the ensemble decide, not hard-coded thresholds
#   - 5-minute decision interval (every bar) instead of 15-minute
#   - Minimal cooldowns: re-enter immediately after exits
#   - Fast labels (short horizon, tight TP/SL) so models learn scalp patterns
#   - Full allocation per trade for maximum capital efficiency
#   - Profit-voting with ultra-loose thresholds: any model agreement = go
#   - Per-model accuracy tracked in journal for post-backtest assessment
#
# RISK WARNING: This profile is EXTREMELY aggressive. Expect very high
# trade count, significant fee drag, and large drawdowns. Use for:
#   1. Backtesting to gather large trade samples for model assessment
#   2. Identifying which models have genuine predictive power
#   3. Stress-testing the ensemble under maximum activity
#
# Activate via:  risk_profile=gatling  (QC parameter panel)

# ── Entry gates (near-zero to maximize trade frequency) ──────────────────────
GATLING_SCORE_MIN               = 0.10   # almost any signal passes
GATLING_MIN_EV                  = -0.005 # allow slightly negative EV trades
GATLING_PRED_RETURN_MIN         = -0.010 # almost never veto on regression
GATLING_MAX_DISPERSION          = 0.50   # even highly disagreeing models pass
GATLING_MIN_AGREE               = 0      # zero agreement required
GATLING_EV_GAP                  = 0.0    # no gap required between top candidates
GATLING_COST_BPS                = 15     # optimistic cost assumption

# ── Position sizing (full allocation, no Kelly shrinkage) ────────────────────
GATLING_ALLOCATION              = 0.95   # near-full portfolio per trade
GATLING_MAX_ALLOC               = 1.00   # allow 100% allocation
GATLING_MIN_ALLOC               = 0.80   # Kelly cannot shrink below 80%
GATLING_USE_KELLY               = False  # flat allocation — no Kelly
GATLING_KELLY_FRAC              = 1.00   # if Kelly enabled, full-Kelly

# ── Exit parameters (tight scalp targets for fast turnover) ──────────────────
GATLING_TAKE_PROFIT             = 0.015  # +1.5% TP — fast scalp
GATLING_STOP_LOSS               = 0.010  # -1.0% SL — tight stop
GATLING_TIMEOUT_HOURS           = 2.0    # 2h max hold — force turnover
GATLING_MIN_HOLD_MINUTES        = 0      # no minimum hold
GATLING_EMERGENCY_SL            = 0.025  # 2.5% emergency stop

# ── Cooldowns (near-zero for maximum re-entry speed) ─────────────────────────
GATLING_COOLDOWN_MINS           = 0      # zero global cooldown
GATLING_SL_COOLDOWN_MINS        = 0      # zero per-coin SL cooldown
GATLING_PENALTY_COOLDOWN_LOSSES = 100    # effectively never triggers penalty
GATLING_PENALTY_COOLDOWN_HOURS  = 0      # no penalty cooldown duration
GATLING_MAX_DAILY_SL            = 50     # extremely high daily SL cap
GATLING_MAX_DD_PCT              = 0.50   # 50% drawdown before circuit-breaker

# ── Decision frequency ───────────────────────────────────────────────────────
GATLING_DECISION_INTERVAL_MIN   = 5      # every 5-min bar = decision point

# ── Runner mode OFF (instant TP exit, no trailing) ───────────────────────────
GATLING_RUNNER_MODE             = False  # take profit immediately
GATLING_TRAIL_AFTER_TP          = 0.0
GATLING_TRAIL_PCT               = 0.0

# ── Anti-chop / loss-streak (extremely relaxed) ──────────────────────────────
GATLING_LOSS_WINDOW_HOURS       = 24
GATLING_LOSS_LIMIT              = 50     # near-impossible to trigger
GATLING_LOSS_BLOCK_HOURS        = 0.0    # no block if triggered
GATLING_PORTFOLIO_LOSS_STREAK   = 50     # near-impossible to trigger
GATLING_PORTFOLIO_PAUSE_HOURS   = 0.0    # no pause if triggered

# ── Confirmation gate (disabled — let ensemble alone decide) ─────────────────
GATLING_CONFIRM_EV_MIN          = -1.0   # any EV passes
GATLING_CONFIRM_PROBA_MIN       = 0.0    # any proba passes
GATLING_CONFIRM_AGREE_MIN       = 0      # zero agreement needed
GATLING_CONFIRM_RET4_MIN        = -1.0   # any momentum passes
GATLING_CONFIRM_RET16_MIN       = -1.0
GATLING_CONFIRM_VOLR_MIN        = 0.0    # any volume passes

# ── Label parameters (fast scalp labels for model training) ──────────────────
GATLING_LABEL_TP                = 0.010  # +1.0% — achievable scalp TP
GATLING_LABEL_SL                = 0.008  # -0.8% — tight SL for binary
GATLING_LABEL_HORIZON_BARS      = 24     # 2h at 5-min bars

# ── Profit-voting (ultra-loose — any model signal fires) ─────────────────────
GATLING_PROFIT_VOTING_MODE      = True   # use vote-based ranking
GATLING_VOTE_THRESHOLD          = 0.35   # very low threshold for "yes" vote
GATLING_VOTE_YES_FRACTION_MIN   = 0.10   # 10% of models saying yes = go
GATLING_TOP3_MEAN_MIN           = 0.30   # low top-3 mean required
GATLING_VOTE_EV_FLOOR           = 0.0    # no EV floor in profit voting

# ── Chop thresholds (also ultra-loose) ───────────────────────────────────────
GATLING_CHOP_VOTE_YES_FRAC_MIN  = 0.15
GATLING_CHOP_TOP3_MEAN_MIN      = 0.35
GATLING_CHOP_PRED_RETURN_MIN    = -1.0   # never blocks
GATLING_CHOP_EV_MIN             = -1.0   # never blocks

# ── Meta-filter (disabled for gatling) ───────────────────────────────────────
GATLING_META_FILTER_ENABLED     = False  # no meta-filter veto
GATLING_META_MIN_PROBA          = 0.0    # pass-through if somehow enabled

# ── Market mode (disabled — trade in all regimes) ────────────────────────────
GATLING_MARKET_MODE_ENABLED     = False
GATLING_ALLOWED_MODES           = ["risk_on_trend", "pump", "chop", "selloff", "high_vol_reversal"]

# ── Breakeven / momentum-fail (disabled for maximum activity) ────────────────
GATLING_BREAKEVEN_AFTER         = 1.0    # effectively never triggers
GATLING_BREAKEVEN_BUFFER        = 0.001
GATLING_MOM_FAIL_ENABLED        = False  # no early momentum-fail exit
GATLING_MOM_FAIL_MIN_HOLD       = 999
GATLING_MOM_FAIL_LOSS           = -1.0

# ── Timeout extension (disabled) ─────────────────────────────────────────────
GATLING_TIMEOUT_MIN_PROFIT      = 1.0    # never extends
GATLING_TIMEOUT_EXTEND_HOURS    = 0
GATLING_MAX_TIMEOUT_HOURS       = 2.0    # hard cap matches timeout

# ── V2 model pool (15 cutting-edge models, max diversity) ────────────────────
GATLING_USE_ENSEMBLE_V2 = True   # use ensemble_v2.py models instead of legacy
GATLING_ACTIVE_MODELS = [
    "catboost", "xgb_hist", "lgbm_goss", "hgbc",
    "tabnet", "ebm", "ngboost", "rf_bal",
    "et_depth5", "ridge_cal", "svc_rbf", "mlp",
    "lgbm_dart",
]
GATLING_VETO_MODELS = ["iforest_veto"]  # block trade when anomaly detected
GATLING_DIAGNOSTIC_MODELS = []
GATLING_SHADOW_MODELS = ["stack_meta"]  # meta-learner, shadow until validated

# ── Diag logging (frequent for analysis) ─────────────────────────────────────
GATLING_DIAG_INTERVAL_HOURS     = 0.5   # log diagnostics every 30 min
GATLING_SKIP_DIAG_INTERVAL_SECS = 1800  # routine skip diags every 30 min

# ── Model assessment tracking ────────────────────────────────────────────────
GATLING_TRACK_MODEL_ACCURACY    = True  # enable per-model accuracy tracking
GATLING_MIN_TRADES_FOR_ASSESS   = 10    # minimum trades before model assessment

# ── Vote logging (MUST be True for model assessment to work) ─────────────────
GATLING_LOG_MODEL_VOTES         = True  # force per-model vote logging in trade log
