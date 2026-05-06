# ── core.py: config + market_mode + momentum + meta_model ──────────────────
import numpy as np
from collections import deque
from datetime import timedelta



# ===============================================================================
# config
# ===============================================================================

# ── Strategy constants ───────────────────────────────────────────────────
TAKE_PROFIT          = 0.030   # +3.0 %  close long on gain
STOP_LOSS            = 0.015   # −1.5 %  close long on loss
TIMEOUT_HOURS        = 6.0     # close after this many hours regardless
ATR_TP_MULT          = 2.0     # TP = entry + ATR_TP_MULT × ATR
ATR_SL_MULT          = 1.2     # SL = entry − ATR_SL_MULT × ATR
SCORE_MIN            = 0.50    # minimum class_proba — balanced default
SCORE_MIN_FLOOR      = 0.15    # floor for the effective base-rate-aware score threshold
SCORE_MIN_MULT       = 3.0     # adaptive multiplier: eff_score_min = SCORE_MIN_MULT * base_rate
SCORE_GAP            = 0.02    # probability gap: required lead of top coin over runner-up
MAX_DISPERSION       = 0.22    # max std_proba across models — relaxed default
MIN_AGREE            = 2       # min models with proba >= agree_thr — relaxed default
ALLOCATION           = 0.50    # fallback fraction of portfolio if Kelly disabled
KELLY_FRAC           = 0.25    # fractional-Kelly multiplier
MAX_ALLOC            = 0.80    # hard ceiling on any single trade allocation
USE_KELLY            = True    # set False to use flat ALLOCATION
MAX_DAILY_SL         = 2       # halt new entries after this many SL hits per day
COOLDOWN_MINS        = 20      # minutes to wait after any exit before re-entering
SL_COOLDOWN_MINS     = 60      # per-coin cooldown specifically after an SL exit
MAX_DD_PCT           = 0.08    # drawdown circuit-breaker: halt if equity drops > 8 %
CASH_BUFFER          = 0.99    # keep 1 % cash headroom for fees/rounding
EXIT_QTY_BUFFER_LOTS = 1       # safety lots subtracted from sell qty
COST_BPS             = 35      # estimated round-trip fee+slippage in basis points
MIN_EV               = 0.001   # minimum expected-value (return fraction) after costs
EV_GAP               = 0.0001  # required EV lead of top coin over second-best
PRED_RETURN_MIN      = -0.0005 # regression veto: block only clearly bad predicted returns
MIN_HOLD_MINUTES     = 15      # suppress ordinary TP/SL/timeout exits before this many minutes
EMERGENCY_SL         = 0.030   # allow early exit only if loss exceeds this: 3.0 %
CONSERVATIVE_MODE    = False   # when True, applies stricter gates
PENALTY_COOLDOWN_LOSSES = 3    # consecutive SL exits before penalty cooldown triggers
PENALTY_COOLDOWN_HOURS  = 48   # hours a symbol is blocked after repeated losses
MAX_EXIT_RETRY_COUNT = 3       # max consecutive INVALID exit retries before treating as dust
RESOLUTION_MINUTES   = 5       # subscribe at 5-min bars
DECISION_INTERVAL_MIN = 15     # only evaluate entries at 15-min boundaries
WARMUP_DAYS          = 90      # bars of history needed before trading
MAX_HISTORY_BARS     = 30000   # safety cap on history bars fetched per symbol
SKIP_DIAG_INTERVAL_SECS = 21600  # routine skip diagnostics logged at most once per 6 hours
DIAG_INTERVAL_HOURS  = 6       # no-candidate summary logged at most once every N hours
MIN_RETRAIN_INTERVAL_HOURS = 20  # skip scheduled retrain if last retrain was within this many hours

# ── Risk profile ─────────────────────────────────────────────────────────
RISK_PROFILE = "balanced"

# ── Aggressive profile defaults ──────────────────────────────────────────
AGGRESSIVE_SCORE_MIN               = 0.48
AGGRESSIVE_MIN_EV                  = 0.0005
AGGRESSIVE_PRED_RETURN_MIN         = -0.0010
AGGRESSIVE_MAX_DISPERSION          = 0.28
AGGRESSIVE_MIN_AGREE               = 1
AGGRESSIVE_EV_GAP                  = 0.0
AGGRESSIVE_COST_BPS                = 25
AGGRESSIVE_ALLOCATION              = 0.75
AGGRESSIVE_MAX_ALLOC               = 0.95
AGGRESSIVE_KELLY_FRAC              = 0.50
AGGRESSIVE_TAKE_PROFIT             = 0.045
AGGRESSIVE_STOP_LOSS               = 0.020
AGGRESSIVE_TIMEOUT_HOURS           = 8
AGGRESSIVE_MIN_HOLD_MINUTES        = 5
AGGRESSIVE_EMERGENCY_SL            = 0.025
AGGRESSIVE_MAX_DAILY_SL            = 3
AGGRESSIVE_COOLDOWN_MINS           = 5
AGGRESSIVE_SL_COOLDOWN_MINS        = 20
AGGRESSIVE_PENALTY_COOLDOWN_LOSSES = 3
AGGRESSIVE_PENALTY_COOLDOWN_HOURS  = 24
AGGRESSIVE_MAX_DD_PCT              = 0.20

# ── Ruthless V2 ─────────────────────────────────────────────────────────
RUTHLESS_V2_MODE                 = False  # default off; activated by profile/param

# ── APEX PREDATOR constants ──────────────────────────────────────────────
APEX_SCORE_ENTRY        = 0.50   # minimum apex_score to trigger entry (was 0.55)
APEX_SCORE_PYRAMID      = 0.55   # minimum apex_score to add a pyramid tranche
APEX_BASE_ALLOC         = 0.20   # baseline allocation (20% of equity)
APEX_MAX_GROSS          = 2.0    # maximum total gross exposure (2× equity)
APEX_MAX_CONCURRENT     = 8      # maximum simultaneous open positions
APEX_MAX_PER_SYMBOL     = 2      # maximum concurrent positions per symbol
APEX_COOLDOWN_MIN       = 15     # per-symbol reentry cooldown in minutes
APEX_TIME_STOP_HRS      = 48     # close position if open > this without +1% MFE
APEX_ATR_SL_MULT        = 1.25   # SL = entry − APEX_ATR_SL_MULT × ATR(14); floor 0.8%, ceil 4%
APEX_ATR_TP_MULT        = 4.0    # TP = entry + APEX_ATR_TP_MULT × ATR(14); floor 2.5%, ceil 15%
APEX_TRAIL_ARM_PCT      = 0.010  # trailing stop arms once unrealised PnL >= 1.0%
APEX_TRAIL_ATR_MULT     = 0.8    # trail distance = max(APEX_TRAIL_ATR_MULT × ATR, 0.6%)
APEX_BREAKEVEN_MFE      = 0.02   # move stop to breakeven once MFE >= 2%

APEX_ENTRY_PATH4_PROBA_MIN   = 0.50
APEX_ENTRY_PATH4_N_AGREE_MIN = 1
APEX_ENTRY_LGBM_BAL_MIN      = 0.50
APEX_BREAKOUT_NBARS          = 20
APEX_BREAKOUT_VOL_MULT       = 1.5
APEX_PULLBACK_RSI_MAX        = 35
APEX_PULLBACK_TREND_BARS     = 10
APEX_MOMENTUM_CONT_BARS      = 3
APEX_MOMENTUM_CONT_VOL_MULT  = 1.5
# ── Apex Predator profile defaults ───────────────────────────────────────
APEX_PROFILE_SCORE_MIN               = 0.15
APEX_PROFILE_PRED_RETURN_MIN         = -0.015
APEX_PROFILE_VOTE_THRESHOLD          = 0.40
APEX_PROFILE_VOTE_YES_FRACTION_MIN   = 0.20
APEX_PROFILE_TOP3_MEAN_MIN           = 0.35
APEX_PROFILE_CHOP_YES_FRAC_MIN       = 0.25
APEX_PROFILE_CHOP_TOP3_MEAN_MIN      = 0.35
APEX_PROFILE_SL_COOLDOWN_MINS        = 10
APEX_PROFILE_LOSS_LIMIT              = 8
APEX_PROFILE_LOSS_WINDOW_HOURS       = 4
APEX_PROFILE_LOSS_BLOCK_HOURS        = 1
APEX_PROFILE_PORTFOLIO_LOSS_STREAK   = 10
APEX_PROFILE_PORTFOLIO_PAUSE_HOURS   = 0.5
APEX_PROFILE_CONFIRM_EV_MIN          = 0.0001
APEX_PROFILE_CONFIRM_PROBA_MIN       = 0.38
APEX_PROFILE_CONFIRM_RET4_MIN        = 0.001
APEX_PROFILE_META_MIN_PROBA          = 0.42
APEX_PROFILE_GOOD_MODE_META_MIN_PROBA = 0.38
APEX_PROFILE_GOOD_MODE_VOLUME_MIN    = 0.8
APEX_PROFILE_LABEL_TP                = 0.020   # achievable TP → positive_rate rises
APEX_PROFILE_LABEL_SL                = 0.010
APEX_PROFILE_LABEL_HORIZON_BARS      = 36      # 3h label horizon at 5-min bars


RUTHLESS_SCORE_MIN               = 0.20
RUTHLESS_MIN_EV                  = 0.0000
RUTHLESS_PRED_RETURN_MIN         = -0.0040   # was -0.0020; very loose veto
RUTHLESS_MAX_DISPERSION          = 0.35
RUTHLESS_MIN_AGREE               = 1
RUTHLESS_EV_GAP                  = 0.0
RUTHLESS_COST_BPS                = 20
RUTHLESS_ALLOCATION              = 0.90
RUTHLESS_MAX_ALLOC               = 1.00
RUTHLESS_KELLY_FRAC              = 0.75
RUTHLESS_MIN_ALLOC               = 0.75     # Kelly cannot shrink below 75 %
RUTHLESS_USE_KELLY               = False    # flat allocation at 90 % by default
RUTHLESS_TAKE_PROFIT             = 0.09    # was 0.060; +9 % target
RUTHLESS_STOP_LOSS               = 0.03    # was 0.025; −3 % stop
RUTHLESS_TIMEOUT_HOURS           = 24      # was 12; winners get room to run
RUTHLESS_MIN_HOLD_MINUTES        = 10      # was 3; less instant chop
RUTHLESS_EMERGENCY_SL            = 0.05    # was 0.040; 5 % catastrophic stop
RUTHLESS_MAX_DAILY_SL            = 5
RUTHLESS_COOLDOWN_MINS           = 0
RUTHLESS_SL_COOLDOWN_MINS        = 30
RUTHLESS_PENALTY_COOLDOWN_LOSSES = 5
RUTHLESS_PENALTY_COOLDOWN_HOURS  = 12
RUTHLESS_MAX_DD_PCT              = 0.35
# Runner / trailing-profit parameters
RUTHLESS_RUNNER_MODE             = True
RUTHLESS_TRAIL_AFTER_TP          = 0.07
RUTHLESS_TRAIL_PCT               = 0.03

# ── Ruthless v4: breakeven stop ──────────────────────────────────────────
RUTHLESS_BREAKEVEN_AFTER         = 0.03   # move stop to entry after +3% return seen
RUTHLESS_BREAKEVEN_BUFFER        = 0.003  # effective stop = entry + this buffer

# ── Ruthless v4: momentum-failure early exit ─────────────────────────────
RUTHLESS_MOM_FAIL_ENABLED        = True
RUTHLESS_MOM_FAIL_MIN_HOLD_MINUTES = 30   # must hold at least this long
RUTHLESS_MOM_FAIL_LOSS           = -0.012 # exit if return <= this with broken momentum

# ── Ruthless v4: smarter timeout extension ───────────────────────────────
RUTHLESS_TIMEOUT_MIN_PROFIT      = 0.03   # allow timeout exit if return >= this
RUTHLESS_TIMEOUT_EXTEND_HOURS    = 12     # extend hold by this many hours
RUTHLESS_MAX_TIMEOUT_HOURS       = 48     # hard cap on total hold time

# ── Market mode detection ────────────────────────────────────────────────
MARKET_MODE_ENABLED              = True
RUTHLESS_ALLOWED_MODES           = ["risk_on_trend", "pump"]

# ── Ruthless meta-filter ─────────────────────────────────────────────────
RUTHLESS_META_FILTER_ENABLED     = True
RUTHLESS_META_MIN_PROBA          = 0.55

# ── Optional entry limit orders ──────────────────────────────────────────
RUTHLESS_USE_ENTRY_LIMIT_ORDERS  = False  # disabled by default; enable to reduce slippage
ENTRY_LIMIT_OFFSET               = 0.001  # buy limit at price * (1 - offset)
ENTRY_LIMIT_TTL_MINUTES          = 3      # cancel unfilled limits after this many minutes

# ── Optional exit limit orders ───────────────────────────────────────────
USE_EXIT_LIMIT_ORDERS            = False  # disabled by default
EXIT_LIMIT_OFFSET                = 0.0005 # sell limit at price * (1 + offset)
EXIT_LIMIT_TTL_MINUTES           = 1      # cancel unfilled exit limits after 1 minute
EXIT_LIMIT_FALLBACK_TO_MARKET    = True   # fallback to market if limit not filled in TTL

# ── Optional external ML models ──────────────────────────────────────────
USE_LIGHTGBM                     = False  # enable if lightgbm installed
USE_XGBOOST                      = False  # enable if xgboost installed
USE_CATBOOST                     = False  # disabled: CatBoost is not available in the QC cloud environment

# ── Per-model static weights ─────────────────────────────────────────────
# Default 1.0 for active models → unweighted mean (preserves current behavior).
# GNB and LR are 0.0 by default: both are diagnostic-only (always-bullish /
# always-bearish observed in live data) and must NOT inflate active consensus.
MODEL_WEIGHT_LR           = 0.0   # diagnostic-only: was always-bearish (~0.006-0.023)
MODEL_WEIGHT_HGBC         = 1.0
MODEL_WEIGHT_ET           = 1.0
MODEL_WEIGHT_RF           = 1.0
MODEL_WEIGHT_GNB          = 0.0   # diagnostic-only: always-bullish (vote_gnb=1.0)
MODEL_WEIGHT_LGBM         = 1.0
MODEL_WEIGHT_XGB          = 1.0
MODEL_WEIGHT_CATBOOST     = 1.0
# Shadow model weights used in ruthless active pool if promoted
MODEL_WEIGHT_HGBC_L2      = 2.0   # promoted: strong regularised booster
MODEL_WEIGHT_CAL_ET       = 0.75
MODEL_WEIGHT_CAL_RF       = 0.75
MODEL_WEIGHT_LGBM_BAL     = 2.0   # promoted: best calibration on imbalanced data

# ── Model roles ──────────────────────────────────────────────────────────
# Roles: "active" | "shadow" | "diagnostic" | "disabled"
#
#   active     — contributes to ensemble vote; affects trading confidence.
#   shadow     — predicted and logged but NEVER affects trading decisions.
#   diagnostic — predicted and logged for risk/veto/debug only.
#   disabled   — skipped entirely (not trained or predicted).
#
# Backward-compat: class_proba / std_proba / n_agree map to ACTIVE values only.
# GNB is diagnostic by default because vote_gnb=1.0 on every entry (degenerate).
# LR is diagnostic because it was always bearish (~0.006-0.023 on all entries).
MODEL_ROLE_LR       = "diagnostic"  # always-bearish observed; diagnostic-only
MODEL_ROLE_HGBC     = "active"
MODEL_ROLE_ET       = "active"
MODEL_ROLE_RF       = "active"
MODEL_ROLE_GNB      = "diagnostic"  # degenerate: always-bullish (vote_gnb=1.0)
MODEL_ROLE_LGBM     = "shadow"
MODEL_ROLE_XGB      = "shadow"
MODEL_ROLE_CATBOOST = "shadow"

# ── Ruthless profit-voting mode ──────────────────────────────────────────
# When True and risk_profile=ruthless, use explicit profit-voting/vote-ranking
# behavior instead of balanced-like cautious thresholds.
# Has no effect on balanced / conservative / aggressive profiles.
RUTHLESS_PROFIT_VOTING_MODE     = True

# Ruthless active voting pool — promoted from shadow in ruthless PV mode.
# Models listed here that are available in shadow_votes are moved into
# active_votes before the profit-voting gate is evaluated.
# GNB, LR, LR_BAL remain diagnostic-only (never promoted).
RUTHLESS_ACTIVE_MODELS    = ["rf", "et", "hgbc_l2", "lgbm_bal", "catboost_bal", "lgbm_dart", "gbc", "ada"]
RUTHLESS_DIAGNOSTIC_MODELS = ["gnb", "lr", "lr_bal", "cal_et", "cal_rf"]
RUTHLESS_SHADOW_MODELS     = ["et_shallow", "rf_shallow", "xgb_bal"]

# ── Profit-voting gate thresholds ────────────────────────────────────────
# Bootstrap / test defaults — intentionally relaxed to restore trading and gather
# candidate journal / reject diagnostic data.  Tighten after analysing results.
# vote_yes_fraction = fraction of active models with P >= RUTHLESS_VOTE_THRESHOLD.
# top3_mean         = mean of the top-3 active model probabilities.
RUTHLESS_VOTE_THRESHOLD         = 0.45   # per-model yes/no split threshold
RUTHLESS_VOTE_YES_FRACTION_MIN  = 0.30   # min yes-fraction for trend/pump entries
RUTHLESS_TOP3_MEAN_MIN          = 0.45   # min top-3 mean for trend/pump entries
RUTHLESS_VOTE_EV_FLOOR          = 0.001  # minimum EV required for profit-voting pass

# ── Chop supermajority requirements ──────────────────────────────────────
# Chop entries require a stricter vote than trend (still bootstrap-relaxed).
RUTHLESS_CHOP_VOTE_YES_FRAC_MIN = 0.40   # yes-fraction in chop
RUTHLESS_CHOP_TOP3_MEAN_MIN     = 0.50   # top-3-mean required in chop
RUTHLESS_CHOP_PRED_RETURN_MIN   = 0.000  # positive pred_return (disabled)
RUTHLESS_CHOP_EV_MIN            = 0.002  # EV floor required in chop

# ── Ruthless payoff floors ───────────────────────────────────────────────
# Prevent ruthless mode from scalping tiny +0.2% wins (fixes collapsed avg_win).
# These are MINIMUM values; ATR-based TP/SL may be larger.
RUTHLESS_MIN_TP                 = 0.04   # do not intentionally target wins < +4 %
# Trail and timeout parameters are already in the ruthless section above (v4).

# ── Multi-position scaffold ──────────────────────────────────────────────
# SAFE SCAFFOLD — current implementation is single-position only.
# Set RUTHLESS_MAX_CONCURRENT_POSITIONS = 1 for current behavior.
# Set to 2 to opt into experimental 2-position mode (TODO: full multi-pos refactor).
RUTHLESS_MAX_CONCURRENT_POSITIONS = 1    # scaffold only; >1 requires refactor
RUTHLESS_MAX_NEW_ENTRIES_PER_DAY  = 3    # reference cap (informational for now)
RUTHLESS_MAX_SYMBOL_ALLOCATION    = 0.45 # reference single-symbol cap (informational)

# ── Candidate journal ────────────────────────────────────────────────────
# When True, each decision cycle records the top-N candidates (selected + skipped)
# for post-hoc analysis of missed opportunities.
PERSIST_CANDIDATE_JOURNAL  = True
CANDIDATE_JOURNAL_TOP_N    = 5     # candidates journaled per cycle
CANDIDATE_JOURNAL_MAX_SIZE = 2000  # rolling memory cap

# ── Shadow model lab ─────────────────────────────────────────────────────
# When True, additional shadow models (ET/RF variants, calibrated variants, etc.)
# are trained and predicted alongside the active ensemble.  Shadow predictions
# are logged but never affect trading.  Disable to reduce training time.
# shadow_lab_extended: also enables gbc/ada/markov/hmm/kmeans/isoforest models.
ENABLE_SHADOW_MODEL_LAB    = True
SHADOW_MODEL_MAX_COUNT     = 16   # increased cap for new shadow+diagnostic models
ENABLE_SHADOW_LAB_EXTENDED = True  # enable gbc/ada/regime diagnostic models

# ── Model health diagnostics ─────────────────────────────────────────────
# Track per-model rolling probability statistics to flag degenerate models.
# Flags: degenerate_bullish | degenerate_bearish | low_variance
MODEL_HEALTH_ENABLED        = True
MODEL_HEALTH_MIN_OBS        = 20    # minimum observations before flagging
MODEL_HEALTH_EXTREME_PROBA  = 0.95  # threshold for "extreme" probability
MODEL_HEALTH_DEGENERATE_FRAC = 0.90 # fraction of obs above/below that triggers flag
MODEL_HEALTH_LOW_STD        = 0.01  # std below this → low_variance flag

# ── Optional active std/disagreement gate ────────────────────────────────
# Disabled by default — trade count is already low; enabling would reduce it.
# When enabled, ruthless ML entries are blocked if active_std > threshold.
RUTHLESS_USE_ACTIVE_STD_GATE   = False
RUTHLESS_MAX_ACTIVE_STD_PROBA  = 0.30

# ── Trade journal config ─────────────────────────────────────────────────
# persist_trade_journal: when True, journal records accumulate in memory and
# are available via self._trade_journal.get_records() / to_json().
PERSIST_TRADE_JOURNAL     = True
TRADE_JOURNAL_MAX_SIZE    = 500   # rolling cap — oldest records dropped first

# ── Vote log config ──────────────────────────────────────────────────────
# log_model_votes: when True, each entry emits a compact [vote] line listing
# per-model probabilities.  Disabled by default to protect QC's 100KB log cap.
LOG_MODEL_VOTES           = False

# ── Ruthless good-market-mode relaxation ─────────────────────────────────
# When market_mode is pump or risk_on_trend, slightly relax ruthless gates to
# increase sample size without returning to chop overtrading.
# Set RUTHLESS_GOOD_MODE_RELAXATION = False to disable all relaxation.
RUTHLESS_GOOD_MODE_RELAXATION         = True
RUTHLESS_GOOD_MODE_META_MIN_PROBA     = 0.48   # relaxed from 0.52
RUTHLESS_GOOD_MODE_MIN_EV             = 0.002   # relaxed confirm ev
RUTHLESS_GOOD_MODE_VOLUME_MIN         = 1.0    # relaxed from 1.3

# ── Ruthless anti-chop: 2-SL-in-24h block ────────────────────────────────
RUTHLESS_LOSS_WINDOW_HOURS       = 12     # rolling window for counting SL exits
RUTHLESS_LOSS_LIMIT              = 4      # SL exits in window that trigger long block
RUTHLESS_LOSS_BLOCK_HOURS        = 6      # block duration when limit exceeded

# ── Ruthless portfolio loss-streak brake ─────────────────────────────────
RUTHLESS_PORTFOLIO_LOSS_STREAK   = 6     # consecutive losses before pause
RUTHLESS_PORTFOLIO_PAUSE_HOURS   = 2     # hours to pause all new entries

# ── Ruthless confirmation gate thresholds ────────────────────────────────
# Ruthless entries must pass one of three confirmation paths:
#   momentum_override | strong_ml | trend_momentum
RUTHLESS_CONFIRM_EV_MIN          = 0.002  # strong_ml: minimum ev_score
RUTHLESS_CONFIRM_PROBA_MIN       = 0.52   # strong_ml: minimum class_proba
RUTHLESS_CONFIRM_AGREE_MIN       = 2      # strong_ml: minimum n_agree
RUTHLESS_CONFIRM_RET4_MIN        = 0.004  # trend_momentum: minimum ret_4
RUTHLESS_CONFIRM_RET16_MIN       = 0.020  # trend_momentum: minimum ret_16
RUTHLESS_CONFIRM_VOLR_MIN        = 1.5    # trend_momentum: minimum vol_r

# ── Ruthless label parameters ────────────────────────────────────────────
RUTHLESS_LABEL_TP                = 0.035  # achievable TP for ruthless training labels
RUTHLESS_LABEL_SL                = 0.015  # matched SL for ruthless training labels
RUTHLESS_LABEL_HORIZON_HOURS     = 12     # 12h horizon — more labels fire within window
# Derived bars: horizon_hours × (60 / DECISION_INTERVAL_MIN) = 12×4 = 48 bars
RUTHLESS_LABEL_HORIZON_BARS      = 48

# ── Momentum breakout override ───────────────────────────────────────────
# Enabled by default for aggressive/ruthless profiles; disabled otherwise.
MOMENTUM_RET4_MIN          = 0.015   # minimum 4-bar return for override
MOMENTUM_RET16_MIN         = 0.025   # minimum 16-bar return for override
MOMENTUM_VOLUME_MIN        = 2.0     # minimum volume ratio (current / 15-bar avg)
MOMENTUM_BTC_REL_MIN       = 0.005   # minimum BTC-relative 4-bar outperformance
MOMENTUM_OVERRIDE_MIN_EV   = -0.002  # momentum override blocked if EV < this threshold

# ── External profile modules ─────────────────────────────────────────────
import active_research_config as _ar  # noqa: E402
import gatling_config as _gatling  # noqa: E402
# Re-export all active_research constants (used by entry_logic, tests)
for _k, _v in vars(_ar).items():
    if _k.startswith("ACTIVE_RESEARCH_"):
        globals()[_k] = _v


def setup_risk_profile(algo):
    """Resolve risk profile from QC parameters and apply gate/sizing overrides.

    Reads ruthless_mode / aggressive_mode / conservative_mode / risk_profile
    parameters, determines the effective profile, applies the corresponding
    overrides to algo's trading attributes, and configures momentum override
    settings.  Call once inside initialize() after base parameters are set.
    """
    _rp_raw          = algo.get_parameter("risk_profile")
    _rm_raw          = algo.get_parameter("ruthless_mode")
    _am_raw          = algo.get_parameter("aggressive_mode")
    _cm_raw          = algo.get_parameter("conservative_mode")
    _ruthless_mode   = (str(_rm_raw).lower() in ("true", "1", "yes")) if _rm_raw else False
    _aggressive_mode = (str(_am_raw).lower() in ("true", "1", "yes")) if _am_raw else False
    algo._conservative_mode = (
        str(_cm_raw).lower() in ("true", "1", "yes")
    ) if _cm_raw else CONSERVATIVE_MODE

    # Detect explicit V2 mode request (via ruthless_v2_mode param or risk_profile=ruthless_v2)
    _v2_raw = algo.get_parameter("ruthless_v2_mode")
    _v2_explicit = (str(_v2_raw).lower() in ("true", "1", "yes")) if _v2_raw else False

    # Priority: ruthless_mode > aggressive_mode > conservative_mode
    #           > risk_profile param > RISK_PROFILE default
    if _ruthless_mode:
        algo._risk_profile = "ruthless"
    elif _aggressive_mode:
        algo._risk_profile = "aggressive"
    elif algo._conservative_mode:
        algo._risk_profile = "conservative"
    elif _rp_raw:
        _rp_val = _rp_raw.lower().strip()
        if _rp_val in ("conservative", "balanced", "aggressive", "ruthless",
                       "ruthless_v2", "apex_predator", "active_research",
                       "gatling"):
            algo._risk_profile = _rp_val
            if _rp_val == "conservative":
                algo._conservative_mode = True
            if _rp_val == "ruthless_v2":
                _v2_explicit = True
        else:
            algo._risk_profile = RISK_PROFILE
    else:
        algo._risk_profile = RISK_PROFILE

    # Normalize ruthless_v2 -> ruthless.
    # V2 is an extension of ruthless, not a separate profile branch.
    # The profile is normalized to "ruthless" first so all ruthless V1 settings
    # are applied; the V2 flag then determines the behavior variant.
    if algo._risk_profile == "ruthless_v2":
        algo._risk_profile = "ruthless"

    # apex_predator: run as ruthless first, then override gates afterward.
    _apex_mode = algo._risk_profile == "apex_predator"
    if _apex_mode:
        algo._risk_profile = "ruthless"

    # Store V2 mode flag on algo — used by execution path.
    # V2 is only meaningful when combined with ruthless profile.
    algo._ruthless_v2_mode = _v2_explicit and (algo._risk_profile == "ruthless")

    if algo._risk_profile == "conservative":
        algo._conservative_mode = True
        algo._s_min            = max(algo._s_min,            0.55)
        algo._max_disp         = min(algo._max_disp,          0.15)
        algo._min_agr          = max(algo._min_agr,           3)
        algo._min_ev           = max(algo._min_ev,            0.004)
        algo._cost_bps         = max(algo._cost_bps,          50.0)
        algo._pred_return_min  = max(algo._pred_return_min,   0.003)
        algo._max_sl           = min(algo._max_sl,            1)
        algo._cd_mins          = max(algo._cd_mins,           30)
        algo._tp               = max(algo._tp,                0.035)
        algo._sl               = max(algo._sl,                0.020)
        algo._min_hold_minutes = max(algo._min_hold_minutes,  20)
        algo.log("[vox] Conservative mode enabled: stricter gate overrides applied.")

    elif algo._risk_profile == "aggressive":
        algo._s_min            = AGGRESSIVE_SCORE_MIN
        algo._min_ev           = AGGRESSIVE_MIN_EV
        algo._pred_return_min  = AGGRESSIVE_PRED_RETURN_MIN
        algo._max_disp         = AGGRESSIVE_MAX_DISPERSION
        algo._min_agr          = AGGRESSIVE_MIN_AGREE
        algo._ev_gap           = AGGRESSIVE_EV_GAP
        algo._cost_bps         = AGGRESSIVE_COST_BPS
        algo._alloc            = AGGRESSIVE_ALLOCATION
        algo._max_alloc        = AGGRESSIVE_MAX_ALLOC
        algo._kf               = AGGRESSIVE_KELLY_FRAC
        algo._tp               = AGGRESSIVE_TAKE_PROFIT
        algo._sl               = AGGRESSIVE_STOP_LOSS
        algo._toh              = AGGRESSIVE_TIMEOUT_HOURS
        algo._min_hold_minutes = AGGRESSIVE_MIN_HOLD_MINUTES
        algo._emergency_sl     = AGGRESSIVE_EMERGENCY_SL
        algo._max_sl           = AGGRESSIVE_MAX_DAILY_SL
        algo._cd_mins          = AGGRESSIVE_COOLDOWN_MINS
        algo._sl_cd            = AGGRESSIVE_SL_COOLDOWN_MINS
        algo._penalty_losses   = AGGRESSIVE_PENALTY_COOLDOWN_LOSSES
        algo._penalty_hours    = AGGRESSIVE_PENALTY_COOLDOWN_HOURS
        algo._max_dd           = AGGRESSIVE_MAX_DD_PCT
        algo.log("[vox] Aggressive mode enabled: high-risk/high-upside settings active.")

    elif algo._risk_profile == "active_research":
        _a = _ar  # shorthand
        algo._s_min = _a.ACTIVE_RESEARCH_SCORE_MIN
        algo._ev_gap = _a.ACTIVE_RESEARCH_SCORE_GAP
        algo._min_agr = _a.ACTIVE_RESEARCH_MIN_AGREE
        algo._max_disp = _a.ACTIVE_RESEARCH_MAX_DISPERSION
        algo._min_ev = _a.ACTIVE_RESEARCH_MIN_EV
        algo._pred_return_min = _a.ACTIVE_RESEARCH_PRED_RETURN_MIN
        algo._cd_mins = _a.ACTIVE_RESEARCH_COOLDOWN_MINS
        algo._sl_cd = _a.ACTIVE_RESEARCH_SL_COOLDOWN_MINS
        algo._max_sl = _a.ACTIVE_RESEARCH_MAX_DAILY_SL
        algo._alloc = _a.ACTIVE_RESEARCH_ALLOCATION
        algo._max_alloc = _a.ACTIVE_RESEARCH_MAX_ALLOC
        algo._min_alloc = _a.ACTIVE_RESEARCH_MIN_ALLOC
        algo._use_kelly = _a.ACTIVE_RESEARCH_USE_KELLY
        algo._tp = _a.ACTIVE_RESEARCH_TAKE_PROFIT
        algo._sl = _a.ACTIVE_RESEARCH_STOP_LOSS
        algo._toh = _a.ACTIVE_RESEARCH_TIMEOUT_HOURS
        algo._min_hold_minutes = _a.ACTIVE_RESEARCH_MIN_HOLD_MINUTES
        algo._emergency_sl = _a.ACTIVE_RESEARCH_EMERGENCY_SL
        algo._penalty_losses = _a.ACTIVE_RESEARCH_PENALTY_LOSSES
        algo._penalty_hours = _a.ACTIVE_RESEARCH_PENALTY_HOURS
        algo._max_dd = _a.ACTIVE_RESEARCH_MAX_DD_PCT
        algo.log("[active_research] DATA-COLLECTION mode — loose gates.")

    elif algo._risk_profile == "gatling":
        # ── GATLING: extremely active gatling-gun for max trade frequency ──
        _g = _gatling
        algo._s_min = _g.GATLING_SCORE_MIN; algo._min_ev = _g.GATLING_MIN_EV
        algo._pred_return_min = _g.GATLING_PRED_RETURN_MIN
        algo._max_disp = _g.GATLING_MAX_DISPERSION; algo._min_agr = _g.GATLING_MIN_AGREE
        algo._ev_gap = _g.GATLING_EV_GAP; algo._cost_bps = _g.GATLING_COST_BPS
        algo._alloc = _g.GATLING_ALLOCATION; algo._max_alloc = _g.GATLING_MAX_ALLOC
        algo._min_alloc = _g.GATLING_MIN_ALLOC; algo._use_kelly = _g.GATLING_USE_KELLY
        algo._kf = _g.GATLING_KELLY_FRAC
        algo._tp = _g.GATLING_TAKE_PROFIT; algo._sl = _g.GATLING_STOP_LOSS
        algo._toh = _g.GATLING_TIMEOUT_HOURS
        algo._min_hold_minutes = _g.GATLING_MIN_HOLD_MINUTES
        algo._emergency_sl = _g.GATLING_EMERGENCY_SL
        algo._cd_mins = _g.GATLING_COOLDOWN_MINS; algo._sl_cd = _g.GATLING_SL_COOLDOWN_MINS
        algo._penalty_losses = _g.GATLING_PENALTY_COOLDOWN_LOSSES
        algo._penalty_hours = _g.GATLING_PENALTY_COOLDOWN_HOURS
        algo._max_sl = _g.GATLING_MAX_DAILY_SL; algo._max_dd = _g.GATLING_MAX_DD_PCT
        algo._runner_mode = _g.GATLING_RUNNER_MODE
        algo._trail_after_tp = _g.GATLING_TRAIL_AFTER_TP
        algo._trail_pct = _g.GATLING_TRAIL_PCT
        algo._ruthless_loss_window_hours = _g.GATLING_LOSS_WINDOW_HOURS
        algo._ruthless_loss_limit = _g.GATLING_LOSS_LIMIT
        algo._ruthless_loss_block_hours = _g.GATLING_LOSS_BLOCK_HOURS
        algo._ruthless_portfolio_loss_streak = _g.GATLING_PORTFOLIO_LOSS_STREAK
        algo._ruthless_portfolio_pause_hours = _g.GATLING_PORTFOLIO_PAUSE_HOURS
        algo._ruthless_confirm_ev_min = _g.GATLING_CONFIRM_EV_MIN
        algo._ruthless_confirm_proba_min = _g.GATLING_CONFIRM_PROBA_MIN
        algo._ruthless_confirm_agree_min = _g.GATLING_CONFIRM_AGREE_MIN
        algo._ruthless_confirm_ret4_min = _g.GATLING_CONFIRM_RET4_MIN
        algo._ruthless_confirm_ret16_min = _g.GATLING_CONFIRM_RET16_MIN
        algo._ruthless_confirm_volr_min = _g.GATLING_CONFIRM_VOLR_MIN
        algo._label_tp = _g.GATLING_LABEL_TP; algo._label_sl = _g.GATLING_LABEL_SL
        algo._label_horizon = _g.GATLING_LABEL_HORIZON_BARS
        algo._ruthless_profit_voting_mode = _g.GATLING_PROFIT_VOTING_MODE
        algo._ruthless_active_models = list(_g.GATLING_ACTIVE_MODELS)
        algo._ruthless_diagnostic_models = list(_g.GATLING_DIAGNOSTIC_MODELS)
        algo._ruthless_shadow_models = list(_g.GATLING_SHADOW_MODELS)
        algo._pv_vote_threshold = _g.GATLING_VOTE_THRESHOLD
        algo._pv_vote_yes_frac_min = _g.GATLING_VOTE_YES_FRACTION_MIN
        algo._pv_top3_mean_min = _g.GATLING_TOP3_MEAN_MIN
        algo._pv_vote_ev_floor = _g.GATLING_VOTE_EV_FLOOR
        algo._pv_chop_yes_frac_min = _g.GATLING_CHOP_VOTE_YES_FRAC_MIN
        algo._pv_chop_top3_mean_min = _g.GATLING_CHOP_TOP3_MEAN_MIN
        algo._pv_chop_pred_return_min = _g.GATLING_CHOP_PRED_RETURN_MIN
        algo._pv_chop_ev_min = _g.GATLING_CHOP_EV_MIN
        algo._meta_filter_enabled = _g.GATLING_META_FILTER_ENABLED
        algo._meta_min_proba = _g.GATLING_META_MIN_PROBA
        algo._market_mode_enabled = _g.GATLING_MARKET_MODE_ENABLED
        algo._ruthless_allowed_modes = list(_g.GATLING_ALLOWED_MODES)
        algo._breakeven_after = _g.GATLING_BREAKEVEN_AFTER
        algo._breakeven_buffer = _g.GATLING_BREAKEVEN_BUFFER
        algo._mom_fail_enabled = _g.GATLING_MOM_FAIL_ENABLED
        algo._mom_fail_min_hold = _g.GATLING_MOM_FAIL_MIN_HOLD
        algo._mom_fail_loss = _g.GATLING_MOM_FAIL_LOSS
        algo._timeout_min_profit = _g.GATLING_TIMEOUT_MIN_PROFIT
        algo._timeout_extend_hours = _g.GATLING_TIMEOUT_EXTEND_HOURS
        algo._max_timeout_hours = _g.GATLING_MAX_TIMEOUT_HOURS
        algo._good_mode_relaxation = True
        algo._good_mode_meta_min_proba = 0.0
        algo._good_mode_min_ev = -1.0; algo._good_mode_volume_min = 0.0
        algo._gatling_decision_interval = _g.GATLING_DECISION_INTERVAL_MIN
        algo._gatling_track_model_accuracy = _g.GATLING_TRACK_MODEL_ACCURACY
        algo._ruthless_min_tp = 0.0
        algo._log_model_votes = True  # force vote logging for model assessment
        # Apply gatling model weights (winning models weighted higher)
        try:
            from gatling_config import GATLING_MODEL_WEIGHTS
            algo._ensemble_model_weights = dict(GATLING_MODEL_WEIGHTS)
        except Exception:
            pass
        # Regime-adaptive sizing flag
        algo._gatling_regime_sizing = _g.GATLING_REGIME_SIZING if hasattr(_g, 'GATLING_REGIME_SIZING') else False
        algo.log(
            "[gatling] GATLING GUN mode active — 5-min decisions, all gates near-zero."
            " Vote logging ON for model assessment."
        )

    elif algo._risk_profile == "ruthless":
        algo._s_min            = RUTHLESS_SCORE_MIN
        algo._min_ev           = RUTHLESS_MIN_EV
        algo._pred_return_min  = RUTHLESS_PRED_RETURN_MIN
        algo._max_disp         = RUTHLESS_MAX_DISPERSION
        algo._min_agr          = RUTHLESS_MIN_AGREE
        algo._ev_gap           = RUTHLESS_EV_GAP
        algo._cost_bps         = RUTHLESS_COST_BPS
        algo._alloc            = RUTHLESS_ALLOCATION
        algo._max_alloc        = RUTHLESS_MAX_ALLOC
        algo._kf               = RUTHLESS_KELLY_FRAC
        # Sizing floor + Kelly override — prevent Kelly from shrinking positions
        # below 75 % of portfolio.  Default: Kelly disabled for flat 90 % sizing.
        if getattr(algo, "_min_alloc", 0.0) == 0.0:
            algo._min_alloc    = RUTHLESS_MIN_ALLOC
        _uk_raw = algo.get_parameter("use_kelly")
        if not _uk_raw:   # no explicit QC override → use ruthless default
            algo._use_kelly = RUTHLESS_USE_KELLY
        algo._tp               = RUTHLESS_TAKE_PROFIT
        algo._sl               = RUTHLESS_STOP_LOSS
        algo._toh              = RUTHLESS_TIMEOUT_HOURS
        algo._min_hold_minutes = RUTHLESS_MIN_HOLD_MINUTES
        algo._emergency_sl     = RUTHLESS_EMERGENCY_SL
        algo._max_sl           = RUTHLESS_MAX_DAILY_SL
        algo._cd_mins          = RUTHLESS_COOLDOWN_MINS
        algo._sl_cd            = RUTHLESS_SL_COOLDOWN_MINS
        algo._penalty_losses   = RUTHLESS_PENALTY_COOLDOWN_LOSSES
        algo._penalty_hours    = RUTHLESS_PENALTY_COOLDOWN_HOURS
        algo._max_dd           = RUTHLESS_MAX_DD_PCT
        # Runner mode — trailing stop replaces instant TP exit
        if not getattr(algo, "_runner_mode", False):
            _rm_param = algo.get_parameter("runner_mode")
            algo._runner_mode = (
                str(_rm_param).lower() in ("true", "1", "yes")
                if _rm_param else RUTHLESS_RUNNER_MODE
            )
        if not getattr(algo, "_trail_after_tp", None):
            algo._trail_after_tp = RUTHLESS_TRAIL_AFTER_TP
        if not getattr(algo, "_trail_pct", None):
            algo._trail_pct      = RUTHLESS_TRAIL_PCT
        # Anti-chop: per-symbol 2-SL-in-24h extended block
        algo._ruthless_loss_window_hours = RUTHLESS_LOSS_WINDOW_HOURS
        algo._ruthless_loss_limit        = RUTHLESS_LOSS_LIMIT
        algo._ruthless_loss_block_hours  = RUTHLESS_LOSS_BLOCK_HOURS
        # Portfolio loss-streak brake
        algo._ruthless_portfolio_loss_streak = RUTHLESS_PORTFOLIO_LOSS_STREAK
        algo._ruthless_portfolio_pause_hours = RUTHLESS_PORTFOLIO_PAUSE_HOURS
        # Ruthless confirmation gate thresholds
        algo._ruthless_confirm_ev_min    = RUTHLESS_CONFIRM_EV_MIN
        algo._ruthless_confirm_proba_min = RUTHLESS_CONFIRM_PROBA_MIN
        algo._ruthless_confirm_agree_min = RUTHLESS_CONFIRM_AGREE_MIN
        algo._ruthless_confirm_ret4_min  = RUTHLESS_CONFIRM_RET4_MIN
        algo._ruthless_confirm_ret16_min = RUTHLESS_CONFIRM_RET16_MIN
        algo._ruthless_confirm_volr_min  = RUTHLESS_CONFIRM_VOLR_MIN
        # Ruthless label parameters — override defaults for profile-aligned training
        algo._label_tp      = RUTHLESS_LABEL_TP
        algo._label_sl      = RUTHLESS_LABEL_SL
        algo._label_horizon = RUTHLESS_LABEL_HORIZON_BARS
        # Ruthless v4 new parameters
        algo._breakeven_after         = RUTHLESS_BREAKEVEN_AFTER
        algo._breakeven_buffer        = RUTHLESS_BREAKEVEN_BUFFER
        algo._mom_fail_enabled        = RUTHLESS_MOM_FAIL_ENABLED
        algo._mom_fail_min_hold       = RUTHLESS_MOM_FAIL_MIN_HOLD_MINUTES
        algo._mom_fail_loss           = RUTHLESS_MOM_FAIL_LOSS
        algo._timeout_min_profit      = RUTHLESS_TIMEOUT_MIN_PROFIT
        algo._timeout_extend_hours    = RUTHLESS_TIMEOUT_EXTEND_HOURS
        algo._max_timeout_hours       = RUTHLESS_MAX_TIMEOUT_HOURS
        algo._use_entry_limit_orders  = RUTHLESS_USE_ENTRY_LIMIT_ORDERS
        algo._market_mode_enabled     = MARKET_MODE_ENABLED
        algo._ruthless_allowed_modes  = RUTHLESS_ALLOWED_MODES
        algo._meta_filter_enabled     = RUTHLESS_META_FILTER_ENABLED
        algo._meta_min_proba          = RUTHLESS_META_MIN_PROBA
        # Good-market-mode relaxation
        algo._good_mode_relaxation          = RUTHLESS_GOOD_MODE_RELAXATION
        algo._good_mode_meta_min_proba      = RUTHLESS_GOOD_MODE_META_MIN_PROBA
        algo._good_mode_min_ev              = RUTHLESS_GOOD_MODE_MIN_EV
        algo._good_mode_volume_min          = RUTHLESS_GOOD_MODE_VOLUME_MIN
        # Profit-voting mode — explicit vote-ranking instead of cautious thresholds
        _pv_raw = algo.get_parameter("ruthless_profit_voting_mode")
        if _pv_raw is not None:
            algo._ruthless_profit_voting_mode = str(_pv_raw).lower() in ("true","1","yes")
        else:
            algo._ruthless_profit_voting_mode = RUTHLESS_PROFIT_VOTING_MODE
        # Active / diagnostic / shadow model lists for ruthless promotion
        algo._ruthless_active_models      = list(RUTHLESS_ACTIVE_MODELS)
        algo._ruthless_diagnostic_models  = list(RUTHLESS_DIAGNOSTIC_MODELS)
        algo._ruthless_shadow_models      = list(RUTHLESS_SHADOW_MODELS)
        # Payoff floor — ruthless should not scalp tiny wins
        algo._ruthless_min_tp = RUTHLESS_MIN_TP
        # Profit-voting gate thresholds
        algo._pv_vote_threshold        = RUTHLESS_VOTE_THRESHOLD
        algo._pv_vote_yes_frac_min     = RUTHLESS_VOTE_YES_FRACTION_MIN
        algo._pv_top3_mean_min         = RUTHLESS_TOP3_MEAN_MIN
        algo._pv_vote_ev_floor         = RUTHLESS_VOTE_EV_FLOOR
        # Chop supermajority thresholds
        algo._pv_chop_yes_frac_min     = RUTHLESS_CHOP_VOTE_YES_FRAC_MIN
        algo._pv_chop_top3_mean_min    = RUTHLESS_CHOP_TOP3_MEAN_MIN
        algo._pv_chop_pred_return_min  = RUTHLESS_CHOP_PRED_RETURN_MIN
        algo._pv_chop_ev_min           = RUTHLESS_CHOP_EV_MIN
        # Enforce TP floor for ruthless (prevents tiny scalp exits)
        if algo._tp < RUTHLESS_MIN_TP:
            algo._tp = max(algo._tp, RUTHLESS_MIN_TP)
        # ── Ruthless V2 engine initialization ────────────────────────────────
        # V2 adds multi-position management, dynamic voter weighting, and
        # multi-horizon scoring.  It activates only when _ruthless_v2_mode=True.
        if getattr(algo, "_ruthless_v2_mode", False):
            try:
                from strategy_ext import (
                    MultiPositionManager,
                    DynamicVoterWeighting,
                    RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
                    RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
                    RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY,
                    RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
                    RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
                    RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
                    RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
                    RUTHLESS_V2_ACTIVE_MODELS,
                    RUTHLESS_V2_BASE_WEIGHTS,
                    RUTHLESS_V2_MACHINE_GUN_MODE,
                    RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES,
                    RUTHLESS_V2_MIN_SCORE_TO_TRADE,
                    RUTHLESS_V2_REGIME_HARD_BLOCK,
                    RUTHLESS_V2_META_HARD_FILTER,
                    RUTHLESS_V2_META_AS_SCORE_PENALTY,
                    RUTHLESS_V2_ALLOW_CHOP_SCALPS,
                    RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC,
                    format_v2_startup_log,
                )
                # ── QC parameter overrides for machine-gun mode ───────────────
                def _bool_param(name, default):
                    raw = algo.get_parameter(name)
                    if raw is not None:
                        return str(raw).lower() in ("true", "1", "yes")
                    return default

                def _float_param(name, default):
                    raw = algo.get_parameter(name)
                    if raw is not None:
                        try:
                            return float(raw)
                        except (ValueError, TypeError):
                            pass
                    return default

                def _int_param(name, default):
                    raw = algo.get_parameter(name)
                    if raw is not None:
                        try:
                            return int(raw)
                        except (ValueError, TypeError):
                            pass
                    return default

                algo._v2_machine_gun_mode = _bool_param(
                    "ruthless_v2_machine_gun_mode", RUTHLESS_V2_MACHINE_GUN_MODE)
                algo._v2_force_top_n = _int_param(
                    "ruthless_v2_force_top_n_when_candidates", RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES)
                algo._v2_min_score_to_trade = _float_param(
                    "ruthless_v2_min_score_to_trade", RUTHLESS_V2_MIN_SCORE_TO_TRADE)
                algo._v2_regime_hard_block = _bool_param(
                    "ruthless_v2_regime_hard_block", RUTHLESS_V2_REGIME_HARD_BLOCK)
                algo._v2_meta_hard_filter = _bool_param(
                    "ruthless_v2_meta_hard_filter", RUTHLESS_V2_META_HARD_FILTER)
                algo._v2_meta_as_score_penalty = _bool_param(
                    "ruthless_v2_meta_as_score_penalty", RUTHLESS_V2_META_AS_SCORE_PENALTY)
                algo._v2_allow_chop_scalps = _bool_param(
                    "ruthless_v2_allow_chop_scalps", RUTHLESS_V2_ALLOW_CHOP_SCALPS)

                algo._v2_position_mgr = MultiPositionManager(
                    max_concurrent=RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
                    max_new_per_day=RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
                    max_per_symbol_per_day=RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY,
                    max_symbol_alloc=RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
                    min_symbol_alloc=RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
                    max_total_exposure=RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
                    reentry_cooldown_min=RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
                )
                algo._v2_voter_weighting = DynamicVoterWeighting(
                    base_weights=RUTHLESS_V2_BASE_WEIGHTS,
                )
                # Override active model list with V2 aggressive pool
                algo._ruthless_active_models = list(RUTHLESS_V2_ACTIVE_MODELS)
                # Emit machine-gun startup log
                for line in format_v2_startup_log(
                    risk_profile="ruthless",
                    v2_mode=True,
                    max_positions=RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
                    active_models=RUTHLESS_V2_ACTIVE_MODELS,
                    machine_gun_mode=algo._v2_machine_gun_mode,
                    regime_hard_block=algo._v2_regime_hard_block,
                    meta_hard_filter=algo._v2_meta_hard_filter,
                    force_top_n=algo._v2_force_top_n,
                    min_score_to_trade=algo._v2_min_score_to_trade,
                ):
                    algo.log(line)
            except ImportError:
                algo.log("[profile] WARNING: ruthless_v2 module not found; V2 disabled.")
                algo._ruthless_v2_mode = False

    # ── Apex Predator gate overrides (applied after ruthless base setup) ─────
    if _apex_mode:
        algo._risk_profile   = "apex_predator"
        algo._apex_predator_mode = True
        # Entry gates — ultra-loose to break the "predict NO" equilibrium
        algo._s_min                          = APEX_PROFILE_SCORE_MIN
        algo._pred_return_min                = APEX_PROFILE_PRED_RETURN_MIN
        algo._sl_cd                          = APEX_PROFILE_SL_COOLDOWN_MINS
        algo._meta_min_proba                 = APEX_PROFILE_META_MIN_PROBA
        algo._good_mode_meta_min_proba       = APEX_PROFILE_GOOD_MODE_META_MIN_PROBA
        algo._good_mode_min_ev               = 0.0
        algo._good_mode_volume_min           = APEX_PROFILE_GOOD_MODE_VOLUME_MIN
        algo._ruthless_loss_window_hours     = APEX_PROFILE_LOSS_WINDOW_HOURS
        algo._ruthless_loss_limit            = APEX_PROFILE_LOSS_LIMIT
        algo._ruthless_loss_block_hours      = APEX_PROFILE_LOSS_BLOCK_HOURS
        algo._ruthless_portfolio_loss_streak = APEX_PROFILE_PORTFOLIO_LOSS_STREAK
        algo._ruthless_portfolio_pause_hours = APEX_PROFILE_PORTFOLIO_PAUSE_HOURS
        algo._ruthless_confirm_ev_min        = APEX_PROFILE_CONFIRM_EV_MIN
        algo._ruthless_confirm_proba_min     = APEX_PROFILE_CONFIRM_PROBA_MIN
        algo._ruthless_confirm_agree_min     = 1
        algo._ruthless_confirm_ret4_min      = APEX_PROFILE_CONFIRM_RET4_MIN
        algo._ruthless_confirm_ret16_min     = RUTHLESS_CONFIRM_RET16_MIN * 0.3
        algo._ruthless_confirm_volr_min      = 1.0
        algo._label_tp                       = APEX_PROFILE_LABEL_TP
        algo._label_sl                       = APEX_PROFILE_LABEL_SL
        algo._label_horizon                  = APEX_PROFILE_LABEL_HORIZON_BARS
        algo._pv_vote_threshold              = APEX_PROFILE_VOTE_THRESHOLD
        algo._pv_vote_yes_frac_min           = APEX_PROFILE_VOTE_YES_FRACTION_MIN
        algo._pv_top3_mean_min               = APEX_PROFILE_TOP3_MEAN_MIN
        algo._pv_vote_ev_floor               = 0.0
        algo._pv_chop_yes_frac_min           = APEX_PROFILE_CHOP_YES_FRAC_MIN
        algo._pv_chop_top3_mean_min          = APEX_PROFILE_CHOP_TOP3_MEAN_MIN
        algo._pv_chop_pred_return_min        = -1.0
        algo._pv_chop_ev_min                 = 0.0

    # ── Momentum override setup ───────────────────────────────────────────────
    _mo_raw = algo.get_parameter("momentum_override")
    if _mo_raw:
        algo._momentum_override = str(_mo_raw).lower() in ("true", "1", "yes")
    else:
        algo._momentum_override = algo._risk_profile in ("aggressive", "ruthless", "apex_predator", "gatling")

    algo._momentum_ret4_min = float(
        algo.get_parameter("momentum_ret4_min") or MOMENTUM_RET4_MIN
    )
    algo._momentum_ret16_min = float(
        algo.get_parameter("momentum_ret16_min") or MOMENTUM_RET16_MIN
    )
    algo._momentum_volume_min = float(
        algo.get_parameter("momentum_volume_min") or MOMENTUM_VOLUME_MIN
    )
    algo._momentum_btc_rel_min = float(
        algo.get_parameter("momentum_btc_rel_min") or MOMENTUM_BTC_REL_MIN
    )
    algo._momentum_override_min_ev = float(
        algo.get_parameter("momentum_override_min_ev") or MOMENTUM_OVERRIDE_MIN_EV
    )
    algo._use_momentum_score = algo._risk_profile in ("aggressive", "ruthless", "apex_predator", "gatling")

    # ── Startup profile audit log (unthrottled — important config visibility) ───
    _runner_mode_val   = getattr(algo, "_runner_mode",   False)
    _trail_after_tp    = getattr(algo, "_trail_after_tp", 0.0)
    _trail_pct         = getattr(algo, "_trail_pct",      0.0)
    _min_alloc         = getattr(algo, "_min_alloc",      0.0)
    _pv_mode           = getattr(algo, "_ruthless_profit_voting_mode", False)
    _v2_mode           = getattr(algo, "_ruthless_v2_mode", False)
    algo.log(
        f"[profile] risk_profile={algo._risk_profile}"
        f" profit_voting={_pv_mode}"
        f" v2={_v2_mode}"
    )
    algo.log(
        f"[profile] tp={algo._tp:.3f} sl={algo._sl:.3f}"
        f" timeout={algo._toh}h"
        f" trail_after={_trail_after_tp:.3f}"
        f" trail_pct={_trail_pct:.3f}"
    )
    algo.log(
        f"[profile] alloc={algo._alloc:.2f}"
        f" min_alloc={_min_alloc:.2f}"
        f" max_alloc={algo._max_alloc:.2f}"
        f" use_kelly={algo._use_kelly}"
        f" runner_mode={_runner_mode_val}"
    )
    # Model role audit (active vs shadow vs diagnostic) — built from config lists
    if algo._risk_profile in ("ruthless", "apex_predator") and getattr(algo, "_ruthless_profit_voting_mode", False):
        _active_str = ",".join(getattr(algo, "_ruthless_active_models", RUTHLESS_ACTIVE_MODELS))
        _diag_str   = ",".join(getattr(algo, "_ruthless_diagnostic_models", RUTHLESS_DIAGNOSTIC_MODELS))
        _shadow_str = ",".join(getattr(algo, "_ruthless_shadow_models", RUTHLESS_SHADOW_MODELS))
        _active_cnt = len(getattr(algo, "_ruthless_active_models", RUTHLESS_ACTIVE_MODELS))
        algo.log(
            f"[profile] effective_active_models={_active_str}"
            f" active_count={_active_cnt}"
        )
        algo.log(
            f"[profile] effective_diagnostic_models={_diag_str}"
            f" effective_shadow_models={_shadow_str}"
        )
    else:
        _active_str = ",".join(getattr(algo, "_ruthless_active_models",
                                       ["rf", "et", "hgbc"]) if algo._risk_profile in ("ruthless", "apex_predator")
                               else ["rf", "et", "hgbc"])
        _diag_str   = ",".join(getattr(algo, "_ruthless_diagnostic_models",
                                       ["gnb", "lr"]) if algo._risk_profile in ("ruthless", "apex_predator")
                               else ["gnb", "lr"])
        algo.log(
            f"[profile] active_models={_active_str}"
            f" diagnostic_models={_diag_str}"
            f" shadow_models=et_shallow,rf_shallow,hgbc_l2,cal_et,cal_rf,lgbm_bal,catboost_bal,lgbm_dart"
        )
    if algo._risk_profile in ("ruthless", "apex_predator"):
        _chop_rule = "supermajority_only" if _pv_mode else "standard"
        algo.log(
            f"[profile] chop_rule={_chop_rule}"
            f" vote_threshold={getattr(algo, '_pv_vote_threshold', RUTHLESS_VOTE_THRESHOLD)}"
            f" yes_frac_min={getattr(algo, '_pv_vote_yes_frac_min', RUTHLESS_VOTE_YES_FRACTION_MIN)}"
            f" top3_mean_min={getattr(algo, '_pv_top3_mean_min', RUTHLESS_TOP3_MEAN_MIN)}"
        )
        algo.log(
            f"[profile] chop_yes_frac_min={getattr(algo, '_pv_chop_yes_frac_min', RUTHLESS_CHOP_VOTE_YES_FRAC_MIN)}"
            f" chop_top3_min={getattr(algo, '_pv_chop_top3_mean_min', RUTHLESS_CHOP_TOP3_MEAN_MIN)}"
            f" chop_ev_min={getattr(algo, '_pv_chop_ev_min', RUTHLESS_CHOP_EV_MIN)}"
        )
        _is_apex = algo._risk_profile == "apex_predator"
        algo.log(
            f"[vox] PROFILE ACTIVE: risk_profile={algo._risk_profile}"
            + (" [APEX PREDATOR — ultra-loose gates]" if _is_apex else "")
            + f"  score_min={algo._s_min}"
            f"  min_ev={algo._min_ev:.5f}"
            f"  pred_return_min={algo._pred_return_min:.5f}"
            f"  allocation={algo._alloc}"
            f"  take_profit={algo._tp}"
            f"  stop_loss={algo._sl}"
            f"  timeout_hours={algo._toh}"
        )
        algo.log(
            f"[vox] {'APEX PREDATOR' if _is_apex else 'RUTHLESS'} v4 ACTIVE:"
            f"  sl_cooldown_mins={algo._sl_cd}"
            f"  loss_window_h={getattr(algo, '_ruthless_loss_window_hours', 24)}"
            f"  loss_limit={getattr(algo, '_ruthless_loss_limit', 2)}"
            f"  loss_block_h={getattr(algo, '_ruthless_loss_block_hours', 24)}"
            f"  portfolio_loss_streak={getattr(algo, '_ruthless_portfolio_loss_streak', 4)}"
            f"  portfolio_pause_h={getattr(algo, '_ruthless_portfolio_pause_hours', 6)}"
            f"  confirm_ev>={getattr(algo, '_ruthless_confirm_ev_min', 0.002)}"
            f"  confirm_proba>={getattr(algo, '_ruthless_confirm_proba_min', 0.52)}"
            f"  confirm_ret4>={getattr(algo, '_ruthless_confirm_ret4_min', 0.004)}"
            f"  label_tp={getattr(algo, '_label_tp', 0.035)}"
            f"  label_sl={getattr(algo, '_label_sl', 0.015)}"
            f"  label_horizon_bars={getattr(algo, '_label_horizon', 48)}"
        )
        algo.log(
            f"[vox] RUTHLESS good-mode-relaxation:"
            f"  enabled={getattr(algo, '_good_mode_relaxation', True)}"
            f"  meta_min_proba={getattr(algo, '_good_mode_meta_min_proba', RUTHLESS_GOOD_MODE_META_MIN_PROBA)}"
            f"  min_ev={getattr(algo, '_good_mode_min_ev', RUTHLESS_GOOD_MODE_MIN_EV)}"
            f"  volume_min={getattr(algo, '_good_mode_volume_min', RUTHLESS_GOOD_MODE_VOLUME_MIN)}"
        )
        algo.log(
            f"[vox] RUTHLESS entry_limit_orders:"
            f"  use={getattr(algo, '_use_entry_limit_orders', False)}"
            f"  offset={ENTRY_LIMIT_OFFSET}"
            f"  ttl_min={ENTRY_LIMIT_TTL_MINUTES}"
        )
    else:
        algo.log(
            f"[vox] PROFILE ACTIVE: risk_profile={algo._risk_profile}"
            f"  score_min={algo._s_min}"
            f"  min_ev={algo._min_ev:.5f}"
            f"  pred_return_min={algo._pred_return_min:.5f}"
            f"  allocation={algo._alloc}"
            f"  min_alloc={_min_alloc}"
            f"  max_alloc={algo._max_alloc}"
            f"  use_kelly={algo._use_kelly}"
            f"  take_profit={algo._tp}"
            f"  stop_loss={algo._sl}"
            f"  timeout_hours={algo._toh}"
            f"  runner_mode={_runner_mode_val}"
            f"  trail_after_tp={_trail_after_tp}"
            f"  trail_pct={_trail_pct}"
        )
        if algo._risk_profile == "active_research":
            algo.log(
                f"[active_research] relaxed thresholds:"
                f"  score_min={algo._s_min}"
                f"  score_gap={algo._ev_gap}"
                f"  min_agree={algo._min_agr}"
                f"  max_disp={algo._max_disp}"
                f"  min_ev={algo._min_ev:.5f}"
                f"  pred_return_min={algo._pred_return_min:.5f}"
                f"  cooldown={algo._cd_mins}min"
                f"  sl_cooldown={algo._sl_cd}min"
                f"  max_daily_sl={algo._max_sl}"
                f"  alloc={algo._alloc:.3f}"
                f"  regime=soft_pass"
            )


# ===============================================================================
# market_mode — moved to market_mode.py for QuantConnect 63KB file limit
# Re-exported here to preserve backward-compatible imports.
# ===============================================================================

from market_mode import MarketModeDetector, MARKET_MODES  # noqa: E402,F401


# ===============================================================================
# momentum
# ===============================================================================

"""Momentum override and scoring helpers for Vox aggressive/ruthless mode."""


def check_momentum_override_conditions(feat, ret4_min, ret16_min, vol_min, btc_rel_min):
    """Return True if momentum breakout override conditions are met.

    Feature layout: [ret_1, ret_4, ret_8, ret_16, rsi_14, atr_n,
                     vol_r, btc_rel, hour, ...]
    """
    return (
        float(feat[1]) >= ret4_min
        and float(feat[3]) >= ret16_min
        and float(feat[6]) >= vol_min
        and float(feat[7]) >= btc_rel_min
    )


def compute_momentum_score(feat):
    """Compute bounded momentum score for aggressive/ruthless ranking.

    Combines ret_4, ret_16, normalised volume excess and btc_rel.
    Volume excess is normalised to [0, 1] and capped to prevent explosion.
    Returns a float clipped to [-0.05, 0.10].
    """
    vol_excess = min(max(float(feat[6]) - 1.0, 0.0), 4.0) / 4.0  # [0, 1]
    raw = (
        0.40 * float(feat[1])    # ret_4
        + 0.30 * float(feat[3])  # ret_16
        + 0.20 * vol_excess      # normalised volume spike
        + 0.10 * float(feat[7])  # btc_rel
    )
    return float(np.clip(raw, -0.05, 0.10))


# ===============================================================================
# meta_model
# ===============================================================================

"""
Lightweight meta-filter / veto model for Vox ruthless entries.

Uses a rules-based meta-score to veto low-conviction entry signals
before committing large ruthless allocations.
"""


class MetaFilter:
    """Rules-based meta-filter that vets entry candidates.

    Given base model signals and contextual features, computes a
    meta-score in [0, 1] and vetoes entries that fall below a threshold.

    The meta-score combines:
      - Model confidence (class_proba, n_agree, std_proba)
      - Expected value (ev_score)
      - Short-term momentum (ret_4, ret_16)
      - Volume confirmation (volume_ratio)
      - Market mode alignment
    """

    def __init__(self, min_proba=0.55, enabled=True):
        self.min_proba = min_proba
        self.enabled   = enabled

    def compute_score(
        self,
        class_proba,
        ev_score,
        n_agree,
        std_proba,
        pred_return,
        feat,
        market_mode=None,
        ruthless_allowed_modes=None,
    ):
        """Compute meta-score in [0, 1].

        Parameters
        ----------
        class_proba   : float — weighted ensemble probability
        ev_score      : float — expected value after costs
        n_agree       : int   — number of agreeing models
        std_proba     : float — standard deviation of model probabilities
        pred_return   : float — regressor ensemble return prediction
        feat          : array-like — feature vector (at least 7 elements)
        market_mode   : str or None — current detected market mode
        ruthless_allowed_modes : list[str] or None

        Returns
        -------
        float  — meta-score in [0, 1]
        """
        score = 0.0

        # Confidence component (0–0.35)
        score += min(0.35, class_proba * 0.35 / 0.65)

        # Model agreement bonus (0–0.20)
        if n_agree >= 3:
            score += 0.20
        elif n_agree >= 2:
            score += 0.10

        # Dispersion penalty
        score -= min(0.15, std_proba * 0.5)

        # EV component (0–0.20)
        score += min(0.20, max(0.0, ev_score * 20.0))

        # Momentum confirmation (0–0.15) from feat
        if feat is not None and len(feat) >= 7:
            ret4  = float(feat[1])
            ret16 = float(feat[3])
            vol_r = float(feat[6])
            if ret4 > 0.01 and ret16 > 0.02:
                score += 0.10
            elif ret4 > 0.005:
                score += 0.05
            if vol_r > 1.5:
                score += 0.05

        # Market mode alignment (0–0.10)
        if market_mode is not None:
            allowed = ruthless_allowed_modes or ["risk_on_trend", "pump"]
            if market_mode in allowed:
                score += 0.10
            elif market_mode in ("chop", "selloff"):
                score -= 0.10

        return float(np.clip(score, 0.0, 1.0))

    def approve(
        self,
        class_proba,
        ev_score,
        n_agree,
        std_proba,
        pred_return,
        feat,
        market_mode=None,
        ruthless_allowed_modes=None,
    ):
        """Returns (approved: bool, meta_score: float).

        When disabled, always returns (True, 1.0).
        """
        if not self.enabled:
            return True, 1.0
        score = self.compute_score(
            class_proba, ev_score, n_agree, std_proba,
            pred_return, feat, market_mode, ruthless_allowed_modes,
        )
        return score >= self.min_proba, score
