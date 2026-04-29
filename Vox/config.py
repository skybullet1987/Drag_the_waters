# ── Strategy constants (all overridable via the QC parameter panel) ───────────
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

# ── Risk profile ───────────────────────────────────────────────────────────────
# Values: "balanced" (default), "conservative", "aggressive", "ruthless"
# Convenience aliases: aggressive_mode=true, ruthless_mode=true
RISK_PROFILE = "balanced"

# ── Aggressive profile defaults ───────────────────────────────────────────────
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

# ── Ruthless v2 profile defaults ──────────────────────────────────────────────
# WARNING: can produce fast large drawdowns.  Use for high-risk experimentation.
# Ruthless v2 targets large asymmetric winners:
#   • Wider TP/SL (9% / 3%) → P/L ratio ≈ 3.0 — meaningfully above break-even
#   • 24h timeout — winners have room to run
#   • Runner mode — trailing stop replaces instant TP exit
#   • Kelly disabled / allocation floor — positions sized ruthlessly (~90%)
#   • Looser predicted-return gate — regressor rarely vetoes now
RUTHLESS_SCORE_MIN               = 0.45
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
RUTHLESS_SL_COOLDOWN_MINS        = 5
RUTHLESS_PENALTY_COOLDOWN_LOSSES = 5
RUTHLESS_PENALTY_COOLDOWN_HOURS  = 12
RUTHLESS_MAX_DD_PCT              = 0.35
# Runner / trailing-profit parameters
RUTHLESS_RUNNER_MODE             = True    # trailing stop instead of instant TP exit
RUTHLESS_TRAIL_AFTER_TP          = 0.04   # activate trailing once return ≥ +4 %
RUTHLESS_TRAIL_PCT               = 0.025  # exit if price drops 2.5 % from trailing high

# ── Momentum breakout override ─────────────────────────────────────────────────
# Enabled by default for aggressive/ruthless profiles; disabled otherwise.
MOMENTUM_RET4_MIN          = 0.015   # minimum 4-bar return for override
MOMENTUM_RET16_MIN         = 0.025   # minimum 16-bar return for override
MOMENTUM_VOLUME_MIN        = 2.0     # minimum volume ratio (current / 15-bar avg)
MOMENTUM_BTC_REL_MIN       = 0.005   # minimum BTC-relative 4-bar outperformance
MOMENTUM_OVERRIDE_MIN_EV   = -0.002  # momentum override blocked if EV < this threshold


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
        if _rp_val in ("conservative", "balanced", "aggressive", "ruthless"):
            algo._risk_profile = _rp_val
            if _rp_val == "conservative":
                algo._conservative_mode = True
        else:
            algo._risk_profile = RISK_PROFILE
    else:
        algo._risk_profile = RISK_PROFILE

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
        if not hasattr(algo, "_min_alloc") or algo._min_alloc == 0.0:
            algo._min_alloc    = RUTHLESS_MIN_ALLOC
        if not hasattr(algo, "_use_kelly"):
            algo._use_kelly    = RUTHLESS_USE_KELLY
        else:
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
        if not hasattr(algo, "_runner_mode") or not algo._runner_mode:
            _rm_param = algo.get_parameter("runner_mode")
            algo._runner_mode = (
                str(_rm_param).lower() in ("true", "1", "yes")
                if _rm_param else RUTHLESS_RUNNER_MODE
            )
        if not hasattr(algo, "_trail_after_tp"):
            algo._trail_after_tp = RUTHLESS_TRAIL_AFTER_TP
        if not hasattr(algo, "_trail_pct"):
            algo._trail_pct      = RUTHLESS_TRAIL_PCT

    # ── Momentum override setup ───────────────────────────────────────────────
    _mo_raw = algo.get_parameter("momentum_override")
    if _mo_raw:
        algo._momentum_override = str(_mo_raw).lower() in ("true", "1", "yes")
    else:
        algo._momentum_override = algo._risk_profile in ("aggressive", "ruthless")

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
    algo._use_momentum_score = algo._risk_profile in ("aggressive", "ruthless")

    # ── Startup profile log (unthrottled — important config visibility) ────────
    _runner_mode_val   = getattr(algo, "_runner_mode",   False)
    _trail_after_tp    = getattr(algo, "_trail_after_tp", 0.0)
    _trail_pct         = getattr(algo, "_trail_pct",      0.0)
    _min_alloc         = getattr(algo, "_min_alloc",      0.0)
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
