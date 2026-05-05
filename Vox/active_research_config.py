# active_research_config.py — constants for the active_research risk profile
#
# WARNING: active_research is a DATA-COLLECTION profile, NOT a production profile.
# Gates are intentionally loose so the ensemble generates many trades for
# model-vote diagnostics and ensemble selection.  Use small allocation only.
#
# Activate via:  risk_profile=active_research  (QC parameter panel)
# Analyse results using vox/model_vote_outcomes.jsonl + research diagnostics script.

ACTIVE_RESEARCH_SCORE_MIN            = 0.17
ACTIVE_RESEARCH_SCORE_GAP            = 0.0
ACTIVE_RESEARCH_MIN_AGREE            = 1
ACTIVE_RESEARCH_MAX_DISPERSION       = 0.50
ACTIVE_RESEARCH_MIN_EV               = -0.003
ACTIVE_RESEARCH_PRED_RETURN_MIN      = -0.005
ACTIVE_RESEARCH_COOLDOWN_MINS        = 0
ACTIVE_RESEARCH_SL_COOLDOWN_MINS     = 20
ACTIVE_RESEARCH_MAX_DAILY_SL         = 10
ACTIVE_RESEARCH_ALLOCATION           = 0.04
ACTIVE_RESEARCH_MAX_ALLOC            = 0.08
ACTIVE_RESEARCH_MIN_ALLOC            = 0.0
ACTIVE_RESEARCH_USE_KELLY            = False
ACTIVE_RESEARCH_TAKE_PROFIT          = 0.012
ACTIVE_RESEARCH_STOP_LOSS            = 0.010
ACTIVE_RESEARCH_TIMEOUT_HOURS        = 2.0
ACTIVE_RESEARCH_MIN_HOLD_MINUTES     = 5
ACTIVE_RESEARCH_EMERGENCY_SL         = 0.030
ACTIVE_RESEARCH_PENALTY_LOSSES       = 20
ACTIVE_RESEARCH_PENALTY_HOURS        = 0
ACTIVE_RESEARCH_MAX_DD_PCT           = 0.40
ACTIVE_RESEARCH_REGIME_SIZE_MULT     = 0.50
ACTIVE_RESEARCH_DIAG_INTERVAL_HOURS  = 1
