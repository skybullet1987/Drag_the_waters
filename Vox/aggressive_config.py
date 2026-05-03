# ── Apex Predator — aggressive strategy configuration ────────────────────────
#
# Named constants for the "apex predator / gatling gun" strategy profile.
# Evidence source: combined_export_20260503_132454.txt model_accuracy_summary
#   - vote_lgbm_bal, shadow_lgbm_bal, active_lgbm_bal: PF 1.7–3.4 at thr 0.50
#   - hgbc_l2: PF 3.35 at thr 0.55–0.60
#   - lr_bal:  PF 7.99 at thr 0.50 (small N but highest discriminating power)
#   - vote_gnb: always near 1.0 → useless as discriminator → weight 0
#   - vote_lr, vote_xgb_bal, vote_cal_*: near-zero medians → weight 0 / 0.5
#
# Import from here in strategy code; do NOT hardcode thresholds inline.
# ─────────────────────────────────────────────────────────────────────────────

# ── Section D: weighted vote aggregator weights ───────────────────────────────
# Derived from profit_factor_yes_50 column of model_accuracy_summary_hotfix.
# Models with evidence of positive expectancy get higher weights.
# Dead voters (gnb, lr) are zeroed; weak voters (xgb_bal, cal_*) kept low.
APEX_WEIGHTED_VOTE_WEIGHTS = {
    "lgbm_bal":    2.5,   # PF 1.7–3.4 at thr 0.50 — consistent performer
    "hgbc_l2":     2.0,   # PF 3.35 at thr 0.55–0.60
    "rf":          1.5,   # PF 2.76 at thr 0.60
    "et":          1.0,   # diversifier
    "lr_bal":      1.5,   # PF 7.99 at thr 0.50 (small N, high signal)
    "lgbm":        1.0,   # base lgbm (non-balanced)
    "gnb":         0.0,   # constant ~1.0 → no discrimination power
    "lr":          0.0,   # near-zero votes → dead voter
    "xgb_bal":     0.5,   # weak signal, occasional contribution
    "cal_et":      0.5,   # calibrated ET — modest contribution
    "cal_rf":      0.5,   # calibrated RF — modest contribution
    "rf_shallow":  0.3,   # near-zero median → minimal weight
    "et_shallow":  0.3,   # near-zero median → minimal weight
}

# Weighted yes-fraction threshold: trade fires if weighted_yes_fraction >= this
APEX_WEIGHTED_YES_THRESHOLD = 0.45   # lower than majority vote → more trades

# Alternative fire conditions (OR logic with threshold):
#   momentum_override = True                         → always fire
#   hgbc_l2 >= APEX_COMBO_HGBC_MIN AND lgbm_bal >= APEX_COMBO_LGBM_MIN → fire
APEX_COMBO_HGBC_MIN  = 0.55   # hgbc_l2 proba floor for combo trigger
APEX_COMBO_LGBM_MIN  = 0.55   # lgbm_bal proba floor for combo trigger

# ── Section A: gating / fire-rate constants ───────────────────────────────────
# Lower these vs. defaults to allow far more orders (target ≥ 200 per 17 months)
APEX_GATE_MIN_CLASS_PROBA  = 0.45   # was ~0.55+; lower floor fires more signals
APEX_GATE_MIN_FINAL_SCORE  = 0.0    # no minimum final-score veto
APEX_GATE_MIN_N_AGREE      = 1      # at least 1 model must agree (was 2–3)
APEX_GATE_COOLDOWN_MIN     = 15     # per-symbol reentry cooldown (min)
APEX_GATE_MAX_CONCURRENT   = 12     # concurrent open positions
APEX_GATE_MAX_PER_SYMBOL   = 3      # max simultaneous positions per symbol

# ── Section B: conviction-weighted sizing ────────────────────────────────────
# size = base_alloc * (1 + k * (final_score - 0.5)), clipped to [0.05, 0.25]
APEX_SIZE_BASE_ALLOC   = 0.10   # baseline per-trade allocation (10% equity)
APEX_SIZE_CONV_K       = 4.0    # conviction scaling factor
APEX_SIZE_MIN_FRAC     = 0.05   # minimum position size (5% equity)
APEX_SIZE_MAX_FRAC     = 0.25   # maximum position size (25% equity)
APEX_USE_LEVERAGE      = True   # allow up to 3x total notional
APEX_MAX_LEVERAGE      = 3.0    # total notional leverage cap

# ── Section C: stop-loss / trailing / time-stop ───────────────────────────────
APEX_SL_ATR_MULT       = 1.5    # SL = entry - 1.5 * ATR (tighter than prev)
APEX_SL_PCT_FLOOR      = 0.025  # 2.5% min SL distance (whichever is tighter)
APEX_TP_USE_FIXED      = False  # no fixed TP — let trail run
APEX_TRAIL_MULT        = 4.0    # ATR trail distance multiplier
APEX_TIME_STOP_DAYS    = 30     # close if open > 30 days without sufficient MFE
APEX_PYRAMID_MFE_ATR   = 1.0    # add 50% tranche after +1 ATR unrealised
APEX_PYRAMID_ADD_FRAC  = 0.50   # size of each pyramid add vs original
APEX_PYRAMID_MAX_ADDS  = 2      # maximum pyramid add-ons per position
