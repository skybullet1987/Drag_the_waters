# ── Apex Voting — weighted ensemble vote aggregator ──────────────────────────
#
# Implements the Section D smarter ensemble vote from the apex-predator spec.
#
# Evidence (diagnostics export combined_export_20260503_132454.txt):
#   - vote_lr, vote_gnb, vote_xgb_bal, vote_cal_* are near-zero or constant
#     → dead voters dragging the ensemble vote fraction down.
#   - lgbm_bal, hgbc_l2, lr_bal have the highest profit-factor evidence.
#   - Previous simple majority vote was vetoed by dead-weight voters.
#
# New rule (OR logic):
#   FIRE if:
#     weighted_yes_fraction >= APEX_WEIGHTED_YES_THRESHOLD (0.45)
#     OR momentum_override is True
#     OR (hgbc_l2 >= APEX_COMBO_HGBC_MIN AND lgbm_bal >= APEX_COMBO_LGBM_MIN)
#
# All weights/thresholds are named constants in aggressive_config.py.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aggressive_config import (
        APEX_WEIGHTED_VOTE_WEIGHTS,
        APEX_WEIGHTED_YES_THRESHOLD,
        APEX_COMBO_HGBC_MIN,
        APEX_COMBO_LGBM_MIN,
    )
except ImportError:
    # Fallback defaults (same as aggressive_config.py)
    APEX_WEIGHTED_VOTE_WEIGHTS = {
        "lgbm_bal":    2.5,
        "hgbc_l2":     2.0,
        "rf":          1.5,
        "et":          1.0,
        "lr_bal":      1.5,
        "lgbm":        1.0,
        "gnb":         0.0,
        "lr":          0.0,
        "xgb_bal":     0.5,
        "cal_et":      0.5,
        "cal_rf":      0.5,
        "rf_shallow":  0.3,
        "et_shallow":  0.3,
    }
    APEX_WEIGHTED_YES_THRESHOLD = 0.45
    APEX_COMBO_HGBC_MIN  = 0.55
    APEX_COMBO_LGBM_MIN  = 0.55


def compute_weighted_yes_fraction(
    model_votes,
    vote_threshold=0.50,
    weights=None,
):
    """Compute the weighted yes-fraction from a dict of per-model vote probabilities.

    For each model in *model_votes*:
      - if its probability >= vote_threshold  → counts as YES with its weight
      - otherwise                             → counts as NO with its weight

    weighted_yes_fraction = sum(weight_i for yes models)
                            / sum(weight_i for all present models)

    Models with zero weight in the weight map are present but contribute 0.
    Models absent from *model_votes* are excluded from both numerator and
    denominator so the result correctly normalises to the present voters.

    Parameters
    ----------
    model_votes    : dict[str, float]
        Per-model probability values, e.g. {"lgbm_bal": 0.72, "hgbc_l2": 0.58, ...}
    vote_threshold : float
        Probability floor to count a model as voting YES (default 0.50).
    weights        : dict[str, float] or None
        Model weights.  If None, uses APEX_WEIGHTED_VOTE_WEIGHTS.

    Returns
    -------
    dict with keys:
        "weighted_yes_fraction" : float — 0.0 if no present models
        "yes_weight"            : float — sum of weights for YES voters
        "total_weight"          : float — sum of weights for present voters
        "yes_models"            : list[str] — models that voted YES
        "no_models"             : list[str] — models that voted NO
        "zero_weight_models"    : list[str] — present but zero-weight models
    """
    if weights is None:
        weights = APEX_WEIGHTED_VOTE_WEIGHTS

    yes_w   = 0.0
    total_w = 0.0
    yes_models        = []
    no_models         = []
    zero_weight_models = []

    for model_id, proba in model_votes.items():
        w = float(weights.get(model_id, 1.0))  # unknown models default to 1.0
        if w == 0.0:
            zero_weight_models.append(model_id)
            # Still accumulate to total so 0-weight models don't inflate fraction
            total_w += 0.0
            continue

        total_w += w
        if float(proba) >= vote_threshold:
            yes_w += w
            yes_models.append(model_id)
        else:
            no_models.append(model_id)

    frac = yes_w / total_w if total_w > 0.0 else 0.0

    return {
        "weighted_yes_fraction": round(frac, 6),
        "yes_weight":            round(yes_w, 6),
        "total_weight":          round(total_w, 6),
        "yes_models":            yes_models,
        "no_models":             no_models,
        "zero_weight_models":    zero_weight_models,
    }


def apex_voting_decision(
    model_votes,
    momentum_override=False,
    vote_threshold=0.50,
    yes_threshold=None,
    combo_hgbc_min=None,
    combo_lgbm_min=None,
    weights=None,
):
    """Evaluate the apex voting decision using the weighted ensemble rule.

    Fires if ANY of:
      1. weighted_yes_fraction >= yes_threshold (0.45)
      2. momentum_override is True
      3. hgbc_l2 >= combo_hgbc_min (0.55) AND lgbm_bal >= combo_lgbm_min (0.55)

    Parameters
    ----------
    model_votes      : dict[str, float] — per-model probability values
    momentum_override: bool — if True, always fires
    vote_threshold   : float — per-model YES/NO threshold (default 0.50)
    yes_threshold    : float or None — weighted-fraction threshold; defaults to
                       APEX_WEIGHTED_YES_THRESHOLD (0.45)
    combo_hgbc_min   : float or None — hgbc_l2 combo floor; defaults to 0.55
    combo_lgbm_min   : float or None — lgbm_bal combo floor; defaults to 0.55
    weights          : dict or None — model weight map; defaults to
                       APEX_WEIGHTED_VOTE_WEIGHTS

    Returns
    -------
    dict with keys:
        "triggered"              : bool
        "trigger_path"           : str or None — which path fired first
        "weighted_yes_fraction"  : float
        "yes_threshold"          : float
        "momentum_override"      : bool
        "combo_fired"            : bool
        "yes_models"             : list[str]
        "no_models"              : list[str]
        "zero_weight_models"     : list[str]
        "reject_reason"          : str or None
    """
    if yes_threshold is None:
        yes_threshold = APEX_WEIGHTED_YES_THRESHOLD
    if combo_hgbc_min is None:
        combo_hgbc_min = APEX_COMBO_HGBC_MIN
    if combo_lgbm_min is None:
        combo_lgbm_min = APEX_COMBO_LGBM_MIN

    vote_result = compute_weighted_yes_fraction(
        model_votes,
        vote_threshold=vote_threshold,
        weights=weights,
    )
    frac = vote_result["weighted_yes_fraction"]

    hgbc_l2  = float(model_votes.get("hgbc_l2",  0.0))
    lgbm_bal = float(model_votes.get("lgbm_bal", 0.0))

    path1 = frac >= yes_threshold
    path2 = bool(momentum_override)
    path3 = hgbc_l2 >= combo_hgbc_min and lgbm_bal >= combo_lgbm_min

    triggered = path1 or path2 or path3

    trigger_path = None
    if path1:
        trigger_path = "weighted_yes_fraction"
    elif path2:
        trigger_path = "momentum_override"
    elif path3:
        trigger_path = "hgbc_lgbm_combo"

    reject_reason = None
    if not triggered:
        reject_reason = (
            f"weighted_yes_fraction={frac:.3f}<{yes_threshold} | "
            f"momentum_override={momentum_override} | "
            f"hgbc_l2={hgbc_l2:.3f}<{combo_hgbc_min} or "
            f"lgbm_bal={lgbm_bal:.3f}<{combo_lgbm_min}"
        )

    return {
        "triggered":             triggered,
        "trigger_path":          trigger_path,
        "weighted_yes_fraction": frac,
        "yes_threshold":         yes_threshold,
        "momentum_override":     bool(momentum_override),
        "combo_fired":           path3,
        "yes_models":            vote_result["yes_models"],
        "no_models":             vote_result["no_models"],
        "zero_weight_models":    vote_result["zero_weight_models"],
        "reject_reason":         reject_reason,
    }
