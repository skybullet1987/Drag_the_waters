# ── Vox Model Registry ────────────────────────────────────────────────────────
#
# Stable model identifiers, role constants, and weighted ensemble helpers.
#
# Model IDs are intentionally short for compact log output.
# The core sklearn stack (lr, hgbc, et, rf) is always present.
# Optional external models (lgbm, xgb, catboost) are guarded by import checks.
# ─────────────────────────────────────────────────────────────────────────────

# ── Core model IDs ────────────────────────────────────────────────────────────
MODEL_ID_LR       = "lr"
MODEL_ID_HGBC     = "hgbc"     # HistGradientBoostingClassifier
MODEL_ID_ET       = "et"       # ExtraTreesClassifier
MODEL_ID_RF       = "rf"       # RandomForestClassifier
MODEL_ID_GNB      = "gnb"      # GaussianNB (diagnostic-only by default)
MODEL_ID_LGBM     = "lgbm"
MODEL_ID_XGB      = "xgb"
MODEL_ID_CATBOOST = "catboost"

# Ordered list of all possible model IDs (core + optional)
ALL_MODEL_IDS = [
    MODEL_ID_LR,
    MODEL_ID_HGBC,
    MODEL_ID_ET,
    MODEL_ID_RF,
    MODEL_ID_LGBM,
    MODEL_ID_XGB,
    MODEL_ID_CATBOOST,
]

# Human-readable descriptions for startup logging
MODEL_DESCRIPTIONS = {
    MODEL_ID_LR:       "LogisticRegression (linear baseline)",
    MODEL_ID_HGBC:     "HistGradientBoostingClassifier (strong sklearn booster)",
    MODEL_ID_ET:       "ExtraTreesClassifier (randomised trees, diverse)",
    MODEL_ID_RF:       "RandomForestClassifier (bagged trees)",
    MODEL_ID_GNB:      "GaussianNB (diagnostic-only; degenerate on crypto)",
    MODEL_ID_LGBM:     "LGBMClassifier (external, optional)",
    MODEL_ID_XGB:      "XGBClassifier (external, optional)",
    MODEL_ID_CATBOOST: "CatBoostClassifier (external, optional)",
}

# ── Model role constants ───────────────────────────────────────────────────────
# active     — contributes to ensemble vote; affects trading confidence.
# shadow     — predicted and logged but NEVER affects trading decisions.
# diagnostic — predicted and logged for risk/veto/debug only.
# disabled   — skipped entirely (not trained or predicted).
ROLE_ACTIVE     = "active"
ROLE_SHADOW     = "shadow"
ROLE_DIAGNOSTIC = "diagnostic"
ROLE_DISABLED   = "disabled"

# Default role for each core model ID
DEFAULT_MODEL_ROLES = {
    MODEL_ID_LR:       ROLE_DIAGNOSTIC,  # was always-bearish; diagnostic only
    MODEL_ID_HGBC:     ROLE_ACTIVE,
    MODEL_ID_ET:       ROLE_ACTIVE,
    MODEL_ID_RF:       ROLE_ACTIVE,
    MODEL_ID_GNB:      ROLE_DIAGNOSTIC,  # always-bullish (vote_gnb=1.0); diagnostic
    MODEL_ID_LGBM:     ROLE_SHADOW,
    MODEL_ID_XGB:      ROLE_SHADOW,
    MODEL_ID_CATBOOST: ROLE_SHADOW,
}


# ── Registry entry structure ──────────────────────────────────────────────────
#
# Each entry is a dict with:
#   id       : str   — stable model ID
#   model    : estimator or None
#   enabled  : bool
#   weight   : float — used in weighted mean computation
#   role     : str   — one of ROLE_* constants

def make_registry_entry(model_id, model, enabled=True, weight=1.0, role=ROLE_ACTIVE):
    """Create a model registry entry dict.

    Parameters
    ----------
    model_id : str
    model    : estimator or None
    enabled  : bool
    weight   : float
    role     : str — one of ROLE_ACTIVE / ROLE_SHADOW / ROLE_DIAGNOSTIC / ROLE_DISABLED
    """
    return {
        "id":      model_id,
        "model":   model,
        "enabled": enabled,
        "weight":  float(weight),
        "role":    role,
    }


def build_registry_from_estimators(estimators, weights_dict=None, roles_dict=None):
    """Build a registry list from VotingClassifier estimators.

    Parameters
    ----------
    estimators   : list of (name, estimator) — from VotingClassifier
    weights_dict : dict[str, float] or None — optional per-model weights
    roles_dict   : dict[str, str] or None   — optional per-model roles

    Returns
    -------
    list of registry entry dicts
    """
    registry = []
    for name, model in estimators:
        w = (weights_dict or {}).get(name, 1.0)
        r = (roles_dict or {}).get(name, ROLE_ACTIVE)
        registry.append(make_registry_entry(
            model_id=name,
            model=model,
            enabled=True,
            weight=w,
            role=r,
        ))
    return registry


# ── Weighted mean computation ──────────────────────────────────────────────────

def compute_weighted_mean(votes, weights_dict=None):
    """Compute weighted mean probability from a per-model votes dict.

    Parameters
    ----------
    votes        : dict[str, float] — model_id -> P(class=1)
    weights_dict : dict[str, float] or None — model_id -> weight.
                   If None or empty, falls back to unweighted mean.

    Returns
    -------
    float — weighted (or unweighted) mean in [0, 1].
    """
    if not votes:
        return 0.5

    if not weights_dict:
        # Unweighted mean
        return sum(votes.values()) / len(votes)

    total_w = 0.0
    weighted_sum = 0.0
    for model_id, proba in votes.items():
        w = weights_dict.get(model_id, 1.0)
        if w > 0:
            weighted_sum += w * proba
            total_w += w

    if total_w <= 0:
        # Fallback: unweighted mean
        return sum(votes.values()) / len(votes)

    return weighted_sum / total_w


def weights_are_uniform(weights_dict):
    """Return True if all weights in the dict are equal (no custom weighting)."""
    if not weights_dict:
        return True
    vals = list(weights_dict.values())
    if len(vals) <= 1:
        return True
    return all(abs(v - vals[0]) < 1e-9 for v in vals)


# ── Role-aware vote splitting ──────────────────────────────────────────────────

def split_votes_by_role(votes, roles_dict):
    """Split a per-model votes dict into role-separated sub-dicts.

    Parameters
    ----------
    votes      : dict[str, float] — model_id -> P(class=1)
    roles_dict : dict[str, str]   — model_id -> role string

    Returns
    -------
    tuple of (active_votes, shadow_votes, diagnostic_votes)
        Each is a dict[str, float] containing only models of that role.
    """
    active     = {}
    shadow     = {}
    diagnostic = {}
    for mid, proba in votes.items():
        role = roles_dict.get(mid, ROLE_ACTIVE)
        if role == ROLE_ACTIVE:
            active[mid] = proba
        elif role == ROLE_SHADOW:
            shadow[mid] = proba
        elif role == ROLE_DIAGNOSTIC:
            diagnostic[mid] = proba
        # ROLE_DISABLED models should not appear in votes at all
    return active, shadow, diagnostic


def compute_vote_score(active_votes, vote_thr=0.55):
    """Compute profit-voting score fields from active-model votes.

    Parameters
    ----------
    active_votes : dict[str, float]
        Model-id -> P(class=1) for active-role models only.
    vote_thr : float
        Per-model yes/no threshold.

    Returns
    -------
    dict with keys:
        active_model_count  — int
        vote_yes_fraction   — float in [0, 1]
        top3_mean           — float: mean of top-3 active probabilities
        vote_score          — float: weighted composite
    """
    import numpy as _np
    if not active_votes:
        return {"active_model_count": 0, "vote_yes_fraction": 0.0,
                "top3_mean": 0.0, "vote_score": 0.0}
    vals = sorted(active_votes.values(), reverse=True)
    n    = len(vals)
    am   = float(_np.mean(vals))
    yf   = sum(1 for v in vals if v >= vote_thr) / n
    t3   = float(_np.mean(vals[:3])) if vals else 0.0
    vs   = 0.40 * am + 0.30 * yf + 0.30 * t3
    return {"active_model_count": n, "vote_yes_fraction": yf,
            "top3_mean": t3, "vote_score": vs}


def compute_active_stats(active_votes, agree_thr=0.5):
    """Compute mean / std / n_agree from active-role votes only.

    Parameters
    ----------
    active_votes : dict[str, float] — active-model votes
    agree_thr    : float            — threshold for agreement counting

    Returns
    -------
    dict with keys: active_mean, active_std, active_n_agree
    """
    import numpy as _np
    if not active_votes:
        return {"active_mean": 0.5, "active_std": 0.0, "active_n_agree": 0}
    vals = list(active_votes.values())
    return {
        "active_mean":    float(_np.mean(vals)),
        "active_std":     float(_np.std(vals)),
        "active_n_agree": int(sum(1 for v in vals if v >= agree_thr)),
    }


# ── Startup log helper ────────────────────────────────────────────────────────

def format_model_registry_log(estimators, weights_dict=None, roles_dict=None):
    """Format a startup log line listing enabled model IDs, weights, and roles.

    Example output:
        [model_registry] active=hgbc(w=1.0),et(w=1.0),rf(w=1.0) diag=lr
    """
    active_parts = []
    shadow_parts = []
    diag_parts   = []
    for name, _ in estimators:
        w    = (weights_dict or {}).get(name, 1.0)
        role = (roles_dict or {}).get(name, ROLE_ACTIVE)
        tag  = f"{name}(w={w:.2g})"
        if role == ROLE_SHADOW:
            shadow_parts.append(tag)
        elif role == ROLE_DIAGNOSTIC:
            diag_parts.append(tag)
        else:
            active_parts.append(tag)
    parts = []
    if active_parts:
        parts.append("active=" + ",".join(active_parts))
    if shadow_parts:
        parts.append("shadow=" + ",".join(shadow_parts))
    if diag_parts:
        parts.append("diag=" + ",".join(diag_parts))
    return "[model_registry] " + " ".join(parts) if parts else "[model_registry] (none)"


def format_vote_summary(votes, vote_threshold=0.5):
    """Format a compact per-model vote string.

    Example: lr:0.55,hgbc:0.62,et:0.70,rf:0.58
    """
    if not votes:
        return ""
    return ",".join(f"{mid}:{v:.2f}" for mid, v in votes.items())


# ── Default model weights from config ─────────────────────────────────────────

def build_weights_dict_from_config(config_module):
    """Extract per-model weights from a config module.

    Looks for MODEL_WEIGHT_LR, MODEL_WEIGHT_HGBC, etc. constants.
    Returns a dict only containing models with non-default (!=1.0) weights,
    or all weights if any differ.
    """
    mapping = {
        MODEL_ID_LR:       "MODEL_WEIGHT_LR",
        MODEL_ID_HGBC:     "MODEL_WEIGHT_HGBC",
        MODEL_ID_ET:       "MODEL_WEIGHT_ET",
        MODEL_ID_RF:       "MODEL_WEIGHT_RF",
        MODEL_ID_GNB:      "MODEL_WEIGHT_GNB",
        MODEL_ID_LGBM:     "MODEL_WEIGHT_LGBM",
        MODEL_ID_XGB:      "MODEL_WEIGHT_XGB",
        MODEL_ID_CATBOOST: "MODEL_WEIGHT_CATBOOST",
        # Shadow/promoted models
        "hgbc_l2":         "MODEL_WEIGHT_HGBC_L2",
        "cal_et":          "MODEL_WEIGHT_CAL_ET",
        "cal_rf":          "MODEL_WEIGHT_CAL_RF",
        "lgbm_bal":        "MODEL_WEIGHT_LGBM_BAL",
    }
    result = {}
    for model_id, attr in mapping.items():
        val = getattr(config_module, attr, 1.0)
        result[model_id] = float(val)
    return result


# ── Default model roles from config ───────────────────────────────────────────

def build_roles_dict_from_config(config_module):
    """Extract per-model roles from a config module.

    Looks for MODEL_ROLE_LR, MODEL_ROLE_HGBC, etc. constants.
    Falls back to DEFAULT_MODEL_ROLES if the constant is missing.
    """
    mapping = {
        MODEL_ID_LR:       "MODEL_ROLE_LR",
        MODEL_ID_HGBC:     "MODEL_ROLE_HGBC",
        MODEL_ID_ET:       "MODEL_ROLE_ET",
        MODEL_ID_RF:       "MODEL_ROLE_RF",
        MODEL_ID_GNB:      "MODEL_ROLE_GNB",
        MODEL_ID_LGBM:     "MODEL_ROLE_LGBM",
        MODEL_ID_XGB:      "MODEL_ROLE_XGB",
        MODEL_ID_CATBOOST: "MODEL_ROLE_CATBOOST",
    }
    result = {}
    for model_id, attr in mapping.items():
        val = getattr(config_module, attr, DEFAULT_MODEL_ROLES.get(model_id, ROLE_ACTIVE))
        result[model_id] = str(val)
    return result
