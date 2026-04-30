# ── Vox Model Registry ────────────────────────────────────────────────────────
#
# Stable model identifiers and weighted ensemble helpers.
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
    MODEL_ID_LGBM:     "LGBMClassifier (external, optional)",
    MODEL_ID_XGB:      "XGBClassifier (external, optional)",
    MODEL_ID_CATBOOST: "CatBoostClassifier (external, optional)",
}


# ── Registry entry structure ──────────────────────────────────────────────────
#
# Each entry is a dict with:
#   id       : str   — stable model ID
#   model    : estimator or None
#   enabled  : bool
#   weight   : float — used in weighted mean computation

def make_registry_entry(model_id, model, enabled=True, weight=1.0):
    """Create a model registry entry dict."""
    return {
        "id":      model_id,
        "model":   model,
        "enabled": enabled,
        "weight":  float(weight),
    }


def build_registry_from_estimators(estimators, weights_dict=None):
    """Build a registry list from VotingClassifier estimators.

    Parameters
    ----------
    estimators   : list of (name, estimator) — from VotingClassifier
    weights_dict : dict[str, float] or None — optional per-model weights

    Returns
    -------
    list of registry entry dicts
    """
    registry = []
    for name, model in estimators:
        w = (weights_dict or {}).get(name, 1.0)
        registry.append(make_registry_entry(
            model_id=name,
            model=model,
            enabled=True,
            weight=w,
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


# ── Startup log helper ────────────────────────────────────────────────────────

def format_model_registry_log(estimators, weights_dict=None):
    """Format a startup log line listing enabled model IDs and their weights.

    Example output:
        [model_registry] enabled=lr(w=1.0),hgbc(w=1.0),et(w=1.0),rf(w=1.0)
    """
    parts = []
    for name, _ in estimators:
        w = (weights_dict or {}).get(name, 1.0)
        parts.append(f"{name}(w={w:.2g})")
    return "[model_registry] enabled=" + ",".join(parts)


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
        MODEL_ID_LGBM:     "MODEL_WEIGHT_LGBM",
        MODEL_ID_XGB:      "MODEL_WEIGHT_XGB",
        MODEL_ID_CATBOOST: "MODEL_WEIGHT_CATBOOST",
    }
    result = {}
    for model_id, attr in mapping.items():
        val = getattr(config_module, attr, 1.0)
        result[model_id] = float(val)
    return result
