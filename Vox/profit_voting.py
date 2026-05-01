# ── Vox Profit-Voting Gate ────────────────────────────────────────────────────
#
# Active vote-score / ranking system for ruthless profit-voting mode.
#
# This module provides:
#   - compute_vote_score()  : calculate vote_score, vote_yes_fraction, top3_mean
#   - check_profit_voting_gate() : entry-gate check for ruthless profit-voting mode
#   - chop supermajority requirements
#
# None of these functions affect balanced/conservative/aggressive profiles.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np


# ── Vote score constants (defaults; override via config) ──────────────────────
DEFAULT_VOTE_THRESHOLD          = 0.55   # probability floor to count as "yes" vote
DEFAULT_VOTE_YES_FRACTION_MIN   = 0.50   # min fraction of active models voting yes
DEFAULT_TOP3_MEAN_MIN           = 0.62   # min mean of top-3 active probabilities

# Chop market mode — stricter supermajority
DEFAULT_CHOP_VOTE_YES_FRAC_MIN  = 0.70   # stricter yes-fraction requirement in chop
DEFAULT_CHOP_TOP3_MEAN_MIN      = 0.75   # stricter top-3-mean requirement in chop
DEFAULT_CHOP_PRED_RETURN_MIN    = 0.01   # positive pred_return required in chop
DEFAULT_CHOP_EV_MIN             = 0.01   # stronger EV floor required in chop

# Weight allocation for vote_score composite
_W_ACTIVE_MEAN   = 0.40
_W_YES_FRAC      = 0.30
_W_TOP3_MEAN     = 0.30


def compute_vote_score(active_votes, vote_thr=DEFAULT_VOTE_THRESHOLD):
    """Compute profit-voting score fields from active-model votes.

    Parameters
    ----------
    active_votes : dict[str, float]
        Model-id → P(class=1) for active-role models only.
    vote_thr : float
        Probability threshold used to classify a model as "voting yes".

    Returns
    -------
    dict with keys:
        active_model_count  — int
        vote_yes_fraction   — float in [0, 1]
        top3_mean           — float: mean of top-3 active probabilities
        vote_score          — float: weighted composite (active_mean, yes_frac, top3_mean)
    """
    if not active_votes:
        return {
            "active_model_count": 0,
            "vote_yes_fraction":  0.0,
            "top3_mean":          0.0,
            "vote_score":         0.0,
        }

    vals = sorted(active_votes.values(), reverse=True)
    n    = len(vals)
    am   = float(np.mean(vals))
    yes_frac = sum(1 for v in vals if v >= vote_thr) / n
    top3_mean = float(np.mean(vals[:3])) if vals else 0.0
    vote_score = _W_ACTIVE_MEAN * am + _W_YES_FRAC * yes_frac + _W_TOP3_MEAN * top3_mean

    return {
        "active_model_count": n,
        "vote_yes_fraction":  yes_frac,
        "top3_mean":          top3_mean,
        "vote_score":         vote_score,
    }


def check_profit_voting_gate(
    conf,
    market_mode,
    vote_thr=DEFAULT_VOTE_THRESHOLD,
    vote_yes_frac_min=DEFAULT_VOTE_YES_FRACTION_MIN,
    top3_mean_min=DEFAULT_TOP3_MEAN_MIN,
    chop_vote_yes_frac_min=DEFAULT_CHOP_VOTE_YES_FRAC_MIN,
    chop_top3_mean_min=DEFAULT_CHOP_TOP3_MEAN_MIN,
    chop_pred_return_min=DEFAULT_CHOP_PRED_RETURN_MIN,
    chop_ev_min=DEFAULT_CHOP_EV_MIN,
    ev_score=0.0,
    require_min_active_models=3,
):
    """Check profit-voting entry gate for ruthless profit-voting mode.

    Parameters
    ----------
    conf : dict
        Output of VoxEnsemble.predict_with_confidence — must contain
        active_votes, vote_yes_fraction, top3_mean, pred_return, active_model_count.
    market_mode : str or None
        Current BTC market mode (e.g. "risk_on_trend", "pump", "chop").
    vote_thr : float
        Per-model yes/no threshold.
    vote_yes_frac_min : float
        Minimum yes-fraction for trend/pump markets.
    top3_mean_min : float
        Minimum top-3 mean for trend/pump markets.
    chop_vote_yes_frac_min : float
        Minimum yes-fraction in chop (supermajority).
    chop_top3_mean_min : float
        Minimum top-3 mean in chop.
    chop_pred_return_min : float
        Minimum predicted return in chop.
    chop_ev_min : float
        Minimum EV floor in chop.
    ev_score : float
        Candidate ev_after_costs (used for chop EV check).
    require_min_active_models : int
        Minimum number of active models required for a valid vote.

    Returns
    -------
    (approved: bool, reason: str)
    """
    active_votes     = conf.get("active_votes", {})
    active_count     = conf.get("active_model_count", len(active_votes))
    vote_yes_frac    = conf.get("vote_yes_fraction", 0.0)
    top3_mean        = conf.get("top3_mean", 0.0)
    pred_return      = conf.get("pred_return", 0.0)

    # Recompute if fields not pre-populated
    if "vote_yes_fraction" not in conf and active_votes:
        vs = compute_vote_score(active_votes, vote_thr)
        active_count  = vs["active_model_count"]
        vote_yes_frac = vs["vote_yes_fraction"]
        top3_mean     = vs["top3_mean"]

    # Require minimum active model count
    if active_count < require_min_active_models:
        return False, f"active_count={active_count} < {require_min_active_models}"

    is_chop = (market_mode is not None and "chop" in str(market_mode).lower())

    if is_chop:
        # Supermajority required in chop
        if vote_yes_frac < chop_vote_yes_frac_min:
            return (
                False,
                f"chop: vote_yes_frac={vote_yes_frac:.2f} < {chop_vote_yes_frac_min}"
            )
        if top3_mean < chop_top3_mean_min:
            return (
                False,
                f"chop: top3_mean={top3_mean:.3f} < {chop_top3_mean_min}"
            )
        if pred_return < chop_pred_return_min:
            return (
                False,
                f"chop: pred_return={pred_return:.4f} < {chop_pred_return_min}"
            )
        if ev_score < chop_ev_min:
            return (
                False,
                f"chop: ev_score={ev_score:.4f} < {chop_ev_min}"
            )
        return True, "chop_supermajority"

    # Trend / pump / unknown market mode
    if vote_yes_frac < vote_yes_frac_min:
        return (
            False,
            f"vote_yes_frac={vote_yes_frac:.2f} < {vote_yes_frac_min}"
        )
    if top3_mean < top3_mean_min:
        return (
            False,
            f"top3_mean={top3_mean:.3f} < {top3_mean_min}"
        )
    return True, "profit_vote_pass"


def format_profit_vote_log(sym_str, conf, vote_score_fields, market_mode=None, approved=True, reason=""):
    """Format a compact log line for profit-voting decisions.

    Parameters
    ----------
    sym_str           : str
    conf              : dict — predict_with_confidence output
    vote_score_fields : dict — from compute_vote_score
    market_mode       : str or None
    approved          : bool
    reason            : str

    Returns
    -------
    str
    """
    am  = conf.get("active_mean", conf.get("class_proba", 0.0))
    n   = vote_score_fields.get("active_model_count", 0)
    yf  = vote_score_fields.get("vote_yes_fraction", 0.0)
    t3  = vote_score_fields.get("top3_mean", 0.0)
    vs  = vote_score_fields.get("vote_score", 0.0)
    mode_str = f" mode={market_mode}" if market_mode else ""
    status   = "OK" if approved else f"BLOCKED:{reason}"
    return (
        f"[profit_vote] {sym_str}{mode_str}"
        f" active_count={n} active_mean={am:.3f}"
        f" yes_frac={yf:.2f} top3={t3:.3f}"
        f" vote_score={vs:.4f}"
        f" status={status}"
    )
