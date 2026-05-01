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
# Bootstrap defaults — intentionally relaxed to restore trading and gather data.
# Tighten after observing candidate journal / reject diagnostics.
DEFAULT_VOTE_THRESHOLD          = 0.50   # probability floor to count as "yes" vote
DEFAULT_VOTE_YES_FRACTION_MIN   = 0.34   # min fraction of active models voting yes
DEFAULT_TOP3_MEAN_MIN           = 0.55   # min mean of top-3 active probabilities
DEFAULT_VOTE_EV_FLOOR           = 0.001  # minimum EV for all profit-voting entries

# Chop market mode — stricter supermajority (still stricter than trend, but relaxed)
DEFAULT_CHOP_VOTE_YES_FRAC_MIN  = 0.50   # yes-fraction requirement in chop
DEFAULT_CHOP_TOP3_MEAN_MIN      = 0.60   # top-3-mean requirement in chop
DEFAULT_CHOP_PRED_RETURN_MIN    = 0.000  # pred_return floor in chop (disabled)
DEFAULT_CHOP_EV_MIN             = 0.002  # EV floor required in chop

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
    ev_floor=0.0,
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
        Candidate ev_after_costs (used for chop EV check and ev_floor check).
    ev_floor : float
        Minimum EV required for all profit-voting entries (trend + chop).
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

    # EV floor — applies to all market modes
    if ev_floor > 0.0 and ev_score < ev_floor:
        return False, f"ev_floor={ev_score:.4f} < {ev_floor}"

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


# ── Ruthless active-model promotion ──────────────────────────────────────────
# In ruthless profit-voting mode, models listed in RUTHLESS_ACTIVE_MODELS that
# are currently in shadow_votes are promoted into active_votes for the purpose
# of vote-score / gate computation.  Backward-compat fields (class_proba,
# std_proba, n_agree) are updated to reflect the promoted active pool.
# Models in RUTHLESS_DIAGNOSTIC_MODELS are never promoted.

def apply_ruthless_active_promotion(conf, active_models, diagnostic_models=None):
    """Promote shadow models to active pool in ruthless profit-voting mode.

    Modifies *conf* in-place.  Models listed in *active_models* that are
    currently in conf['shadow_votes'] are moved to conf['active_votes'] for
    vote-score computation.  Models in *diagnostic_models* are never promoted.

    Parameters
    ----------
    conf              : dict — from predict_with_confidence (modified in-place)
    active_models     : list[str] — model IDs to include as active
    diagnostic_models : list[str] or None — model IDs never promoted (gnb, lr, lr_bal)

    Returns
    -------
    int — number of models added to active pool by promotion
    """
    if not active_models:
        return 0

    diag_set    = set(diagnostic_models or [])
    shadow      = conf.get("shadow_votes", {})
    current_act = conf.get("active_votes", {})

    # Build promoted set: existing active + shadow models in active_models list
    promoted = {}
    for mid, proba in current_act.items():
        if mid not in diag_set:
            promoted[mid] = proba

    added = 0
    for mid in active_models:
        if mid in diag_set:
            continue
        if mid in promoted:
            continue  # already active
        if mid in shadow:
            promoted[mid] = shadow[mid]
            added += 1

    if added == 0:
        return 0  # nothing to do

    # Recompute vote-score fields from promoted active pool
    vs = compute_vote_score(promoted)

    conf["active_votes"]         = promoted
    conf["active_model_count"]   = vs["active_model_count"]
    conf["vote_yes_fraction"]    = vs["vote_yes_fraction"]
    conf["top3_mean"]            = vs["top3_mean"]
    conf["vote_score"]           = vs["vote_score"]

    # Update backward-compat fields to reflect promoted active pool
    if promoted:
        vals = list(promoted.values())
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        nagree = int(sum(1 for v in vals if v >= 0.5))
        conf["class_proba"]    = mean_v
        conf["mean_proba"]     = mean_v
        conf["std_proba"]      = std_v
        conf["n_agree"]        = nagree
        conf["active_mean"]    = mean_v
        conf["active_std"]     = std_v
        conf["active_n_agree"] = nagree

    return added


# ── Profit-voting reject counters ─────────────────────────────────────────────

def make_pv_counters():
    """Return a fresh profit-voting reject counter dict."""
    return {
        "candidates":         0,
        "pass":               0,
        "fail_active_count":  0,
        "fail_ev_floor":      0,
        "fail_yes_frac":      0,
        "fail_top3":          0,
        "fail_chop_yes_frac": 0,
        "fail_chop_top3":     0,
        "fail_chop_pred":     0,
        "fail_chop_ev":       0,
        "no_active_votes":    0,
    }


def increment_pv_counter(pv_counters, reason):
    """Increment the appropriate reject counter from a gate reject reason string."""
    if "active_count" in reason:
        pv_counters["fail_active_count"] += 1
    elif "ev_floor" in reason:
        pv_counters["fail_ev_floor"] += 1
    elif "chop: vote_yes_frac" in reason:
        pv_counters["fail_chop_yes_frac"] += 1
    elif "chop: top3_mean" in reason:
        pv_counters["fail_chop_top3"] += 1
    elif "chop: pred_return" in reason:
        pv_counters["fail_chop_pred"] += 1
    elif "chop: ev_score" in reason:
        pv_counters["fail_chop_ev"] += 1
    elif "vote_yes_frac" in reason:
        pv_counters["fail_yes_frac"] += 1
    elif "top3_mean" in reason:
        pv_counters["fail_top3"] += 1
    elif "no_active" in reason:
        pv_counters["no_active_votes"] += 1


def format_pv_reject_log(sym_str, conf, market_mode, reason):
    """Format a compact per-candidate profit-voting reject log line.

    Example:
        [pv_reject] ADAUSD mode=chop reason=top3_mean yes_frac=0.50 top3=0.54 ev=0.003 active_n=5
    """
    yf  = conf.get("vote_yes_fraction", 0.0)
    t3  = conf.get("top3_mean", 0.0)
    n   = conf.get("active_model_count", len(conf.get("active_votes", {})))
    ev  = 0.0  # caller may inject via conf if needed
    pr  = conf.get("pred_return", 0.0)
    mode_str = f" mode={market_mode}" if market_mode else ""
    return (
        f"[pv_reject] {sym_str}{mode_str}"
        f" reason={reason}"
        f" yes_frac={yf:.2f} top3={t3:.3f}"
        f" pred={pr:.4f}"
        f" active_n={n}"
    )


def format_pv_summary_log(pv_counters):
    """Format a one-line profit-voting cycle summary log.

    Example:
        [pv_summary] candidates=18 pass=0 fail_active=2 fail_yes_frac=7 fail_top3=5 fail_chop=4
    """
    c   = pv_counters
    return (
        f"[pv_summary] candidates={c['candidates']}"
        f" pass={c['pass']}"
        f" fail_active_count={c['fail_active_count']}"
        f" fail_ev_floor={c['fail_ev_floor']}"
        f" fail_yes_frac={c['fail_yes_frac']}"
        f" fail_top3={c['fail_top3']}"
        f" fail_chop_yes_frac={c['fail_chop_yes_frac']}"
        f" fail_chop_top3={c['fail_chop_top3']}"
        f" fail_chop_pred={c['fail_chop_pred']}"
        f" fail_chop_ev={c['fail_chop_ev']}"
        f" no_active={c['no_active_votes']}"
    )

