# ── Ruthless V2 — multi-horizon scoring functions ────────────────────────────
#
# compute_multihorizon_scores()    — scalp / continuation / runner lane scores
# compute_v2_opportunity_score()   — cross-sectional ranking formula
# compute_breakout_score()         — breakout potential score
# compute_volume_expansion_score() — volume expansion score
# compute_regime_score()           — regime quality score
# compute_relative_strength_scores() — cross-sectional relative strength
# ─────────────────────────────────────────────────────────────────────────────


def compute_multihorizon_scores(feat, conf, ev_score, pred_return, market_mode=None):
    """Compute scalp / continuation / runner lane scores.

    Uses existing features and model signals — no separate model training needed.
    Scores are heuristic approximations:
      - scalp_score        : fast short-term momentum + low dispersion
      - continuation_score : medium-term momentum + volume expansion + model agreement
      - runner_score       : strong breakout potential + high EV + regime alignment

    Parameters
    ----------
    feat        : list/array — feature vector from build_features()
                  [ret_1, ret_4, ret_8, ret_16, ret_32, vol_price, vol_r,
                   btc_rel, rsi, bb_pos, bb_width, ...]
    conf        : dict — predict_with_confidence output
    ev_score    : float — ev_after_costs
    pred_return : float — predicted return from regressor
    market_mode : str or None

    Returns
    -------
    dict with keys: scalp_score, continuation_score, runner_score, lane_selected
    """
    if feat is None or len(feat) < 12:
        return {
            "scalp_score":        0.0,
            "continuation_score": 0.0,
            "runner_score":       0.0,
            "lane_selected":      "scalp",
        }

    try:
        ret_1  = float(feat[0])
        ret_4  = float(feat[1])
        ret_8  = float(feat[2])
        ret_16 = float(feat[3])
        vol_r  = float(feat[6])
        rsi    = float(feat[8])   if len(feat) > 8  else 50.0
        bb_pos = float(feat[9])   if len(feat) > 9  else 0.0
        bb_w   = float(feat[10])  if len(feat) > 10 else 0.02
    except (TypeError, IndexError, ValueError):
        return {
            "scalp_score":        0.0,
            "continuation_score": 0.0,
            "runner_score":       0.0,
            "lane_selected":      "scalp",
        }

    active_count  = conf.get("active_model_count", 0)
    vote_score    = conf.get("vote_score", 0.0)
    top3_mean     = conf.get("top3_mean", 0.0)
    n_agree       = conf.get("active_n_agree", conf.get("n_agree", 0))
    class_proba   = conf.get("class_proba", 0.0)

    # ── Scalp lane (30–90 min) ────────────────────────────────────────────────
    scalp_momentum = max(0.0, min(ret_4 * 10.0, 1.0))
    scalp_rsi_ok   = 1.0 - max(0.0, min((rsi - 70.0) / 30.0, 1.0))
    scalp_vol_ok   = max(0.0, min((vol_r - 1.0) / 2.0, 1.0))
    scalp_bb_ok    = max(0.0, min(bb_pos, 1.0))
    scalp_score    = (
        0.40 * scalp_momentum
        + 0.25 * scalp_rsi_ok
        + 0.20 * scalp_vol_ok
        + 0.15 * scalp_bb_ok
    )
    scalp_score = max(0.0, min(scalp_score, 1.0))

    # ── Continuation lane (2–8h) ──────────────────────────────────────────────
    cont_momentum  = max(0.0, min((ret_16 * 5.0), 1.0))
    cont_vol       = max(0.0, min((vol_r - 1.0) / 3.0, 1.0))
    cont_model     = max(0.0, min((top3_mean - 0.5) * 2.0, 1.0))
    cont_ev        = max(0.0, min(ev_score * 50.0, 1.0))
    continuation_score = (
        0.35 * cont_momentum
        + 0.25 * cont_vol
        + 0.25 * cont_model
        + 0.15 * cont_ev
    )
    continuation_score = max(0.0, min(continuation_score, 1.0))

    # ── Runner lane (12–48h) ──────────────────────────────────────────────────
    runner_momentum = max(0.0, min((ret_16 + ret_8 * 0.5) * 4.0, 1.0))
    runner_vol      = max(0.0, min((vol_r - 1.5) / 3.0, 1.0))
    runner_model    = max(0.0, min((vote_score - 0.5) * 2.0, 1.0))
    runner_ev       = max(0.0, min(ev_score * 40.0, 1.0))
    regime_bonus    = 0.15 if market_mode in ("pump", "risk_on_trend") else 0.0
    runner_score    = (
        0.30 * runner_momentum
        + 0.20 * runner_vol
        + 0.25 * runner_model
        + 0.15 * runner_ev
        + 0.10 * max(0.0, min((class_proba - 0.5) * 2.0, 1.0))
        + regime_bonus
    )
    runner_score = max(0.0, min(runner_score, 1.0))

    scores = {
        "scalp":        scalp_score,
        "continuation": continuation_score,
        "runner":       runner_score,
    }
    lane_selected = max(scores, key=scores.__getitem__)

    return {
        "scalp_score":        round(scalp_score, 4),
        "continuation_score": round(continuation_score, 4),
        "runner_score":       round(runner_score, 4),
        "lane_selected":      lane_selected,
    }


def compute_v2_opportunity_score(
    dynamic_vote_score,
    continuation_score,
    runner_score,
    breakout_score,
    volume_expansion_score,
    regime_score,
    cost_penalty=0.0,
    exhaustion_penalty=0.0,
    relative_strength_score=0.0,
):
    """Compute V2 cross-sectional opportunity score.

    Formula::
        ruthless_v2_score = (
            0.25 * dynamic_vote_score
            + 0.20 * continuation_score
            + 0.20 * runner_score
            + 0.15 * breakout_score
            + 0.10 * volume_expansion_score
            + 0.10 * regime_score
            - cost_penalty
            - exhaustion_penalty
        )
    """
    raw = (
        0.25 * dynamic_vote_score
        + 0.20 * continuation_score
        + 0.20 * runner_score
        + 0.15 * breakout_score
        + 0.10 * volume_expansion_score
        + 0.10 * regime_score
        - cost_penalty
        - exhaustion_penalty
    )
    return raw


def compute_breakout_score(feat):
    """Compute a simple breakout/breakout-potential score from features."""
    if feat is None or len(feat) < 11:
        return 0.0
    try:
        ret_4  = float(feat[1])
        ret_16 = float(feat[3])
        vol_r  = float(feat[6])
        bb_pos = float(feat[9])   if len(feat) > 9  else 0.0
        bb_w   = float(feat[10])  if len(feat) > 10 else 0.02
    except (TypeError, IndexError, ValueError):
        return 0.0

    mom_score   = max(0.0, min((ret_4 * 8.0 + ret_16 * 4.0), 1.0))
    vol_score   = max(0.0, min((vol_r - 1.0) / 3.0, 1.0))
    bb_score    = max(0.0, min(bb_pos * 1.5, 1.0))
    width_score = max(0.0, min(bb_w / 0.10, 1.0))
    return max(0.0, min(
        0.40 * mom_score + 0.30 * vol_score + 0.20 * bb_score + 0.10 * width_score,
        1.0,
    ))


def compute_volume_expansion_score(feat):
    """Return a [0, 1] score for current volume expansion."""
    if feat is None or len(feat) < 7:
        return 0.0
    try:
        vol_r = float(feat[6])
    except (TypeError, IndexError, ValueError):
        return 0.0
    return max(0.0, min((vol_r - 1.0) / 4.0, 1.0))


def compute_regime_score(market_mode):
    """Convert market_mode string to a numeric [0, 1] regime quality score."""
    if market_mode is None:
        return 0.4
    mm = str(market_mode).lower()
    if "pump" in mm:
        return 1.0
    if "risk_on_trend" in mm:
        return 0.85
    if "trend" in mm:
        return 0.65
    if "chop" in mm:
        return 0.30
    if "selloff" in mm or "bear" in mm:
        return 0.10
    return 0.40


def compute_relative_strength_scores(candidates_feat_map):
    """Compute cross-sectional relative strength scores.

    Parameters
    ----------
    candidates_feat_map : dict[str, list/array]
        symbol -> feature vector

    Returns
    -------
    dict[str, float] — symbol -> relative_strength_score in [0, 1]
    dict[str, int]   — symbol -> rank (1 = strongest)
    """
    if not candidates_feat_map:
        return {}, {}

    sym_ret = {}
    for sym, feat in candidates_feat_map.items():
        if feat is None or len(feat) < 4:
            sym_ret[sym] = 0.0
        else:
            try:
                sym_ret[sym] = float(feat[3])
            except (TypeError, IndexError, ValueError):
                sym_ret[sym] = 0.0

    vals = list(sym_ret.values())
    if not vals:
        return {}, {}

    v_min = min(vals)
    v_max = max(vals)
    v_range = v_max - v_min

    rs_scores = {}
    for sym, v in sym_ret.items():
        rs_scores[sym] = (v - v_min) / v_range if v_range > 1e-9 else 0.5

    sorted_syms = sorted(sym_ret, key=sym_ret.__getitem__, reverse=True)
    ranks = {sym: i + 1 for i, sym in enumerate(sorted_syms)}

    return rs_scores, ranks
