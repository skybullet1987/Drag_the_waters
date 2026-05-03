# ── Ruthless V2 — machine-gun mode helpers ───────────────────────────────────
#
# apply_regime_soft_penalty()    — soft regime block in machine-gun mode
# apply_meta_soft_penalty()      — soft meta-filter in machine-gun mode
# select_top_n_machine_gun()     — force top-N entries when slots are open
# compute_machine_gun_allocation() — tier-based allocation in machine-gun mode
# format_v2_startup_log()        — startup audit log formatter
# ─────────────────────────────────────────────────────────────────────────────

from .cfg import (
    RUTHLESS_V2_SOFT_REGIME_PENALTY,
    RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER,
    RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT,
    RUTHLESS_V2_HARD_BLOCK_MODES,
    RUTHLESS_V2_META_SCORE_WEIGHT,
    RUTHLESS_V2_LOW_META_ALLOC_MULT,
    RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES,
    RUTHLESS_V2_MIN_SCORE_TO_TRADE,
    RUTHLESS_V2_BASE_ALLOCATION,
    RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION,
    RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
    RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC,
    RUTHLESS_V2_MACHINE_GUN_MODE,
    RUTHLESS_V2_REGIME_HARD_BLOCK,
    RUTHLESS_V2_META_HARD_FILTER,
    RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
    RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
    RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
)


def apply_regime_soft_penalty(
    symbol,
    score,
    market_mode,
    machine_gun_mode=True,
    regime_hard_block=False,
    soft_penalty=None,
    hard_block_modes=None,
    _log_fn=None,
):
    """Apply regime filter in machine-gun mode.

    In machine-gun mode with regime_hard_block=False, normal regime blocks
    (e.g. chop) become soft score penalties instead of hard rejects.  Only
    truly dangerous modes (risk_off_crash, dump, emergency) remain hard blocks.

    Returns
    -------
    (score: float, blocked: bool, log_msg: str or None)
    """
    if soft_penalty is None:
        soft_penalty = RUTHLESS_V2_SOFT_REGIME_PENALTY
    if hard_block_modes is None:
        hard_block_modes = RUTHLESS_V2_HARD_BLOCK_MODES

    mm = str(market_mode).lower() if market_mode else ""

    for danger in hard_block_modes:
        if danger in mm:
            msg = f"[v2_hard_regime] {symbol} mode={market_mode} hard_block=True"
            if _log_fn:
                _log_fn(msg)
            return score, True, msg

    if not machine_gun_mode or regime_hard_block:
        if mm and mm not in ("risk_on_trend", "pump", "trend"):
            return score, True, None
        return score, False, None

    is_chop    = "chop" in mm
    is_selloff = "selloff" in mm or "bear" in mm
    is_weak    = mm and mm not in ("risk_on_trend", "pump", "trend")

    penalty = 0.0
    if is_selloff:
        penalty = soft_penalty * RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER
    elif is_chop:
        penalty = soft_penalty
    elif is_weak:
        penalty = soft_penalty * RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT

    if penalty > 0.0:
        score_before = score
        score        = score - penalty
        msg = (
            f"[v2_soft_regime] {symbol} mode={market_mode}"
            f" penalty={penalty:.2f}"
            f" score_before={score_before:.4f}"
            f" score_after={score:.4f}"
        )
        if _log_fn:
            _log_fn(msg)
        return score, False, msg

    return score, False, None


def apply_meta_soft_penalty(
    symbol,
    score,
    allocation,
    meta_score,
    machine_gun_mode=True,
    meta_hard_filter=False,
    meta_as_score_penalty=True,
    meta_score_weight=None,
    low_meta_alloc_mult=None,
    meta_score_floor=-1.0,
    _log_fn=None,
):
    """Apply meta-filter in machine-gun mode.

    In machine-gun mode with meta_hard_filter=False, meta-score modifies
    score and allocation instead of hard-rejecting the candidate.

    Returns
    -------
    (score: float, allocation: float, blocked: bool, log_msg: str or None)
    """
    if meta_score_weight is None:
        meta_score_weight = RUTHLESS_V2_META_SCORE_WEIGHT
    if low_meta_alloc_mult is None:
        low_meta_alloc_mult = RUTHLESS_V2_LOW_META_ALLOC_MULT

    if meta_score < meta_score_floor:
        msg = f"[v2_meta_emergency_reject] {symbol} meta={meta_score:.3f} < floor={meta_score_floor:.3f}"
        if _log_fn:
            _log_fn(msg)
        return score, allocation, True, msg

    if not machine_gun_mode or meta_hard_filter:
        return score, allocation, False, None

    alloc_mult = 1.0
    if meta_as_score_penalty:
        score = score + meta_score_weight * meta_score

    if meta_score < 0.0:
        alloc_mult = low_meta_alloc_mult
        allocation = allocation * alloc_mult

    msg = (
        f"[v2_soft_meta] {symbol}"
        f" meta={meta_score:.3f}"
        f" score_after={score:.4f}"
        f" alloc_mult={alloc_mult:.2f}"
    )
    if _log_fn:
        _log_fn(msg)
    return score, allocation, False, msg


def select_top_n_machine_gun(
    ranked_candidates,
    open_slots,
    force_top_n=None,
    min_score=None,
    _log_fn=None,
):
    """Select up to force_top_n candidates from ranked list for machine-gun entries.

    Returns
    -------
    list of dict — the selected candidate dicts (may be empty)
    """
    if force_top_n is None:
        force_top_n = RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES
    if min_score is None:
        min_score = RUTHLESS_V2_MIN_SCORE_TO_TRADE

    n_candidates = len(ranked_candidates)
    take          = min(force_top_n, open_slots, n_candidates)

    eligible = [
        c for c in ranked_candidates
        if c.get("v2_opportunity_score", 0.0) >= min_score
    ]

    selected = eligible[:take]

    msg = (
        f"[v2_rank] candidates={n_candidates}"
        f" eligible={len(eligible)}"
        f" slots={open_slots}"
        f" force_top_n={force_top_n}"
        f" taking={len(selected)}"
    )
    if _log_fn:
        _log_fn(msg)

    return selected


def compute_machine_gun_allocation(
    score,
    lane,
    market_mode=None,
    base_alloc=None,
    high_conviction_alloc=None,
    max_symbol_alloc=None,
    chop_scalp_max_alloc=None,
    allow_chop_scalps=True,
):
    """Compute allocation for machine-gun mode based on score and lane.

    Allocation tiers:
      low score scalp/chop      -> 0.08–0.10
      medium score              -> 0.12–0.18
      high score continuation   -> 0.20–0.25
      high conviction runner    -> up to high_conviction_alloc

    Returns
    -------
    float — allocation fraction in [RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, max_symbol_alloc]
    """
    if base_alloc is None:
        base_alloc = RUTHLESS_V2_BASE_ALLOCATION
    if high_conviction_alloc is None:
        high_conviction_alloc = RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION
    if max_symbol_alloc is None:
        max_symbol_alloc = RUTHLESS_V2_MAX_SYMBOL_ALLOCATION
    if chop_scalp_max_alloc is None:
        chop_scalp_max_alloc = RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC

    mm = str(market_mode).lower() if market_mode else ""
    is_chop = "chop" in mm

    if lane == "runner" and score >= 0.50:
        alloc = min(high_conviction_alloc + (score - 0.50) * 0.10, max_symbol_alloc)
    elif lane == "runner" and score >= 0.30:
        alloc = 0.20 + (score - 0.30) * 0.25
    elif lane == "continuation" and score >= 0.35:
        alloc = 0.14 + (score - 0.35) * 0.22
    elif score >= 0.20:
        alloc = base_alloc + (score - 0.20) * 0.10
    elif score >= 0.05:
        alloc = base_alloc
    else:
        alloc = max(RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, 0.08 + max(score, 0.0) * 0.20)

    if is_chop and allow_chop_scalps:
        alloc = min(alloc, chop_scalp_max_alloc)

    alloc = max(RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, min(alloc, max_symbol_alloc))
    return round(alloc, 4)


def format_v2_startup_log(
    risk_profile, v2_mode, max_positions, active_models,
    dynamic_weights=None,
    machine_gun_mode=None,
    regime_hard_block=None,
    meta_hard_filter=None,
    force_top_n=None,
    min_score_to_trade=None,
):
    """Format V2 startup audit log lines (returns list of strings)."""
    lines = [
        f"[profile] risk_profile={risk_profile} v2={v2_mode}"
        f" max_positions={max_positions}"
        f" active_models={','.join(active_models)}",
    ]
    if dynamic_weights:
        w_str = " ".join(f"{m}={w:.3f}" for m, w in sorted(dynamic_weights.items()))
        lines.append(f"[v2_weights] {w_str}")

    mg = machine_gun_mode if machine_gun_mode is not None else RUTHLESS_V2_MACHINE_GUN_MODE
    rh = regime_hard_block if regime_hard_block is not None else RUTHLESS_V2_REGIME_HARD_BLOCK
    mh = meta_hard_filter  if meta_hard_filter  is not None else RUTHLESS_V2_META_HARD_FILTER
    fn = force_top_n       if force_top_n        is not None else RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES
    ms = min_score_to_trade if min_score_to_trade is not None else RUTHLESS_V2_MIN_SCORE_TO_TRADE
    lines.append(
        f"[v2_machine_gun] enabled={mg}"
        f" regime_hard_block={rh}"
        f" meta_hard_filter={mh}"
        f" force_top_n={fn}"
        f" min_score={ms}"
        f" max_entries_day={RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY}"
        f" max_concurrent={RUTHLESS_V2_MAX_CONCURRENT_POSITIONS}"
        f" reentry_cooldown_min={RUTHLESS_V2_REENTRY_COOLDOWN_MIN}"
    )
    return lines
