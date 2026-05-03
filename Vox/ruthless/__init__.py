# ── Ruthless V2 package ───────────────────────────────────────────────────────
#
# Re-exports the complete public surface of the ruthless V2 engine so that
# ``from Vox.ruthless import ...`` and the backwards-compat shim
# ``Vox/ruthless_v2.py`` both work without modification.
# ─────────────────────────────────────────────────────────────────────────────

from .cfg import (
    RUTHLESS_V2_MODE,
    RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
    RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
    RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY,
    RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
    RUTHLESS_V2_DECISION_INTERVAL_MIN,
    RUTHLESS_V2_MIN_SCORE_TO_TRADE,
    RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
    RUTHLESS_V2_MACHINE_GUN_MODE,
    RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES,
    RUTHLESS_V2_REGIME_HARD_BLOCK,
    RUTHLESS_V2_META_HARD_FILTER,
    RUTHLESS_V2_META_AS_SCORE_PENALTY,
    RUTHLESS_V2_META_SCORE_WEIGHT,
    RUTHLESS_V2_LOW_META_ALLOC_MULT,
    RUTHLESS_V2_ALLOW_CHOP_SCALPS,
    RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC,
    RUTHLESS_V2_BASE_ALLOCATION,
    RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION,
    RUTHLESS_V2_SOFT_REGIME_PENALTY,
    RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER,
    RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT,
    RUTHLESS_V2_HARD_BLOCK_MODES,
    RUTHLESS_V2_PARTIAL_TP_ENABLED,
    RUTHLESS_V2_PARTIAL_TP_FRACTION,
    RUTHLESS_V2_SCALP_TP,
    RUTHLESS_V2_CONTINUATION_TP,
    RUTHLESS_V2_RUNNER_INITIAL_TP,
    RUTHLESS_V2_RUNNER_TRAIL_PCT,
    RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT,
    RUTHLESS_V2_ACTIVE_MODELS,
    RUTHLESS_V2_OPTIONAL_MODELS,
    RUTHLESS_V2_DIAGNOSTIC_MODELS,
    RUTHLESS_V2_BASE_WEIGHTS,
    RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST,
    RUTHLESS_V2_DECAY_FACTOR,
)

from .positions import (
    MultiPositionManager,
    DynamicVoterWeighting,
    _date_key,
)

from .scoring import (
    compute_multihorizon_scores,
    compute_v2_opportunity_score,
    compute_breakout_score,
    compute_volume_expansion_score,
    compute_regime_score,
    compute_relative_strength_scores,
)

from .pump import (
    compute_pump_scores,
    exhaustion_override_allowed,
)

from .meta import (
    compute_meta_entry_score,
    SplitExitHelper,
    rank_candidates_v2,
)

from .machine_gun import (
    apply_regime_soft_penalty,
    apply_meta_soft_penalty,
    select_top_n_machine_gun,
    compute_machine_gun_allocation,
    format_v2_startup_log,
)

from .apex import (
    _APEX_WEIGHTS,
    compute_apex_score,
    apex_entry_decision,
    compute_apex_size,
    compute_apex_atr_stops,
    apex_breakout_signal,
    apex_pullback_signal,
    apex_momentum_continuation_signal,
    apex_rejected_entry_log,
)

__all__ = [
    # cfg
    "RUTHLESS_V2_MODE",
    "RUTHLESS_V2_MAX_CONCURRENT_POSITIONS",
    "RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY",
    "RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY",
    "RUTHLESS_V2_MAX_SYMBOL_ALLOCATION",
    "RUTHLESS_V2_MIN_SYMBOL_ALLOCATION",
    "RUTHLESS_V2_MAX_TOTAL_EXPOSURE",
    "RUTHLESS_V2_DECISION_INTERVAL_MIN",
    "RUTHLESS_V2_MIN_SCORE_TO_TRADE",
    "RUTHLESS_V2_REENTRY_COOLDOWN_MIN",
    "RUTHLESS_V2_MACHINE_GUN_MODE",
    "RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES",
    "RUTHLESS_V2_REGIME_HARD_BLOCK",
    "RUTHLESS_V2_META_HARD_FILTER",
    "RUTHLESS_V2_META_AS_SCORE_PENALTY",
    "RUTHLESS_V2_META_SCORE_WEIGHT",
    "RUTHLESS_V2_LOW_META_ALLOC_MULT",
    "RUTHLESS_V2_ALLOW_CHOP_SCALPS",
    "RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC",
    "RUTHLESS_V2_BASE_ALLOCATION",
    "RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION",
    "RUTHLESS_V2_SOFT_REGIME_PENALTY",
    "RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER",
    "RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT",
    "RUTHLESS_V2_HARD_BLOCK_MODES",
    "RUTHLESS_V2_PARTIAL_TP_ENABLED",
    "RUTHLESS_V2_PARTIAL_TP_FRACTION",
    "RUTHLESS_V2_SCALP_TP",
    "RUTHLESS_V2_CONTINUATION_TP",
    "RUTHLESS_V2_RUNNER_INITIAL_TP",
    "RUTHLESS_V2_RUNNER_TRAIL_PCT",
    "RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT",
    "RUTHLESS_V2_ACTIVE_MODELS",
    "RUTHLESS_V2_OPTIONAL_MODELS",
    "RUTHLESS_V2_DIAGNOSTIC_MODELS",
    "RUTHLESS_V2_BASE_WEIGHTS",
    "RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER",
    "RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER",
    "RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST",
    "RUTHLESS_V2_DECAY_FACTOR",
    # positions
    "MultiPositionManager",
    "DynamicVoterWeighting",
    "_date_key",
    # scoring
    "compute_multihorizon_scores",
    "compute_v2_opportunity_score",
    "compute_breakout_score",
    "compute_volume_expansion_score",
    "compute_regime_score",
    "compute_relative_strength_scores",
    # pump
    "compute_pump_scores",
    "exhaustion_override_allowed",
    # meta
    "compute_meta_entry_score",
    "SplitExitHelper",
    "rank_candidates_v2",
    # machine_gun
    "apply_regime_soft_penalty",
    "apply_meta_soft_penalty",
    "select_top_n_machine_gun",
    "compute_machine_gun_allocation",
    "format_v2_startup_log",
    # apex
    "_APEX_WEIGHTS",
    "compute_apex_score",
    "apex_entry_decision",
    "compute_apex_size",
    "compute_apex_atr_stops",
    "apex_breakout_signal",
    "apex_pullback_signal",
    "apex_momentum_continuation_signal",
    "apex_rejected_entry_log",
]
