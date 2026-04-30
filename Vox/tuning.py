# ── Vox Tuning — Good-Market-Mode Relaxation ─────────────────────────────────
#
# Controlled parameter relaxation for ruthless mode in favorable market modes.
#
# Problem: ruthless mode had only 11 round trips in ~16 months.
# Goal: carefully increase sample size only when market is in pump/risk_on_trend.
#
# This module provides helpers to compute slightly relaxed confirmation/filter
# thresholds when the market is in a good mode, without touching chop/selloff.
# ─────────────────────────────────────────────────────────────────────────────

# ── Default good-mode constants ───────────────────────────────────────────────
# These are applied only when risk_profile=ruthless AND market_mode is favorable.
# Overridable via config.py / QC parameters.

RUTHLESS_GOOD_MODES = ["risk_on_trend", "pump"]

# Slightly lower meta-filter threshold in good modes (0.55 -> 0.52)
RUTHLESS_GOOD_MODE_META_MIN_PROBA = 0.52

# Slightly lower minimum EV in good modes (0.006 -> 0.004 for confirm, 0.0 base)
RUTHLESS_GOOD_MODE_MIN_EV = 0.004

# Slightly lower volume-ratio requirement in good modes (1.5 -> 1.3)
RUTHLESS_GOOD_MODE_VOLUME_MIN = 1.3

# Master switch — set False to disable all good-mode relaxation
RUTHLESS_GOOD_MODE_RELAXATION = True


# ── Core relaxation helper ────────────────────────────────────────────────────

def get_relaxed_thresholds(
    market_mode,
    risk_profile,
    base_confirm_ev_min,
    base_confirm_volr_min,
    base_meta_min_proba,
    good_modes=None,
    relaxation_enabled=True,
    relaxed_ev_min=RUTHLESS_GOOD_MODE_MIN_EV,
    relaxed_volr_min=RUTHLESS_GOOD_MODE_VOLUME_MIN,
    relaxed_meta_min_proba=RUTHLESS_GOOD_MODE_META_MIN_PROBA,
):
    """Return (confirm_ev_min, confirm_volr_min, meta_min_proba) for the current bar.

    In favorable market modes (pump / risk_on_trend), returns slightly relaxed
    thresholds.  In all other modes, returns the base (strict) thresholds.

    Only active when ``risk_profile == 'ruthless'`` and ``relaxation_enabled``.

    Parameters
    ----------
    market_mode           : str or None
    risk_profile          : str
    base_confirm_ev_min   : float — strict EV threshold
    base_confirm_volr_min : float — strict volume-ratio threshold
    base_meta_min_proba   : float — strict meta-filter threshold
    good_modes            : list[str] or None — defaults to RUTHLESS_GOOD_MODES
    relaxation_enabled    : bool
    relaxed_ev_min        : float
    relaxed_volr_min      : float
    relaxed_meta_min_proba: float

    Returns
    -------
    tuple (confirm_ev_min, confirm_volr_min, meta_min_proba)
    """
    if risk_profile != "ruthless" or not relaxation_enabled:
        return base_confirm_ev_min, base_confirm_volr_min, base_meta_min_proba

    allowed = good_modes if good_modes is not None else RUTHLESS_GOOD_MODES
    if market_mode in allowed:
        return (
            min(base_confirm_ev_min,   relaxed_ev_min),
            min(base_confirm_volr_min, relaxed_volr_min),
            min(base_meta_min_proba,   relaxed_meta_min_proba),
        )

    # Strict modes (chop, selloff, high_vol_reversal, None)
    return base_confirm_ev_min, base_confirm_volr_min, base_meta_min_proba


def is_good_mode(market_mode, good_modes=None):
    """Return True if market_mode is in the list of favorable modes.

    Parameters
    ----------
    market_mode : str or None
    good_modes  : list[str] or None — defaults to RUTHLESS_GOOD_MODES
    """
    allowed = good_modes if good_modes is not None else RUTHLESS_GOOD_MODES
    return market_mode in allowed


def format_relaxation_log(
    market_mode,
    base_ev,
    eff_ev,
    base_volr,
    eff_volr,
    base_meta,
    eff_meta,
):
    """Format a log line showing when good-mode relaxation is applied.

    Example::

        [relax] mode=pump ev=0.006->0.004 volr=1.5->1.3 meta=0.55->0.52
    """
    parts = [f"[relax] mode={market_mode}"]
    if eff_ev < base_ev:
        parts.append(f"ev={base_ev:.3f}->{eff_ev:.3f}")
    if eff_volr < base_volr:
        parts.append(f"volr={base_volr:.2f}->{eff_volr:.2f}")
    if eff_meta < base_meta:
        parts.append(f"meta={base_meta:.2f}->{eff_meta:.2f}")
    if len(parts) == 1:
        parts.append("(no change)")
    return " ".join(parts)


# ── Parameter resolution from config/algo ────────────────────────────────────

def resolve_good_mode_params(algo, config_module):
    """Resolve good-mode relaxation parameters from algo QC params or config.

    Returns
    -------
    dict with keys:
        enabled, good_modes, relaxed_ev_min, relaxed_volr_min, relaxed_meta_min_proba
    """
    enabled = getattr(config_module, "RUTHLESS_GOOD_MODE_RELAXATION", RUTHLESS_GOOD_MODE_RELAXATION)
    # Allow QC param override
    _raw = None
    try:
        _raw = algo.get_parameter("ruthless_good_mode_relaxation")
    except Exception:
        pass
    if _raw:
        enabled = str(_raw).lower() in ("true", "1", "yes")

    relaxed_ev   = getattr(config_module, "RUTHLESS_GOOD_MODE_MIN_EV",         RUTHLESS_GOOD_MODE_MIN_EV)
    relaxed_volr = getattr(config_module, "RUTHLESS_GOOD_MODE_VOLUME_MIN",     RUTHLESS_GOOD_MODE_VOLUME_MIN)
    relaxed_meta = getattr(config_module, "RUTHLESS_GOOD_MODE_META_MIN_PROBA", RUTHLESS_GOOD_MODE_META_MIN_PROBA)

    # Allow QC param overrides
    try:
        _v = algo.get_parameter("ruthless_good_mode_min_ev")
        if _v:
            relaxed_ev = float(_v)
    except Exception:
        pass
    try:
        _v = algo.get_parameter("ruthless_good_mode_volume_min")
        if _v:
            relaxed_volr = float(_v)
    except Exception:
        pass
    try:
        _v = algo.get_parameter("ruthless_good_mode_meta_min_proba")
        if _v:
            relaxed_meta = float(_v)
    except Exception:
        pass

    return {
        "enabled":                 enabled,
        "good_modes":              list(RUTHLESS_GOOD_MODES),
        "relaxed_ev_min":          relaxed_ev,
        "relaxed_volr_min":        relaxed_volr,
        "relaxed_meta_min_proba":  relaxed_meta,
    }
