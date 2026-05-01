# ── Vox Diagnostics ───────────────────────────────────────────────────────────
#
# Formatting helpers for vote logs and exit diagnostics.
#
# These are pure functions — no side effects, no QC dependencies.
# All formatters return strings suitable for self.log() or self.debug().
# ─────────────────────────────────────────────────────────────────────────────


# ── Feature diagnostics ───────────────────────────────────────────────────────

def _feature_diag_suffix(ft):
    """Return a compact r4/r16/volume-ratio suffix string from a feature vector.

    Safe against None, too-short vectors, and NumPy arrays (avoids ambiguous
    truth-value checks on multi-element arrays).

    Parameters
    ----------
    ft : array-like or None

    Returns
    -------
    str — e.g. " r4=0.0123 r16=0.0234 vr=1.45", or "" when unavailable.
    """
    try:
        if ft is None or len(ft) <= 6:
            return ""
        return f" r4={float(ft[1]):.4f} r16={float(ft[3]):.4f} vr={float(ft[6]):.2f}"
    except Exception:
        return ""


# ── Vote log formatting ───────────────────────────────────────────────────────

def format_vote_log(symbol, conf, meta_score=None, market_mode=None):
    """Format a compact per-model vote log line with role-separated vote groups.

    Example output (with roles)::

        [vote] ADAUSD active_mean=0.62 active_std=0.05 agree=3/3 mode=pump
               active=hgbc:0.70,et:0.67,rf:0.61
               shadow=et_shallow:0.64,cal_et:0.65 diag=lr:0.01

    Falls back to legacy format when role fields are absent::

        [vote] ADAUSD mean=0.64 std=0.05 agree=5/6 meta=0.59
               votes=lr:0.55,hgbc:0.62,et:0.70,rf:0.58

    Parameters
    ----------
    symbol     : str
    conf       : dict — confidence dict from predict_with_confidence[_batch]
    meta_score : float or None
    market_mode: str or None

    Returns
    -------
    str
    """
    # Prefer active-role statistics when available
    if "active_mean" in conf:
        mean    = conf["active_mean"]
        std     = conf["active_std"]
        n_agree = conf["active_n_agree"]
        total   = len(conf.get("active_votes", {}))
        line = (
            f"[vote] {symbol}"
            f" active_mean={mean:.2f} active_std={std:.2f}"
            f" agree={n_agree}/{total}"
        )
        if meta_score is not None:
            line += f" meta={meta_score:.2f}"
        if market_mode is not None:
            line += f" mode={market_mode}"
        av = conf.get("active_votes", {})
        if av:
            line += " active=" + ",".join(f"{m}:{v:.2f}" for m, v in av.items())
        sv = conf.get("shadow_votes", {})
        if sv:
            line += " shadow=" + ",".join(f"{m}:{v:.2f}" for m, v in sv.items())
        dv = conf.get("diagnostic_votes", {})
        if dv:
            line += " diag=" + ",".join(f"{m}:{v:.2f}" for m, v in dv.items())
        excl = conf.get("excluded_models", {})
        if excl:
            line += " excluded=" + ",".join(f"{m}:{r}" for m, r in excl.items())
        return line

    # Legacy format (no role fields)
    mean    = conf.get("class_proba", conf.get("mean_proba", 0.0))
    std     = conf.get("std_proba", 0.0)
    n_agree = conf.get("n_agree", 0)
    votes   = conf.get("per_model", {})
    total   = len(votes)

    line = (
        f"[vote] {symbol}"
        f" mean={mean:.2f} std={std:.2f}"
        f" agree={n_agree}/{total}"
    )
    if meta_score is not None:
        line += f" meta={meta_score:.2f}"
    if market_mode is not None:
        line += f" mode={market_mode}"
    if votes:
        vote_str = ",".join(f"{mid}:{v:.2f}" for mid, v in votes.items())
        line += f" votes={vote_str}"
    return line


def format_entry_tag(mean, n_agree, total, meta_score, market_mode):
    """Format a compact entry order tag suitable for order.tag field.

    Example::

        ENTRY|ml|mean=0.64|agree=5/6|meta=0.59|mode=pump
    """
    return (
        f"ENTRY|ml"
        f"|mean={mean:.2f}"
        f"|agree={n_agree}/{total}"
        f"|meta={meta_score:.2f}"
        f"|mode={market_mode or 'n/a'}"
    )


# ── Exit diagnostic formatting ────────────────────────────────────────────────

def format_exit_diagnostic(
    symbol,
    entry_price,
    exit_fill_price,
    exit_reason,
    realized_return,
    sl_use,
    tp_use,
    max_return_seen,
    elapsed_minutes,
    trail_active=False,
    trail_high_px=None,
    breakeven_active=False,
    stop_price=None,
):
    """Format a detailed exit diagnostic string.

    This resolves ambiguous EXIT_SL tags by logging full context.
    Useful for diagnosing suspicious exits (e.g. positive-fill tagged EXIT_SL,
    or losses much larger than the configured stop).

    Example output::

        [exit_diag] SOLUSD entry=268.23 fill=268.34 ret=+0.04%
                    tag=EXIT_SL sl=0.0300 tp=0.0900
                    max_ret=+0.04% held=12.5m breakeven=active

    Parameters
    ----------
    symbol           : str
    entry_price      : float
    exit_fill_price  : float
    exit_reason      : str — raw exit tag
    realized_return  : float
    sl_use           : float — configured stop fraction
    tp_use           : float — configured TP fraction
    max_return_seen  : float — high-water mark return seen
    elapsed_minutes  : float
    trail_active     : bool
    trail_high_px    : float or None
    breakeven_active : bool
    stop_price       : float or None — explicit stop price if available

    Returns
    -------
    str
    """
    # Classify why EXIT_SL may be firing on what looks like a flat/positive fill
    notes = []
    if exit_reason == "EXIT_SL":
        if realized_return >= 0.0:
            notes.append("warn:ret>=0_tagged_sl")
        configured_sl_price = entry_price * (1.0 - sl_use) if entry_price > 0 else None
        if (
            configured_sl_price is not None
            and exit_fill_price > 0
            and exit_fill_price < configured_sl_price - 0.001 * entry_price
        ):
            notes.append("warn:fill_below_configured_sl")
        if breakeven_active and realized_return >= -sl_use * 0.5:
            notes.append("info:breakeven_stop")
    elif exit_reason == "EXIT_TRAIL":
        if trail_active and trail_high_px is not None and trail_high_px > 0:
            drawdown_from_high = (exit_fill_price - trail_high_px) / trail_high_px
            notes.append(f"trail_drawdown={drawdown_from_high:.3%}")

    parts = [
        f"[exit_diag] {symbol}",
        f"entry={entry_price:.5f}",
        f"fill={exit_fill_price:.5f}",
        f"ret={realized_return:+.3%}",
        f"tag={exit_reason}",
        f"sl={sl_use:.4f}",
        f"tp={tp_use:.4f}",
        f"max_ret={max_return_seen:+.3%}",
        f"held={elapsed_minutes:.1f}m",
    ]
    if trail_active and trail_high_px is not None:
        parts.append(f"trail_high={trail_high_px:.5f}")
    if breakeven_active:
        parts.append("breakeven=active")
    if stop_price is not None:
        parts.append(f"stop_px={stop_price:.5f}")
    if notes:
        parts.append("(" + " ".join(notes) + ")")
    return " ".join(parts)


# ── Model attribution summary formatting ─────────────────────────────────────

def format_model_attribution_summary(attribution_dict, n_trades):
    """Format a multi-line per-model attribution summary for logging.

    Parameters
    ----------
    attribution_dict : dict — output of TradeJournal.compute_model_attribution()
    n_trades         : int  — total completed trades used

    Returns
    -------
    str
    """
    if not attribution_dict:
        return f"[model_attr] No attribution data ({n_trades} trades, no votes logged)."

    lines = [f"[model_attr] Per-model attribution ({n_trades} trades — small sample, noisy):"]
    for mid, s in sorted(attribution_dict.items()):
        wr  = s.get("win_rate_when_yes")
        ar  = s.get("avg_return_when_yes")
        arn = s.get("avg_return_when_no")
        line = (
            f"  {mid:<12}"
            f"  yes={s.get('vote_yes_count', 0):>3}"
            f"  no={s.get('vote_no_count', 0):>3}"
        )
        if wr is not None:
            line += f"  wr_yes={wr:.0%}"
        if ar is not None:
            line += f"  avg_ret_yes={ar:+.2%}"
        if arn is not None:
            line += f"  avg_ret_no={arn:+.2%}"
        lines.append(line)
    lines.append(
        "  WARNING: sample size < 50 makes these estimates unreliable."
        " Run multiple windows (2023/2024/2025/bull/chop/selloff)."
    )
    return "\n".join(lines)


# ── Startup configuration log ─────────────────────────────────────────────────

def format_limit_order_startup_log(
    use_entry_limit_orders,
    entry_limit_offset,
    entry_limit_ttl_minutes,
    entry_limit_chase=False,
    use_exit_limit_orders=False,
    exit_limit_offset=None,
    exit_limit_ttl_minutes=None,
):
    """Format a startup log line for entry/exit limit order configuration."""
    return (
        f"[limit_orders]"
        f" entry_limit={use_entry_limit_orders}"
        f" offset={entry_limit_offset}"
        f" ttl_min={entry_limit_ttl_minutes}"
        f" chase={entry_limit_chase}"
        f" exit_limit={use_exit_limit_orders}"
        + (
            f" exit_offset={exit_limit_offset}"
            f" exit_ttl={exit_limit_ttl_minutes}"
            if use_exit_limit_orders else ""
        )
    )
