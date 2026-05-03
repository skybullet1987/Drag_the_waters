# ── Ruthless V2 — pump continuation and exhaustion scoring ───────────────────


def compute_pump_scores(
    symbol,
    same_symbol_entries_2h=0,
    same_symbol_trail_wins_2h=0,
    same_symbol_realized_return_2h=0.0,
    minutes_since_last_exit=None,
    prior_exit_reason=None,
    feat=None,
    conf=None,
):
    """Compute pump continuation and exhaustion scores for a symbol.

    Parameters
    ----------
    symbol                          : str
    same_symbol_entries_2h          : int — entries in the same symbol in last 2h
    same_symbol_trail_wins_2h       : int — trail wins in same symbol in last 2h
    same_symbol_realized_return_2h  : float — realized PnL from same symbol in 2h
    minutes_since_last_exit         : float or None
    prior_exit_reason               : str or None — e.g. "EXIT_TRAIL", "EXIT_SL"
    feat                            : feature vector or None
    conf                            : model confidence dict or None

    Returns
    -------
    dict with keys:
        pump_continuation_score : float [0, 1] — higher = stronger pump, re-entry ok
        pump_exhaustion_score   : float [0, 1] — higher = late/exhausted, avoid entry
    """
    cont_vol        = 0.0
    cont_momentum   = 0.0
    cont_model      = 0.0
    if feat is not None and len(feat) >= 7:
        try:
            ret_4 = float(feat[1])
            vol_r = float(feat[6])
            cont_vol      = max(0.0, min((vol_r - 1.5) / 3.0, 1.0))
            cont_momentum = max(0.0, min(ret_4 * 10.0, 1.0))
        except (TypeError, IndexError, ValueError):
            pass
    if conf is not None:
        top3_mean = conf.get("top3_mean", 0.0)
        cont_model = max(0.0, min((top3_mean - 0.50) * 2.5, 1.0))

    trail_wins_bonus = min(same_symbol_trail_wins_2h * 0.15, 0.30)

    pump_continuation_score = max(0.0, min(
        0.35 * cont_momentum
        + 0.30 * cont_vol
        + 0.25 * cont_model
        + trail_wins_bonus,
        1.0,
    ))

    entry_exhaustion = min(same_symbol_entries_2h * 0.20, 0.60)
    win_exhaustion   = min(same_symbol_trail_wins_2h * 0.15, 0.45)
    time_exhaustion  = 0.0
    if minutes_since_last_exit is not None:
        time_exhaustion = max(0.0, min((10.0 - minutes_since_last_exit) / 10.0, 0.40))
    prior_sl_penalty = 0.20 if prior_exit_reason and "SL" in str(prior_exit_reason) else 0.0
    vol_fade = 0.0
    if feat is not None and len(feat) >= 7:
        try:
            vol_r = float(feat[6])
            if vol_r < 0.8:
                vol_fade = min((0.8 - vol_r) * 2.0, 0.30)
        except (TypeError, IndexError, ValueError):
            pass

    pump_exhaustion_score = max(0.0, min(
        entry_exhaustion + win_exhaustion + time_exhaustion
        + prior_sl_penalty + vol_fade,
        1.0,
    ))

    return {
        "pump_continuation_score": round(pump_continuation_score, 4),
        "pump_exhaustion_score":   round(pump_exhaustion_score, 4),
    }


def exhaustion_override_allowed(pump_continuation_score, pump_exhaustion_score,
                                continuation_threshold=0.55, exhaustion_threshold=0.60):
    """Return True if continuation is strong enough to override simple cooldown.

    Allows re-entry even when reentry_cooldown is active if:
      - continuation score is high (real pump strength confirmed)
      - exhaustion score is below threshold
    """
    return (
        pump_continuation_score >= continuation_threshold
        and pump_exhaustion_score < exhaustion_threshold
    )
