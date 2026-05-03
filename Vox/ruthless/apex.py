# ── Ruthless V2 — Apex Predator helpers ──────────────────────────────────────
#
# compute_apex_score()                  — weighted ensemble vote
# apex_entry_decision()                 — five trigger paths
# compute_apex_size()                   — Kelly-lite sizing + pyramiding
# compute_apex_atr_stops()              — ATR-based SL/TP/trail/breakeven
# apex_breakout_signal()                — price × N-bar high + volume spike
# apex_pullback_signal()                — RSI < 35 in uptrend
# apex_momentum_continuation_signal()   — N consec. higher closes + vol
# apex_rejected_entry_log()             — structured rejection log builder
# ─────────────────────────────────────────────────────────────────────────────

# Apex vote column weights (must sum to 1.0)
_APEX_WEIGHTS = {
    "vote_lr_bal":      0.35,
    "vote_hgbc_l2":     0.25,
    "active_rf":        0.15,
    "active_hgbc_l2":   0.10,
    "active_lgbm_bal":  0.10,
    "vote_et":          0.05,
}


def compute_apex_score(votes):
    """Compute the weighted Apex Predator ensemble vote score.

    Parameters
    ----------
    votes : dict[str, float]
        Per-column model probability/vote values.  Missing keys have their
        weight redistributed pro-rata across the present columns.

    Returns
    -------
    apex_score : float
    weight_present : float
    """
    present = {col: w for col, w in _APEX_WEIGHTS.items() if col in votes}
    if not present:
        return 0.0, 0.0

    total_w = sum(present.values())
    if total_w <= 0.0:
        return 0.0, 0.0

    score = sum(
        (w / total_w) * float(votes[col])
        for col, w in present.items()
    )
    return max(0.0, min(1.0, score)), total_w


def apex_entry_decision(
    votes,
    mean_proba=0.0,
    n_agree=0,
    apex_score_entry=None,
):
    """Evaluate the five Apex Predator entry trigger paths (v2 — aggressive gates).

    Entry fires when **ANY** of the following is True:
      1. apex_score >= apex_score_entry        (default 0.50)
      2. vote_lr_bal >= 0.50                   (proven PF ~8)
      3. vote_hgbc_l2 >= 0.55 AND active_lgbm_bal >= 0.55
      4. mean_proba >= APEX_ENTRY_PATH4_PROBA_MIN (0.50) AND
         n_agree >= APEX_ENTRY_PATH4_N_AGREE_MIN (1)
      5. active_lgbm_bal >= APEX_ENTRY_LGBM_BAL_MIN (0.50)

    Returns
    -------
    dict with keys: triggered, apex_score, weight_present, path, path_detail, reject_reason
    """
    try:
        from config import (
            APEX_SCORE_ENTRY           as _cfg_entry,
            APEX_ENTRY_PATH4_PROBA_MIN as _p4_proba,
            APEX_ENTRY_PATH4_N_AGREE_MIN as _p4_agree,
            APEX_ENTRY_LGBM_BAL_MIN    as _p5_lgbm,
        )
    except ImportError:
        _cfg_entry = 0.50
        _p4_proba  = 0.50
        _p4_agree  = 1
        _p5_lgbm   = 0.50

    if apex_score_entry is None:
        apex_score_entry = _cfg_entry

    apex_score, weight_present = compute_apex_score(votes)

    lr_bal      = float(votes.get("vote_lr_bal",     0.0))
    hgbc_l2     = float(votes.get("vote_hgbc_l2",    0.0))
    lgbm_bal    = float(votes.get("active_lgbm_bal", 0.0))

    path1 = apex_score >= apex_score_entry
    path2 = lr_bal >= 0.50
    path3 = hgbc_l2 >= 0.55 and lgbm_bal >= 0.55
    path4 = float(mean_proba) >= _p4_proba and int(n_agree) >= _p4_agree
    path5 = lgbm_bal >= _p5_lgbm

    triggered = path1 or path2 or path3 or path4 or path5

    path_name = None
    if path1:
        path_name = "apex_score"
    elif path2:
        path_name = "vote_lr_bal"
    elif path3:
        path_name = "hgbc_l2_x_lgbm_bal"
    elif path4:
        path_name = "strong_ml_backstop"
    elif path5:
        path_name = "lgbm_bal_direct"

    reject_reason = None
    if not triggered:
        reject_reason = (
            f"apex_score={apex_score:.3f}<{apex_score_entry} | "
            f"vote_lr_bal={lr_bal:.3f}<0.50 | "
            f"hgbc_l2={hgbc_l2:.3f}<0.55 or lgbm_bal={lgbm_bal:.3f}<0.55 | "
            f"mean_proba={mean_proba:.3f}<{_p4_proba} or n_agree={n_agree}<{_p4_agree} | "
            f"lgbm_bal={lgbm_bal:.3f}<{_p5_lgbm}"
        )

    return {
        "triggered":       triggered,
        "apex_score":      apex_score,
        "weight_present":  weight_present,
        "path":            path_name,
        "reject_reason":   reject_reason,
        "path_detail": {
            "apex_score_gate":      path1,
            "vote_lr_bal_gate":     path2,
            "hgbc_lgbm_gate":       path3,
            "strong_ml_backstop":   path4,
            "lgbm_bal_gate":        path5,
        },
    }


def compute_apex_size(
    apex_score,
    n_agree=0,
    current_total_exposure=0.0,
    base_alloc=None,
    max_gross=None,
):
    """Compute Apex Predator position size (Kelly-lite with pyramiding support).

    Formula::
        edge_mult = clip((apex_score - 0.50) / 0.30, 0.0, 1.5)
        conf_mult = 1.0 + 0.5 * (n_agree >= 4)
        size_frac = clip(base_alloc * (1 + edge_mult) * conf_mult, 0.05, 0.45)

    Returns
    -------
    float — allocation fraction in [0.05, 0.45], capped by remaining headroom.
    """
    try:
        from config import APEX_BASE_ALLOC as _ba, APEX_MAX_GROSS as _mg
    except ImportError:
        _ba, _mg = 0.20, 2.0

    if base_alloc is None:
        base_alloc = _ba
    if max_gross is None:
        max_gross = _mg

    edge_mult = max(0.0, min(1.5, (float(apex_score) - 0.50) / 0.30))
    conf_mult = 1.0 + 0.5 * (int(n_agree) >= 4)
    size_frac = base_alloc * (1.0 + edge_mult) * conf_mult

    size_frac = max(0.05, min(0.45, size_frac))

    remaining = float(max_gross) - float(current_total_exposure)
    size_frac = max(0.0, min(size_frac, remaining))

    return round(size_frac, 4)


def compute_apex_atr_stops(
    entry_price,
    atr,
    atr_sl_mult=None,
    atr_tp_mult=None,
    sl_floor_pct=0.008,
    sl_ceil_pct=0.040,
    tp_floor_pct=0.025,
    tp_ceil_pct=0.150,
):
    """Compute Apex Predator ATR-based stop-loss and take-profit levels.

    SL = entry - atr_sl_mult * ATR; clamped to [sl_floor_pct, sl_ceil_pct]
    TP = entry + atr_tp_mult * ATR; clamped to [tp_floor_pct, tp_ceil_pct]

    Returns
    -------
    dict with keys: sl_price, tp_price, sl_pct, tp_pct,
                    trail_arm_pct, trail_dist_pct, breakeven_mfe_pct, time_stop_hrs
    """
    try:
        from config import (
            APEX_ATR_SL_MULT   as _sl_m,
            APEX_ATR_TP_MULT   as _tp_m,
            APEX_TRAIL_ARM_PCT as _arm,
            APEX_TRAIL_ATR_MULT as _trail_m,
            APEX_BREAKEVEN_MFE as _be,
            APEX_TIME_STOP_HRS as _tsh,
        )
    except ImportError:
        _sl_m, _tp_m, _arm, _trail_m, _be, _tsh = 1.25, 4.0, 0.010, 0.8, 0.02, 48

    if atr_sl_mult is None:
        atr_sl_mult = _sl_m
    if atr_tp_mult is None:
        atr_tp_mult = _tp_m

    ep = float(entry_price)
    if ep <= 0.0:
        return {
            "sl_price": ep, "tp_price": ep,
            "sl_pct": sl_floor_pct, "tp_pct": tp_floor_pct,
            "trail_arm_pct": _arm, "trail_dist_pct": 0.006,
            "breakeven_mfe_pct": _be, "time_stop_hrs": _tsh,
        }

    atr_v = float(atr) if atr and float(atr) > 0 else ep * 0.01

    raw_sl_pct = atr_sl_mult * atr_v / ep
    sl_pct     = max(sl_floor_pct, min(sl_ceil_pct, raw_sl_pct))
    sl_price   = ep * (1.0 - sl_pct)

    raw_tp_pct = atr_tp_mult * atr_v / ep
    tp_pct     = max(tp_floor_pct, min(tp_ceil_pct, raw_tp_pct))
    tp_price   = ep * (1.0 + tp_pct)

    raw_trail_pct = _trail_m * atr_v / ep
    trail_pct     = max(0.006, raw_trail_pct)

    return {
        "sl_price":          round(sl_price, 8),
        "tp_price":          round(tp_price, 8),
        "sl_pct":            round(sl_pct,   6),
        "tp_pct":            round(tp_pct,   6),
        "trail_arm_pct":     _arm,
        "trail_dist_pct":    round(trail_pct, 6),
        "breakeven_mfe_pct": _be,
        "time_stop_hrs":     _tsh,
    }


def apex_breakout_signal(closes, volumes, n_bars=None, vol_mult=None):
    """Return True when price breaks above the rolling N-bar high with a volume spike.

    A breakout is confirmed when:
      1. closes[-1] > max(closes[-n_bars-1:-1])   (crosses prior high)
      2. volumes[-1] >= vol_mult × mean(volumes[-n_bars-1:-1])
    """
    try:
        from config import (
            APEX_BREAKOUT_NBARS    as _nb,
            APEX_BREAKOUT_VOL_MULT as _vm,
        )
    except ImportError:
        _nb, _vm = 20, 1.5

    if n_bars is None:
        n_bars = _nb
    if vol_mult is None:
        vol_mult = _vm

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed or len(volumes) < needed:
        return False

    try:
        prior_highs = [float(c) for c in closes[-(n_bars + 1):-1]]
        current_close = float(closes[-1])
        prior_vols = [float(v) for v in volumes[-(n_bars + 1):-1]]
        current_vol = float(volumes[-1])
    except (TypeError, ValueError, IndexError):
        return False

    if not prior_highs or not prior_vols:
        return False

    rolling_high = max(prior_highs)
    avg_vol = sum(prior_vols) / len(prior_vols)

    price_breakout = current_close > rolling_high
    vol_confirmed  = avg_vol > 0 and current_vol >= vol_mult * avg_vol

    return price_breakout and vol_confirmed


def apex_pullback_signal(closes, rsi, n_bars=None, rsi_max=None):
    """Return True when RSI is oversold inside a confirmed uptrend (pullback long).

    Conditions:
      1. rsi <= rsi_max                     (oversold / pullback)
      2. closes[-1] > closes[-n_bars-1]    (price higher than n_bars ago)
    """
    try:
        from config import (
            APEX_PULLBACK_TREND_BARS as _nb,
            APEX_PULLBACK_RSI_MAX    as _rmax,
        )
    except ImportError:
        _nb, _rmax = 10, 35

    if n_bars is None:
        n_bars = _nb
    if rsi_max is None:
        rsi_max = _rmax

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed:
        return False

    try:
        current_close = float(closes[-1])
        past_close    = float(closes[-(n_bars + 1)])
        rsi_val       = float(rsi)
    except (TypeError, ValueError, IndexError):
        return False

    oversold      = rsi_val <= rsi_max
    in_uptrend    = current_close > past_close

    return oversold and in_uptrend


def apex_momentum_continuation_signal(closes, volumes, n_bars=None, vol_mult=None):
    """Return True on momentum continuation: N consecutive higher closes + volume spike.

    Conditions:
      1. Each of the last n_bars closes is strictly higher than the one before it.
      2. volumes[-1] >= vol_mult × mean(volumes[-n_bars-1:-1])
    """
    try:
        from config import (
            APEX_MOMENTUM_CONT_BARS     as _nb,
            APEX_MOMENTUM_CONT_VOL_MULT as _vm,
        )
    except ImportError:
        _nb, _vm = 3, 1.5

    if n_bars is None:
        n_bars = _nb
    if vol_mult is None:
        vol_mult = _vm

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed or len(volumes) < needed:
        return False

    try:
        recent_closes = [float(c) for c in closes[-(n_bars + 1):]]
        prior_vols    = [float(v) for v in volumes[-(n_bars + 1):-1]]
        current_vol   = float(volumes[-1])
    except (TypeError, ValueError, IndexError):
        return False

    if len(recent_closes) < n_bars + 1 or not prior_vols:
        return False

    consecutive_higher = all(
        recent_closes[i] > recent_closes[i - 1]
        for i in range(1, len(recent_closes))
    )

    avg_vol       = sum(prior_vols) / len(prior_vols)
    vol_confirmed = avg_vol > 0 and current_vol >= vol_mult * avg_vol

    return consecutive_higher and vol_confirmed


def apex_rejected_entry_log(votes, mean_proba, n_agree, decision):
    """Build a structured log entry for a rejected Apex Predator entry attempt.

    Returns a dict with the specific gate(s) that caused the rejection.
    """
    return {
        "triggered":      decision.get("triggered", False),
        "reject_reason":  decision.get("reject_reason"),
        "path_detail":    decision.get("path_detail", {}),
        "apex_score":     decision.get("apex_score", 0.0),
        "mean_proba":     float(mean_proba),
        "n_agree":        int(n_agree),
        "votes_snapshot": {k: float(v) for k, v in votes.items()},
    }
