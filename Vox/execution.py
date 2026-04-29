"""Exit and entry execution helpers for Vox ruthless mode."""
import numpy as np
from datetime import timedelta


# ── Breakeven stop ──────────────────────────────────────────────────────────

def apply_breakeven(ret, max_return_seen, breakeven_after, breakeven_buffer):
    """Return True if breakeven stop should trigger (position reversing through entry
    after having hit +breakeven_after)."""
    if max_return_seen >= breakeven_after:
        return ret < breakeven_buffer
    return False


def is_breakeven_active(max_return_seen, breakeven_after):
    """Return True if the breakeven protection has been activated."""
    return max_return_seen >= breakeven_after


# ── Momentum-failure exit ────────────────────────────────────────────────────

def should_exit_momentum_fail(elapsed_minutes, ret, feat,
                               min_hold_minutes=30, fail_loss=-0.012):
    """Return True if the trade should be cut early due to momentum failure.

    Conditions:
      - held >= min_hold_minutes
      - current return <= fail_loss
      - ret_4 (feat[1]) < 0
      - ret_16 (feat[3]) < 0
    """
    if elapsed_minutes < min_hold_minutes:
        return False
    if ret > fail_loss:
        return False
    if feat is None or len(feat) < 4:
        return False
    return float(feat[1]) < 0.0 and float(feat[3]) < 0.0


# ── Timeout extension ────────────────────────────────────────────────────────

def evaluate_timeout(
    elapsed_hours, ret, feat, toh,
    timeout_min_profit=0.03,
    timeout_extend_hours=12,
    max_timeout_hours=48,
    extension_hours_used=0.0,
):
    """Evaluate timeout logic for ruthless runner mode.

    Returns one of: 'exit', 'extend', 'hold'
    """
    if elapsed_hours < toh:
        return 'hold'

    if ret >= timeout_min_profit:
        return 'exit'

    if (
        extension_hours_used < (max_timeout_hours - toh)
        and feat is not None
        and len(feat) >= 4
        and ret > -0.01
        and float(feat[1]) > 0.0
    ):
        return 'extend'

    return 'exit'


# ── Entry limit order TTL tracker ────────────────────────────────────────────

class LimitOrderTracker:
    """Track a pending entry limit order and manage TTL expiry."""

    def __init__(self):
        self._order_id    = None
        self._submit_time = None
        self._ttl_minutes = 3

    def start(self, order_id, submit_time, ttl_minutes=3):
        self._order_id    = order_id
        self._submit_time = submit_time
        self._ttl_minutes = ttl_minutes

    def is_pending(self):
        return self._order_id is not None

    def is_expired(self, current_time):
        if self._submit_time is None:
            return False
        elapsed = (current_time - self._submit_time).total_seconds() / 60.0
        return elapsed >= self._ttl_minutes

    def cancel_and_clear(self, algo):
        """Cancel the limit order (if possible) and reset state."""
        if self._order_id is not None:
            try:
                algo.transactions.cancel_order(self._order_id)
            except Exception:
                pass
        self._order_id    = None
        self._submit_time = None

    def clear(self):
        self._order_id    = None
        self._submit_time = None

    @property
    def order_id(self):
        return self._order_id


# ── Candidate scoring helper ─────────────────────────────────────────────────

def evaluate_candidate(
    sym, feat, conf, price, atr,
    risk_profile, tp_base, sl_base, atr_tp_mult, atr_sl_mult,
    cost_fraction,
    momentum_override_enabled, momentum_ret4_min, momentum_ret16_min,
    momentum_volume_min, momentum_btc_rel_min, momentum_override_min_ev,
    ruthless_confirm_ev_min, ruthless_confirm_proba_min, ruthless_confirm_agree_min,
    ruthless_confirm_ret4_min, ruthless_confirm_ret16_min, ruthless_confirm_volr_min,
    use_momentum_score, reg_fitted,
    score_min_eff, max_disp, min_agr, min_ev, pred_return_min,
    compute_momentum_score_fn,
    counters,
    market_mode=None,
    ruthless_allowed_modes=None,
):
    """Evaluate a single candidate for entry. Returns a result dict or None if filtered.

    counters: dict with keys n_pass_disp, n_pass_agree, n_pass_score,
              n_pass_ev, n_pass_pred_ret, n_momentum_override — mutated in-place.
    """
    class_proba = conf["class_proba"]
    std_proba   = conf["std_proba"]
    n_agree     = conf["n_agree"]
    pred_return = conf["pred_return"]

    passed_disp  = std_proba  <= max_disp
    passed_agree = n_agree    >= min_agr
    passed_score = class_proba >= score_min_eff
    if passed_disp:  counters["n_pass_disp"]  += 1
    if passed_agree: counters["n_pass_agree"] += 1
    if passed_score: counters["n_pass_score"] += 1

    ml_gates_pass = passed_disp and passed_agree and passed_score
    entry_path = "ml"

    if not ml_gates_pass:
        if not momentum_override_enabled:
            return None
        if not (
            float(feat[1]) >= momentum_ret4_min
            and float(feat[3]) >= momentum_ret16_min
            and float(feat[6]) >= momentum_volume_min
            and float(feat[7]) >= momentum_btc_rel_min
        ):
            return None
        entry_path = "momentum_override"

    # ATR-based TP/SL
    if atr > 0:
        tp_use = (atr * atr_tp_mult) / price
        sl_use = (atr * atr_sl_mult) / price
    else:
        tp_use = tp_base
        sl_use = sl_base

    # Ruthless TP/SL floors
    tp_floor_applied = False
    sl_floor_applied = False
    if risk_profile == "ruthless":
        if tp_use < tp_base:
            tp_use = tp_base
            tp_floor_applied = True
        if sl_use < sl_base:
            sl_use = sl_base
            sl_floor_applied = True

    # EV
    ev_after_costs = (
        class_proba * tp_use
        - (1.0 - class_proba) * sl_use
        - cost_fraction
    )

    if entry_path == "momentum_override":
        if ev_after_costs < momentum_override_min_ev:
            return None
        counters["n_momentum_override"] += 1
        counters["n_pass_ev"] += 1
        counters["n_pass_pred_ret"] += 1
    else:
        if ev_after_costs <= min_ev:
            return None
        counters["n_pass_ev"] += 1
        if reg_fitted and pred_return < pred_return_min:
            return None
        counters["n_pass_pred_ret"] += 1

    # Ruthless confirmation gate
    confirm_reason = None
    if risk_profile == "ruthless":
        if entry_path == "momentum_override":
            confirm_reason = "momentum_override"
        elif (
            ev_after_costs >= ruthless_confirm_ev_min
            and class_proba >= ruthless_confirm_proba_min
            and n_agree    >= ruthless_confirm_agree_min
        ):
            confirm_reason = "strong_ml"
        elif (
            float(feat[1]) >= ruthless_confirm_ret4_min
            and float(feat[3]) >= ruthless_confirm_ret16_min
            and float(feat[6]) >= ruthless_confirm_volr_min
        ):
            confirm_reason = "trend_momentum"
        if confirm_reason is None and market_mode is not None:
            allowed = ruthless_allowed_modes or ["risk_on_trend", "pump"]
            if market_mode in allowed:
                confirm_reason = "market_mode"
        if confirm_reason is None:
            return None

    # Final score
    if use_momentum_score:
        momentum_score = compute_momentum_score_fn(feat)
        if reg_fitted and pred_return != 0.0:
            final_score = (
                0.50 * ev_after_costs
                + 0.25 * pred_return
                + 0.25 * momentum_score
            )
        else:
            confidence_adj = max(0.0, 1.0 - std_proba)
            final_score = (
                0.75 * ev_after_costs * confidence_adj
                + 0.25 * momentum_score
            )
    else:
        if reg_fitted and pred_return != 0.0:
            final_score = 0.6 * ev_after_costs + 0.4 * pred_return
        else:
            confidence_adj = max(0.0, 1.0 - std_proba)
            final_score = ev_after_costs * confidence_adj

    return {
        "sym":               sym,
        "final_score":       final_score,
        "ev":                ev_after_costs,
        "tp_use":            tp_use,
        "sl_use":            sl_use,
        "atr":               atr,
        "price":             price,
        "tp_floor_applied":  tp_floor_applied,
        "sl_floor_applied":  sl_floor_applied,
        "entry_path":        entry_path,
        "confirm_reason":    confirm_reason,
    }
