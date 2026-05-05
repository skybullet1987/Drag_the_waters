# ── entry_logic.py: exit and entry logic for VoxAlgorithm ────────────────────
#
# Extracted from main.py to keep main.py under the QuantConnect 63KB file limit.
# Called from VoxAlgorithm._check_exit() and VoxAlgorithm._try_enter().
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import core as _cfg_module
from core import SCORE_MIN_MULT, DIAG_INTERVAL_HOURS
from infra import OrderHelper
from models import build_features, compute_atr
from strategy import (
    apply_breakeven,
    should_exit_momentum_fail,
    evaluate_candidate,
    compute_qty,
)
from journals import (
    format_vote_log,
    _feature_diag_suffix,
    build_candidate_records,
    build_rejected_candidate_records,
)
from audit_utils import audit_safe_float, audit_trim_votes
from core import compute_momentum_score


def check_exit(algo, price):
    """Evaluate TP / SL / timeout and submit market sell if triggered.

    Extracted from VoxAlgorithm._check_exit to reduce main.py file size.
    """
    sym        = algo._pos_sym
    entry_px   = algo._entry_px
    entry_time = algo._entry_time

    if sym is None or entry_time is None or entry_px <= 0:
        return

    ret             = (price - entry_px) / entry_px
    elapsed_sec     = (algo.time - entry_time).total_seconds()
    elapsed_hours   = elapsed_sec / 3600.0
    elapsed_minutes = elapsed_sec / 60.0

    tp_use = algo._tp_dyn if algo._tp_dyn > 0 else algo._tp
    sl_use = algo._sl_dyn if algo._sl_dyn > 0 else algo._sl
    # Ruthless v4: effective timeout with extension
    _ext   = getattr(algo, "_timeout_ext_hours", 0.0)
    toh_eff = algo._toh + _ext

    # ── Ruthless v4: update breakeven high-water mark ────────────────────
    if ret > algo._max_return_seen:
        algo._max_return_seen = ret

    # ── Ruthless v4: breakeven stop ───────────────────────────────────────
    be_after  = getattr(algo, "_breakeven_after",  0.03)
    be_buffer = getattr(algo, "_breakeven_buffer", 0.003)
    if not algo._breakeven_active and algo._max_return_seen >= be_after:
        algo._breakeven_active = True
    if algo._breakeven_active and apply_breakeven(
        ret, algo._max_return_seen, be_after, be_buffer
    ):
        _be_qty = OrderHelper.safe_crypto_sell_qty(
            algo, sym,
            OrderHelper.get_lot_size(algo.securities[sym]),
            OrderHelper.get_min_order_size(algo.securities[sym]),
            exit_qty_buffer_lots=algo._exit_qty_buffer,
        )
        # Breakeven exits are profit-protection, NOT stop-loss hits.
        # Tag as EXIT_BE so they don't inflate daily SL count or cooldowns.
        # Submit order first — let on_order_event handle fill, risk recording,
        # state clearing, and audit logging (avoids clearing _pos_sym before fill).
        if _be_qty > 0:
            algo._exiting = True
            algo._exit_retry_count = 0
            algo.market_order(sym, -_be_qty, tag="EXIT_BE")
        else:
            # Dust position — can't submit order; clear state now.
            algo._risk.record_exit(sym, is_sl=False, exit_time=algo.time)
            algo._clear_position_state(include_retry=True)
        return

    # ── Ruthless v4: momentum-fail early exit ────────────────────────────
    if getattr(algo, "_mom_fail_enabled", False):
        feat_now = algo._last_feat.get(sym)
        if feat_now is not None and should_exit_momentum_fail(
            elapsed_minutes=elapsed_minutes,
            ret=ret,
            feat=feat_now,
            min_hold_minutes=getattr(algo, "_mom_fail_min_hold", 30),
            fail_loss=getattr(algo, "_mom_fail_loss", -0.012),
        ):
            qty_mf = OrderHelper.safe_crypto_sell_qty(
                algo, sym,
                OrderHelper.get_lot_size(algo.securities[sym]),
                OrderHelper.get_min_order_size(algo.securities[sym]),
                exit_qty_buffer_lots=algo._exit_qty_buffer,
            )
            if qty_mf > 0:
                algo._exiting = True
                algo._exit_retry_count = 0
                algo.market_order(sym, -qty_mf, tag="EXIT_MOM_FAIL")
                return

    # ── Minimum hold period ────────────────────────────────────────────────
    # During the minimum hold window, suppress ordinary exits.  Only the
    # emergency SL is allowed to exit early to protect against large gaps.
    if elapsed_minutes < algo._min_hold_minutes:
        if ret <= -algo._emergency_sl:
            reason = "EXIT_SL"   # emergency stop — override min-hold
        else:
            return               # hold — suppress normal TP/SL/timeout
    elif algo._runner_mode:
        # ── Runner / trailing-profit mode ─────────────────────────────────
        # Hard SL and timeout always take priority over trailing logic.
        if ret <= -sl_use:
            reason = "EXIT_SL"
        elif elapsed_hours >= toh_eff:
            reason = "EXIT_TIMEOUT"
        elif algo._trail_active:
            # Update the high-water mark in price space
            if price > algo._trail_high_px:
                algo._trail_high_px = price
            # Trail stop: exit if price falls trail_pct from the high
            if price <= algo._trail_high_px * (1.0 - algo._trail_pct):
                reason = "EXIT_TRAIL"
            else:
                return   # still running — keep holding
        else:
            # Check if we should activate the trailing stop.
            # Activate at whichever threshold is reached first: the
            # configured trail_after_tp (e.g. +4%) or the ATR-derived
            # tp_use (in case ATR produces a target below trail_after_tp).
            trail_trigger = min(tp_use, algo._trail_after_tp)
            if ret >= trail_trigger:
                algo._trail_active  = True
                algo._trail_high_px = price
                return   # don't exit yet — trail is now live
            else:
                return   # below trigger — keep holding
    else:
        # ── Normal (balanced / conservative / aggressive) exit logic ───────
        reason = None
        if ret >= tp_use:
            reason = "EXIT_TP"
        elif ret <= -sl_use:
            reason = "EXIT_SL"
        elif elapsed_hours >= toh_eff:
            reason = "EXIT_TIMEOUT"

        if not reason:
            return

    sec      = algo.securities[sym]
    lot_size = OrderHelper.get_lot_size(sec)
    min_ord  = OrderHelper.get_min_order_size(sec)
    qty = OrderHelper.safe_crypto_sell_qty(
        algo, sym, lot_size, min_ord,
        exit_qty_buffer_lots=algo._exit_qty_buffer,
    )

    if qty > 0:
        algo._exiting          = True
        algo._exit_retry_count = 0
        algo.debug(
            f"EXIT order {sym.value}  reason={reason}"
            f"  qty={qty:.6f}  ret={ret:.3%}"
            f"  elapsed_min={elapsed_minutes:.1f}"
            + (f"  trail_high={algo._trail_high_px:.4f}" if algo._trail_active else "")
        )
        algo.market_order(sym, -qty, tag=reason)
    else:
        # Portfolio is flat or remaining position is non-actionable dust.
        portfolio_qty = float(algo.portfolio[sym].quantity)
        algo.debug(
            f"EXIT {sym.value}: safe sell qty=0 (dust/flat),"
            f" portfolio_qty={portfolio_qty:.8f}  reason={reason}"
            f" — clearing state"
        )
        is_sl = (reason == "EXIT_SL") or (reason == "EXIT_MOM_FAIL" and ret < 0.0)
        algo._risk.record_exit(sym, is_sl=is_sl, exit_time=algo.time)
        algo._clear_position_state(include_retry=True)


def try_enter(algo):
    """Score all symbols; if a clear winner passes all gates, place a buy order.

    Extracted from VoxAlgorithm._try_enter to reduce main.py file size.
    """
    scores         = {}   # sym -> final_score
    conf_data      = {}   # sym -> confidence dict from ensemble
    tp_sl_data     = {}   # sym -> (tp_use, sl_use, atr, price) for re-use
    ev_data        = {}   # sym -> ev_after_costs
    entry_path_data = {}  # sym -> "ml" | "momentum_override"

    cost_fraction = algo._cost_bps * 1e-4
    reg_fitted    = getattr(algo._ensemble, "_reg_fitted", False)

    btc_closes = (
        list(algo._state[algo._btc_sym]["closes"])
        if algo._btc_sym else []
    )

    candidates = []   # list of (sym, feat)
    for sym in algo._symbols:
        st = algo._state.get(sym)
        if st is None:
            continue

        # ── Penalty cooldown gate ─────────────────────────────────────────
        # Skip symbols that are in their post-repeated-loss cooldown window.
        if algo._is_in_penalty_cooldown(sym):
            continue

        closes  = list(st["closes"])
        volumes = list(st["volumes"])

        feat = build_features(
            closes     = closes,
            volumes    = volumes,
            btc_closes = btc_closes,
            hour       = algo.time.hour,
        )
        if feat is None:
            continue

        candidates.append((sym, feat))

    if not candidates:
        # Only log occasionally to avoid spamming the 100KB log cap
        if algo.time.minute == 0 and algo.time.hour % 6 == 0:
            algo.log(
                f"[diag] no_features symbols={len(algo._symbols)} "
                f"(build_features returned None for all)"
            )
        return

    X_all = np.vstack([c[1] for c in candidates])
    try:
        confs = algo._ensemble.predict_with_confidence_batch(X_all)
    except Exception as exc:
        algo.debug(f"[vox] batch predict failed: {exc}")
        return

    # Base-rate-aware effective thresholds (computed once per tick)
    pr            = algo._ensemble.base_rate
    score_min_eff = float(np.clip(
        max(algo._s_min_floor, SCORE_MIN_MULT * pr), algo._s_min_floor, algo._s_min
    ))
    agree_thr     = algo._ensemble._agree_threshold()

    # Determine active_research mode once — used in regime gate and diagnostics
    _is_active_research = getattr(algo, "_risk_profile", "") == "active_research"

    # Per-gate pass counters and best-candidate values for diagnostics
    n_pass_disp     = 0
    n_pass_agree    = 0
    n_pass_score    = 0
    n_pass_ev       = 0
    n_pass_pred_ret = 0
    n_momentum_override = 0
    _best_ev_diag   = float("-inf")   # best ev_after_costs seen (for diagnostics)

    _market_mode = None
    if algo._market_mode_det is not None:
        try:
            _btc_c = list(algo._state[algo._btc_sym]["closes"]) if algo._btc_sym else []
            _btc_v = list(algo._state[algo._btc_sym]["volumes"]) if algo._btc_sym else []
            _market_mode = algo._market_mode_det.detect(_btc_c, _btc_v)
        except Exception:
            pass

    counters = {
        "n_pass_disp": 0, "n_pass_agree": 0, "n_pass_score": 0,
        "n_pass_ev": 0, "n_pass_pred_ret": 0, "n_momentum_override": 0,
    }
    all_cand_conf = {}   # all evaluated sym→conf (for journal when scores empty)

    _ruthless_allowed_modes = getattr(algo, "_ruthless_allowed_modes", [])
    for (sym, feat), conf in zip(candidates, confs):
        price = float(algo.securities[sym].price)
        if price <= 0:
            continue
        st  = algo._state[sym]
        atr = compute_atr(
            highs  = list(st["highs"]),
            lows   = list(st["lows"]),
            closes = list(st["closes"]),
        )
        result = evaluate_candidate(
            sym=sym, feat=feat, conf=conf, price=price, atr=atr,
            risk_profile=algo._risk_profile,
            tp_base=algo._tp, sl_base=algo._sl,
            atr_tp_mult=algo._atr_tp, atr_sl_mult=algo._atr_sl,
            cost_fraction=cost_fraction,
            momentum_override_enabled=algo._momentum_override,
            momentum_ret4_min=algo._momentum_ret4_min,
            momentum_ret16_min=algo._momentum_ret16_min,
            momentum_volume_min=algo._momentum_volume_min,
            momentum_btc_rel_min=algo._momentum_btc_rel_min,
            momentum_override_min_ev=algo._momentum_override_min_ev,
            ruthless_confirm_ev_min=getattr(algo, "_ruthless_confirm_ev_min", 0.006),
            ruthless_confirm_proba_min=getattr(algo, "_ruthless_confirm_proba_min", 0.60),
            ruthless_confirm_agree_min=getattr(algo, "_ruthless_confirm_agree_min", 2),
            ruthless_confirm_ret4_min=getattr(algo, "_ruthless_confirm_ret4_min", 0.010),
            ruthless_confirm_ret16_min=getattr(algo, "_ruthless_confirm_ret16_min", 0.020),
            ruthless_confirm_volr_min=getattr(algo, "_ruthless_confirm_volr_min", 1.5),
            use_momentum_score=algo._use_momentum_score,
            reg_fitted=reg_fitted,
            score_min_eff=score_min_eff,
            max_disp=algo._max_disp,
            min_agr=algo._min_agr,
            min_ev=algo._min_ev,
            pred_return_min=algo._pred_return_min,
            compute_momentum_score_fn=compute_momentum_score,
            counters=counters,
            market_mode=_market_mode,
            ruthless_allowed_modes=_ruthless_allowed_modes,
            ruthless_good_mode_relaxation=getattr(algo, "_good_mode_relaxation", True),
            ruthless_good_mode_ev_min=getattr(algo, "_good_mode_min_ev", 0.004),
            ruthless_good_mode_volr_min=getattr(algo, "_good_mode_volume_min", 1.3),
            ruthless_profit_voting_mode=getattr(algo, "_ruthless_profit_voting_mode", False),
            pv_vote_threshold=getattr(algo, "_pv_vote_threshold", 0.50),
            pv_vote_yes_frac_min=getattr(algo, "_pv_vote_yes_frac_min", 0.34),
            pv_top3_mean_min=getattr(algo, "_pv_top3_mean_min", 0.55),
            pv_vote_ev_floor=getattr(algo, "_pv_vote_ev_floor", 0.001),
            pv_chop_yes_frac_min=getattr(algo, "_pv_chop_yes_frac_min", 0.50),
            pv_chop_top3_mean_min=getattr(algo, "_pv_chop_top3_mean_min", 0.60),
            pv_chop_pred_return_min=getattr(algo, "_pv_chop_pred_return_min", 0.000),
            pv_chop_ev_min=getattr(algo, "_pv_chop_ev_min", 0.002),
            ruthless_active_models=getattr(algo, "_ruthless_active_models", None),
            ruthless_diagnostic_models=getattr(algo, "_ruthless_diagnostic_models", None),
        )
        all_cand_conf[sym] = conf
        if result is None:
            continue
        ev_cand = result.get("ev", float("-inf"))
        if ev_cand > _best_ev_diag:
            _best_ev_diag = ev_cand
        scores[sym]          = result["final_score"]
        conf_data[sym]       = conf
        tp_sl_data[sym]      = (
            result["tp_use"], result["sl_use"], result["atr"], result["price"],
            result["tp_floor_applied"], result["sl_floor_applied"],
        )
        ev_data[sym]         = ev_cand
        entry_path_data[sym] = result["entry_path"]
        if not hasattr(algo, "_ruthless_confirm_reasons"):
            algo._ruthless_confirm_reasons = {}
        if result.get("confirm_reason"):
            algo._ruthless_confirm_reasons[sym] = result["confirm_reason"]

    n_pass_disp = counters["n_pass_disp"]
    n_pass_agree = counters["n_pass_agree"]
    n_pass_score = counters["n_pass_score"]
    n_pass_ev = counters["n_pass_ev"]
    n_pass_pred_ret = counters["n_pass_pred_ret"]
    n_momentum_override = counters["n_momentum_override"]

    if scores:
        ranked      = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sym_d   = ranked[0][0]
        top_sc_d    = ranked[0][1]
        second_sc_d = ranked[1][1] if len(ranked) > 1 else 0.0
        top_cd      = conf_data[top_sym_d]
        top_tp, top_sl, top_atr, top_px, _tp_fl, _sl_fl = tp_sl_data[top_sym_d]
        algo.log(
            f"[diag] candidates={len(scores)}"
            f" top={top_sym_d.value}"
            f" path={entry_path_data.get(top_sym_d, 'ml')}"
            f" final_score={top_sc_d:.5f}"
            f" ev_score={ev_data[top_sym_d]:.5f}"
            f" pred_ret={top_cd['pred_return']:.5f}"
            f" gap={top_sc_d-second_sc_d:.5f}"
            f" class_proba={top_cd['class_proba']:.3f}"
            f" std_proba={top_cd['std_proba']:.3f}"
            f" n_agree={top_cd['n_agree']}"
            f" vote_score={top_cd.get('vote_score',0):.4f}"
            f" tp={top_tp:.4f} sl={top_sl:.4f}"
            + (f" tp_floor={_tp_fl}" if _tp_fl else "")
            + (f" sl_floor={_sl_fl}" if _sl_fl else "")
            + (f" mo_overrides={n_momentum_override}" if n_momentum_override else "")
        )
    else:
        _diag_interval_h = (
            getattr(_cfg_module, "ACTIVE_RESEARCH_DIAG_INTERVAL_HOURS", 1)
            if _is_active_research else DIAG_INTERVAL_HOURS
        )
        _emit_diag = (
            algo._last_nocandidate_diag_time is None
            or (algo.time - algo._last_nocandidate_diag_time).total_seconds()
               >= _diag_interval_h * 3600
        )
        if _emit_diag:
            algo._last_nocandidate_diag_time = algo.time
            best_proba  = max(c["class_proba"] for c in confs)
            best_nagree = max(c["n_agree"]     for c in confs)
            best_std    = min(c["std_proba"]   for c in confs)
            best_pred   = max(c["pred_return"] for c in confs)
            best_ev_str = f"{_best_ev_diag:.5f}" if _best_ev_diag > float("-inf") else "n/a"
            _last_rej   = getattr(algo, "_last_gate_rejection", None)
            algo.log(
                f"[diag] eval={len(candidates)} "
                f"pass_disp={n_pass_disp} pass_agree={n_pass_agree} "
                f"pass_score={n_pass_score} pass_ev={n_pass_ev} "
                f"pass_pred_ret={n_pass_pred_ret} "
                f"best_proba={best_proba:.3f} best_agree={best_nagree} "
                f"best_disp={best_std:.3f} best_pred_ret={best_pred:.5f} "
                f"best_ev={best_ev_str} "
                f"(thresh: score>={score_min_eff:.3f} "
                f"agree>={algo._min_agr} disp<={algo._max_disp} "
                f"ev>{algo._min_ev:.5f} pred_ret>={algo._pred_return_min:.5f} "
                f"cost={cost_fraction:.4f})"
                + (f" last_top_rejection={_last_rej}" if _last_rej else "")
            )

    if not scores:
        # Journal top-N rejected candidates so the user can diagnose gate rejects
        if getattr(_cfg_module, "PERSIST_CANDIDATE_JOURNAL", True) and all_cand_conf:
            _rej = build_rejected_candidate_records(
                all_cand_conf,
                market_mode=_market_mode,
                top_n=getattr(_cfg_module, "CANDIDATE_JOURNAL_TOP_N", 5),
            )
            if _rej:
                algo._candidate_journal.record_cycle(algo.time, _rej)
        return

    ranked   = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_sym, top_sc = ranked[0]
    second_sc       = ranked[1][1] if len(ranked) > 1 else 0.0

    # Track final rejection reason for diagnostics
    _final_rejection = None

    # ── Meta-filter check ─────────────────────────────────────────────────
    _top_cd  = conf_data.get(top_sym, {})
    _top_feat = next((f for s, f in candidates if s == top_sym), None)
    _mf_approved, _mf_score = algo._meta_filter.approve(
        class_proba=_top_cd.get("class_proba", 0.0),
        ev_score=ev_data.get(top_sym, 0.0),
        n_agree=_top_cd.get("n_agree", 0),
        std_proba=_top_cd.get("std_proba", 1.0),
        pred_return=_top_cd.get("pred_return", 0.0),
        feat=_top_feat,
        market_mode=_market_mode,
        ruthless_allowed_modes=getattr(algo, "_ruthless_allowed_modes", []),
    )
    if not _mf_approved:
        _final_rejection = f"meta_filter:score={_mf_score:.3f}"
        algo._throttled_skip_debug(
            f"[vox] Meta-filter rejected {top_sym.value} score={_mf_score:.3f}"
        )
        algo._last_gate_rejection = _final_rejection
        return

    # Gap check on final score
    ev_gap_actual = top_sc - second_sc
    if len(ranked) > 1 and ev_gap_actual < algo._ev_gap:
        _final_rejection = f"score_gap:actual={ev_gap_actual:.5f}<required={algo._ev_gap:.5f}"
        algo._throttled_skip_debug(
            f"[vox] Score gap too small: top={top_sc:.5f}"
            f"  second={second_sc:.5f}"
            f"  gap={ev_gap_actual:.5f}"
            f"  required={algo._ev_gap:.5f}"
        )
        algo._last_gate_rejection = _final_rejection
        return

    # ── Regime gate ───────────────────────────────────────────────────────
    _regime_risk_on = algo._regime.is_risk_on(algo._btc_sym, sym=top_sym)
    if not _regime_risk_on:
        if _is_active_research:
            # Soft pass: allow entry but apply a size multiplier to keep risk small.
            algo.debug(
                f"[active_research] soft regime pass for {top_sym.value}"
                f" (non-risk-on regime; size will be reduced)"
            )
        else:
            _final_rejection = f"regime:non_risk_on"
            algo._throttled_skip_debug(f"[vox] Regime block for {top_sym.value}")
            algo._last_gate_rejection = _final_rejection
            return

    # ── Risk manager gate ──────────────────────────────────────────────────
    pv = algo.portfolio.total_portfolio_value
    can, reason = algo._risk.can_enter(
        sym=top_sym, current_time=algo.time, portfolio_value=pv
    )
    if not can:
        _final_rejection = f"risk_mgr:{reason}"
        algo._throttled_skip_debug(f"[vox] Risk block for {top_sym.value}: {reason}")
        algo._last_gate_rejection = _final_rejection
        return

    # ── Pre-trade validation ───────────────────────────────────────────────
    tp_use, sl_use, atr, price, tp_floor_applied, sl_floor_applied = tp_sl_data[top_sym]
    class_proba_top  = conf_data[top_sym]["class_proba"]
    pred_return_top  = conf_data[top_sym]["pred_return"]
    ev_top           = ev_data[top_sym]
    _top_confirm     = getattr(algo, "_ruthless_confirm_reasons", {}).get(top_sym, "n/a")

    # Kelly / flat sizing (uses class_proba and ATR TP/SL for Kelly edge)
    qty, alloc = compute_qty(
        mean_proba      = class_proba_top,
        tp              = tp_use,
        sl              = sl_use,
        price           = price,
        portfolio_value = pv,
        kelly_frac      = algo._kf,
        max_alloc       = algo._max_alloc,
        cash_buffer     = algo._cb,
        use_kelly       = algo._use_kelly,
        allocation      = algo._alloc,
        min_alloc       = algo._min_alloc,
    )

    # ── Active-research regime size reduction ──────────────────────────────
    # When regime is not risk-on and we are in active_research (soft pass),
    # halve the position size to limit downside while still collecting data.
    if _is_active_research and not _regime_risk_on:
        from core import ACTIVE_RESEARCH_REGIME_SIZE_MULT
        qty   = qty   * ACTIVE_RESEARCH_REGIME_SIZE_MULT
        alloc = alloc * ACTIVE_RESEARCH_REGIME_SIZE_MULT

    # Cash check
    cash = algo.portfolio.cash
    if qty * price > cash * algo._cb:
        algo.debug(
            f"[vox] ENTRY skip {top_sym.value}: insufficient cash"
            f" (need {qty*price:.2f}, have {cash:.2f})"
        )
        algo._last_gate_rejection = f"cash:need={qty*price:.2f}>have={cash:.2f}"
        return

    # Lot-size rounding and minimum order validation
    sec      = algo.securities[top_sym]
    lot_size = OrderHelper.get_lot_size(sec)
    min_ord  = OrderHelper.get_min_order_size(sec)
    qty      = OrderHelper.round_qty(qty, lot_size)

    if not OrderHelper.validate_qty(qty, min_ord):
        algo.debug(
            f"[vox] ENTRY skip {top_sym.value}: qty={qty:.8f}"
            f" < min_order={min_ord}"
        )
        algo._last_gate_rejection = f"min_order:qty={qty:.8f}<min={min_ord}"
        return
    if qty <= 0:
        algo.debug(
            f"[vox] ENTRY skip {top_sym.value}: computed qty={qty:.8f}"
        )
        algo._last_gate_rejection = f"qty_zero:qty={qty:.8f}"
        return

    # Place entry order — set _pending_sym BEFORE market_order() (fills synchronously).
    algo._pending_sym        = top_sym
    algo._tp_dyn             = tp_use
    algo._sl_dyn             = sl_use
    algo._trail_active       = False
    algo._trail_high_px      = 0.0
    algo._max_return_seen    = 0.0
    algo._breakeven_active   = False
    algo._timeout_ext_hours  = 0.0
    algo._timeout_ext_logged = False
    algo._last_gate_rejection = None   # entry succeeded; clear last rejection
    algo._last_feat[top_sym] = next((f for s, f in candidates if s == top_sym), None)
    order = algo.market_order(top_sym, qty, tag="ENTRY")
    algo._pending_oid = order.order_id
    algo._fill_tracker.start_order(order.order_id, qty)

    # Store entry predictions for realized-EV logging at exit.
    _cd_entry = conf_data[top_sym]
    _ep_str   = str(algo.time).replace("-","").replace(" ","").replace(":","")[:12]
    algo._entry_predictions[top_sym] = {
        "trade_id":          f"{top_sym.value[:6]}_{_ep_str}",
        "symbol":            top_sym.value,
        "risk_profile":      algo._risk_profile,
        "market_mode":       _market_mode,
        "confirm":           _top_confirm,
        "entry_path":        entry_path_data.get(top_sym, "ml"),
        "class_proba":       class_proba_top,
        "pred_return":       pred_return_top,
        "ev":                ev_top,
        "final_score":       top_sc,
        "tp":                tp_use,
        "sl":                sl_use,
        "vote_score":        _cd_entry.get("vote_score", 0.0),
        "vote_yes_fraction": _cd_entry.get("vote_yes_fraction", 0.0),
        "top3_mean":         _cd_entry.get("top3_mean", 0.0),
        "n_agree":           _cd_entry.get("n_agree", 0),
        "std_proba":         _cd_entry.get("std_proba", 0.0),
        "time":              algo.time,
        "model_votes":       _cd_entry.get("per_model", {}),
        "active_votes":      _cd_entry.get("active_votes", {}),
        "shadow_votes":      _cd_entry.get("shadow_votes", {}),
        "diagnostic_votes":  _cd_entry.get("diagnostic_votes", {}),
    }

    _top_entry_path = entry_path_data.get(top_sym, "ml")
    if algo._risk_profile in ("ruthless", "gatling"):
        _ft = next((f for s, f in candidates if s == top_sym), None)
        _cd = conf_data[top_sym]
        algo.log(
            f"[ruthless] ENTRY {top_sym.value} path={_top_entry_path} confirm={_top_confirm}"
            f" proba={class_proba_top:.3f} agree={_cd['n_agree']}"
            f" std={_cd['std_proba']:.3f} ev={ev_top:.5f} pred={pred_return_top:.5f}"
            + _feature_diag_suffix(_ft)
            + f" tp={tp_use:.4f}(fl={tp_floor_applied}) sl={sl_use:.4f}(fl={sl_floor_applied})"
            + f" alloc={alloc:.3f} px={price:.4f} qty={qty:.6f}"
        )
    else:
        algo.debug(
            f"ENTRY order {top_sym.value}"
            f"  path={_top_entry_path}"
            f"  final_score={top_sc:.5f}"
            f"  ev={ev_top:.5f}"
            f"  pred_ret={pred_return_top:.5f}"
            f"  class_proba={class_proba_top:.3f}"
            f"  std_proba={conf_data[top_sym]['std_proba']:.3f}"
            f"  n_agree={conf_data[top_sym]['n_agree']}"
            f"  price={price:.4f}  qty={qty:.6f}"
            f"  alloc={alloc:.3f}  tp={tp_use:.4f}  sl={sl_use:.4f}"
        )

    # Log trade attempt to persistence
    _cd_top = conf_data[top_sym]
    algo._persistence.log_trade({
        "event":        "entry_attempt",
        "time":         str(algo.time),
        "symbol":       top_sym.value,
        "price":        price,
        "qty":          qty,
        "alloc":        alloc,
        "class_proba":  class_proba_top,
        "pred_return":  pred_return_top,
        "n_agree":      _cd_top["n_agree"],
        "std_proba":    _cd_top["std_proba"],
        "model_votes":  _cd_top.get("per_model", {}),
        "active_votes": _cd_top.get("active_votes", {}),
        "shadow_votes": _cd_top.get("shadow_votes", {}),
        "vote_score":         _cd_top.get("vote_score", 0.0),
        "vote_yes_fraction":  _cd_top.get("vote_yes_fraction", 0.0),
        "top3_mean":          _cd_top.get("top3_mean", 0.0),
        "tp":           tp_use,
        "sl":           sl_use,
        "ev_score":     ev_top,
        "final_score":  top_sc,
        "cost_bps":     algo._cost_bps,
        "entry_path":   _top_entry_path,
        "market_mode":  _market_mode,
        "confirm":      _top_confirm,
    })
    if algo._log_model_votes and _cd_top.get("per_model"):
        algo.log(format_vote_log(top_sym.value, _cd_top, market_mode=_market_mode))

    # Record candidate journal (top-N candidates for post-hoc analysis)
    if getattr(_cfg_module, "PERSIST_CANDIDATE_JOURNAL", True):
        _cj_records = build_candidate_records(
            ranked_results=ranked[:getattr(_cfg_module, "CANDIDATE_JOURNAL_TOP_N", 5)],
            conf_data=conf_data, ev_data=ev_data, entry_path_data=entry_path_data,
            scores=scores, market_mode=_market_mode,
            confirm_reasons=getattr(algo, "_ruthless_confirm_reasons", {}),
            selected_sym=top_sym,
        )
        algo._candidate_journal.record_cycle(algo.time, _cj_records)
