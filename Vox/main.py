# region imports
from AlgorithmImports import *
import json
import numpy as np
import random
from collections import deque
from datetime import timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Vox/main.py — PRODUCTION ENTRY POINT (Full ML-Powered Vox Algorithm)
#
# This is the primary production algorithm for the Vox ML ensemble strategy.
# It uses:
#   - VoxEnsemble: heterogeneous soft-voting classifier (HGBC/ET/RF/LR)
#   - Regime filters, risk manager, and profit-voting gates from strategy.py
#   - Triple-barrier labeling and walk-forward training from models.py
#   - Shadow model lab (optional diagnostic models)
#   - Full position/risk/exit logic with breakeven/trailing stops
#
# ► The simple baseline/benchmark algorithm is at: /main.py (root)
#   (KrakenTopCoinAlgorithm — momentum/RSI/volume only, no ML)
#
# ─────────────────────────────────────────────────────────────────────────────

from infra   import add_universe, KRAKEN_PAIRS, OrderHelper, PartialFillTracker, PersistenceManager
from models  import (
    build_features, compute_atr, VoxEnsemble, build_training_data, walk_forward_train,
    LABEL_TP, LABEL_SL, LABEL_HORIZON_BARS, LABEL_COST_BPS,
    check_label_execution_alignment, derive_training_hour,
)
from strategy import RegimeFilter, RiskManager, compute_qty
# endregion

from core   import *                       # noqa: F401,F403  strategy constants
from core   import setup_risk_profile
from core import check_momentum_override_conditions, compute_momentum_score
from strategy import (
    apply_breakeven, is_breakeven_active, should_exit_momentum_fail,
    evaluate_timeout, LimitOrderTracker, evaluate_candidate,
)
from core    import MarketModeDetector
from core     import MetaFilter
from infra          import hydrate_state_from_history
from journals    import format_vote_log, _feature_diag_suffix
from infra import format_model_registry_log, build_roles_dict_from_config
from journals import CandidateJournal, build_candidate_records, build_rejected_candidate_records
from audit_utils import audit_safe_float, audit_trim_votes
import core as _cfg_module
from entry_logic import check_exit as _check_exit_fn, try_enter as _try_enter_fn

np.random.seed(42)
random.seed(42)


class VoxAlgorithm(QCAlgorithm):
    """Execution-first ML ensemble strategy for Kraken spot crypto.

    See README.md for architecture details and risk-profile documentation.
    """

    _MODEL_VOTE_OUTCOME_KEY       = "vox/model_vote_outcomes.jsonl"
    _MODEL_VOTE_OUTCOME_MAX_BYTES = 90_000

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(10_000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.set_time_zone(TimeZones.UTC)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # ── Parameters — overridable via QC parameter panel ───────────────────
        self._tp       = float(self.get_parameter("take_profit")      or TAKE_PROFIT)
        self._sl       = float(self.get_parameter("stop_loss")        or STOP_LOSS)
        self._toh      = float(self.get_parameter("timeout_hours")    or TIMEOUT_HOURS)
        self._atr_tp   = float(self.get_parameter("atr_tp_mult")      or ATR_TP_MULT)
        self._atr_sl   = float(self.get_parameter("atr_sl_mult")      or ATR_SL_MULT)
        self._s_min    = float(self.get_parameter("score_min")        or SCORE_MIN)
        self._s_min_floor = SCORE_MIN_FLOOR
        self._s_gap    = float(self.get_parameter("score_gap")        or SCORE_GAP)
        self._max_disp = float(self.get_parameter("max_dispersion")   or MAX_DISPERSION)
        self._min_agr  = int(self.get_parameter("min_agree")          or MIN_AGREE)
        self._alloc    = float(self.get_parameter("allocation")       or ALLOCATION)
        self._kf       = float(self.get_parameter("kelly_frac")       or KELLY_FRAC)
        self._max_alloc= float(self.get_parameter("max_alloc")        or MAX_ALLOC)
        _uk_raw        = self.get_parameter("use_kelly")
        self._use_kelly= (str(_uk_raw).lower() in ("true","1","yes")) if _uk_raw else USE_KELLY
        self._max_sl   = int(self.get_parameter("max_daily_sl")       or MAX_DAILY_SL)
        self._cd_mins  = int(self.get_parameter("cooldown_mins")      or COOLDOWN_MINS)
        self._sl_cd    = int(self.get_parameter("sl_cooldown_mins")   or SL_COOLDOWN_MINS)
        self._max_dd   = float(self.get_parameter("max_dd_pct")       or MAX_DD_PCT)
        self._cb       = float(self.get_parameter("cash_buffer")      or CASH_BUFFER)
        self._exit_qty_buffer = int(
            self.get_parameter("exit_qty_buffer_lots") or EXIT_QTY_BUFFER_LOTS
        )
        self._cost_bps = float(self.get_parameter("cost_bps")         or COST_BPS)
        self._min_ev   = float(self.get_parameter("min_ev")           or MIN_EV)
        self._ev_gap   = float(self.get_parameter("ev_gap")           or EV_GAP)
        self._pred_return_min = float(self.get_parameter("pred_return_min") or PRED_RETURN_MIN)
        self._min_hold_minutes = int(self.get_parameter("min_hold_minutes") or MIN_HOLD_MINUTES)
        self._emergency_sl = float(self.get_parameter("emergency_sl") or EMERGENCY_SL)
        self._penalty_losses = int(self.get_parameter("penalty_cooldown_losses") or PENALTY_COOLDOWN_LOSSES)
        self._penalty_hours  = float(self.get_parameter("penalty_cooldown_hours") or PENALTY_COOLDOWN_HOURS)
        _uc_raw        = self.get_parameter("use_calibration")
        self._use_calibration = (str(_uc_raw).lower() in ("true", "1", "yes")) if _uc_raw else True
        self._min_alloc = float(self.get_parameter("min_alloc") or 0.0)
        self._runner_mode    = False
        self._trail_after_tp = 0.04
        self._trail_pct      = 0.025
        self._label_tp      = float(self.get_parameter("label_tp")           or LABEL_TP)
        self._label_sl      = float(self.get_parameter("label_sl")           or LABEL_SL)
        self._label_horizon = int(self.get_parameter("label_horizon_bars")   or LABEL_HORIZON_BARS)
        self._label_cost_bps = float(self.get_parameter("label_cost_bps")    or LABEL_COST_BPS)

        setup_risk_profile(self)

        self._market_mode_det  = None
        self._meta_filter      = MetaFilter(
            enabled  = getattr(self, "_meta_filter_enabled", False),
            min_proba= getattr(self, "_meta_min_proba", 0.55),
        )

        # ── Universe ──────────────────────────────────────────────────────────
        self._symbols  = add_universe(self)
        self._btc_sym  = next(
            (s for s in self._symbols if s.value.upper().startswith("BTC")), None
        )

        for sym in self._symbols:
            self.securities[sym].set_slippage_model(ConstantSlippageModel(0.001))

        # ── Per-symbol state deques ───────────────────────────────────────────
        _bars_per_day = int(24 * 60 / RESOLUTION_MINUTES)
        _state_max    = int(WARMUP_DAYS * _bars_per_day * 1.2)
        _dq = lambda n=_state_max: deque(maxlen=n)
        self._state = {}
        for sym in self._symbols:
            self._state[sym] = {"closes": _dq(), "highs": _dq(), "lows": _dq(), "volumes": _dq()}

        for sym in self._symbols:
            self.consolidate(sym, timedelta(minutes=RESOLUTION_MINUTES),
                             lambda bar, s=sym: self._on_5m(s, bar))

        self._regime = RegimeFilter()
        if self._btc_sym:
            self._regime.update_btc(self, self._btc_sym)

        _mm_enabled = getattr(self, "_market_mode_enabled", False)
        if _mm_enabled and self._btc_sym:
            self._market_mode_det = MarketModeDetector()
            self._market_mode_det.update_btc(self, self._btc_sym)

        self._risk = RiskManager(
            max_daily_sl=self._max_sl, cooldown_mins=self._cd_mins,
            sl_cooldown_mins=self._sl_cd, max_dd_pct=self._max_dd, cash_buffer=self._cb,
        )

        self._ensemble    = VoxEnsemble(logger=self.log, use_calibration=self._use_calibration)
        self._persistence = PersistenceManager(self)
        self._fill_tracker= PartialFillTracker()
        self._model_ready = False

        self._audit_clear_model_vote_outcomes_for_backtest()

        saved = self._persistence.load_model()
        if saved is not None:
            self._ensemble.load_state(saved)
            if hasattr(self._ensemble, "set_logger"):
                self._ensemble.set_logger(self.log)
            self._model_ready = self._ensemble.is_fitted
            self.log("[vox] Loaded pre-trained model from ObjectStore.")
        # Apply model roles from config (active/shadow/diagnostic)
        self._ensemble.set_model_roles(build_roles_dict_from_config(_cfg_module))
        self.log(format_model_registry_log(self._ensemble._estimators, roles_dict=self._ensemble._model_roles))

        # ── Position state ────────────────────────────────────────────────────
        self._pos_sym          = None
        self._entry_px         = 0.0
        self._entry_time       = None
        self._tp_dyn           = 0.0
        self._sl_dyn           = 0.0
        self._pending_sym      = None
        self._pending_oid      = None
        self._exiting          = False
        self._exit_time        = None
        self._exit_retry_count = 0
        self._trail_active     = False
        self._trail_high_px    = 0.0

        # ── Ruthless v4 position state ────────────────────────────────────────
        self._max_return_seen     = 0.0
        self._breakeven_active    = False
        self._timeout_ext_hours   = 0.0
        self._timeout_ext_logged  = False
        self._last_feat           = {}
        self._entry_limit_tracker = None

        self._entry_predictions = {}
        self._log_model_votes   = LOG_MODEL_VOTES
        self._candidate_journal = CandidateJournal(
            max_size=getattr(_cfg_module, "CANDIDATE_JOURNAL_MAX_SIZE", 2000),
            top_n=getattr(_cfg_module, "CANDIDATE_JOURNAL_TOP_N", 5),
        )

        # ── Per-symbol penalty cooldown ───────────────────────────────────────
        from collections import deque as _deque
        self._sym_outcomes     = {sym: _deque(maxlen=10) for sym in self._symbols}
        self._sym_penalty_until = {}
        self._sym_sl_times     = {}
        self._portfolio_loss_streak  = 0
        self._portfolio_pause_until  = None

        # ── Diagnostic throttling ─────────────────────────────────────────────
        self._last_skip_diag_time = None
        self._last_nocandidate_diag_time = None
        self._last_retrain_time = None
        self._last_gate_rejection = None  # last post-ranking gate rejection reason

        # ── Warm-up ───────────────────────────────────────────────────────────
        self.set_warm_up(timedelta(days=WARMUP_DAYS))

        # ── Scheduled events ──────────────────────────────────────────────────
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(0, 0),
            self._reset_daily,
        )
        # Weekly retrain on Monday at 00:05 UTC
        self.train(
            self.date_rules.week_start(),
            self.time_rules.at(0, 5),
            self._retrain,
        )

    # ── 5-minute bar handler ──────────────────────────────────────────────────

    def on_warmup_finished(self):
        """Schedule initial hydration + training off the time loop via self.train()."""
        self.log("[vox] Warmup finished — scheduling initial train off-loop.")
        self.train(self._initial_train)

    def _initial_train(self):
        """Off-loop initial training: hydrate history, then train."""
        import time
        t0 = time.time()
        try:
            self._hydrate_state_from_history(WARMUP_DAYS)
            self.log(f"[vox] hydrate took {time.time()-t0:.1f}s")
        except Exception as exc:
            self.log(f"[vox] history hydrate failed: {exc}")
        t1 = time.time()
        try:
            self._retrain()
            self.log(f"[vox] initial _retrain took {time.time()-t1:.1f}s")
        except Exception as exc:
            self.log(f"[vox] Initial train failed: {exc}")

    # ── 5-minute bar handler ──────────────────────────────────────────────────

    def _on_5m(self, sym, bar):
        """Append OHLCV from a freshly closed 5-min bar to the state deques."""
        st = self._state[sym]
        st["closes"].append(float(bar.close))
        st["highs"].append(float(bar.high))
        st["lows"].append(float(bar.low))
        st["volumes"].append(float(bar.volume))

    # ── 4h BTC bar handler (wired via RegimeFilter.update_btc) ───────────────
    # (RegimeFilter registers its own consolidator callback; the method below
    #  is kept for documentation / direct wiring if needed)

    def _on_4h_btc(self, bar):
        """Relay a 4h BTC bar to the regime filter (not used — wired inside RegimeFilter)."""
        pass

    # ── Main data handler ─────────────────────────────────────────────────────

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Update rolling high for drawdown circuit-breaker on every bar
        self._risk.update_rolling_high(self.portfolio.total_portfolio_value)

        # Reconcile internal state against actual holdings
        self._reconcile()

        # ── Exit check — minute-level precision ───────────────────────────────
        if self._pos_sym is not None and not self._exiting:
            if self._pos_sym in data.bars:
                self._check_exit(float(data.bars[self._pos_sym].close))
            elif self._entry_time is not None:
                # No bar for this symbol this tick (illiquid / low-volume pair).
                # Normal TP/SL checks are skipped, but still check the timeout
                # so the hold is bounded even when bars are absent.
                elapsed = (self.time - self._entry_time).total_seconds() / 3600.0
                if elapsed >= self._toh:
                    fallback_px = float(self.securities[self._pos_sym].price)
                    if fallback_px > 0:
                        self._check_exit(fallback_px)

        # ── Entry logic — fire only at decision interval boundaries ──────────
        _decision_interval = getattr(self, "_gatling_decision_interval", DECISION_INTERVAL_MIN)
        if self.time.minute % _decision_interval != 0:
            return
        if self._pos_sym is not None or self._pending_sym is not None:
            return
        # Block new entries while an exit order is still in flight.
        # This prevents double-entry in the same bar when an exit fills slowly.
        if self._exiting:
            return

        # Kill switch
        if self._persistence.is_kill_switch_active():
            self.log("[vox] Kill switch active — skipping entry.")
            return

        # Ruthless portfolio loss-streak brake — pause all new entries briefly
        if self._portfolio_pause_until is not None:
            if self.time < self._portfolio_pause_until:
                self._throttled_skip_debug(
                    f"[vox] Portfolio loss-streak pause active until {self._portfolio_pause_until}"
                )
                return
            else:
                self._portfolio_pause_until = None   # pause expired

        # Model must be trained
        if not self._model_ready:
            return

        # Risk guards
        pv = self.portfolio.total_portfolio_value
        can, reason = self._risk.can_enter(
            sym=None, current_time=self.time, portfolio_value=pv
        )
        if not can:
            self._throttled_skip_debug(f"[vox] Entry blocked: {reason}")
            return

        self._try_enter()

    def _clear_position_state(self, include_retry=False):
        """Reset all position state fields after an exit or cancellation.

        Note: _last_feat is intentionally NOT cleared here because it is a
        per-symbol dict used for exit decisions (momentum-fail) and the
        stale feature value is harmless — it will be overwritten on the next
        entry bar.  Clearing it here would also risk a KeyError if _check_exit
        reads it for the same symbol during the same tick.
        """
        self._pos_sym           = None
        self._entry_px          = 0.0
        self._entry_time        = None
        self._tp_dyn            = 0.0
        self._sl_dyn            = 0.0
        self._exiting           = False
        self._exit_time         = self.time
        self._trail_active      = False
        self._trail_high_px     = 0.0
        self._max_return_seen   = 0.0
        self._breakeven_active  = False
        self._timeout_ext_hours = 0.0
        self._timeout_ext_logged= False
        if include_retry:
            self._exit_retry_count = 0

    # ── Audit helpers ─────────────────────────────────────────────────────────

    def _audit_clear_model_vote_outcomes_for_backtest(self):
        """Clear the model vote outcome log at backtest start (not in live)."""
        if not self.live_mode:
            try:
                self.object_store.save(self._MODEL_VOTE_OUTCOME_KEY, "")
            except Exception as exc:
                self.debug(f"[audit] clear_model_vote_outcomes failed: {exc}")

    def _audit_append_model_vote_outcome(self, record):
        """Append one JSON record to vox/model_vote_outcomes.jsonl, capping to max bytes."""
        try:
            line = json.dumps(record, default=str) + "\n"
            existing = ""
            if self.object_store.contains_key(self._MODEL_VOTE_OUTCOME_KEY):
                existing = self.object_store.read(self._MODEL_VOTE_OUTCOME_KEY)
            combined = existing + line
            if len(combined.encode()) > self._MODEL_VOTE_OUTCOME_MAX_BYTES:
                lines = [l for l in combined.splitlines() if l.strip()]
                while lines and len("\n".join(lines).encode()) > self._MODEL_VOTE_OUTCOME_MAX_BYTES:
                    lines.pop(0)
                combined = "\n".join(lines) + "\n"
            self.object_store.save(self._MODEL_VOTE_OUTCOME_KEY, combined)
        except Exception as exc:
            self.debug(f"[audit] append_model_vote_outcome failed: {exc}")

    # ── Order-event state machine ──────────────────────────────────────────────

    def on_order_event(self, order_event):
        oid   = order_event.order_id
        order = self.transactions.get_order_by_id(oid)
        if order is None:
            return
        sym = order.symbol
        tag = order.tag or ""

        if order_event.status == OrderStatus.FILLED:
            if tag == "ENTRY" and sym == self._pending_sym:
                # Entry fill confirmed — mark position active
                self._pos_sym    = sym
                self._entry_px   = float(order_event.fill_price)
                self._entry_time = self.time
                self._fill_tracker.on_fill(oid, abs(float(order_event.fill_quantity)))
                self._pending_sym = None
                self._pending_oid = None
                self.debug(
                    f"FILL ENTRY {sym.value}  px={self._entry_px:.4f}"
                    f"  qty={order_event.fill_quantity:.6f}"
                )

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                # Exit fill confirmed — clear position state
                ret = (
                    (order_event.fill_price - self._entry_px) / self._entry_px
                    if self._entry_px else 0.0
                )
                # EXIT_BE (breakeven) is profit-protection — never a stop-loss hit.
                # EXIT_MOM_FAIL counts as a risk/SL exit when the return is negative.
                is_sl = (
                    tag == "EXIT_SL"
                    or (tag == "EXIT_MOM_FAIL" and ret < 0.0)
                )
                is_loss = ret < 0.0

                # ── Ruthless/gatling exit diagnostics ─────────────────────────
                if self._risk_profile in ("ruthless", "gatling"):
                    _ti = (f"  trail_high={self._trail_high_px:.4f}" if self._trail_active else "")
                    self.log(
                        f"[exit_diag] {sym.value}  tag={tag}"
                        f"  entry={self._entry_px:.5f}  fill={order_event.fill_price:.5f}"
                        f"  ret={ret:+.3%}  max={self._max_return_seen:+.3%}"
                        f"  elapsed_min={(self.time - self._entry_time).total_seconds()/60 if self._entry_time else 0:.1f}"
                        + (f"  trail=active{_ti}" if self._trail_active else "")
                    )
                else:
                    self.debug(
                        f"FILL EXIT {sym.value}  px={order_event.fill_price:.4f}"
                        f"  ret={ret:.3%}  tag={tag}"
                    )

                self._risk.record_exit(sym, is_sl=is_sl, exit_time=self.time)
                self._fill_tracker.clear(oid)

                # ── Realized EV logging ────────────────────────────────────────
                entry_pred = self._entry_predictions.pop(sym, None)
                self._persistence.log_trade({
                    "event":              "exit",
                    "time":               str(self.time),
                    "symbol":             sym.value,
                    "exit_reason":        tag,
                    "realized_return":    round(ret, 6),
                    "max_return_seen":    round(self._max_return_seen, 6),
                    "predicted_class_proba": round(entry_pred["class_proba"], 4) if entry_pred else None,
                    "predicted_return":   round(entry_pred["pred_return"], 6)    if entry_pred else None,
                    "predicted_ev":       round(entry_pred["ev"], 6)             if entry_pred else None,
                    "final_score":        round(entry_pred["final_score"], 6)    if entry_pred else None,
                    "tp":                 round(entry_pred["tp"], 4)             if entry_pred else None,
                    "sl":                 round(entry_pred["sl"], 4)             if entry_pred else None,
                    "entry_path":         entry_pred.get("entry_path", "ml")     if entry_pred else None,
                    "model_votes":        entry_pred.get("model_votes", {})      if entry_pred else {},
                })

                # ── Compact closed-trade audit (model vote outcomes) ───────────
                if entry_pred:
                    _hold_min = None
                    _et = entry_pred.get("time")
                    if _et is not None:
                        try:
                            _hold_min = round((self.time - _et).total_seconds() / 60.0, 1)
                        except Exception:
                            pass
                    self._audit_append_model_vote_outcome({
                        "trade_id":          entry_pred.get("trade_id", ""),
                        "symbol":            sym.value,
                        "risk_profile":      entry_pred.get("risk_profile", ""),
                        "market_mode":       entry_pred.get("market_mode"),
                        "entry_path":        entry_pred.get("entry_path", "ml"),
                        "confirm":           entry_pred.get("confirm", ""),
                        "entry_time":        str(_et or ""),
                        "exit_time":         str(self.time),
                        "entry_price":       audit_safe_float(self._entry_px, 6),
                        "exit_price":        audit_safe_float(order_event.fill_price, 6),
                        "exit_reason":       tag,
                        "realized_return":   audit_safe_float(ret, 6),
                        "winner":            ret > 0,
                        "hold_minutes":      _hold_min,
                        "max_return_seen":   audit_safe_float(self._max_return_seen, 6),
                        "class_proba":       audit_safe_float(entry_pred.get("class_proba"), 4),
                        "pred_return":       audit_safe_float(entry_pred.get("pred_return"), 6),
                        "ev":                audit_safe_float(entry_pred.get("ev"), 6),
                        "final_score":       audit_safe_float(entry_pred.get("final_score"), 6),
                        "tp":                audit_safe_float(entry_pred.get("tp"), 4),
                        "sl":                audit_safe_float(entry_pred.get("sl"), 4),
                        "vote_score":        audit_safe_float(entry_pred.get("vote_score"), 4),
                        "vote_yes_fraction": audit_safe_float(entry_pred.get("vote_yes_fraction"), 4),
                        "top3_mean":         audit_safe_float(entry_pred.get("top3_mean"), 4),
                        "n_agree":           entry_pred.get("n_agree"),
                        "std_proba":         audit_safe_float(entry_pred.get("std_proba"), 4),
                        "model_votes":       audit_trim_votes(entry_pred.get("model_votes", {})),
                        "active_votes":      audit_trim_votes(entry_pred.get("active_votes", {})),
                        "shadow_votes":      audit_trim_votes(entry_pred.get("shadow_votes", {})),
                        "diagnostic_votes":  audit_trim_votes(entry_pred.get("diagnostic_votes", {})),
                    })

                # ── Per-symbol outcome tracking (penalty cooldown) ─────────────
                if sym not in self._sym_outcomes:
                    self._sym_outcomes[sym] = deque(maxlen=10)
                self._sym_outcomes[sym].append(ret)
                self._update_penalty_cooldown(sym, is_sl)

                # ── Anti-chop: per-symbol SL timestamp tracking ────────────────
                if self._risk_profile in ("ruthless", "gatling") and is_sl:
                    if sym not in self._sym_sl_times:
                        self._sym_sl_times[sym] = deque(maxlen=20)
                    self._sym_sl_times[sym].append(self.time)
                    # Check if 2+ SLs occurred within the loss window → extended block
                    _window_h   = getattr(self, "_ruthless_loss_window_hours", 24)
                    _loss_limit = getattr(self, "_ruthless_loss_limit", 2)
                    _block_h    = getattr(self, "_ruthless_loss_block_hours", 24)
                    _cutoff     = self.time - timedelta(hours=_window_h)
                    _recent_sls = sum(1 for t in self._sym_sl_times[sym] if t >= _cutoff)
                    if _recent_sls >= _loss_limit:
                        _block_end = self.time + timedelta(hours=_block_h)
                        self._sym_penalty_until[sym] = _block_end
                        self.log(
                            f"[ruthless] ANTI-CHOP BLOCK: {sym.value} had"
                            f" {_recent_sls} SL exits in {_window_h}h"
                            f" — blocked until {_block_end}"
                        )

                # ── Portfolio loss-streak brake ────────────────────────────────
                if self._risk_profile in ("ruthless", "gatling"):
                    _streak_limit = getattr(self, "_ruthless_portfolio_loss_streak", 4)
                    _pause_h      = getattr(self, "_ruthless_portfolio_pause_hours", 6)
                    if is_loss:
                        self._portfolio_loss_streak += 1
                        if self._portfolio_loss_streak >= _streak_limit:
                            _pause_end = self.time + timedelta(hours=_pause_h)
                            self._portfolio_pause_until = _pause_end
                            self.log(
                                f"[ruthless] LOSS-STREAK PAUSE: {self._portfolio_loss_streak}"
                                f" consecutive losses — all entries paused until {_pause_end}"
                            )
                            self._portfolio_loss_streak = 0  # reset after triggering
                    else:
                        self._portfolio_loss_streak = 0   # any win resets streak

                self._clear_position_state()

        elif order_event.status == OrderStatus.PARTIALLY_FILLED:
            self._fill_tracker.on_fill(oid, abs(float(order_event.fill_quantity)))
            if tag == "ENTRY" and sym == self._pending_sym and self._pos_sym is None:
                self._pos_sym    = sym
                self._entry_px   = float(order_event.fill_price)
                self._entry_time = self.time
                self.debug(
                    f"PARTIAL FILL ENTRY {sym.value}  px={self._entry_px:.4f}"
                    f"  qty_so_far={self._fill_tracker.get_filled(oid):.6f}"
                )
            if self._fill_tracker.is_complete(oid):
                self._pending_sym = None
                self._pending_oid = None

        elif order_event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            if tag == "ENTRY" and sym == self._pending_sym:
                self.debug(
                    f"ENTRY order {oid} for {sym.value} — status={order_event.status},"
                    f" clearing pending"
                )
                self._fill_tracker.clear(oid)
                self._pending_sym   = None
                self._pending_oid   = None
                self._clear_position_state()

            elif tag.startswith("EXIT") and sym == self._pos_sym:
                self._exit_retry_count += 1
                _sec      = self.securities[sym]
                _lot_size = OrderHelper.get_lot_size(_sec)
                _min_ord  = OrderHelper.get_min_order_size(_sec)
                _safe_qty = OrderHelper.safe_crypto_sell_qty(
                    self, sym, _lot_size, _min_ord,
                    exit_qty_buffer_lots=self._exit_qty_buffer,
                )
                if _safe_qty <= 0 or self._exit_retry_count >= MAX_EXIT_RETRY_COUNT:
                    self.debug(
                        f"EXIT order {oid} for {sym.value} —"
                        f" status={order_event.status},"
                        f" safe_qty={_safe_qty:.8f}"
                        f" retry={self._exit_retry_count}"
                        f" — clearing as dust"
                    )
                    is_sl = (tag == "EXIT_SL")  # EXIT_BE and EXIT_MOM_FAIL are not SL here
                    self._risk.record_exit(sym, is_sl=is_sl, exit_time=self.time)
                    self._clear_position_state(include_retry=True)
                else:
                    self.debug(
                        f"EXIT order {oid} for {sym.value} —"
                        f" status={order_event.status},"
                        f" safe_qty={_safe_qty:.8f}"
                        f" retry={self._exit_retry_count} — will retry"
                    )
                    self._exiting = False

    # ── Reconciliation ────────────────────────────────────────────────────────

    def _reconcile(self):
        """Compare internal state against portfolio; clears stale state on divergence."""
        if self._pos_sym is not None:
            qty = self.portfolio[self._pos_sym].quantity
            if qty <= 0:
                sym = self._pos_sym   # capture before clearing
                self.debug(
                    f"RECONCILE: tracking {sym.value} but qty={qty:.6f}"
                    f" — clearing stale state"
                )
                self._clear_position_state()
                # Inform the risk manager so cooldown accounting is not bypassed.
                self._risk.record_exit(sym, is_sl=False, exit_time=self.time)

        # Safety net for stale pending orders.
        if self._pending_oid is not None:
            order = self.transactions.get_order_by_id(self._pending_oid)
            if order is not None and order.status in (
                OrderStatus.FILLED, OrderStatus.INVALID, OrderStatus.CANCELED
            ):
                if order.status == OrderStatus.FILLED:
                    if self._pos_sym is None and self._pending_sym is not None:
                        held_qty = self.portfolio[self._pending_sym].quantity
                        if held_qty > 0:
                            self._pos_sym    = self._pending_sym
                            self._entry_time = self.time
                            self._entry_px   = float(
                                self.securities[self._pending_sym].price
                            )
                            self.debug(
                                f"RECONCILE: recovered missed ENTRY fill for"
                                f" {self._pending_sym.value}"
                                f"  approx_px={self._entry_px:.4f}"
                            )
                    self._fill_tracker.clear(self._pending_oid)
                    self._pending_sym = None
                    self._pending_oid = None
                else:   # INVALID or CANCELED
                    self._fill_tracker.clear(self._pending_oid)
                    self._pending_sym = None
                    self._pending_oid = None

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, price):
        """Evaluate TP / SL / timeout and submit market sell if triggered."""
        _check_exit_fn(self, price)

    # ── Entry logic ───────────────────────────────────────────────────────────
    def _try_enter(self):
        """Score all symbols; if a clear winner passes all gates, place a buy order."""
        _try_enter_fn(self)


    # ── Retrain ───────────────────────────────────────────────────────────────

    def _retrain(self):
        """Weekly retrain on all accumulated history.  Saves model to ObjectStore."""
        # Skip during warmup — _initial_train() handles the first fit after warmup.
        if self.is_warming_up:
            return

        # Avoid duplicate retrain shortly after the initial train or a recent
        # scheduled retrain.  Skip if the last retrain completed within
        # MIN_RETRAIN_INTERVAL_HOURS hours.
        if self._last_retrain_time is not None:
            elapsed_h = (self.time - self._last_retrain_time).total_seconds() / 3600.0
            if elapsed_h < MIN_RETRAIN_INTERVAL_HOURS:
                self.debug(
                    f"[vox] Scheduled retrain skipped: last retrain was"
                    f" {elapsed_h:.1f}h ago (min={MIN_RETRAIN_INTERVAL_HOURS}h)"
                )
                return

        self.log("[vox] Starting scheduled retrain …")
        timeout_bars = int(self._toh * 60 / DECISION_INTERVAL_MIN)

        X, y_class, y_return = build_training_data(
            algorithm             = self,
            symbols               = self._symbols,
            state_dict            = self._state,
            tp                    = self._tp,
            sl                    = self._sl,
            timeout_bars          = timeout_bars,
            decision_interval_min = DECISION_INTERVAL_MIN,
            label_tp              = self._label_tp,
            label_sl              = self._label_sl,
            label_horizon_bars    = self._label_horizon,
            cost_bps              = self._label_cost_bps,
        )
        if X is None or len(X) < 50:
            self.log("[vox] Retrain skipped: insufficient training data.")
            return

        # ── Label-vs-execution alignment check ────────────────────────────────
        check_label_execution_alignment(
            label_tp             = self._label_tp,
            label_sl             = self._label_sl,
            label_horizon_bars   = self._label_horizon,
            exec_tp              = self._tp,
            exec_sl              = self._sl,
            exec_timeout_hours   = self._toh,
            decision_interval_min = DECISION_INTERVAL_MIN,
            logger               = self.log,
        )

        try:
            self._ensemble = walk_forward_train(self._ensemble, X, y_class, y_return)
            self._model_ready = self._ensemble.is_fitted
            self._persistence.save_model(self._ensemble)
            reg_trained = getattr(self._ensemble, "_reg_fitted", False)
            self._last_retrain_time = self.time
            self.log(
                f"[vox] Retrain complete. Samples={len(X)}, fitted={self._model_ready}"
                f", reg_fitted={reg_trained}"
            )
        except Exception as exc:
            self.log(f"[vox] Retrain failed: {exc}")

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily(self):
        """Reset daily counters at midnight UTC."""
        self._risk.reset_daily()

    # ── Per-symbol penalty cooldown ───────────────────────────────────────────

    def _is_in_penalty_cooldown(self, sym):
        """Return True if sym is currently blocked by the penalty cooldown."""
        penalty_until = self._sym_penalty_until.get(sym)
        if penalty_until is None:
            return False
        if self.time >= penalty_until:
            # Cooldown expired — clean up
            del self._sym_penalty_until[sym]
            return False
        return True

    def _update_penalty_cooldown(self, sym, is_sl):
        """Apply penalty cooldown if sym has had _penalty_losses consecutive SL exits."""
        if not is_sl:
            return   # Only SL exits contribute to penalty counting
        outcomes = list(self._sym_outcomes.get(sym, []))
        if len(outcomes) < self._penalty_losses:
            return   # Not enough history yet

        # Check that the last N outcomes are all losses (SL returns are negative)
        recent = outcomes[-self._penalty_losses:]
        all_losses = all(r < 0 for r in recent)
        if all_losses:
            penalty_end = self.time + timedelta(hours=self._penalty_hours)
            self._sym_penalty_until[sym] = penalty_end
            self.log(
                f"[vox] PENALTY COOLDOWN: {sym.value} has had {self._penalty_losses}"
                f" consecutive losses — blocked until {penalty_end} "
                f"({self._penalty_hours}h)"
            )

    # ── Throttled diagnostic helper ───────────────────────────────────────────

    def _throttled_skip_debug(self, message: str) -> None:
        """Emit debug diagnostic for routine skips, throttled to SKIP_DIAG_INTERVAL_SECS."""
        now = self.time
        if (
            self._last_skip_diag_time is None
            or (now - self._last_skip_diag_time).total_seconds() >= SKIP_DIAG_INTERVAL_SECS
        ):
            self._last_skip_diag_time = now
            self.debug(message)

    # ── History hydration ─────────────────────────────────────────────────────

    def _hydrate_state_from_history(self, days):
        """Delegate to infra.hydrate_state_from_history."""
        hydrate_state_from_history(
            algorithm=self,
            state=self._state,
            symbols=self._symbols,
            days=days,
            resolution_minutes=RESOLUTION_MINUTES,
            max_history_bars=MAX_HISTORY_BARS,
        )

    # ── End of algorithm ──────────────────────────────────────────────────────

    def on_end_of_algorithm(self):
        try:
            self._persistence.flush_trade_log()
        except Exception as exc:
            self.log(f"[vox] Failed to flush trade log on shutdown: {exc}")
