# ── Tests: Ruthless V2 engine ─────────────────────────────────────────────────
#
# Tests for ruthless_v2.py and trade_vote_audit.py functionality.
# Covers the 10 acceptance criteria from the problem statement:
#  1. V2 config defaults are safe/off unless explicitly enabled.
#  2. V2 active pool includes rf, et, hgbc_l2, lgbm_bal, gbc, ada when available.
#  3. V2 active pool excludes gnb, lr, lr_bal, cal_et, cal_rf by default.
#  4. Dynamic voter weight update rewards winners and penalizes losers with caps.
#  5. V2 opportunity score formula ranks stronger candidates higher.
#  6. Pump exhaustion score increases after repeated same-symbol wins/entries.
#  7. Pump continuation can override simple cooldown when strength is high.
#  8. Trade vote audit record shape contains required entry/exit/vote fields.
#  9. Multi-position limit helper respects max positions, per-symbol allocation,
#     total exposure.
# 10. Split-exit sizing helper produces safe partial/runner quantities.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ruthless_v2 import (
    RUTHLESS_V2_MODE,
    RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
    RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
    RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
    RUTHLESS_V2_ACTIVE_MODELS,
    RUTHLESS_V2_DIAGNOSTIC_MODELS,
    RUTHLESS_V2_OPTIONAL_MODELS,
    RUTHLESS_V2_BASE_WEIGHTS,
    RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST,
    MultiPositionManager,
    DynamicVoterWeighting,
    compute_multihorizon_scores,
    compute_v2_opportunity_score,
    compute_breakout_score,
    compute_volume_expansion_score,
    compute_regime_score,
    compute_relative_strength_scores,
    compute_pump_scores,
    exhaustion_override_allowed,
    compute_meta_entry_score,
    SplitExitHelper,
    rank_candidates_v2,
    format_v2_startup_log,
)
from trade_vote_audit import (
    TradeVoteAudit,
    build_entry_snapshot,
    build_exit_outcome,
    ENTRY_RECORD_REQUIRED_FIELDS,
    EXIT_RECORD_REQUIRED_FIELDS,
    AUDIT_STORE_KEY,
)
from config import (
    RUTHLESS_V2_MODE as CFG_V2_MODE,
    RUTHLESS_ACTIVE_MODELS,
    RUTHLESS_DIAGNOSTIC_MODELS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now():
    return datetime.datetime(2025, 3, 2, 12, 0, 0)


def _feat(
    ret_1=0.005, ret_4=0.012, ret_8=0.015, ret_16=0.020, ret_32=0.025,
    vol_p=1.0, vol_r=2.5, btc_rel=0.008, rsi=55.0, bb_pos=0.6, bb_w=0.05,
    extra_zeros=10,
):
    base = [ret_1, ret_4, ret_8, ret_16, ret_32, vol_p, vol_r, btc_rel,
            rsi, bb_pos, bb_w]
    return base + [0.0] * extra_zeros


def _conf(
    class_proba=0.65, std_proba=0.10, n_agree=4,
    vote_score=0.70, vote_yes_fraction=0.80, top3_mean=0.75,
    active_model_count=6, active_mean=0.65, active_std=0.10,
    active_n_agree=5, pred_return=0.015, active_votes=None,
):
    av = active_votes or {
        "rf": 0.72, "et": 0.68, "hgbc_l2": 0.80, "lgbm_bal": 0.75,
        "gbc": 0.65, "ada": 0.60,
    }
    return {
        "class_proba":      class_proba,
        "std_proba":        std_proba,
        "n_agree":          n_agree,
        "vote_score":       vote_score,
        "vote_yes_fraction":vote_yes_fraction,
        "top3_mean":        top3_mean,
        "active_model_count": active_model_count,
        "active_mean":      active_mean,
        "active_std":       active_std,
        "active_n_agree":   active_n_agree,
        "pred_return":      pred_return,
        "active_votes":     av,
        "shadow_votes":     {"xgb_bal": 0.60},
        "diagnostic_votes": {"gnb": 0.99, "lr": 0.02},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. V2 config defaults are safe/off
# ─────────────────────────────────────────────────────────────────────────────

class TestV2ConfigDefaults:
    """Criterion 1: V2 config exists and defaults to safe/off."""

    def test_v2_mode_off_by_default(self):
        """RUTHLESS_V2_MODE default must be False (safe off)."""
        assert RUTHLESS_V2_MODE is False

    def test_config_module_v2_mode_off_by_default(self):
        """config.RUTHLESS_V2_MODE must be False."""
        assert CFG_V2_MODE is False

    def test_max_positions_is_positive(self):
        assert RUTHLESS_V2_MAX_CONCURRENT_POSITIONS >= 1

    def test_max_entries_per_day_is_positive(self):
        assert RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY >= 1

    def test_alloc_bounds(self):
        assert 0 < RUTHLESS_V2_MIN_SYMBOL_ALLOCATION < RUTHLESS_V2_MAX_SYMBOL_ALLOCATION <= 1.0

    def test_total_exposure_above_one(self):
        """V2 max_total_exposure > 1.0 enables multi-position."""
        assert RUTHLESS_V2_MAX_TOTAL_EXPOSURE > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. V2 active pool includes rf, et, hgbc_l2, lgbm_bal, gbc, ada
# ─────────────────────────────────────────────────────────────────────────────

class TestV2ActivePool:
    """Criterion 2: V2 active pool includes the correct models."""

    def test_v2_active_includes_rf(self):
        assert "rf" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v2_active_includes_et(self):
        assert "et" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v2_active_includes_hgbc_l2(self):
        assert "hgbc_l2" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v2_active_includes_lgbm_bal(self):
        assert "lgbm_bal" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v2_active_includes_gbc(self):
        assert "gbc" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v2_active_includes_ada(self):
        assert "ada" in RUTHLESS_V2_ACTIVE_MODELS

    def test_v1_active_pool_also_includes_core_models(self):
        """V1 ruthless active pool still has rf, et, hgbc_l2, lgbm_bal."""
        for model in ["rf", "et", "hgbc_l2", "lgbm_bal"]:
            assert model in RUTHLESS_ACTIVE_MODELS, (
                f"V1 pool missing {model}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. V2 active pool excludes gnb, lr, lr_bal, cal_et, cal_rf
# ─────────────────────────────────────────────────────────────────────────────

class TestV2DiagnosticExclusions:
    """Criterion 3: Conservative/degenerate models not in V2 active pool."""

    def test_gnb_not_in_v2_active(self):
        assert "gnb" not in RUTHLESS_V2_ACTIVE_MODELS

    def test_lr_not_in_v2_active(self):
        assert "lr" not in RUTHLESS_V2_ACTIVE_MODELS

    def test_lr_bal_not_in_v2_active(self):
        assert "lr_bal" not in RUTHLESS_V2_ACTIVE_MODELS

    def test_cal_et_not_in_v2_active(self):
        assert "cal_et" not in RUTHLESS_V2_ACTIVE_MODELS

    def test_cal_rf_not_in_v2_active(self):
        assert "cal_rf" not in RUTHLESS_V2_ACTIVE_MODELS

    def test_gnb_in_v2_diagnostic(self):
        assert "gnb" in RUTHLESS_V2_DIAGNOSTIC_MODELS

    def test_lr_in_v2_diagnostic(self):
        assert "lr" in RUTHLESS_V2_DIAGNOSTIC_MODELS

    def test_cal_et_in_v2_diagnostic(self):
        assert "cal_et" in RUTHLESS_V2_DIAGNOSTIC_MODELS

    def test_cal_rf_in_v2_diagnostic(self):
        assert "cal_rf" in RUTHLESS_V2_DIAGNOSTIC_MODELS

    def test_v1_diagnostic_includes_cal_et_cal_rf(self):
        """V1 active pool should NOT include cal_et and cal_rf (moved to diagnostic)."""
        assert "cal_et" not in RUTHLESS_ACTIVE_MODELS
        assert "cal_rf" not in RUTHLESS_ACTIVE_MODELS
        # cal_et, cal_rf should now be in V1 diagnostic list
        assert "cal_et" in RUTHLESS_DIAGNOSTIC_MODELS
        assert "cal_rf" in RUTHLESS_DIAGNOSTIC_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dynamic voter weight update
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicVoterWeighting:
    """Criterion 4: Dynamic voter weighting rewards winners, penalizes losers."""

    def _make_dv(self):
        return DynamicVoterWeighting(
            base_weights={"rf": 1.35, "et": 0.80, "ada": 0.70},
            min_obs=2,
        )

    def test_initial_weights_equal_base(self):
        dv = self._make_dv()
        assert dv.effective_weight("rf") == pytest.approx(1.35)
        assert dv.effective_weight("et") == pytest.approx(0.80)

    def test_winning_trade_rewards_yes_voters(self):
        dv = self._make_dv()
        # Both models voted yes; trade won
        snap = {"rf": True, "et": True, "ada": True}
        for _ in range(3):  # exceed min_obs=2
            dv.update(snap, realized_return=0.05, winner=True)
        # Weight should increase after wins
        assert dv.effective_weight("rf") > 1.35

    def test_losing_trade_penalizes_yes_voters(self):
        dv = self._make_dv()
        snap = {"rf": True, "et": True, "ada": True}
        for _ in range(3):
            dv.update(snap, realized_return=-0.03, winner=False)
        # Weight should decrease after losses
        assert dv.effective_weight("rf") < 1.35

    def test_weight_cap_max(self):
        dv = self._make_dv()
        snap = {"rf": True}
        for _ in range(20):  # many wins
            dv.update(snap, realized_return=0.10, winner=True)
        # Must not exceed base * max_multiplier
        max_allowed = 1.35 * RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER
        assert dv.effective_weight("rf") <= max_allowed + 1e-6

    def test_weight_cap_min(self):
        dv = self._make_dv()
        snap = {"rf": True}
        for _ in range(20):  # many losses
            dv.update(snap, realized_return=-0.05, winner=False)
        # Must not go below base * min_multiplier
        min_allowed = 1.35 * RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER
        assert dv.effective_weight("rf") >= min_allowed - 1e-6

    def test_no_obs_keeps_neutral_weight(self):
        dv = self._make_dv()
        # Only 1 update (below min_obs=2) — weight should stay neutral
        snap = {"rf": True}
        dv.update(snap, realized_return=0.10, winner=True)
        # perf_score starts at 1.0 → effective_weight = base
        assert dv.effective_weight("rf") == pytest.approx(1.35)

    def test_snapshot_entry_votes_correct(self):
        dv = self._make_dv()
        active_votes = {"rf": 0.70, "et": 0.45}  # rf=yes, et=no at thr=0.50
        snap = dv.snapshot_entry_votes(active_votes)
        assert snap["rf"] is True
        assert snap["et"] is False

    def test_state_summary_has_required_fields(self):
        dv = self._make_dv()
        summary = dv.get_state_summary()
        for mid in ["rf", "et", "ada"]:
            assert mid in summary
            s = summary[mid]
            assert "base_weight" in s
            assert "effective_weight" in s
            assert "yes_count" in s
            assert "yes_win_rate" in s


# ─────────────────────────────────────────────────────────────────────────────
# 5. V2 opportunity score ranks stronger candidates higher
# ─────────────────────────────────────────────────────────────────────────────

class TestV2OpportunityScore:
    """Criterion 5: V2 score formula ranks stronger candidates higher."""

    def test_higher_vote_score_ranks_higher(self):
        s_high = compute_v2_opportunity_score(
            dynamic_vote_score=0.80, continuation_score=0.50, runner_score=0.50,
            breakout_score=0.50, volume_expansion_score=0.50, regime_score=0.50,
        )
        s_low = compute_v2_opportunity_score(
            dynamic_vote_score=0.20, continuation_score=0.50, runner_score=0.50,
            breakout_score=0.50, volume_expansion_score=0.50, regime_score=0.50,
        )
        assert s_high > s_low

    def test_exhaustion_penalty_reduces_score(self):
        s_clean = compute_v2_opportunity_score(
            dynamic_vote_score=0.70, continuation_score=0.60, runner_score=0.60,
            breakout_score=0.60, volume_expansion_score=0.60, regime_score=0.60,
        )
        s_exhaust = compute_v2_opportunity_score(
            dynamic_vote_score=0.70, continuation_score=0.60, runner_score=0.60,
            breakout_score=0.60, volume_expansion_score=0.60, regime_score=0.60,
            exhaustion_penalty=0.20,
        )
        assert s_exhaust < s_clean

    def test_rank_candidates_sorts_descending(self):
        cands = [
            {"sym": "A", "v2_opportunity_score": 0.30},
            {"sym": "B", "v2_opportunity_score": 0.80},
            {"sym": "C", "v2_opportunity_score": 0.55},
        ]
        ranked = rank_candidates_v2(cands)
        assert ranked[0]["sym"] == "B"
        assert ranked[1]["sym"] == "C"
        assert ranked[2]["sym"] == "A"

    def test_pump_regime_gives_highest_regime_score(self):
        assert compute_regime_score("pump") > compute_regime_score("chop")
        assert compute_regime_score("risk_on_trend") > compute_regime_score("chop")

    def test_multihorizon_scores_return_expected_keys(self):
        feat = _feat()
        conf = _conf()
        scores = compute_multihorizon_scores(feat, conf, ev_score=0.01, pred_return=0.015)
        assert "scalp_score" in scores
        assert "continuation_score" in scores
        assert "runner_score" in scores
        assert "lane_selected" in scores

    def test_multihorizon_lane_selected_is_valid(self):
        feat = _feat()
        conf = _conf()
        scores = compute_multihorizon_scores(feat, conf, ev_score=0.01, pred_return=0.015)
        assert scores["lane_selected"] in ("scalp", "continuation", "runner")

    def test_multihorizon_scores_in_unit_range(self):
        feat = _feat()
        conf = _conf()
        scores = compute_multihorizon_scores(feat, conf, ev_score=0.01, pred_return=0.015)
        for k in ("scalp_score", "continuation_score", "runner_score"):
            assert 0.0 <= scores[k] <= 1.0, f"{k}={scores[k]} out of [0, 1]"

    def test_strong_momentum_raises_runner_score(self):
        feat_strong = _feat(ret_16=0.05, vol_r=4.0)
        feat_weak   = _feat(ret_16=0.001, vol_r=0.8)
        conf = _conf()
        s_strong = compute_multihorizon_scores(feat_strong, conf, 0.02, 0.03)
        s_weak   = compute_multihorizon_scores(feat_weak,   conf, 0.001, 0.001)
        assert s_strong["runner_score"] > s_weak["runner_score"]

    def test_relative_strength_normalizes_to_unit_range(self):
        feat_map = {
            "BTCUSD": _feat(ret_16=0.05),
            "ETHUSD": _feat(ret_16=0.02),
            "ADAUSD": _feat(ret_16=-0.01),
        }
        rs_scores, ranks = compute_relative_strength_scores(feat_map)
        for sym, score in rs_scores.items():
            assert 0.0 <= score <= 1.0
        # BTCUSD should have highest RS
        assert rs_scores["BTCUSD"] > rs_scores["ADAUSD"]
        assert ranks["BTCUSD"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pump exhaustion score increases after repeated same-symbol wins/entries
# ─────────────────────────────────────────────────────────────────────────────

class TestPumpScores:
    """Criterion 6: Pump exhaustion increases with repeated same-symbol activity."""

    def test_no_history_gives_low_exhaustion(self):
        scores = compute_pump_scores(
            "ADAUSD",
            same_symbol_entries_2h=0,
            same_symbol_trail_wins_2h=0,
        )
        assert scores["pump_exhaustion_score"] < 0.30

    def test_many_entries_increases_exhaustion(self):
        low  = compute_pump_scores("ADAUSD", same_symbol_entries_2h=1)
        high = compute_pump_scores("ADAUSD", same_symbol_entries_2h=4)
        assert high["pump_exhaustion_score"] > low["pump_exhaustion_score"]

    def test_many_trail_wins_increases_exhaustion(self):
        low  = compute_pump_scores("ADAUSD", same_symbol_trail_wins_2h=0)
        high = compute_pump_scores("ADAUSD", same_symbol_trail_wins_2h=4)
        assert high["pump_exhaustion_score"] > low["pump_exhaustion_score"]

    def test_fast_reentry_increases_exhaustion(self):
        slow  = compute_pump_scores("ADAUSD", minutes_since_last_exit=30.0)
        fast  = compute_pump_scores("ADAUSD", minutes_since_last_exit=1.0)
        assert fast["pump_exhaustion_score"] >= slow["pump_exhaustion_score"]

    def test_prior_sl_increases_exhaustion(self):
        no_sl = compute_pump_scores("ADAUSD", prior_exit_reason="EXIT_TRAIL")
        sl    = compute_pump_scores("ADAUSD", prior_exit_reason="EXIT_SL")
        assert sl["pump_exhaustion_score"] > no_sl["pump_exhaustion_score"]

    def test_exhaustion_scores_in_unit_range(self):
        scores = compute_pump_scores(
            "ADAUSD",
            same_symbol_entries_2h=5,
            same_symbol_trail_wins_2h=5,
            minutes_since_last_exit=0.5,
            prior_exit_reason="EXIT_SL",
        )
        assert 0.0 <= scores["pump_continuation_score"] <= 1.0
        assert 0.0 <= scores["pump_exhaustion_score"] <= 1.0

    def test_strong_momentum_raises_continuation(self):
        feat_strong = _feat(ret_4=0.03, vol_r=4.0)
        strong_conf = _conf(top3_mean=0.85)
        scores = compute_pump_scores(
            "ADAUSD",
            same_symbol_trail_wins_2h=2,
            feat=feat_strong,
            conf=strong_conf,
        )
        weak_scores = compute_pump_scores("ADAUSD", same_symbol_trail_wins_2h=0)
        assert scores["pump_continuation_score"] > weak_scores["pump_continuation_score"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pump continuation can override simple cooldown
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldownOverride:
    """Criterion 7: Strong continuation overrides reentry_cooldown block."""

    def test_strong_continuation_low_exhaustion_allows_override(self):
        assert exhaustion_override_allowed(
            pump_continuation_score=0.70, pump_exhaustion_score=0.30
        ) is True

    def test_weak_continuation_blocks_override(self):
        assert exhaustion_override_allowed(
            pump_continuation_score=0.30, pump_exhaustion_score=0.30
        ) is False

    def test_high_exhaustion_blocks_override(self):
        assert exhaustion_override_allowed(
            pump_continuation_score=0.70, pump_exhaustion_score=0.70
        ) is False

    def test_custom_thresholds_respected(self):
        # Tighter threshold: continuation must be >= 0.80
        assert exhaustion_override_allowed(
            pump_continuation_score=0.75, pump_exhaustion_score=0.30,
            continuation_threshold=0.80,
        ) is False
        assert exhaustion_override_allowed(
            pump_continuation_score=0.85, pump_exhaustion_score=0.30,
            continuation_threshold=0.80,
        ) is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. Trade vote audit record shape
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeVoteAudit:
    """Criterion 8: Trade vote audit records have required entry/exit/vote fields."""

    def _make_entry_snap(self, symbol="ADAUSD", order_id="ord_001"):
        conf = _conf()
        return build_entry_snapshot(
            symbol=symbol,
            entry_order_id=order_id,
            entry_time=_now(),
            entry_price=1.05,
            entry_qty=1000.0,
            allocation=0.20,
            risk_profile="ruthless",
            ruthless_v2_mode=True,
            conf=conf,
            ev_score=0.012,
            final_score=0.015,
            market_mode="risk_on_trend",
            confirm="strong_ml",
            entry_path="ml",
            multihorizon_scores={
                "scalp_score": 0.40, "continuation_score": 0.65, "runner_score": 0.55,
                "lane_selected": "continuation",
            },
            pump_scores={"pump_continuation_score": 0.50, "pump_exhaustion_score": 0.15},
            v2_opportunity_score=0.62,
            relative_strength_score=0.75,
            relative_strength_rank=2,
            meta_entry_score=0.58,
            effective_model_weights={"rf": 1.40, "et": 0.82},
            dynamic_vote_score=0.72,
        )

    def test_entry_record_has_all_required_fields(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        trade_id = audit.record_entry("ADAUSD", snap)
        records = audit.get_records("entry")
        assert len(records) == 1
        rec = records[0]
        for field in ENTRY_RECORD_REQUIRED_FIELDS:
            assert field in rec, f"Entry record missing required field: {field}"

    def test_entry_record_has_vote_dicts(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        audit.record_entry("ADAUSD", snap)
        rec = audit.get_records("entry")[0]
        assert isinstance(rec["active_votes"], dict)
        assert isinstance(rec["shadow_votes"], dict)
        assert isinstance(rec["diagnostic_votes"], dict)
        # active_votes should have the 6 V2 models
        assert len(rec["active_votes"]) == 6

    def test_exit_record_has_all_required_fields(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        trade_id = audit.record_entry("ADAUSD", snap)
        outcome = build_exit_outcome(
            trade_id=trade_id,
            symbol="ADAUSD",
            exit_order_id="exit_001",
            exit_time=_now() + datetime.timedelta(hours=2),
            exit_price=1.10,
            exit_reason="EXIT_TRAIL",
            entry_price=1.05,
            entry_qty=1000.0,
            hold_minutes=120.0,
        )
        audit.record_exit(trade_id, outcome)
        exits = audit.get_records("exit")
        assert len(exits) == 1
        rec = exits[0]
        for field in EXIT_RECORD_REQUIRED_FIELDS:
            assert field in rec, f"Exit record missing required field: {field}"

    def test_realized_return_computed_correctly(self):
        outcome = build_exit_outcome(
            trade_id="tid001",
            symbol="ADAUSD",
            exit_order_id="exit_001",
            exit_time=_now(),
            exit_price=1.10,
            exit_reason="EXIT_TRAIL",
            entry_price=1.00,
            entry_qty=100.0,
        )
        assert outcome["realized_return"] == pytest.approx(0.10, rel=1e-4)
        assert outcome["winner"] is True

    def test_loser_exit_winner_is_false(self):
        outcome = build_exit_outcome(
            trade_id="tid001",
            symbol="ADAUSD",
            exit_order_id="exit_002",
            exit_time=_now(),
            exit_price=0.97,
            exit_reason="EXIT_SL",
            entry_price=1.00,
            entry_qty=100.0,
        )
        assert outcome["winner"] is False

    def test_trade_id_generated_on_entry(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        trade_id = audit.record_entry("ADAUSD", snap)
        assert isinstance(trade_id, str)
        assert len(trade_id) == 16  # short UUID hex

    def test_record_count_tracks_correctly(self):
        audit = TradeVoteAudit()
        assert audit.record_count("entry") == 0
        snap = self._make_entry_snap()
        t = audit.record_entry("ADAUSD", snap)
        assert audit.record_count("entry") == 1
        outcome = build_exit_outcome(
            t, "ADAUSD", "ex", _now(), 1.10, "EXIT_TRAIL", 1.05, 100.0
        )
        audit.record_exit(t, outcome)
        assert audit.record_count("exit") == 1

    def test_model_attribution_uses_selected_trades_only(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        t = audit.record_entry("ADAUSD", snap)
        outcome = build_exit_outcome(
            t, "ADAUSD", "ex", _now(), 1.10, "EXIT_TRAIL", 1.05, 1000.0
        )
        audit.record_exit(t, outcome)
        attr = audit.compute_model_attribution()
        # rf, et, hgbc_l2, etc. should be in attribution
        assert "rf" in attr
        assert "et" in attr

    def test_to_jsonl_produces_one_line_per_record(self):
        audit = TradeVoteAudit()
        snap = self._make_entry_snap()
        t = audit.record_entry("ADAUSD", snap)
        outcome = build_exit_outcome(
            t, "ADAUSD", "ex", _now(), 1.10, "EXIT_TRAIL", 1.05, 1000.0
        )
        audit.record_exit(t, outcome)
        jsonl = audit.to_jsonl()
        lines = [l for l in jsonl.split("\n") if l.strip()]
        assert len(lines) == 2  # one entry, one exit

    def test_audit_store_key_format(self):
        assert AUDIT_STORE_KEY == "vox/trade_vote_audit.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Multi-position limit helper
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPositionManager:
    """Criterion 9: Multi-position manager enforces all limits."""

    def _make_mgr(self):
        return MultiPositionManager(
            max_concurrent=3,
            max_new_per_day=5,
            max_per_symbol_per_day=2,
            max_symbol_alloc=0.30,
            min_symbol_alloc=0.05,
            max_total_exposure=0.90,
            reentry_cooldown_min=10,
        )

    def test_empty_manager_allows_entry(self):
        mgr = self._make_mgr()
        ok, reason = mgr.can_enter("ADAUSD", 0.20, _now())
        assert ok, reason

    def test_max_concurrent_blocks_new_entries(self):
        mgr = self._make_mgr()
        t = _now()
        mgr.open_position("ADAUSD", 0.25, "t1", t)
        mgr.open_position("BTCUSD", 0.25, "t2", t)
        mgr.open_position("ETHUSD", 0.25, "t3", t)
        ok, reason = mgr.can_enter("SOLUSD", 0.20, t)
        assert not ok
        assert "max_concurrent" in reason

    def test_symbol_allocation_cap_blocks_oversized_entry(self):
        mgr = self._make_mgr()
        t = _now()
        mgr.open_position("ADAUSD", 0.30, "t1", t)
        # max_symbol_alloc=0.30 → already at cap
        ok, reason = mgr.can_enter("ADAUSD", 0.10, t)
        assert not ok
        assert "symbol_exposure" in reason

    def test_total_exposure_cap(self):
        mgr = MultiPositionManager(
            max_concurrent=5,       # high enough so concurrent doesn't block first
            max_new_per_day=10,
            max_symbol_alloc=0.35,
            min_symbol_alloc=0.05,
            max_total_exposure=0.90,
            reentry_cooldown_min=0,
        )
        t = _now()
        mgr.open_position("ADAUSD", 0.30, "t1", t)
        mgr.open_position("BTCUSD", 0.30, "t2", t)
        mgr.open_position("ETHUSD", 0.25, "t3", t)
        # total = 0.85; max = 0.90 → 0.10 more would put total at 0.95 > 0.90
        ok, reason = mgr.can_enter("SOLUSD", 0.10, t)
        assert not ok
        assert "total_exposure" in reason

    def test_daily_entry_cap(self):
        mgr = self._make_mgr()
        t = _now()
        for i in range(5):
            mgr.open_position(f"SYM{i}", 0.10, f"t{i}", t)
            mgr.close_position(f"t{i}", t)
            # re-enable by opening (simulate day tracking with open_position)
        # Reset open positions; daily count is what matters
        mgr2 = MultiPositionManager(max_concurrent=10, max_new_per_day=2,
                                    max_symbol_alloc=0.50, max_total_exposure=5.0)
        t2 = _now()
        mgr2.open_position("A", 0.10, "x1", t2)
        mgr2.open_position("B", 0.10, "x2", t2)
        ok, reason = mgr2.can_enter("C", 0.10, t2)
        assert not ok
        assert "max_new_per_day" in reason

    def test_reentry_cooldown_blocks_fast_reentry(self):
        mgr = self._make_mgr()
        t = _now()
        mgr.open_position("ADAUSD", 0.20, "t1", t)
        # Close immediately
        mgr.close_position("t1", t)
        # Try reentry 5 minutes later (cooldown=10)
        t2 = t + datetime.timedelta(minutes=5)
        ok, reason = mgr.can_enter("ADAUSD", 0.20, t2)
        assert not ok
        assert "reentry_cooldown" in reason

    def test_reentry_allowed_after_cooldown(self):
        mgr = self._make_mgr()
        t = _now()
        mgr.open_position("ADAUSD", 0.20, "t1", t)
        mgr.close_position("t1", t)
        t2 = t + datetime.timedelta(minutes=15)  # > cooldown=10
        ok, reason = mgr.can_enter("ADAUSD", 0.20, t2)
        assert ok, reason

    def test_open_position_count_tracks_correctly(self):
        mgr = self._make_mgr()
        t = _now()
        assert mgr.open_position_count() == 0
        mgr.open_position("ADAUSD", 0.20, "t1", t)
        assert mgr.open_position_count() == 1
        mgr.close_position("t1", t)
        assert mgr.open_position_count() == 0

    def test_total_exposure_tracks_correctly(self):
        mgr = self._make_mgr()
        t = _now()
        mgr.open_position("ADAUSD", 0.20, "t1", t)
        mgr.open_position("BTCUSD", 0.25, "t2", t)
        assert mgr.total_exposure() == pytest.approx(0.45)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Split-exit sizing helper
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitExitHelper:
    """Criterion 10: Split-exit sizing produces safe partial/runner quantities."""

    def _helper(self):
        return SplitExitHelper(partial_tp_fraction=0.50)

    def test_basic_split_50_50(self):
        h = self._helper()
        partial, runner = h.compute_split_quantities(1000.0)
        assert partial + runner == pytest.approx(1000.0)
        assert partial == pytest.approx(500.0)
        assert runner  == pytest.approx(500.0)

    def test_no_oversell(self):
        h = self._helper()
        for qty in [1.0, 100.0, 999.99, 0.01]:
            p, r = h.compute_split_quantities(qty)
            assert p + r <= qty + 1e-9, f"over-sold for qty={qty}: {p}+{r}={p+r}"

    def test_zero_qty_returns_zeros(self):
        h = self._helper()
        p, r = h.compute_split_quantities(0.0)
        assert p == 0.0
        assert r == 0.0

    def test_negative_qty_returns_zeros(self):
        h = self._helper()
        p, r = h.compute_split_quantities(-10.0)
        assert p == 0.0
        assert r == 0.0

    def test_min_runner_qty_respected(self):
        h = self._helper()
        # If runner would be < min_runner_qty, sell all in partial
        p, r = h.compute_split_quantities(0.001, min_runner_qty=0.01)
        assert r == 0.0
        assert p == pytest.approx(0.001)

    def test_custom_fraction(self):
        h = self._helper()
        p, r = h.compute_split_quantities(100.0, partial_fraction=0.40)
        assert p == pytest.approx(40.0)
        assert r == pytest.approx(60.0)

    def test_lane_tp_values_are_positive(self):
        h = self._helper()
        for lane in ("scalp", "continuation", "runner"):
            tp = h.get_lane_tp(lane)
            assert tp > 0.0

    def test_scalp_tp_less_than_continuation_tp(self):
        h = self._helper()
        assert h.get_lane_tp("scalp") < h.get_lane_tp("continuation")

    def test_continuation_tp_less_than_runner_tp(self):
        h = self._helper()
        assert h.get_lane_tp("continuation") < h.get_lane_tp("runner")

    def test_pump_mode_trail_wider_than_normal(self):
        h = self._helper()
        trail_normal = h.get_lane_trail_pct("runner", pump_mode=False)
        trail_pump   = h.get_lane_trail_pct("runner", pump_mode=True)
        assert trail_pump >= trail_normal


# ─────────────────────────────────────────────────────────────────────────────
# Meta-entry score
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaEntryScore:
    """Meta score is in [0, 1] and higher signals produce higher scores."""

    def test_score_in_unit_range(self):
        score = compute_meta_entry_score(
            vote_score=0.7, dynamic_vote_score=0.7, model_disagreement=0.1,
            regime_score=0.8, volume_expansion_score=0.7, relative_strength_score=0.8,
            breakout_score=0.6, exhaustion_score=0.1, recent_win_rate=0.65,
            pump_continuation_score=0.5,
        )
        assert 0.0 <= score <= 1.0

    def test_exhaustion_reduces_score(self):
        high_exh = compute_meta_entry_score(
            vote_score=0.7, dynamic_vote_score=0.7, exhaustion_score=0.9,
        )
        low_exh = compute_meta_entry_score(
            vote_score=0.7, dynamic_vote_score=0.7, exhaustion_score=0.0,
        )
        assert low_exh > high_exh

    def test_high_model_quality_raises_score(self):
        high = compute_meta_entry_score(vote_score=0.9, dynamic_vote_score=0.9)
        low  = compute_meta_entry_score(vote_score=0.2, dynamic_vote_score=0.2)
        assert high > low


# ─────────────────────────────────────────────────────────────────────────────
# Startup log helper
# ─────────────────────────────────────────────────────────────────────────────

class TestStartupLog:
    def test_startup_log_contains_v2_true(self):
        lines = format_v2_startup_log(
            risk_profile="ruthless",
            v2_mode=True,
            max_positions=4,
            active_models=RUTHLESS_V2_ACTIVE_MODELS,
        )
        combined = " ".join(lines)
        assert "v2=True" in combined
        assert "max_positions=4" in combined

    def test_startup_log_with_weights(self):
        lines = format_v2_startup_log(
            risk_profile="ruthless",
            v2_mode=True,
            max_positions=4,
            active_models=["rf", "et"],
            dynamic_weights={"rf": 1.40, "et": 0.75},
        )
        combined = " ".join(lines)
        assert "rf=1.400" in combined
        assert "et=0.750" in combined
