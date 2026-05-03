# ── Tests: Apex Predator helpers ──────────────────────────────────────────────
#
# Tests for the Apex Predator regime functions in ruthless_v2.py:
#   - compute_apex_score()     : weighted average with missing-column redistribution
#   - apex_entry_decision()    : four trigger paths (True / False coverage)
#   - compute_apex_size()      : Kelly-lite sizing clamps + gross exposure cap
#   - compute_apex_atr_stops() : ATR-based SL/TP clamps and trail params
#
# Tests match the style of test_ruthless_v2.py (pytest classes).
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy_ext import (
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
from core import (
    APEX_SCORE_ENTRY,
    APEX_SCORE_PYRAMID,
    APEX_BASE_ALLOC,
    APEX_MAX_GROSS,
    APEX_MAX_CONCURRENT,
    APEX_MAX_PER_SYMBOL,
    APEX_COOLDOWN_MIN,
    APEX_TIME_STOP_HRS,
    APEX_ATR_SL_MULT,
    APEX_ATR_TP_MULT,
    APEX_TRAIL_ARM_PCT,
    APEX_TRAIL_ATR_MULT,
    APEX_BREAKEVEN_MFE,
    APEX_ENTRY_PATH4_PROBA_MIN,
    APEX_ENTRY_PATH4_N_AGREE_MIN,
    APEX_ENTRY_LGBM_BAL_MIN,
    APEX_BREAKOUT_NBARS,
    APEX_BREAKOUT_VOL_MULT,
    APEX_PULLBACK_RSI_MAX,
    APEX_PULLBACK_TREND_BARS,
    APEX_MOMENTUM_CONT_BARS,
    APEX_MOMENTUM_CONT_VOL_MULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a full votes dict for testing
# ─────────────────────────────────────────────────────────────────────────────

def _full_votes(
    vote_lr_bal=0.60,
    vote_hgbc_l2=0.65,
    active_rf=0.65,
    active_hgbc_l2=0.60,
    active_lgbm_bal=0.65,
    vote_et=0.55,
):
    return {
        "vote_lr_bal":     vote_lr_bal,
        "vote_hgbc_l2":    vote_hgbc_l2,
        "active_rf":       active_rf,
        "active_hgbc_l2":  active_hgbc_l2,
        "active_lgbm_bal": active_lgbm_bal,
        "vote_et":         vote_et,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config constants sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestApexConfigConstants:
    """Verify APEX_* constants in config are present and in valid ranges."""

    def test_apex_score_entry_in_range(self):
        assert 0.0 < APEX_SCORE_ENTRY < 1.0

    def test_apex_score_pyramid_in_range(self):
        assert 0.0 < APEX_SCORE_PYRAMID < 1.0

    def test_apex_base_alloc_positive(self):
        assert 0.0 < APEX_BASE_ALLOC <= 1.0

    def test_apex_max_gross_above_one(self):
        assert APEX_MAX_GROSS >= 1.0

    def test_apex_max_concurrent_positive(self):
        assert APEX_MAX_CONCURRENT >= 1

    def test_apex_max_per_symbol_lte_concurrent(self):
        assert 1 <= APEX_MAX_PER_SYMBOL <= APEX_MAX_CONCURRENT

    def test_apex_cooldown_min_nonnegative(self):
        assert APEX_COOLDOWN_MIN >= 0

    def test_apex_time_stop_hrs_positive(self):
        assert APEX_TIME_STOP_HRS > 0

    def test_apex_atr_sl_mult_positive(self):
        assert APEX_ATR_SL_MULT > 0

    def test_apex_atr_tp_mult_positive(self):
        assert APEX_ATR_TP_MULT > 0

    def test_apex_trail_arm_pct_within_valid_range(self):
        assert 0.0 < APEX_TRAIL_ARM_PCT < 0.10

    def test_apex_trail_atr_mult_positive(self):
        assert APEX_TRAIL_ATR_MULT > 0

    def test_apex_breakeven_mfe_positive(self):
        assert APEX_BREAKEVEN_MFE > 0

    def test_apex_weights_sum_to_one(self):
        total = sum(_APEX_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"APEX weights sum to {total}, expected 1.0"


# ─────────────────────────────────────────────────────────────────────────────
# compute_apex_score: weighted average and missing-column redistribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeApexScore:
    """Tests for compute_apex_score() weighted-average logic."""

    def test_all_zeros_gives_zero(self):
        votes = _full_votes(
            vote_lr_bal=0.0, vote_hgbc_l2=0.0, active_rf=0.0,
            active_hgbc_l2=0.0, active_lgbm_bal=0.0, vote_et=0.0,
        )
        score, weight = compute_apex_score(votes)
        assert score == pytest.approx(0.0, abs=1e-6)
        assert weight == pytest.approx(1.0, abs=1e-6)

    def test_all_ones_gives_one(self):
        votes = _full_votes(
            vote_lr_bal=1.0, vote_hgbc_l2=1.0, active_rf=1.0,
            active_hgbc_l2=1.0, active_lgbm_bal=1.0, vote_et=1.0,
        )
        score, weight = compute_apex_score(votes)
        assert score == pytest.approx(1.0, abs=1e-6)
        assert weight == pytest.approx(1.0, abs=1e-6)

    def test_full_votes_result_in_unit_range(self):
        score, weight = compute_apex_score(_full_votes())
        assert 0.0 <= score <= 1.0
        assert weight == pytest.approx(1.0, abs=1e-6)

    def test_empty_votes_returns_zero(self):
        score, weight = compute_apex_score({})
        assert score == pytest.approx(0.0, abs=1e-6)
        assert weight == pytest.approx(0.0, abs=1e-6)

    def test_one_column_missing_redistributes_weight(self):
        """With vote_et missing its 0.05 weight should go to others."""
        full_votes = _full_votes(vote_et=0.0)  # present but zero
        partial_votes = {k: v for k, v in full_votes.items() if k != "vote_et"}

        score_partial, w_partial = compute_apex_score(partial_votes)
        score_full, w_full = compute_apex_score(full_votes)

        # partial weight < 1.0 since one column is absent
        assert w_partial == pytest.approx(0.95, abs=1e-6)
        # With all-non-zero votes, partial > full (weight redistributed away from 0)
        # Actually both should differ since the missing 0.05 is redistributed
        # Check: result is still in valid range
        assert 0.0 <= score_partial <= 1.0

    def test_two_columns_missing_redistributes_weight(self):
        """With two columns missing, weight_present should equal sum of remaining."""
        remaining_cols = {"vote_lr_bal": 0.70, "vote_hgbc_l2": 0.60, "active_rf": 0.65}
        expected_weight = _APEX_WEIGHTS["vote_lr_bal"] + _APEX_WEIGHTS["vote_hgbc_l2"] + _APEX_WEIGHTS["active_rf"]

        score, weight = compute_apex_score(remaining_cols)
        assert weight == pytest.approx(expected_weight, abs=1e-6)
        assert 0.0 <= score <= 1.0

    def test_missing_column_result_equals_manual_calculation(self):
        """Score with one missing column equals manual weighted mean of present cols."""
        votes = {"vote_lr_bal": 0.80, "active_rf": 0.60}
        w_lr  = _APEX_WEIGHTS["vote_lr_bal"]
        w_rf  = _APEX_WEIGHTS["active_rf"]
        total_w = w_lr + w_rf
        expected = (w_lr * 0.80 + w_rf * 0.60) / total_w

        score, _ = compute_apex_score(votes)
        assert score == pytest.approx(expected, abs=1e-6)

    def test_high_votes_score_above_entry_threshold(self):
        votes = _full_votes(
            vote_lr_bal=0.80, vote_hgbc_l2=0.75, active_rf=0.70,
            active_hgbc_l2=0.65, active_lgbm_bal=0.70, vote_et=0.60,
        )
        score, _ = compute_apex_score(votes)
        assert score >= APEX_SCORE_ENTRY

    def test_low_votes_score_below_entry_threshold(self):
        votes = _full_votes(
            vote_lr_bal=0.30, vote_hgbc_l2=0.35, active_rf=0.30,
            active_hgbc_l2=0.30, active_lgbm_bal=0.35, vote_et=0.30,
        )
        score, _ = compute_apex_score(votes)
        assert score < APEX_SCORE_ENTRY


# ─────────────────────────────────────────────────────────────────────────────
# apex_entry_decision: four trigger paths
# ─────────────────────────────────────────────────────────────────────────────

class TestApexEntryDecision:
    """Tests for apex_entry_decision() — all four trigger paths and False case."""

    # ── Path 1: apex_score >= APEX_SCORE_ENTRY ────────────────────────────────

    def test_path1_apex_score_triggers_entry(self):
        # Force apex_score well above threshold
        votes = _full_votes(
            vote_lr_bal=0.80, vote_hgbc_l2=0.75, active_rf=0.75,
            active_hgbc_l2=0.70, active_lgbm_bal=0.75, vote_et=0.70,
        )
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["triggered"] is True
        assert result["path"] == "apex_score"
        assert result["path_detail"]["apex_score_gate"] is True

    def test_path1_only_fires_when_score_high_enough(self):
        # All votes well below threshold → path1 should not fire
        votes = _full_votes(
            vote_lr_bal=0.30, vote_hgbc_l2=0.30, active_rf=0.30,
            active_hgbc_l2=0.30, active_lgbm_bal=0.30, vote_et=0.30,
        )
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["path_detail"]["apex_score_gate"] is False

    # ── Path 2: vote_lr_bal >= 0.50 ───────────────────────────────────────────

    def test_path2_vote_lr_bal_triggers_entry(self):
        # Low other votes (so apex_score < 0.55), but vote_lr_bal >= 0.50
        votes = {
            "vote_lr_bal":     0.55,   # path2 fires
            "vote_hgbc_l2":    0.30,
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.30,
            "vote_et":         0.30,
        }
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["triggered"] is True
        assert result["path_detail"]["vote_lr_bal_gate"] is True

    def test_path2_lr_bal_below_threshold_does_not_trigger(self):
        votes = {
            "vote_lr_bal":     0.45,   # below 0.50
            "vote_hgbc_l2":    0.30,
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.30,
            "vote_et":         0.30,
        }
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["path_detail"]["vote_lr_bal_gate"] is False

    # ── Path 3: vote_hgbc_l2 >= 0.55 AND active_lgbm_bal >= 0.55 ─────────────

    def test_path3_both_conditions_triggers_entry(self):
        votes = {
            "vote_lr_bal":     0.30,
            "vote_hgbc_l2":    0.60,   # >= 0.55
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.60,   # >= 0.55
            "vote_et":         0.30,
        }
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["triggered"] is True
        assert result["path_detail"]["hgbc_lgbm_gate"] is True

    def test_path3_only_one_condition_does_not_trigger(self):
        # hgbc_l2 high but lgbm_bal low
        votes = {
            "vote_lr_bal":     0.30,
            "vote_hgbc_l2":    0.60,   # >= 0.55 ✓
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.40,   # < 0.55 ✗
            "vote_et":         0.30,
        }
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["path_detail"]["hgbc_lgbm_gate"] is False

    # ── Path 4: mean_proba >= 0.60 AND n_agree >= 3 ───────────────────────────

    def test_path4_strong_ml_backstop_triggers_entry(self):
        votes = {
            "vote_lr_bal":     0.30,
            "vote_hgbc_l2":    0.30,
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.30,
            "vote_et":         0.30,
        }
        result = apex_entry_decision(
            votes, mean_proba=0.65, n_agree=3  # path4 fires
        )
        assert result["triggered"] is True
        assert result["path_detail"]["strong_ml_backstop"] is True

    def test_path4_proba_high_but_agree_zero_does_not_trigger(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.70, n_agree=0)
        assert result["path_detail"]["strong_ml_backstop"] is False

    def test_path4_agree_high_but_proba_low_does_not_trigger(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=4)
        assert result["path_detail"]["strong_ml_backstop"] is False

    # ── All paths fail → no trigger ───────────────────────────────────────────

    def test_all_paths_fail_returns_not_triggered(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=1)
        assert result["triggered"] is False
        assert result["path"] is None
        for v in result["path_detail"].values():
            assert v is False

    # ── Result dict shape ─────────────────────────────────────────────────────

    def test_result_has_required_keys(self):
        result = apex_entry_decision(_full_votes())
        for key in ("triggered", "apex_score", "weight_present", "path", "path_detail"):
            assert key in result

    def test_apex_score_in_result_in_unit_range(self):
        result = apex_entry_decision(_full_votes())
        assert 0.0 <= result["apex_score"] <= 1.0

    def test_missing_votes_handled_gracefully(self):
        """apex_entry_decision must not raise even with empty votes dict."""
        result = apex_entry_decision({}, mean_proba=0.0, n_agree=0)
        assert isinstance(result["triggered"], bool)

    def test_custom_apex_score_entry_threshold(self):
        """Passing a custom threshold overrides config value."""
        votes = _full_votes(
            vote_lr_bal=0.60, vote_hgbc_l2=0.60, active_rf=0.60,
            active_hgbc_l2=0.60, active_lgbm_bal=0.60, vote_et=0.60,
        )
        # apex_score will be ~0.60; with threshold=0.99 path1 should NOT fire
        result_high_thr = apex_entry_decision(votes, apex_score_entry=0.99)
        # path1 should fail; others may still fire
        assert result_high_thr["path_detail"]["apex_score_gate"] is False

        # With very low threshold path1 definitely fires
        result_low_thr = apex_entry_decision(votes, apex_score_entry=0.01)
        assert result_low_thr["path_detail"]["apex_score_gate"] is True


# ─────────────────────────────────────────────────────────────────────────────
# compute_apex_size: Kelly-lite clamps and gross exposure cap
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeApexSize:
    """Tests for compute_apex_size() sizing formula and clamping."""

    def test_result_in_valid_range(self):
        size = compute_apex_size(apex_score=0.60, n_agree=3)
        assert 0.05 <= size <= 0.45

    def test_floor_clamp_at_0_05(self):
        """Very low apex_score should not produce size below 0.05."""
        size = compute_apex_size(apex_score=0.10, n_agree=0)
        assert size >= 0.05

    def test_ceil_clamp_at_0_45(self):
        """Extremely high apex_score + n_agree should not exceed 0.45."""
        size = compute_apex_size(apex_score=1.0, n_agree=10)
        assert size <= 0.45

    def test_higher_apex_score_gives_larger_size(self):
        s_high = compute_apex_size(apex_score=0.90, n_agree=3)
        s_low  = compute_apex_size(apex_score=0.52, n_agree=3)
        assert s_high > s_low

    def test_high_n_agree_boosts_size(self):
        s_high_agree = compute_apex_size(apex_score=0.60, n_agree=4)
        s_low_agree  = compute_apex_size(apex_score=0.60, n_agree=2)
        assert s_high_agree > s_low_agree

    def test_gross_exposure_cap_limits_size(self):
        """If current exposure nearly at max_gross, size should be capped."""
        max_g = 2.0
        # Only 0.05 headroom remaining
        size = compute_apex_size(
            apex_score=0.90, n_agree=6,
            current_total_exposure=max_g - 0.05,
            max_gross=max_g,
        )
        assert size <= 0.05 + 1e-6

    def test_no_headroom_returns_zero(self):
        """At full capacity (exposure == max_gross), no new position allowed."""
        size = compute_apex_size(
            apex_score=0.90, n_agree=6,
            current_total_exposure=2.0,
            max_gross=2.0,
        )
        assert size == pytest.approx(0.0, abs=1e-6)

    def test_custom_base_alloc(self):
        """Custom base_alloc overrides config default."""
        s_custom = compute_apex_size(apex_score=0.50, n_agree=0, base_alloc=0.10)
        s_default = compute_apex_size(apex_score=0.50, n_agree=0)
        # With a smaller base, custom should be smaller (both at floor edge)
        assert s_custom <= s_default

    def test_obeys_apex_max_gross_constant(self):
        """Total after adding size must not exceed APEX_MAX_GROSS."""
        current = 1.90
        size = compute_apex_size(
            apex_score=0.80, n_agree=4,
            current_total_exposure=current,
        )
        assert current + size <= APEX_MAX_GROSS + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# compute_apex_atr_stops: SL / TP / trail parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeApexAtrStops:
    """Tests for compute_apex_atr_stops() ATR-based levels and clamps."""

    def _stops(self, entry=1.0, atr=0.01, **kwargs):
        return compute_apex_atr_stops(entry_price=entry, atr=atr, **kwargs)

    def test_returns_required_keys(self):
        stops = self._stops()
        for key in ("sl_price", "tp_price", "sl_pct", "tp_pct",
                    "trail_arm_pct", "trail_dist_pct",
                    "breakeven_mfe_pct", "time_stop_hrs"):
            assert key in stops, f"missing key: {key}"

    def test_sl_below_entry(self):
        stops = self._stops(entry=100.0, atr=1.5)
        assert stops["sl_price"] < 100.0

    def test_tp_above_entry(self):
        stops = self._stops(entry=100.0, atr=1.5)
        assert stops["tp_price"] > 100.0

    def test_sl_pct_floor(self):
        """Very tight ATR should floor sl_pct at 0.8%."""
        stops = self._stops(entry=1000.0, atr=0.001)
        assert stops["sl_pct"] >= 0.008

    def test_sl_pct_ceil(self):
        """Very wide ATR should ceil sl_pct at 4%."""
        stops = self._stops(entry=1.0, atr=10.0)
        assert stops["sl_pct"] <= 0.04

    def test_tp_pct_floor(self):
        """Very tight ATR should floor tp_pct at 2.5%."""
        stops = self._stops(entry=1000.0, atr=0.001)
        assert stops["tp_pct"] >= 0.025

    def test_tp_pct_ceil(self):
        """Very wide ATR should ceil tp_pct at 15%."""
        stops = self._stops(entry=1.0, atr=10.0)
        assert stops["tp_pct"] <= 0.15

    def test_trail_arm_equals_config(self):
        stops = self._stops()
        assert stops["trail_arm_pct"] == pytest.approx(APEX_TRAIL_ARM_PCT)

    def test_trail_dist_minimum_0_6_pct(self):
        """Trail distance floor is 0.6%."""
        stops = self._stops(entry=10000.0, atr=0.001)
        assert stops["trail_dist_pct"] >= 0.006

    def test_breakeven_mfe_equals_config(self):
        stops = self._stops()
        assert stops["breakeven_mfe_pct"] == pytest.approx(APEX_BREAKEVEN_MFE)

    def test_time_stop_hrs_equals_config(self):
        stops = self._stops()
        assert stops["time_stop_hrs"] == APEX_TIME_STOP_HRS

    def test_zero_entry_price_does_not_raise(self):
        stops = compute_apex_atr_stops(entry_price=0.0, atr=0.01)
        assert "sl_price" in stops

    def test_normal_atr_gives_wider_stops_than_floor(self):
        """With normal ATR the stops should be wider than the floor values."""
        stops = self._stops(entry=1.0, atr=0.02)  # ATR = 2% of entry
        # sl = 1.25 * 0.02 / 1.0 = 2.5% > floor 0.8%
        assert stops["sl_pct"] > 0.008
        # tp = 4.0 * 0.02 / 1.0 = 8% > floor 2.5%
        assert stops["tp_pct"] > 0.025

    def test_custom_multipliers_respected(self):
        stops_tight = self._stops(entry=1.0, atr=0.02, atr_sl_mult=0.5, atr_tp_mult=2.0)
        stops_wide  = self._stops(entry=1.0, atr=0.02, atr_sl_mult=2.0, atr_tp_mult=6.0)
        # Wider multipliers should produce larger sl/tp (within clamp bounds)
        assert stops_wide["sl_pct"] >= stops_tight["sl_pct"]
        assert stops_wide["tp_pct"] >= stops_tight["tp_pct"]


# ─────────────────────────────────────────────────────────────────────────────
# Config constants: new v2 aggressive-gate constants
# ─────────────────────────────────────────────────────────────────────────────

class TestApexV2AggressiveConfigConstants:
    """Verify newly-added v2 aggressive-gate constants exist and are in valid ranges."""

    def test_apex_score_entry_lowered_to_050(self):
        assert APEX_SCORE_ENTRY == pytest.approx(0.50, abs=1e-6)

    def test_path4_proba_min_in_range(self):
        assert 0.0 < APEX_ENTRY_PATH4_PROBA_MIN <= 1.0

    def test_path4_n_agree_min_at_least_one(self):
        assert APEX_ENTRY_PATH4_N_AGREE_MIN >= 1

    def test_lgbm_bal_min_in_range(self):
        assert 0.0 < APEX_ENTRY_LGBM_BAL_MIN <= 1.0

    def test_breakout_nbars_positive(self):
        assert APEX_BREAKOUT_NBARS >= 1

    def test_breakout_vol_mult_above_one(self):
        assert APEX_BREAKOUT_VOL_MULT >= 1.0

    def test_pullback_rsi_max_in_valid_range(self):
        assert 0 < APEX_PULLBACK_RSI_MAX < 50

    def test_pullback_trend_bars_positive(self):
        assert APEX_PULLBACK_TREND_BARS >= 1

    def test_momentum_cont_bars_positive(self):
        assert APEX_MOMENTUM_CONT_BARS >= 1

    def test_momentum_cont_vol_mult_above_one(self):
        assert APEX_MOMENTUM_CONT_VOL_MULT >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# apex_entry_decision: path-5 (vote_lgbm_bal direct gate) and updated path-4
# ─────────────────────────────────────────────────────────────────────────────

class TestApexEntryDecisionV2Paths:
    """Tests for the new path-5 and the relaxed path-4 in apex_entry_decision()."""

    # ── Path 5: active_lgbm_bal >= APEX_ENTRY_LGBM_BAL_MIN ───────────────────

    def test_path5_lgbm_bal_triggers_entry(self):
        # All votes low except active_lgbm_bal — path5 should fire
        votes = {
            "vote_lr_bal":     0.30,
            "vote_hgbc_l2":    0.30,
            "active_rf":       0.30,
            "active_hgbc_l2":  0.30,
            "active_lgbm_bal": 0.55,   # >= 0.50 → path5
            "vote_et":         0.30,
        }
        result = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        assert result["triggered"] is True
        assert result["path_detail"]["lgbm_bal_gate"] is True
        assert result["path"] == "lgbm_bal_direct"

    def test_path5_lgbm_bal_at_threshold_triggers(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        votes["active_lgbm_bal"] = APEX_ENTRY_LGBM_BAL_MIN
        result = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        assert result["path_detail"]["lgbm_bal_gate"] is True

    def test_path5_lgbm_bal_below_threshold_does_not_trigger(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        votes["active_lgbm_bal"] = APEX_ENTRY_LGBM_BAL_MIN - 0.01
        result = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        assert result["path_detail"]["lgbm_bal_gate"] is False

    def test_path5_lgbm_bal_present_in_path_detail(self):
        result = apex_entry_decision(_full_votes(), mean_proba=0.40, n_agree=0)
        assert "lgbm_bal_gate" in result["path_detail"]

    # ── Path 4 relaxed thresholds (0.50 / n_agree=1) ─────────────────────────

    def test_path4_fires_at_relaxed_proba_05_agree_1(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.50, n_agree=1)
        assert result["path_detail"]["strong_ml_backstop"] is True

    def test_path4_fires_with_proba_above_05_agree_1(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.65, n_agree=1)
        assert result["path_detail"]["strong_ml_backstop"] is True

    def test_path4_does_not_fire_proba_below_05(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.40, n_agree=3)
        assert result["path_detail"]["strong_ml_backstop"] is False

    def test_path4_does_not_fire_agree_zero(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.70, n_agree=0)
        assert result["path_detail"]["strong_ml_backstop"] is False

    # ── reject_reason populated when not triggered ────────────────────────────

    def test_reject_reason_is_none_when_triggered(self):
        votes = _full_votes(
            vote_lr_bal=0.80, vote_hgbc_l2=0.75, active_rf=0.70,
            active_hgbc_l2=0.65, active_lgbm_bal=0.70, vote_et=0.60,
        )
        result = apex_entry_decision(votes, mean_proba=0.70, n_agree=3)
        assert result["triggered"] is True
        assert result["reject_reason"] is None

    def test_reject_reason_is_string_when_not_triggered(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        assert result["triggered"] is False
        assert isinstance(result["reject_reason"], str)
        assert len(result["reject_reason"]) > 0

    def test_reject_reason_mentions_apex_score(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        reason = result["reject_reason"]
        # All five gate labels must appear in the rejection reason
        for gate in ("apex_score", "vote_lr_bal", "hgbc_l2", "lgbm_bal", "mean_proba"):
            assert gate in reason, f"reject_reason missing gate '{gate}': {reason}"

    def test_result_has_reject_reason_key(self):
        result = apex_entry_decision(_full_votes())
        assert "reject_reason" in result


# ─────────────────────────────────────────────────────────────────────────────
# apex_breakout_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestApexBreakoutSignal:
    """Tests for apex_breakout_signal() — price × N-bar high + volume spike."""

    def _make_data(self, n=25, base_close=100.0, base_vol=1000.0):
        closes  = [base_close + i * 0.05 for i in range(n)]  # gently rising
        volumes = [base_vol] * n
        return closes, volumes

    def test_clear_breakout_returns_true(self):
        closes, volumes = self._make_data(n=25, base_close=100.0)
        # Force current bar well above all prior highs and double the volume
        closes[-1]  = max(closes[:-1]) + 1.0
        volumes[-1] = 2.0 * max(volumes[:-1])
        assert apex_breakout_signal(closes, volumes) is True

    def test_no_price_breakout_returns_false(self):
        closes, volumes = self._make_data(n=25)
        closes[-1]  = min(closes[:-1]) - 0.5   # current bar below all prior
        volumes[-1] = 2.0 * volumes[0]
        assert apex_breakout_signal(closes, volumes) is False

    def test_price_breakout_but_vol_too_low_returns_false(self):
        closes, volumes = self._make_data(n=25)
        closes[-1]  = max(closes[:-1]) + 1.0   # above prior high
        volumes[-1] = 0.5 * volumes[0]          # volume well below average
        assert apex_breakout_signal(closes, volumes) is False

    def test_too_few_bars_returns_false(self):
        closes  = [100.0, 101.0]   # only 2 bars, need n_bars+1
        volumes = [1000.0, 2000.0]
        assert apex_breakout_signal(closes, volumes, n_bars=20) is False

    def test_custom_n_bars_and_vol_mult(self):
        n = 10
        closes  = [100.0 + i * 0.1 for i in range(n + 1)]
        volumes = [1000.0] * (n + 1)
        closes[-1]  = max(closes[:-1]) + 2.0
        volumes[-1] = 3.0 * 1000.0   # 3× average with vol_mult=2.0 → fire
        result = apex_breakout_signal(closes, volumes, n_bars=n, vol_mult=2.0)
        assert result is True

    def test_equal_to_rolling_high_not_a_breakout(self):
        closes  = [100.0] * 22
        volumes = [1000.0] * 22
        closes[-1] = 100.0   # equal, not strictly greater
        volumes[-1] = 2000.0
        assert apex_breakout_signal(closes, volumes) is False

    def test_empty_sequences_returns_false(self):
        assert apex_breakout_signal([], []) is False


# ─────────────────────────────────────────────────────────────────────────────
# apex_pullback_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestApexPullbackSignal:
    """Tests for apex_pullback_signal() — oversold RSI in uptrend."""

    def _make_closes(self, n=15, slope=0.5):
        return [100.0 + i * slope for i in range(n)]

    def test_oversold_in_uptrend_returns_true(self):
        closes = self._make_closes(n=15, slope=0.5)   # clearly rising
        assert apex_pullback_signal(closes, rsi=25) is True

    def test_rsi_above_threshold_returns_false(self):
        closes = self._make_closes(n=15, slope=0.5)
        assert apex_pullback_signal(closes, rsi=50) is False

    def test_oversold_but_downtrend_returns_false(self):
        closes = self._make_closes(n=15, slope=-0.5)   # falling
        assert apex_pullback_signal(closes, rsi=25) is False

    def test_too_few_bars_returns_false(self):
        closes = [100.0, 101.0]
        assert apex_pullback_signal(closes, rsi=20, n_bars=10) is False

    def test_rsi_exactly_at_max_triggers(self):
        closes = self._make_closes(n=15, slope=1.0)
        assert apex_pullback_signal(closes, rsi=APEX_PULLBACK_RSI_MAX) is True

    def test_rsi_one_above_max_does_not_trigger(self):
        closes = self._make_closes(n=15, slope=1.0)
        assert apex_pullback_signal(closes, rsi=APEX_PULLBACK_RSI_MAX + 1) is False

    def test_custom_parameters_respected(self):
        closes = self._make_closes(n=20, slope=0.5)
        # With custom rsi_max=40, RSI=38 should fire
        assert apex_pullback_signal(closes, rsi=38, rsi_max=40) is True

    def test_empty_closes_returns_false(self):
        assert apex_pullback_signal([], rsi=20) is False


# ─────────────────────────────────────────────────────────────────────────────
# apex_momentum_continuation_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestApexMomentumContinuationSignal:
    """Tests for apex_momentum_continuation_signal() — N consecutive higher closes + vol."""

    def _make_cont_data(self, n=5, vol=1000.0, vol_spike=2.0):
        # n+1 strictly rising closes; spike on last bar
        closes  = [100.0 + i * 0.5 for i in range(n + 1)]
        volumes = [vol] * (n + 1)
        volumes[-1] = vol * vol_spike
        return closes, volumes

    def test_clear_continuation_returns_true(self):
        closes, volumes = self._make_cont_data(n=4, vol_spike=2.0)
        assert apex_momentum_continuation_signal(closes, volumes) is True

    def test_one_lower_close_breaks_streak_returns_false(self):
        closes, volumes = self._make_cont_data(n=4, vol_spike=2.0)
        closes[-2] = closes[-3] - 0.1   # break the streak
        assert apex_momentum_continuation_signal(closes, volumes) is False

    def test_consecutive_higher_closes_but_vol_low_returns_false(self):
        closes, volumes = self._make_cont_data(n=4, vol_spike=0.5)
        # volume[-1] = 0.5 × avg, below 1.5 threshold
        assert apex_momentum_continuation_signal(closes, volumes) is False

    def test_too_few_bars_returns_false(self):
        assert apex_momentum_continuation_signal([100.0, 101.0], [1000.0, 2000.0], n_bars=5) is False

    def test_custom_n_bars_and_vol_mult(self):
        n = 2
        closes  = [100.0, 101.0, 102.0]
        volumes = [1000.0, 1000.0, 2500.0]   # 2.5× avg → fires at vol_mult=2.0
        assert apex_momentum_continuation_signal(closes, volumes, n_bars=n, vol_mult=2.0) is True

    def test_flat_closes_do_not_trigger(self):
        closes  = [100.0] * 5
        volumes = [1000.0] * 4 + [3000.0]
        assert apex_momentum_continuation_signal(closes, volumes) is False

    def test_empty_sequences_returns_false(self):
        assert apex_momentum_continuation_signal([], []) is False


# ─────────────────────────────────────────────────────────────────────────────
# apex_rejected_entry_log
# ─────────────────────────────────────────────────────────────────────────────

class TestApexRejectedEntryLog:
    """Tests for apex_rejected_entry_log() diagnostic helper."""

    def _make_rejected(self):
        votes    = {k: 0.30 for k in _APEX_WEIGHTS}
        decision = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        return apex_rejected_entry_log(votes, 0.30, 0, decision)

    def _make_triggered(self):
        votes    = _full_votes()
        decision = apex_entry_decision(votes, mean_proba=0.70, n_agree=3)
        return apex_rejected_entry_log(votes, 0.70, 3, decision)

    def test_returns_dict(self):
        assert isinstance(self._make_rejected(), dict)

    def test_required_keys_present(self):
        log = self._make_rejected()
        for key in ("triggered", "reject_reason", "path_detail",
                    "apex_score", "mean_proba", "n_agree", "votes_snapshot"):
            assert key in log, f"missing key: {key}"

    def test_triggered_false_for_rejected(self):
        assert self._make_rejected()["triggered"] is False

    def test_reject_reason_is_string_for_rejected(self):
        log = self._make_rejected()
        assert isinstance(log["reject_reason"], str)
        assert len(log["reject_reason"]) > 0

    def test_triggered_true_reject_reason_none(self):
        log = self._make_triggered()
        assert log["triggered"] is True
        assert log["reject_reason"] is None

    def test_votes_snapshot_is_copy(self):
        votes    = {k: 0.30 for k in _APEX_WEIGHTS}
        decision = apex_entry_decision(votes, mean_proba=0.30, n_agree=0)
        log      = apex_rejected_entry_log(votes, 0.30, 0, decision)
        # Mutate original; snapshot must not change
        votes["vote_lr_bal"] = 0.99
        assert log["votes_snapshot"]["vote_lr_bal"] == pytest.approx(0.30, abs=1e-6)

    def test_mean_proba_and_n_agree_stored(self):
        votes    = {k: 0.30 for k in _APEX_WEIGHTS}
        decision = apex_entry_decision(votes, mean_proba=0.45, n_agree=2)
        log      = apex_rejected_entry_log(votes, 0.45, 2, decision)
        assert log["mean_proba"]  == pytest.approx(0.45, abs=1e-6)
        assert log["n_agree"]     == 2

    def test_apex_score_in_range(self):
        log = self._make_rejected()
        assert 0.0 <= log["apex_score"] <= 1.0

