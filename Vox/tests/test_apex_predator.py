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

from ruthless_v2 import (
    _APEX_WEIGHTS,
    compute_apex_score,
    apex_entry_decision,
    compute_apex_size,
    compute_apex_atr_stops,
)
from config import (
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

    def test_apex_trail_arm_pct_small(self):
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

    def test_path4_proba_high_but_agree_low_does_not_trigger(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.70, n_agree=2)
        assert result["path_detail"]["strong_ml_backstop"] is False

    def test_path4_agree_high_but_proba_low_does_not_trigger(self):
        votes = {k: 0.30 for k in _APEX_WEIGHTS}
        result = apex_entry_decision(votes, mean_proba=0.55, n_agree=4)
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
