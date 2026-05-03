# ── Tests: apex_voting weighted vote aggregator ───────────────────────────────
#
# Tests for Vox/apex_voting.py:
#   - compute_weighted_yes_fraction(): correct weighting, zero-weight models,
#     missing models, edge cases
#   - apex_voting_decision(): all three trigger paths, rejection logging
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy import (  # type: ignore  # noqa: E402
    compute_weighted_yes_fraction,
    apex_voting_decision,
    APEX_WEIGHTED_VOTE_WEIGHTS,
    APEX_WEIGHTED_YES_THRESHOLD,
    APEX_COMBO_HGBC_MIN,
    APEX_COMBO_LGBM_MIN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _votes_all_yes(proba=0.80):
    """Return a vote dict where every key in APEX_WEIGHTED_VOTE_WEIGHTS is YES."""
    return {k: proba for k in APEX_WEIGHTED_VOTE_WEIGHTS}


def _votes_all_no(proba=0.10):
    """Return a vote dict where every key is below threshold."""
    return {k: proba for k in APEX_WEIGHTED_VOTE_WEIGHTS}


# ─────────────────────────────────────────────────────────────────────────────
# compute_weighted_yes_fraction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeWeightedYesFraction:

    def test_empty_votes_returns_zero(self):
        result = compute_weighted_yes_fraction({})
        assert result["weighted_yes_fraction"] == 0.0
        assert result["total_weight"] == 0.0

    def test_all_yes_returns_one(self):
        result = compute_weighted_yes_fraction(_votes_all_yes(0.80))
        assert result["weighted_yes_fraction"] == pytest.approx(1.0)

    def test_all_no_returns_zero(self):
        result = compute_weighted_yes_fraction(_votes_all_no(0.10))
        assert result["weighted_yes_fraction"] == pytest.approx(0.0)

    def test_only_high_weight_yes_exceeds_threshold(self):
        # lgbm_bal (w=2.5) votes YES; everything else NO.
        votes = _votes_all_no(0.10)
        votes["lgbm_bal"] = 0.80
        result = compute_weighted_yes_fraction(votes)
        # lgbm_bal contributes 2.5 out of total non-zero weights
        total_w = sum(w for w in APEX_WEIGHTED_VOTE_WEIGHTS.values() if w > 0)
        expected = 2.5 / total_w
        assert result["weighted_yes_fraction"] == pytest.approx(expected, abs=1e-5)

    def test_zero_weight_models_excluded_from_numerator(self):
        # gnb and lr have weight 0 → voting YES should not change fraction
        votes = {"gnb": 1.0, "lr": 1.0}
        result = compute_weighted_yes_fraction(votes)
        assert result["weighted_yes_fraction"] == pytest.approx(0.0)
        assert "gnb" in result["zero_weight_models"]
        assert "lr" in result["zero_weight_models"]

    def test_unknown_model_defaults_to_weight_one(self):
        votes = {"unknown_model_xyz": 0.90}
        result = compute_weighted_yes_fraction(votes)
        # weight 1.0, voted YES → fraction = 1.0
        assert result["weighted_yes_fraction"] == pytest.approx(1.0)
        assert "unknown_model_xyz" in result["yes_models"]

    def test_yes_models_list_correct(self):
        votes = {"lgbm_bal": 0.75, "hgbc_l2": 0.30, "rf": 0.80}
        result = compute_weighted_yes_fraction(votes, vote_threshold=0.50)
        assert "lgbm_bal" in result["yes_models"]
        assert "rf" in result["yes_models"]
        assert "hgbc_l2" in result["no_models"]

    def test_no_models_list_correct(self):
        votes = {"lgbm_bal": 0.20, "hgbc_l2": 0.10}
        result = compute_weighted_yes_fraction(votes)
        assert "lgbm_bal" in result["no_models"]
        assert "hgbc_l2" in result["no_models"]
        assert result["yes_models"] == []

    def test_custom_vote_threshold(self):
        votes = {"lgbm_bal": 0.60, "hgbc_l2": 0.75}
        result_50 = compute_weighted_yes_fraction(votes, vote_threshold=0.50)
        result_70 = compute_weighted_yes_fraction(votes, vote_threshold=0.70)
        assert result_50["weighted_yes_fraction"] > result_70["weighted_yes_fraction"]

    def test_custom_weight_map(self):
        custom_weights = {"modelA": 1.0, "modelB": 1.0}
        votes = {"modelA": 0.80, "modelB": 0.20}
        result = compute_weighted_yes_fraction(votes, weights=custom_weights)
        # modelA YES (w=1), modelB NO (w=1) → fraction = 0.5
        assert result["weighted_yes_fraction"] == pytest.approx(0.5)

    def test_high_weight_lr_bal_alone_may_trigger(self):
        # lr_bal has w=1.5, above threshold
        votes = {"lr_bal": 0.80}
        result = compute_weighted_yes_fraction(votes)
        # Only lr_bal present with w=1.5 and votes YES → fraction = 1.0
        assert result["weighted_yes_fraction"] == pytest.approx(1.0)

    def test_fraction_between_zero_and_one(self):
        votes = _votes_all_yes(0.80)
        votes["hgbc_l2"] = 0.10   # flip one to NO
        result = compute_weighted_yes_fraction(votes)
        assert 0.0 < result["weighted_yes_fraction"] < 1.0

    def test_yes_plus_no_weights_sum_to_total(self):
        votes = {"lgbm_bal": 0.80, "hgbc_l2": 0.30, "rf": 0.70}
        result = compute_weighted_yes_fraction(votes)
        yes_w   = result["yes_weight"]
        total_w = result["total_weight"]
        # yes_weight + no_weight = total_weight
        no_w = sum(
            APEX_WEIGHTED_VOTE_WEIGHTS.get(m, 1.0)
            for m in result["no_models"]
        )
        assert yes_w + no_w == pytest.approx(total_w, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# apex_voting_decision tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApexVotingDecision:

    def test_triggers_on_high_weighted_fraction(self):
        result = apex_voting_decision(_votes_all_yes(0.80))
        assert result["triggered"] is True
        assert result["trigger_path"] == "weighted_yes_fraction"

    def test_no_trigger_when_all_no(self):
        result = apex_voting_decision(_votes_all_no(0.10), momentum_override=False)
        assert result["triggered"] is False
        assert result["trigger_path"] is None

    def test_momentum_override_always_triggers(self):
        result = apex_voting_decision(_votes_all_no(0.10), momentum_override=True)
        assert result["triggered"] is True
        assert result["trigger_path"] == "momentum_override"

    def test_combo_path_triggers_when_both_above_floor(self):
        votes = {"hgbc_l2": APEX_COMBO_HGBC_MIN, "lgbm_bal": APEX_COMBO_LGBM_MIN}
        result = apex_voting_decision(votes, momentum_override=False)
        assert result["triggered"] is True
        assert result["combo_fired"] is True

    def test_combo_path_does_not_trigger_if_one_below_floor(self):
        votes = {"hgbc_l2": APEX_COMBO_HGBC_MIN, "lgbm_bal": 0.10}
        result = apex_voting_decision(votes, momentum_override=False)
        assert result["combo_fired"] is False

    def test_reject_reason_present_when_not_triggered(self):
        result = apex_voting_decision(_votes_all_no(0.10))
        assert result["triggered"] is False
        assert result["reject_reason"] is not None
        assert "weighted_yes_fraction" in result["reject_reason"]

    def test_no_reject_reason_when_triggered(self):
        result = apex_voting_decision(_votes_all_yes(0.80))
        assert result["triggered"] is True
        assert result["reject_reason"] is None

    def test_weighted_yes_fraction_reported(self):
        result = apex_voting_decision(_votes_all_yes(0.80))
        assert 0.0 <= result["weighted_yes_fraction"] <= 1.0

    def test_yes_threshold_default_is_config_value(self):
        result = apex_voting_decision(_votes_all_no(0.10))
        assert result["yes_threshold"] == APEX_WEIGHTED_YES_THRESHOLD

    def test_custom_yes_threshold(self):
        # With threshold=0.99, even all-YES with low weights may not trigger
        # unless weighted fraction reaches 0.99
        result_high = apex_voting_decision(_votes_all_yes(0.80), yes_threshold=0.99)
        result_low  = apex_voting_decision(_votes_all_yes(0.80), yes_threshold=0.01)
        # low threshold should always trigger (all YES)
        assert result_low["triggered"] is True
        # high threshold may or may not trigger depending on weights
        # (we just check it's a bool)
        assert isinstance(result_high["triggered"], bool)

    def test_gnb_does_not_influence_decision(self):
        # gnb is weight=0; even if it votes YES it should not change the result
        # compared to same votes without gnb
        votes_base = {"lgbm_bal": 0.20}
        votes_with_gnb = {"lgbm_bal": 0.20, "gnb": 1.0}
        r1 = apex_voting_decision(votes_base, momentum_override=False)
        r2 = apex_voting_decision(votes_with_gnb, momentum_override=False)
        assert r1["triggered"] == r2["triggered"]
        assert r1["weighted_yes_fraction"] == pytest.approx(r2["weighted_yes_fraction"])

    def test_lr_dead_voter_does_not_influence_decision(self):
        votes_base = {"lgbm_bal": 0.20}
        votes_with_lr = {"lgbm_bal": 0.20, "lr": 1.0}
        r1 = apex_voting_decision(votes_base)
        r2 = apex_voting_decision(votes_with_lr)
        assert r1["triggered"] == r2["triggered"]

    def test_zero_weight_models_listed(self):
        votes = {"gnb": 1.0, "lr": 1.0, "lgbm_bal": 0.80}
        result = apex_voting_decision(votes)
        assert "gnb" in result["zero_weight_models"]
        assert "lr" in result["zero_weight_models"]

    def test_yes_and_no_models_reported(self):
        votes = {"lgbm_bal": 0.80, "hgbc_l2": 0.20}
        result = apex_voting_decision(votes)
        assert "lgbm_bal" in result["yes_models"]
        assert "hgbc_l2" in result["no_models"]

    def test_momentum_override_field_reflected(self):
        r_true  = apex_voting_decision({}, momentum_override=True)
        r_false = apex_voting_decision({}, momentum_override=False)
        assert r_true["momentum_override"] is True
        assert r_false["momentum_override"] is False

    def test_weighted_yes_fraction_at_exactly_threshold_triggers(self):
        # Construct votes that put weighted_yes_fraction exactly at threshold
        threshold = APEX_WEIGHTED_YES_THRESHOLD
        # Use custom weights to get exact control
        custom_w = {"m1": 1.0, "m2": 1.0}
        # YES: m1 (w=1), NO: m2 (w=1) → fraction = 0.5
        votes = {"m1": 0.80, "m2": 0.10}
        result = apex_voting_decision(
            votes,
            yes_threshold=0.50,
            weights=custom_w,
            momentum_override=False,
        )
        # weighted_yes_fraction = 0.50 >= 0.50 → triggered
        assert result["triggered"] is True

    def test_just_below_threshold_does_not_trigger_without_override(self):
        # weighted_yes_fraction = 0.44 < 0.45 threshold, no override, no combo
        custom_w = {"m1": 44.0, "m2": 56.0}
        votes = {"m1": 0.80, "m2": 0.10}
        result = apex_voting_decision(
            votes,
            yes_threshold=0.45,
            weights=custom_w,
            momentum_override=False,
            combo_hgbc_min=0.99,
            combo_lgbm_min=0.99,
        )
        frac = result["weighted_yes_fraction"]
        # frac = 44/100 = 0.44 → not triggered
        assert frac == pytest.approx(0.44, abs=1e-5)
        assert result["triggered"] is False
