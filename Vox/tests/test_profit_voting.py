# ── Tests: Profit-voting, shadow lab, candidate journal, profile audit ────────
#
# These tests cover the new functionality added in the profit-voting + shadow
# model expansion PR.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import numpy as np
import pytest

# Add Vox to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_registry import (
    compute_vote_score,
    compute_active_stats,
    ROLE_ACTIVE,
    ROLE_SHADOW,
    ROLE_DIAGNOSTIC,
)
from profit_voting import (
    check_profit_voting_gate,
    compute_vote_score as pv_compute_vote_score,
    format_profit_vote_log,
    apply_ruthless_active_promotion,
    make_pv_counters,
    increment_pv_counter,
    format_pv_reject_log,
    format_pv_summary_log,
    DEFAULT_VOTE_THRESHOLD,
    DEFAULT_VOTE_YES_FRACTION_MIN,
    DEFAULT_TOP3_MEAN_MIN,
    DEFAULT_CHOP_VOTE_YES_FRAC_MIN,
    DEFAULT_CHOP_TOP3_MEAN_MIN,
)
from candidate_journal import (
    CandidateJournal,
    build_candidate_records,
    build_rejected_candidate_records,
    CANDIDATE_JOURNAL_TOP_N,
)
from shadow_lab import (
    extend_shadow_estimators,
    MarkovRegimeDiagnostic,
    KMeansRegimeDiagnostic,
    IsoForestRiskDiagnostic,
    ROLE_SHADOW as SL_ROLE_SHADOW,
    ROLE_DIAGNOSTIC as SL_ROLE_DIAGNOSTIC,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_vote_score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeVoteScore:
    """Tests for model_registry.compute_vote_score."""

    def test_empty_votes_returns_zeros(self):
        result = compute_vote_score({})
        assert result["active_model_count"] == 0
        assert result["vote_yes_fraction"] == 0.0
        assert result["top3_mean"] == 0.0
        assert result["vote_score"] == 0.0

    def test_all_high_probabilities(self):
        votes = {"rf": 0.8, "et": 0.75, "hgbc": 0.85}
        result = compute_vote_score(votes, vote_thr=0.55)
        assert result["active_model_count"] == 3
        assert result["vote_yes_fraction"] == pytest.approx(1.0)
        assert result["top3_mean"] > 0.75
        assert result["vote_score"] > 0.7

    def test_all_low_probabilities(self):
        votes = {"rf": 0.3, "et": 0.25, "hgbc": 0.35}
        result = compute_vote_score(votes, vote_thr=0.55)
        assert result["vote_yes_fraction"] == 0.0
        assert result["vote_score"] < 0.5

    def test_mixed_probabilities(self):
        votes = {"rf": 0.8, "et": 0.4, "hgbc": 0.7, "lgbm": 0.6}
        result = compute_vote_score(votes, vote_thr=0.55)
        # 3 out of 4 models above 0.55
        assert result["vote_yes_fraction"] == pytest.approx(0.75)
        assert result["active_model_count"] == 4

    def test_vote_score_composite_formula(self):
        votes = {"rf": 1.0, "et": 1.0, "hgbc": 1.0}
        result = compute_vote_score(votes, vote_thr=0.55)
        # All components should be 1.0, so vote_score = 0.4*1 + 0.3*1 + 0.3*1 = 1.0
        assert result["vote_score"] == pytest.approx(1.0)

    def test_top3_mean_uses_top3_only(self):
        # 6 models: top 3 are 0.9, 0.85, 0.8; bottom 3 are 0.1, 0.2, 0.3
        votes = {"a": 0.9, "b": 0.85, "c": 0.8, "d": 0.1, "e": 0.2, "f": 0.3}
        result = compute_vote_score(votes, vote_thr=0.55)
        expected_top3 = (0.9 + 0.85 + 0.8) / 3
        assert result["top3_mean"] == pytest.approx(expected_top3, abs=1e-6)
        # active_mean includes all 6 (avg ~0.39); top3_mean is much higher
        active_mean = sum(votes.values()) / len(votes)
        assert result["top3_mean"] > active_mean

    def test_pv_compute_vote_score_matches_model_registry(self):
        """profit_voting module's compute_vote_score should match model_registry's."""
        votes = {"rf": 0.7, "et": 0.65, "hgbc": 0.6}
        r1 = compute_vote_score(votes)
        r2 = pv_compute_vote_score(votes)
        assert r1["vote_score"] == pytest.approx(r2["vote_score"])
        assert r1["vote_yes_fraction"] == pytest.approx(r2["vote_yes_fraction"])


# ─────────────────────────────────────────────────────────────────────────────
# check_profit_voting_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestProfitVotingGate:
    """Tests for profit_voting.check_profit_voting_gate."""

    def _make_conf(self, active_votes, pred_return=0.02):
        vs = pv_compute_vote_score(active_votes)
        conf = dict(active_votes=active_votes, pred_return=pred_return)
        conf.update(vs)
        return conf

    def test_passes_with_strong_votes(self):
        conf = self._make_conf({"rf": 0.8, "et": 0.75, "hgbc": 0.85})
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend")
        assert approved

    def test_fails_with_low_yes_fraction(self):
        conf = self._make_conf({"rf": 0.4, "et": 0.38, "hgbc": 0.45})
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend",
                                                     vote_yes_frac_min=0.50)
        assert not approved
        assert "vote_yes_frac" in reason

    def test_fails_with_low_top3_mean(self):
        # High yes fraction but low probability values
        conf = self._make_conf({"rf": 0.56, "et": 0.57, "hgbc": 0.58})
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend",
                                                     top3_mean_min=0.62)
        assert not approved
        assert "top3_mean" in reason

    def test_chop_requires_supermajority(self):
        # 3/4 models above 0.55 → yes_frac = 0.75 ≥ chop_threshold(0.70), but
        # top3_mean = (0.72+0.70+0.65)/3 = 0.69 < chop_top3_min(0.75) → FAIL
        conf = self._make_conf({"rf": 0.70, "et": 0.72, "hgbc": 0.65, "lgbm": 0.35})
        approved, reason = check_profit_voting_gate(conf, market_mode="chop",
                                                     chop_vote_yes_frac_min=0.70,
                                                     chop_top3_mean_min=0.75)
        # yes_frac = 3/4 = 0.75 (passes frac), but top3_mean ≈ 0.69 (fails top3)
        assert not approved
        assert "top3_mean" in reason

    def test_chop_blocks_low_pred_return(self):
        conf = self._make_conf({"rf": 0.9, "et": 0.88, "hgbc": 0.87, "lgbm": 0.86},
                                pred_return=0.005)
        approved, reason = check_profit_voting_gate(conf, market_mode="chop",
                                                     chop_pred_return_min=0.01)
        assert not approved
        assert "pred_return" in reason

    def test_chop_blocks_low_ev(self):
        conf = self._make_conf({"rf": 0.9, "et": 0.88, "hgbc": 0.87, "lgbm": 0.86},
                                pred_return=0.02)
        approved, reason = check_profit_voting_gate(conf, market_mode="chop",
                                                     ev_score=0.005,
                                                     chop_ev_min=0.01)
        assert not approved
        assert "ev_score" in reason

    def test_requires_min_active_models(self):
        # Only 2 active models — below default minimum of 3
        conf = self._make_conf({"rf": 0.9, "et": 0.88})
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend",
                                                     require_min_active_models=3)
        assert not approved
        assert "active_count" in reason

    def test_none_market_mode_treated_as_trend(self):
        # None market mode should use trend thresholds (not chop)
        conf = self._make_conf({"rf": 0.8, "et": 0.75, "hgbc": 0.85})
        approved, reason = check_profit_voting_gate(conf, market_mode=None)
        assert approved

    def test_gnb_degenerate_excluded_doesnt_affect_gate(self):
        # gnb=1.0 (degenerate) — should NOT be in active_votes
        # Gate should pass/fail based only on real active models
        active_votes = {"rf": 0.65, "et": 0.62, "hgbc": 0.70}  # no gnb
        conf = self._make_conf(active_votes)
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend")
        assert approved  # passes on valid active models

    def test_format_log_returns_string(self):
        conf = self._make_conf({"rf": 0.8, "et": 0.75, "hgbc": 0.85})
        vs = pv_compute_vote_score({"rf": 0.8, "et": 0.75, "hgbc": 0.85})
        log = format_profit_vote_log("BTCUSD", conf, vs, market_mode="pump")
        assert "profit_vote" in log
        assert "BTCUSD" in log
        assert "vote_score" in log


# ─────────────────────────────────────────────────────────────────────────────
# Profile audit — ruthless differs from balanced
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileDifferences:
    """Tests proving ruthless and balanced are effectively different."""

    def _get_profile_values(self, profile):
        """Run setup_risk_profile on a mock algo and return key attrs."""
        import config as _cfg

        class MockAlgo:
            def __init__(self, p):
                self._risk_profile = p
                self._prof = p
            def get_parameter(self, name):
                if name == "risk_profile":
                    return self._prof
                return None
            def log(self, msg):
                pass
            def debug(self, msg):
                pass

        algo = MockAlgo(profile)
        # Set defaults that setup_risk_profile will override
        for attr, val in [
            ("_tp", 0.04), ("_sl", 0.02), ("_toh", 24),
            ("_min_ev", 0.002), ("_pred_return_min", 0.0),
            ("_s_min", 0.5), ("_s_min_floor", 0.4),
            ("_alloc", 0.9), ("_max_alloc", 0.9), ("_kf", 0.25),
            ("_use_kelly", False), ("_max_sl", 3), ("_cd_mins", 60),
            ("_sl_cd", 120), ("_max_dd", 0.15), ("_cb", 0.95),
            ("_min_hold_minutes", 15), ("_emergency_sl", 0.15),
            ("_label_tp", 0.04), ("_label_sl", 0.02),
            ("_label_horizon", 48), ("_label_cost_bps", 14),
            ("_penalty_losses", 3), ("_penalty_hours", 24.0),
            ("_min_alloc", 0.0), ("_runner_mode", False),
            ("_trail_after_tp", 0.04), ("_trail_pct", 0.025),
        ]:
            setattr(algo, attr, val)

        _cfg.setup_risk_profile(algo)
        return algo

    def test_ruthless_tp_is_higher_than_balanced(self):
        ruthless = self._get_profile_values("ruthless")
        balanced = self._get_profile_values("balanced")
        assert ruthless._tp >= balanced._tp

    def test_ruthless_has_profit_voting_mode(self):
        ruthless = self._get_profile_values("ruthless")
        balanced = self._get_profile_values("balanced")
        assert getattr(ruthless, "_ruthless_profit_voting_mode", False) is True
        # Balanced should NOT have profit_voting_mode set to True
        assert getattr(balanced, "_ruthless_profit_voting_mode", False) is False

    def test_ruthless_pv_thresholds_are_set(self):
        ruthless = self._get_profile_values("ruthless")
        assert hasattr(ruthless, "_pv_vote_threshold")
        assert hasattr(ruthless, "_pv_vote_yes_frac_min")
        assert hasattr(ruthless, "_pv_top3_mean_min")
        assert hasattr(ruthless, "_pv_chop_yes_frac_min")

    def test_ruthless_chop_thresholds_stricter_than_trend(self):
        ruthless = self._get_profile_values("ruthless")
        # Chop thresholds should be stricter (higher) than trend thresholds
        assert ruthless._pv_chop_yes_frac_min > ruthless._pv_vote_yes_frac_min
        assert ruthless._pv_chop_top3_mean_min > ruthless._pv_top3_mean_min

    def test_balanced_profile_unchanged(self):
        """Balanced profile should not have ruthless settings."""
        balanced = self._get_profile_values("balanced")
        assert balanced._tp < 0.10   # balanced TP is low

    def test_ruthless_stores_active_model_lists(self):
        """Ruthless profile must store active and diagnostic model lists."""
        ruthless = self._get_profile_values("ruthless")
        assert hasattr(ruthless, "_ruthless_active_models")
        assert hasattr(ruthless, "_ruthless_diagnostic_models")
        active = ruthless._ruthless_active_models
        diag   = ruthless._ruthless_diagnostic_models
        assert len(active) >= 3
        # Core promoted models must be present
        for m in ("rf", "et", "hgbc_l2"):
            assert m in active, f"{m} missing from _ruthless_active_models"
        # Diagnostic models must include gnb and lr
        for m in ("gnb", "lr"):
            assert m in diag, f"{m} missing from _ruthless_diagnostic_models"
        # gnb and lr must NOT be in active
        for m in ("gnb", "lr", "lr_bal"):
            assert m not in active, f"{m} must not be in _ruthless_active_models"


# ─────────────────────────────────────────────────────────────────────────────
# CandidateJournal
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateJournal:
    """Tests for candidate_journal.CandidateJournal."""

    def test_empty_journal_has_zero_records(self):
        journal = CandidateJournal()
        assert len(journal) == 0
        assert journal.get_records() == []

    def test_record_cycle_adds_records(self):
        journal = CandidateJournal(top_n=3)
        candidates = [
            {"symbol": "BTCUSD", "rank": 1, "selected": True, "vote_score": 0.7},
            {"symbol": "ETHUSD", "rank": 2, "selected": False, "reject_reason": "lower_rank"},
        ]
        journal.record_cycle("2025-01-01", candidates)
        assert len(journal) == 2

    def test_journal_respects_top_n(self):
        journal = CandidateJournal(top_n=2)
        candidates = [
            {"symbol": f"SYM{i}", "rank": i+1, "selected": i == 0}
            for i in range(5)
        ]
        journal.record_cycle("2025-01-01", candidates)
        # Only top 2 should be recorded
        assert len(journal) == 2

    def test_rolling_cap(self):
        journal = CandidateJournal(max_size=5, top_n=1)
        for i in range(10):
            journal.record_cycle(f"2025-01-0{i}", [{"symbol": f"SYM{i}", "rank": 1}])
        assert len(journal) <= 5

    def test_get_skipped_records(self):
        journal = CandidateJournal()
        journal.record_cycle("2025-01-01", [
            {"symbol": "BTC", "rank": 1, "selected": True},
            {"symbol": "ETH", "rank": 2, "selected": False, "reject_reason": "lower_rank"},
        ])
        skipped = journal.get_skipped_records()
        selected = journal.get_selected_records()
        assert len(skipped) == 1
        assert len(selected) == 1
        assert skipped[0]["symbol"] == "ETH"

    def test_to_json_and_from_json(self):
        journal = CandidateJournal()
        journal.record_cycle("2025-01-01", [
            {"symbol": "BTC", "rank": 1, "selected": True, "vote_score": 0.7},
        ])
        json_str = journal.to_json()
        assert "BTC" in json_str

        journal2 = CandidateJournal()
        journal2.from_json(json_str)
        assert len(journal2) == 1
        assert journal2.get_records()[0]["symbol"] == "BTC"

    def test_clear_removes_all_records(self):
        journal = CandidateJournal()
        journal.record_cycle("2025-01-01", [{"symbol": "BTC", "rank": 1}])
        journal.clear()
        assert len(journal) == 0

    def test_record_contains_expected_fields(self):
        journal = CandidateJournal()
        journal.record_cycle("2025-01-01", [{
            "symbol": "BTCUSD",
            "rank": 1,
            "selected": True,
            "vote_score": 0.71,
            "vote_yes_fraction": 0.67,
            "top3_mean": 0.75,
            "active_mean": 0.68,
            "active_n_agree": 3,
            "ev_score": 0.008,
            "pred_return": 0.025,
        }])
        record = journal.get_records()[0]
        assert record["vote_score"] == pytest.approx(0.71)
        assert record["vote_yes_fraction"] == pytest.approx(0.67)
        assert record["top3_mean"] == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# build_candidate_records
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCandidateRecords:
    """Tests for candidate_journal.build_candidate_records."""

    def test_builds_records_with_correct_shape(self):
        class FakeSym:
            def __init__(self, name):
                self.value = name
            def __hash__(self):
                return hash(self.value)
            def __eq__(self, other):
                return self.value == other.value

        sym1 = FakeSym("BTC")
        sym2 = FakeSym("ETH")
        ranked = [(sym1, 0.9), (sym2, 0.7)]
        conf_data = {sym1: {"vote_score": 0.8, "active_mean": 0.7}, sym2: {"vote_score": 0.6}}
        ev_data = {sym1: 0.01, sym2: 0.005}
        entry_path_data = {sym1: "ml", sym2: "ml"}
        scores = {sym1: 0.9, sym2: 0.7}

        records = build_candidate_records(
            ranked_results=ranked,
            conf_data=conf_data,
            ev_data=ev_data,
            entry_path_data=entry_path_data,
            scores=scores,
            selected_sym=sym1,
        )
        assert len(records) == 2
        assert records[0]["rank"] == 1
        assert records[0]["selected"] is True
        assert records[0]["symbol"] == "BTC"
        assert records[1]["selected"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Shadow lab models
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowLab:
    """Tests for shadow_lab.extend_shadow_estimators and regime diagnostics."""

    def _make_training_data(self, n=200):
        np.random.seed(42)
        X = np.random.randn(n, 20)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_extend_shadow_adds_gbc_and_ada(self):
        shadows = extend_shadow_estimators([], max_count=10)
        shadow_ids = [s[0] for s in shadows]
        assert "gbc" in shadow_ids
        assert "ada" in shadow_ids

    def test_extend_shadow_adds_regime_models(self):
        shadows = extend_shadow_estimators([], max_count=16)
        shadow_ids = [s[0] for s in shadows]
        assert "markov_regime" in shadow_ids
        assert "kmeans_regime" in shadow_ids
        assert "isoforest_risk" in shadow_ids

    def test_gbc_and_ada_are_shadow_role(self):
        shadows = extend_shadow_estimators([], max_count=10)
        for sid, est, role in shadows:
            if sid in ("gbc", "ada"):
                assert role == SL_ROLE_SHADOW

    def test_regime_models_are_diagnostic_role(self):
        shadows = extend_shadow_estimators([], max_count=16)
        for sid, est, role in shadows:
            if sid in ("markov_regime", "hmm_regime", "kmeans_regime", "isoforest_risk"):
                assert role == SL_ROLE_DIAGNOSTIC

    def test_max_count_respected(self):
        shadows = extend_shadow_estimators([], max_count=3)
        assert len(shadows) <= 3

    def test_existing_shadows_preserved(self):
        existing = [("rf_shallow", object(), SL_ROLE_SHADOW)]
        shadows = extend_shadow_estimators(existing, max_count=10)
        shadow_ids = [s[0] for s in shadows]
        assert "rf_shallow" in shadow_ids
        assert "gbc" in shadow_ids

    def test_markov_regime_fit_predict(self):
        X, y = self._make_training_data()
        model = MarkovRegimeDiagnostic()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1.0)

    def test_kmeans_regime_fit_predict(self):
        X, y = self._make_training_data()
        model = KMeansRegimeDiagnostic(n_clusters=3)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_isoforest_risk_fit_predict(self):
        X, y = self._make_training_data()
        model = IsoForestRiskDiagnostic()
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        # Col 1 is anomaly probability
        assert np.all(probs[:, 1] >= 0) and np.all(probs[:, 1] <= 1.0)

    def test_gbc_and_ada_train_and_predict(self):
        from shadow_lab import _make_gbc, _make_ada
        X, y = self._make_training_data()
        gbc = _make_gbc()
        ada = _make_ada()
        assert gbc is not None
        assert ada is not None
        gbc.fit(X, y)
        ada.fit(X, y)
        probs_gbc = gbc.predict_proba(X[:5])[:, 1]
        probs_ada = ada.predict_proba(X[:5])[:, 1]
        assert probs_gbc.shape == (5,)
        assert probs_ada.shape == (5,)


# ─────────────────────────────────────────────────────────────────────────────
# GNB excluded from active decisions
# ─────────────────────────────────────────────────────────────────────────────

class TestGNBExcluded:
    """Tests that gnb=1.0 does NOT inflate active_n_agree / class_proba."""

    def test_gnb_not_in_active_votes_by_default(self):
        from model_registry import DEFAULT_MODEL_ROLES, ROLE_DIAGNOSTIC, MODEL_ID_GNB
        assert DEFAULT_MODEL_ROLES[MODEL_ID_GNB] == ROLE_DIAGNOSTIC

    def test_gnb_weight_zero_in_config(self):
        import config as _cfg
        assert getattr(_cfg, "MODEL_WEIGHT_GNB", 1.0) == 0.0

    def test_lr_weight_zero_in_config(self):
        import config as _cfg
        assert getattr(_cfg, "MODEL_WEIGHT_LR", 1.0) == 0.0

    def test_active_stats_ignores_diagnostic_models(self):
        # Simulate what models.py does: split votes by role, then compute active stats
        from model_registry import split_votes_by_role, compute_active_stats, DEFAULT_MODEL_ROLES

        all_votes = {"rf": 0.65, "et": 0.60, "hgbc": 0.70, "gnb": 1.0, "lr": 0.01}
        roles = DEFAULT_MODEL_ROLES.copy()
        active_votes, shadow_votes, diagnostic_votes = split_votes_by_role(all_votes, roles)

        assert "gnb" not in active_votes
        assert "lr" not in active_votes
        assert "gnb" in diagnostic_votes
        assert "lr" in diagnostic_votes

        stats = compute_active_stats(active_votes)
        # Active mean should NOT include gnb's 1.0
        assert stats["active_mean"] < 0.75
        assert stats["active_n_agree"] <= 3

    def test_active_vote_score_excludes_gnb(self):
        active_votes = {"rf": 0.65, "et": 0.60, "hgbc": 0.70}
        # gnb is NOT in active_votes
        vs = compute_vote_score(active_votes)
        assert vs["active_model_count"] == 3  # not 4

    def test_gnb_degenerate_does_not_pass_profit_gate(self):
        """Even if gnb=1.0 were in active_votes, other low votes should fail the gate."""
        # Simulate degenerate scenario where gnb got into active votes accidentally
        # (this should not happen but validate the gate still catches poor quality)
        # The gate requires vote_yes_fraction of active models
        # If gnb is the only yes vote among many nos:
        degenerate_votes = {"gnb": 1.0, "rf": 0.3, "et": 0.28, "hgbc": 0.32}
        vs = pv_compute_vote_score(degenerate_votes)
        conf = {"active_votes": degenerate_votes, "pred_return": 0.01}
        conf.update(vs)
        approved, reason = check_profit_voting_gate(conf, market_mode="risk_on_trend",
                                                     vote_yes_frac_min=0.50)
        # Only 1/4 votes are "yes" → fails
        assert not approved


# ─────────────────────────────────────────────────────────────────────────────
# Config constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigConstants:
    """Tests for new config constants."""

    def test_ruthless_profit_voting_mode_exists(self):
        import config as _cfg
        assert hasattr(_cfg, "RUTHLESS_PROFIT_VOTING_MODE")
        assert _cfg.RUTHLESS_PROFIT_VOTING_MODE is True

    def test_ruthless_active_models_defined(self):
        import config as _cfg
        assert hasattr(_cfg, "RUTHLESS_ACTIVE_MODELS")
        assert len(_cfg.RUTHLESS_ACTIVE_MODELS) > 0

    def test_ruthless_diagnostic_models_defined(self):
        import config as _cfg
        assert hasattr(_cfg, "RUTHLESS_DIAGNOSTIC_MODELS")
        assert "gnb" in _cfg.RUTHLESS_DIAGNOSTIC_MODELS
        assert "lr" in _cfg.RUTHLESS_DIAGNOSTIC_MODELS

    def test_chop_thresholds_stricter(self):
        import config as _cfg
        assert _cfg.RUTHLESS_CHOP_VOTE_YES_FRAC_MIN > _cfg.RUTHLESS_VOTE_YES_FRACTION_MIN
        assert _cfg.RUTHLESS_CHOP_TOP3_MEAN_MIN > _cfg.RUTHLESS_TOP3_MEAN_MIN

    def test_candidate_journal_config(self):
        import config as _cfg
        assert hasattr(_cfg, "PERSIST_CANDIDATE_JOURNAL")
        assert hasattr(_cfg, "CANDIDATE_JOURNAL_TOP_N")
        assert _cfg.CANDIDATE_JOURNAL_TOP_N >= 1

    def test_shadow_model_max_count_increased(self):
        import config as _cfg
        assert _cfg.SHADOW_MODEL_MAX_COUNT >= 14  # increased for new models

    def test_ruthless_min_tp_exists(self):
        import config as _cfg
        assert hasattr(_cfg, "RUTHLESS_MIN_TP")
        assert _cfg.RUTHLESS_MIN_TP >= 0.03  # at least 3%

    def test_multi_position_scaffold_config(self):
        import config as _cfg
        assert hasattr(_cfg, "RUTHLESS_MAX_CONCURRENT_POSITIONS")
        # Currently scaffold only (single position)
        assert _cfg.RUTHLESS_MAX_CONCURRENT_POSITIONS >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Ruthless active-model promotion
# ─────────────────────────────────────────────────────────────────────────────

class TestRuthlessActivePromotion:
    """Tests for profit_voting.apply_ruthless_active_promotion."""

    def _make_conf(self, active_votes, shadow_votes=None, pred_return=0.02):
        vs = pv_compute_vote_score(active_votes)
        conf = dict(active_votes=dict(active_votes), pred_return=pred_return)
        if shadow_votes:
            conf["shadow_votes"] = dict(shadow_votes)
        conf.update(vs)
        return conf

    def test_promotes_shadow_to_active(self):
        shadow = {"hgbc_l2": 0.72, "cal_et": 0.65, "lgbm_bal": 0.68}
        conf = self._make_conf(
            active_votes={"rf": 0.60, "et": 0.58, "hgbc": 0.65},
            shadow_votes=shadow,
        )
        added = apply_ruthless_active_promotion(conf, ["rf", "et", "hgbc_l2", "cal_et", "lgbm_bal"])
        assert added >= 2
        assert "hgbc_l2" in conf["active_votes"]
        assert "cal_et" in conf["active_votes"]
        assert "lgbm_bal" in conf["active_votes"]

    def test_excludes_diagnostic_models(self):
        shadow = {"gnb": 1.0, "lr": 0.01, "hgbc_l2": 0.72}
        conf = self._make_conf(
            active_votes={"rf": 0.60, "et": 0.58},
            shadow_votes=shadow,
        )
        apply_ruthless_active_promotion(
            conf,
            active_models=["rf", "et", "hgbc_l2", "gnb", "lr"],
            diagnostic_models=["gnb", "lr", "lr_bal"],
        )
        assert "gnb" not in conf["active_votes"]
        assert "lr" not in conf["active_votes"]
        assert "hgbc_l2" in conf["active_votes"]

    def test_backward_compat_fields_updated(self):
        shadow = {"hgbc_l2": 0.80, "cal_et": 0.75}
        conf = self._make_conf(
            active_votes={"rf": 0.60},
            shadow_votes=shadow,
        )
        apply_ruthless_active_promotion(conf, ["rf", "hgbc_l2", "cal_et"])
        assert "class_proba" in conf
        assert "n_agree" in conf
        assert "std_proba" in conf
        # Active pool now has 3 models; mean should be between 0.6 and 0.8
        assert 0.60 <= conf["class_proba"] <= 0.85

    def test_no_promotion_if_nothing_in_shadow(self):
        conf = self._make_conf(active_votes={"rf": 0.60, "et": 0.55})
        original_active = dict(conf["active_votes"])
        added = apply_ruthless_active_promotion(conf, ["hgbc_l2", "lgbm_bal"])
        assert added == 0
        assert conf["active_votes"] == original_active

    def test_empty_active_models_is_noop(self):
        conf = self._make_conf(active_votes={"rf": 0.60})
        original_active = dict(conf["active_votes"])
        added = apply_ruthless_active_promotion(conf, [])
        assert added == 0
        assert conf["active_votes"] == original_active

    def test_vote_score_updated_after_promotion(self):
        shadow = {"hgbc_l2": 0.90, "lgbm_bal": 0.85}
        conf = self._make_conf(
            active_votes={"rf": 0.40, "et": 0.38},  # weak votes
            shadow_votes=shadow,
        )
        old_vs = conf["vote_score"]
        apply_ruthless_active_promotion(conf, ["rf", "et", "hgbc_l2", "lgbm_bal"])
        # vote_score should increase after adding strong shadow models
        assert conf["vote_score"] > old_vs

    def test_gnb_and_lr_never_promoted_by_default(self):
        shadow = {"gnb": 1.0, "lr": 0.02, "lr_bal": 0.50}
        conf = self._make_conf(
            active_votes={"rf": 0.60, "et": 0.55, "hgbc": 0.65},
            shadow_votes=shadow,
        )
        # Try to promote gnb/lr — should be blocked by diagnostic_models
        apply_ruthless_active_promotion(
            conf,
            active_models=["rf", "et", "hgbc", "gnb", "lr"],
            diagnostic_models=["gnb", "lr", "lr_bal"],
        )
        assert "gnb" not in conf["active_votes"]
        assert "lr" not in conf["active_votes"]
        assert "lr_bal" not in conf["active_votes"]

    def test_active_model_count_reflects_promoted_pool(self):
        shadow = {"hgbc_l2": 0.70, "cal_et": 0.65, "cal_rf": 0.68, "lgbm_bal": 0.72}
        conf = self._make_conf(
            active_votes={"rf": 0.60, "et": 0.58, "hgbc": 0.62},
            shadow_votes=shadow,
        )
        apply_ruthless_active_promotion(
            conf,
            active_models=["rf", "et", "hgbc", "hgbc_l2", "cal_et", "cal_rf", "lgbm_bal"],
        )
        assert conf["active_model_count"] == 7

    def test_config_ruthless_active_models_excludes_gnb_lr(self):
        import config as _cfg
        assert "gnb" not in _cfg.RUTHLESS_ACTIVE_MODELS
        assert "lr" not in _cfg.RUTHLESS_ACTIVE_MODELS
        assert "lr_bal" not in _cfg.RUTHLESS_ACTIVE_MODELS

    def test_config_ruthless_diagnostic_models_includes_gnb_lr(self):
        import config as _cfg
        assert "gnb" in _cfg.RUTHLESS_DIAGNOSTIC_MODELS
        assert "lr" in _cfg.RUTHLESS_DIAGNOSTIC_MODELS
        assert "lr_bal" in _cfg.RUTHLESS_DIAGNOSTIC_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Relaxed bootstrap thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestRelaxedBootstrapThresholds:
    """Tests that thresholds are at bootstrap-relaxed levels."""

    def test_config_vote_threshold_relaxed(self):
        import config as _cfg
        assert _cfg.RUTHLESS_VOTE_THRESHOLD <= 0.50

    def test_config_yes_frac_min_relaxed(self):
        import config as _cfg
        assert _cfg.RUTHLESS_VOTE_YES_FRACTION_MIN <= 0.34

    def test_config_top3_mean_relaxed(self):
        import config as _cfg
        assert _cfg.RUTHLESS_TOP3_MEAN_MIN <= 0.55

    def test_config_ev_floor_relaxed(self):
        import config as _cfg
        assert _cfg.RUTHLESS_VOTE_EV_FLOOR <= 0.001

    def test_config_chop_thresholds_still_stricter(self):
        import config as _cfg
        # Chop thresholds must be stricter than trend thresholds
        assert _cfg.RUTHLESS_CHOP_VOTE_YES_FRAC_MIN > _cfg.RUTHLESS_VOTE_YES_FRACTION_MIN
        assert _cfg.RUTHLESS_CHOP_TOP3_MEAN_MIN > _cfg.RUTHLESS_TOP3_MEAN_MIN

    def test_pv_gate_passes_3_model_setup(self):
        """3-model setup should pass at bootstrap thresholds."""
        conf = {"rf": 0.60, "et": 0.55, "hgbc": 0.62}
        vs = pv_compute_vote_score(conf, vote_thr=0.50)
        gate_conf = {"active_votes": conf, "pred_return": 0.01}
        gate_conf.update(vs)
        approved, reason = check_profit_voting_gate(
            gate_conf, market_mode="risk_on_trend",
            vote_thr=0.50, vote_yes_frac_min=0.34, top3_mean_min=0.55,
            require_min_active_models=3,
        )
        assert approved, f"Gate blocked with reason: {reason}"

    def test_pv_gate_passes_6_model_setup(self):
        """6-model promoted pool should pass with typical probas."""
        votes = {"rf": 0.62, "et": 0.58, "hgbc_l2": 0.70, "cal_et": 0.55,
                 "cal_rf": 0.57, "lgbm_bal": 0.66}
        vs = pv_compute_vote_score(votes, vote_thr=0.50)
        gate_conf = {"active_votes": votes, "pred_return": 0.01}
        gate_conf.update(vs)
        approved, reason = check_profit_voting_gate(
            gate_conf, market_mode="risk_on_trend",
            vote_thr=0.50, vote_yes_frac_min=0.34, top3_mean_min=0.55,
            require_min_active_models=3,
        )
        assert approved, f"Gate blocked with reason: {reason}"


# ─────────────────────────────────────────────────────────────────────────────
# Profit-voting reject counters and log formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestPvRejectCounters:
    """Tests for make_pv_counters / increment_pv_counter / format helpers."""

    def test_make_pv_counters_has_expected_keys(self):
        c = make_pv_counters()
        for key in ("candidates", "pass", "fail_active_count", "fail_yes_frac",
                    "fail_top3", "no_active_votes"):
            assert key in c, f"Missing key: {key}"

    def test_increment_active_count_reason(self):
        c = make_pv_counters()
        increment_pv_counter(c, "active_count=2 < 3")
        assert c["fail_active_count"] == 1

    def test_increment_yes_frac_reason(self):
        c = make_pv_counters()
        increment_pv_counter(c, "vote_yes_frac=0.25 < 0.34")
        assert c["fail_yes_frac"] == 1

    def test_increment_top3_reason(self):
        c = make_pv_counters()
        increment_pv_counter(c, "top3_mean=0.50 < 0.55")
        assert c["fail_top3"] == 1

    def test_increment_chop_yes_frac(self):
        c = make_pv_counters()
        increment_pv_counter(c, "chop: vote_yes_frac=0.30 < 0.50")
        assert c["fail_chop_yes_frac"] == 1

    def test_increment_ev_floor(self):
        c = make_pv_counters()
        increment_pv_counter(c, "ev_floor=0.0000 < 0.001")
        assert c["fail_ev_floor"] == 1

    def test_format_pv_reject_log_returns_string(self):
        conf = {"vote_yes_fraction": 0.33, "top3_mean": 0.52,
                "active_model_count": 5, "pred_return": 0.005}
        log = format_pv_reject_log("ADAUSD", conf, "chop", "top3_mean")
        assert "[pv_reject]" in log
        assert "ADAUSD" in log
        assert "top3_mean" in log

    def test_format_pv_summary_log_returns_string(self):
        c = make_pv_counters()
        c["candidates"] = 18
        c["pass"] = 0
        c["fail_yes_frac"] = 7
        log = format_pv_summary_log(c)
        assert "[pv_summary]" in log
        assert "candidates=18" in log
        assert "fail_yes_frac=7" in log


# ─────────────────────────────────────────────────────────────────────────────
# Rejected candidate journal (empty-scores cycle)
# ─────────────────────────────────────────────────────────────────────────────

class TestRejectedCandidateJournal:
    """Tests for build_rejected_candidate_records and empty-scores journaling."""

    class FakeSym:
        def __init__(self, name):
            self.value = name
        def __hash__(self):
            return hash(self.value)
        def __eq__(self, other):
            return self.value == getattr(other, "value", other)

    def _make_conf(self, vote_score=0.0, active_votes=None):
        return {
            "vote_score": vote_score,
            "active_mean": 0.60,
            "class_proba": 0.60,
            "std_proba": 0.05,
            "n_agree": 2,
            "vote_yes_fraction": 0.50,
            "top3_mean": 0.62,
            "pred_return": 0.01,
            "active_model_count": 3,
            "active_votes": active_votes or {"rf": 0.60, "et": 0.58},
            "shadow_votes": {},
            "diagnostic_votes": {},
        }

    def test_build_rejected_records_empty_conf_dict(self):
        records = build_rejected_candidate_records({})
        assert records == []

    def test_build_rejected_records_returns_not_selected(self):
        sym = self.FakeSym("BTCUSD")
        conf_dict = {sym: self._make_conf(vote_score=0.7)}
        records = build_rejected_candidate_records(conf_dict)
        assert len(records) == 1
        assert records[0]["selected"] is False
        assert records[0]["symbol"] == "BTCUSD"
        assert records[0]["reject_reason"] == "pv_no_pass"

    def test_build_rejected_records_sorted_by_vote_score(self):
        s1 = self.FakeSym("LOW")
        s2 = self.FakeSym("HIGH")
        conf_dict = {
            s1: self._make_conf(vote_score=0.30),
            s2: self._make_conf(vote_score=0.80),
        }
        records = build_rejected_candidate_records(conf_dict)
        assert records[0]["symbol"] == "HIGH"
        assert records[1]["symbol"] == "LOW"

    def test_build_rejected_records_respects_top_n(self):
        conf_dict = {self.FakeSym(f"SYM{i}"): self._make_conf(vote_score=i/10) for i in range(10)}
        records = build_rejected_candidate_records(conf_dict, top_n=3)
        assert len(records) == 3

    def test_build_rejected_records_includes_active_votes(self):
        sym = self.FakeSym("ETHUSD")
        av = {"rf": 0.65, "et": 0.70}
        conf_dict = {sym: self._make_conf(active_votes=av)}
        records = build_rejected_candidate_records(conf_dict)
        assert records[0]["active_votes"] == av

    def test_rejected_journal_recorded_via_record_cycle(self):
        journal = CandidateJournal()
        sym = self.FakeSym("SOLUSD")
        conf_dict = {sym: self._make_conf(vote_score=0.65)}
        records = build_rejected_candidate_records(conf_dict, market_mode="chop")
        journal.record_cycle("2025-01-01", records)
        assert len(journal) == 1
        assert journal.get_records()[0]["reject_reason"] == "pv_no_pass"
        assert journal.get_records()[0]["market_mode"] == "chop"

    def test_no_active_votes_produces_safe_record(self):
        sym = self.FakeSym("XRPUSD")
        conf = self._make_conf(vote_score=0.0)
        conf["active_votes"] = {}  # explicitly empty after make_conf
        conf["active_model_count"] = 0
        records = build_rejected_candidate_records({sym: conf})
        assert len(records) == 1
        assert records[0]["active_votes"] == {}

