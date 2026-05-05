"""
test_gatling.py — verify the gatling risk profile: constants, gate setup,
model assessment utility, and ensemble training+prediction under gatling.

No QuantConnect dependency — runs with plain pytest + numpy + scikit-learn.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Constants exist and are ultra-loose
# ═══════════════════════════════════════════════════════════════════════════════

class TestGatlingConstants:
    def test_constants_importable(self):
        from gatling_config import (
            GATLING_SCORE_MIN, GATLING_MIN_EV, GATLING_PRED_RETURN_MIN,
            GATLING_MAX_DISPERSION, GATLING_MIN_AGREE, GATLING_EV_GAP,
            GATLING_COST_BPS, GATLING_ALLOCATION, GATLING_MAX_ALLOC,
            GATLING_TAKE_PROFIT, GATLING_STOP_LOSS, GATLING_TIMEOUT_HOURS,
            GATLING_COOLDOWN_MINS, GATLING_SL_COOLDOWN_MINS,
            GATLING_DECISION_INTERVAL_MIN, GATLING_META_FILTER_ENABLED,
            GATLING_MARKET_MODE_ENABLED, GATLING_ACTIVE_MODELS,
            GATLING_TRACK_MODEL_ACCURACY,
        )
        assert GATLING_SCORE_MIN <= 0.15
        assert GATLING_MIN_EV < 0
        assert GATLING_PRED_RETURN_MIN < -0.005
        assert GATLING_MAX_DISPERSION >= 0.50
        assert GATLING_MIN_AGREE == 0
        assert GATLING_EV_GAP == 0.0
        assert GATLING_ALLOCATION >= 0.90
        assert GATLING_MAX_ALLOC >= 0.95
        assert GATLING_TAKE_PROFIT <= 0.020
        assert GATLING_STOP_LOSS <= 0.015
        assert GATLING_TIMEOUT_HOURS <= 3.0
        assert GATLING_COOLDOWN_MINS == 0
        assert GATLING_SL_COOLDOWN_MINS == 0
        assert GATLING_DECISION_INTERVAL_MIN == 5
        assert GATLING_META_FILTER_ENABLED is False
        assert GATLING_MARKET_MODE_ENABLED is False
        assert len(GATLING_ACTIVE_MODELS) >= 10
        assert GATLING_TRACK_MODEL_ACCURACY is True

    def test_score_min_looser_than_all_profiles(self):
        from gatling_config import GATLING_SCORE_MIN
        from core import (
            SCORE_MIN, RUTHLESS_SCORE_MIN, AGGRESSIVE_SCORE_MIN,
            ACTIVE_RESEARCH_SCORE_MIN,
        )
        assert GATLING_SCORE_MIN <= RUTHLESS_SCORE_MIN
        assert GATLING_SCORE_MIN <= AGGRESSIVE_SCORE_MIN
        assert GATLING_SCORE_MIN <= ACTIVE_RESEARCH_SCORE_MIN
        assert GATLING_SCORE_MIN < SCORE_MIN

    def test_cooldowns_all_zero_or_near_zero(self):
        from gatling_config import (
            GATLING_COOLDOWN_MINS, GATLING_SL_COOLDOWN_MINS,
            GATLING_PENALTY_COOLDOWN_LOSSES, GATLING_PORTFOLIO_LOSS_STREAK,
        )
        assert GATLING_COOLDOWN_MINS == 0
        assert GATLING_SL_COOLDOWN_MINS == 0
        assert GATLING_PENALTY_COOLDOWN_LOSSES >= 50
        assert GATLING_PORTFOLIO_LOSS_STREAK >= 50

    def test_confirmation_gate_effectively_disabled(self):
        from gatling_config import (
            GATLING_CONFIRM_EV_MIN, GATLING_CONFIRM_PROBA_MIN,
            GATLING_CONFIRM_AGREE_MIN,
        )
        assert GATLING_CONFIRM_EV_MIN < -0.5
        assert GATLING_CONFIRM_PROBA_MIN <= 0.0
        assert GATLING_CONFIRM_AGREE_MIN == 0

    def test_active_models_include_v2_models(self):
        from gatling_config import GATLING_ACTIVE_MODELS
        assert "catboost" in GATLING_ACTIVE_MODELS
        assert "xgb_hist" in GATLING_ACTIVE_MODELS
        assert "lgbm_goss" in GATLING_ACTIVE_MODELS
        assert "hgbc" in GATLING_ACTIVE_MODELS
        assert "tabnet" in GATLING_ACTIVE_MODELS
        assert "ebm" in GATLING_ACTIVE_MODELS


# ═══════════════════════════════════════════════════════════════════════════════
# 2. setup_risk_profile applies gatling settings correctly
# ═══════════════════════════════════════════════════════════════════════════════

class MockAlgo:
    def __init__(self, risk_profile="gatling"):
        self._rp = risk_profile
        self._logs = []
        from core import (
            SCORE_MIN, MAX_DISPERSION, MIN_AGREE, MIN_EV, EV_GAP, COST_BPS,
            ALLOCATION, MAX_ALLOC, KELLY_FRAC, TAKE_PROFIT, STOP_LOSS,
            TIMEOUT_HOURS, MIN_HOLD_MINUTES, EMERGENCY_SL, MAX_DAILY_SL,
            COOLDOWN_MINS, SL_COOLDOWN_MINS, PENALTY_COOLDOWN_LOSSES,
            PENALTY_COOLDOWN_HOURS, MAX_DD_PCT, PRED_RETURN_MIN, USE_KELLY,
        )
        self._s_min = SCORE_MIN
        self._max_disp = MAX_DISPERSION
        self._min_agr = MIN_AGREE
        self._min_ev = MIN_EV
        self._ev_gap = EV_GAP
        self._cost_bps = COST_BPS
        self._alloc = ALLOCATION
        self._max_alloc = MAX_ALLOC
        self._kf = KELLY_FRAC
        self._tp = TAKE_PROFIT
        self._sl = STOP_LOSS
        self._toh = TIMEOUT_HOURS
        self._min_hold_minutes = MIN_HOLD_MINUTES
        self._emergency_sl = EMERGENCY_SL
        self._max_sl = MAX_DAILY_SL
        self._cd_mins = COOLDOWN_MINS
        self._sl_cd = SL_COOLDOWN_MINS
        self._penalty_losses = PENALTY_COOLDOWN_LOSSES
        self._penalty_hours = PENALTY_COOLDOWN_HOURS
        self._max_dd = MAX_DD_PCT
        self._pred_return_min = PRED_RETURN_MIN
        self._use_kelly = USE_KELLY
        self._min_alloc = 0.0
        self._runner_mode = False
        self._trail_after_tp = 0.0
        self._trail_pct = 0.0
        self._atr_tp = 2.0
        self._atr_sl = 1.2

    def get_parameter(self, name):
        if name == "risk_profile":
            return self._rp
        return None

    def log(self, msg):
        self._logs.append(msg)

    def debug(self, msg):
        self._logs.append(msg)


class TestGatlingProfile:
    def test_profile_applies_ultra_loose_gates(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._risk_profile == "gatling"
        assert algo._s_min <= 0.15
        assert algo._min_ev < 0
        assert algo._min_agr == 0
        assert algo._max_disp >= 0.50
        assert algo._cd_mins == 0
        assert algo._sl_cd == 0
        assert algo._max_sl >= 50

    def test_meta_filter_disabled(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._meta_filter_enabled is False

    def test_market_mode_disabled(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._market_mode_enabled is False
        assert len(algo._ruthless_allowed_modes) == 5

    def test_decision_interval_5min(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._gatling_decision_interval == 5

    def test_profit_voting_enabled(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._ruthless_profit_voting_mode is True

    def test_all_models_active(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert len(algo._ruthless_active_models) >= 10
        assert "catboost" in algo._ruthless_active_models
        assert "xgb_hist" in algo._ruthless_active_models
        assert "hgbc" in algo._ruthless_active_models

    def test_momentum_override_enabled(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._momentum_override is True
        assert algo._use_momentum_score is True

    def test_runner_mode_off(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._runner_mode is False

    def test_no_ruthless_min_tp(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._ruthless_min_tp == 0.0

    def test_startup_log_emitted(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert any("[gatling]" in msg for msg in algo._logs)

    def test_labels_fast_scalp(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._label_tp <= 0.015
        assert algo._label_sl <= 0.010
        assert algo._label_horizon <= 30

    def test_allocation_near_full(self):
        from core import setup_risk_profile
        algo = MockAlgo("gatling")
        setup_risk_profile(algo)
        assert algo._alloc >= 0.90
        assert algo._max_alloc >= 0.95
        assert algo._use_kelly is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Gatling treated as ruthless-like in evaluate_candidate
# ═══════════════════════════════════════════════════════════════════════════════

class TestGatlingEvaluateCandidate:
    def _make_conf(self, proba=0.3):
        return {
            "class_proba": proba,
            "std_proba": 0.1,
            "n_agree": 1,
            "pred_return": 0.001,
            "return_dispersion": 0.0,
            "per_model": {"hgbc": proba, "et": proba, "rf": proba},
            "active_votes": {"hgbc": proba, "et": proba, "rf": proba},
            "shadow_votes": {},
            "diagnostic_votes": {},
            "vote_score": 0.5,
            "vote_yes_fraction": 0.5,
            "top3_mean": proba,
        }

    def _make_feat(self):
        return np.zeros(20)

    def test_gatling_passes_low_proba_candidate(self):
        from strategy import evaluate_candidate
        result = evaluate_candidate(
            sym="BTCUSD", feat=self._make_feat(), conf=self._make_conf(0.35),
            price=50000.0, atr=500.0,
            risk_profile="gatling",
            tp_base=0.015, sl_base=0.010,
            atr_tp_mult=2.0, atr_sl_mult=1.2,
            cost_fraction=0.0015,
            momentum_override_enabled=True,
            momentum_ret4_min=-1.0, momentum_ret16_min=-1.0,
            momentum_volume_min=0.0, momentum_btc_rel_min=-1.0,
            momentum_override_min_ev=-1.0,
            ruthless_confirm_ev_min=-1.0,
            ruthless_confirm_proba_min=0.0,
            ruthless_confirm_agree_min=0,
            ruthless_confirm_ret4_min=-1.0,
            ruthless_confirm_ret16_min=-1.0,
            ruthless_confirm_volr_min=0.0,
            use_momentum_score=True,
            reg_fitted=False,
            score_min_eff=0.10,
            max_disp=0.50,
            min_agr=0,
            min_ev=-0.005,
            pred_return_min=-0.010,
            compute_momentum_score_fn=lambda f: 0.0,
            counters={"n_pass_disp": 0, "n_pass_agree": 0, "n_pass_score": 0,
                      "n_pass_ev": 0, "n_pass_pred_ret": 0, "n_momentum_override": 0},
            market_mode="chop",
            ruthless_allowed_modes=["risk_on_trend", "pump", "chop", "selloff", "high_vol_reversal"],
            ruthless_good_mode_relaxation=True,
            ruthless_good_mode_ev_min=-1.0,
            ruthless_good_mode_volr_min=0.0,
            ruthless_profit_voting_mode=True,
            pv_vote_threshold=0.35,
            pv_vote_yes_frac_min=0.10,
            pv_top3_mean_min=0.30,
            pv_vote_ev_floor=0.0,
            pv_chop_yes_frac_min=0.15,
            pv_chop_top3_mean_min=0.35,
            pv_chop_pred_return_min=-1.0,
            pv_chop_ev_min=-1.0,
            ruthless_active_models=["rf", "et", "hgbc"],
            ruthless_diagnostic_models=["gnb"],
        )
        assert result is not None
        assert "final_score" in result
        assert "confirm_reason" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Model assessment utility
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelAssessment:
    def test_compute_model_accuracy(self):
        from gatling_config import GATLING_MIN_TRADES_FOR_ASSESS
        from model_assessment import compute_model_accuracy

        trades = [
            {"active_votes": {"rf": 0.7, "et": 0.6, "hgbc": 0.3},
             "realized_return": 0.015, "winner": True},
            {"active_votes": {"rf": 0.8, "et": 0.4, "hgbc": 0.6},
             "realized_return": -0.005, "winner": False},
            {"active_votes": {"rf": 0.6, "et": 0.7, "hgbc": 0.7},
             "realized_return": 0.010, "winner": True},
        ]
        result = compute_model_accuracy(trades, vote_threshold=0.50)
        assert "rf" in result
        assert "et" in result
        assert "hgbc" in result
        assert result["rf"]["yes_count"] >= 2
        assert "win_rate_when_yes" in result["rf"]
        assert "avg_return_when_yes" in result["rf"]
        assert "profit_factor" in result["rf"]

    def test_empty_trades_returns_empty(self):
        from model_assessment import compute_model_accuracy
        result = compute_model_accuracy([], vote_threshold=0.50)
        assert result == {}

    def test_format_assessment_report(self):
        from model_assessment import compute_model_accuracy, format_assessment_report
        trades = [
            {"active_votes": {"rf": 0.7, "et": 0.6}, "realized_return": 0.01, "winner": True},
            {"active_votes": {"rf": 0.3, "et": 0.8}, "realized_return": -0.005, "winner": False},
        ]
        accuracy = compute_model_accuracy(trades, vote_threshold=0.50)
        report = format_assessment_report(accuracy)
        assert isinstance(report, str)
        assert "rf" in report or "et" in report

    def test_rank_models_by_profit_factor(self):
        from model_assessment import compute_model_accuracy, rank_models
        trades = [
            {"active_votes": {"rf": 0.7, "et": 0.6, "hgbc": 0.8},
             "realized_return": 0.02, "winner": True},
            {"active_votes": {"rf": 0.8, "et": 0.3, "hgbc": 0.4},
             "realized_return": -0.01, "winner": False},
            {"active_votes": {"rf": 0.6, "et": 0.7, "hgbc": 0.7},
             "realized_return": 0.015, "winner": True},
        ]
        accuracy = compute_model_accuracy(trades, vote_threshold=0.50)
        ranked = rank_models(accuracy, by="profit_factor")
        assert isinstance(ranked, list)
        assert len(ranked) >= 2
        if len(ranked) >= 2:
            assert ranked[0][1]["profit_factor"] >= ranked[1][1]["profit_factor"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. End-to-end: VoxEnsemble train → predict under gatling config
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleV2:
    def test_make_v2_estimators(self):
        from ensemble_v2 import make_v2_estimators, V2_MODEL_IDS, V2_ACTIVE_IDS
        models = make_v2_estimators()
        ids = [m[0] for m in models]
        assert len(models) >= 8
        for mid in ["hgbc", "et_depth5", "ridge_cal", "mlp", "iforest_veto"]:
            assert mid in ids, f"{mid} missing"
        assert len(V2_ACTIVE_IDS) >= 10
        assert "iforest_veto" not in V2_ACTIVE_IDS

    def test_fit_and_predict_v2(self):
        from ensemble_v2 import make_v2_estimators, fit_v2_models, predict_v2_models
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = (np.random.rand(200) > 0.6).astype(int)
        models = make_v2_estimators()
        fitted, n_ok = fit_v2_models(models, X, y)
        assert n_ok >= 4
        preds = predict_v2_models(fitted, X[0])
        assert len(preds) >= 4
        for mid, prob in preds.items():
            assert 0.0 <= prob <= 1.0, f"{mid} prob={prob}"

    def test_v2_weights_dict(self):
        from ensemble_v2 import V2_WEIGHTS
        assert V2_WEIGHTS["catboost"] == 1.5
        assert V2_WEIGHTS["mlp"] == 0.75
        assert V2_WEIGHTS["stack_meta"] == 0.0

    def test_iforest_is_veto(self):
        from ensemble_v2 import make_v2_estimators
        models = make_v2_estimators()
        for mid, est, role, w in models:
            if mid == "iforest_veto":
                assert role == "veto"


class TestGatlingEnsembleEndToEnd:
    def test_ensemble_trains_with_fast_labels(self):
        from models import VoxEnsemble, build_features, triple_barrier_outcome
        from gatling_config import GATLING_LABEL_TP, GATLING_LABEL_SL, GATLING_LABEL_HORIZON_BARS

        np.random.seed(99)
        n = 300
        prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.003)
        volumes = np.random.uniform(1000, 5000, n)

        X_list = []
        for i in range(50, n):
            fv = build_features(
                closes=prices[max(0, i-40):i+1],
                volumes=volumes[max(0, i-40):i+1],
                btc_closes=prices[max(0, i-40):i+1],
                hour=12,
            )
            if fv is not None:
                X_list.append(fv)

        X = np.array(X_list)
        y_class, y_return = [], []
        for i in range(len(X)):
            remaining = prices[i:]
            if len(remaining) > GATLING_LABEL_HORIZON_BARS:
                label, ret = triple_barrier_outcome(
                    remaining, tp=GATLING_LABEL_TP, sl=GATLING_LABEL_SL,
                    timeout_bars=GATLING_LABEL_HORIZON_BARS, cost_fraction=0.0015,
                )
                y_class.append(label)
                y_return.append(ret)
            else:
                y_class.append(0)
                y_return.append(0.0)

        y_class = np.array(y_class)
        y_return = np.array(y_return)

        ens = VoxEnsemble()
        ens.fit(X, y_class, y_return)
        assert ens._fitted

        conf = ens.predict_with_confidence(X[-1:])
        assert "class_proba" in conf
        assert "per_model" in conf
        assert len(conf["per_model"]) >= 3

    def test_file_size_constraint(self):
        """gatling_config.py must stay under QC 63KB limit."""
        import pathlib
        p = pathlib.Path(__file__).parent.parent / "gatling_config.py"
        assert p.stat().st_size < 63000
