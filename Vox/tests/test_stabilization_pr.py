"""
Tests for the professionalization/stabilization PR changes.

Coverage:
  1. HDBSCAN disabled fallback (Phase 1)
  2. Training hour helper (Phase 2)
  3. Label-vs-execution alignment check (Phase 2)
  4. Version constants (Phase 2)
  5. Voting weights exclude LR from active vote (Phase 3)
  6. Breakeven tag / risk accounting (Phase 4)
  7. RiskManager rolling window (Phase 4)
  8. Position-count size discount helper (Phase 4)
  9. Kraken backtest guard (Phase 5)
 10. VoxEnsemble load_state version checks (Phase 2)
"""
import sys
import os
import math
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategy import (                        # noqa: E402
    HDBSCANRegimeDiagnostic,
    HAS_HDBSCAN,
    RiskManager,
    position_count_size_multiplier,
)
from models import (                          # noqa: E402
    derive_training_hour,
    check_label_execution_alignment,
    CLASSIFIER_WEIGHTS,
    MODEL_VERSION,
    FEATURE_VERSION,
    LABEL_VERSION,
    FEATURE_COUNT,
    VoxEnsemble,
)
from infra import fetch_kraken_top20_usd, KRAKEN_PAIRS  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — HDBSCAN disabled fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestHDBSCANDisabled:
    """HDBSCAN must be permanently disabled for QuantConnect compatibility."""

    def test_has_hdbscan_is_false(self):
        """HAS_HDBSCAN must be False at module level (disabled for QC)."""
        assert HAS_HDBSCAN is False

    def test_predict_proba_unfitted_returns_half(self):
        """Unfitted diagnostic must return neutral 0.5 probas."""
        d = HDBSCANRegimeDiagnostic()
        out = d.predict_proba(np.zeros((4, 20)))
        assert out.shape == (4, 2)
        assert np.allclose(out[:, 0], 0.5)
        assert np.allclose(out[:, 1], 0.5)

    def test_fit_does_not_crash(self):
        """fit() must not raise even when hdbscan is unavailable."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 20))
        y = rng.integers(0, 2, size=40)
        d = HDBSCANRegimeDiagnostic()
        d.fit(X, y)   # must not raise

    def test_fit_leaves_unfitted_when_disabled(self):
        """With HAS_HDBSCAN=False, fit() should be a no-op (fitted=False)."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 20))
        y = rng.integers(0, 2, size=40)
        d = HDBSCANRegimeDiagnostic()
        d.fit(X, y)
        assert not d._fitted   # no-op when disabled

    def test_predict_after_disabled_fit_still_safe(self):
        """predict_proba after a disabled fit must still return valid output."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 20))
        y = rng.integers(0, 2, size=10)
        d = HDBSCANRegimeDiagnostic()
        d.fit(X, y)
        out = d.predict_proba(X[:3])
        assert out.shape == (3, 2)
        assert np.all(out >= 0) and np.all(out <= 1)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Training hour helper
# ─────────────────────────────────────────────────────────────────────────────

class TestDeriveTrainingHour:
    """derive_training_hour must return varied hours, not constant zero."""

    def test_returns_int_in_range(self):
        """Return value must be an int in [0, 23]."""
        for i in range(0, 100, 10):
            h = derive_training_hour(i, 100, decision_interval_min=15)
            assert isinstance(h, int)
            assert 0 <= h <= 23

    def test_not_all_zero(self):
        """Different bar positions must yield different hours (non-constant)."""
        n = 200
        hours = [derive_training_hour(i, n, decision_interval_min=15) for i in range(0, n, 5)]
        assert len(set(hours)) > 1, "All training hours are identical (constant bias)"

    def test_most_recent_bar_returns_zero(self):
        """The most-recent bar (index = n-1) should map to hour 0."""
        n = 100
        h = derive_training_hour(n - 1, n, decision_interval_min=15)
        assert h == 0

    def test_4h_cadence(self):
        """Bars 4h apart should differ by exactly 4 hours (mod 24)."""
        # 4h = 16 bars at 15min cadence
        n = 300
        h0 = derive_training_hour(100, n, decision_interval_min=15)
        h1 = derive_training_hour(100 + 16, n, decision_interval_min=15)
        diff = (h0 - h1) % 24
        assert diff == 4

    def test_24h_cycle(self):
        """96 bars of 15-min cadence = exactly 24h, so hours wrap to same value."""
        n = 300
        bars_per_day = 96  # 24*60/15
        h0 = derive_training_hour(100, n, decision_interval_min=15)
        h1 = derive_training_hour(100 + bars_per_day, n, decision_interval_min=15)
        assert h0 == h1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Label-vs-execution alignment check
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckLabelExecutionAlignment:
    """Alignment checker must warn on material mismatches, pass on small diffs."""

    def test_no_warnings_when_aligned(self):
        """No warnings when label and exec params are similar."""
        # 24 bars * 15min = 6h horizon vs 4h timeout = ratio 1.5 (within 2x threshold)
        warnings = check_label_execution_alignment(
            label_tp=0.030, label_sl=0.015, label_horizon_bars=24,
            exec_tp=0.025,  exec_sl=0.012, exec_timeout_hours=4.0,
        )
        assert len(warnings) == 0

    def test_tp_mismatch_warns(self):
        """Large TP ratio should produce a warning."""
        warnings = check_label_execution_alignment(
            label_tp=0.060, label_sl=0.015, label_horizon_bars=48,
            exec_tp=0.012,  exec_sl=0.012, exec_timeout_hours=4.0,
        )
        assert any("label_tp" in w for w in warnings)

    def test_sl_mismatch_warns(self):
        """Large SL ratio should produce a warning."""
        warnings = check_label_execution_alignment(
            label_tp=0.030, label_sl=0.050, label_horizon_bars=48,
            exec_tp=0.025,  exec_sl=0.007, exec_timeout_hours=4.0,
        )
        assert any("label_sl" in w for w in warnings)

    def test_horizon_mismatch_warns(self):
        """Large horizon ratio should produce a warning."""
        warnings = check_label_execution_alignment(
            label_tp=0.030, label_sl=0.015, label_horizon_bars=288,   # 24h
            exec_tp=0.025,  exec_sl=0.012, exec_timeout_hours=2.0,    # 2h
        )
        assert any("horizon" in w.lower() for w in warnings)

    def test_logger_called_for_warnings(self):
        """Logger must be called once per warning."""
        msgs = []
        check_label_execution_alignment(
            label_tp=0.060, label_sl=0.015, label_horizon_bars=48,
            exec_tp=0.012,  exec_sl=0.012, exec_timeout_hours=4.0,
            logger=msgs.append,
        )
        assert len(msgs) >= 1

    def test_returns_list(self):
        warnings = check_label_execution_alignment(
            label_tp=0.030, label_sl=0.015, label_horizon_bars=48,
            exec_tp=0.025,  exec_sl=0.012, exec_timeout_hours=4.0,
        )
        assert isinstance(warnings, list)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Version constants
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionConstants:
    def test_model_version_is_string(self):
        assert isinstance(MODEL_VERSION, str) and len(MODEL_VERSION) > 0

    def test_feature_version_is_string(self):
        assert isinstance(FEATURE_VERSION, str) and len(FEATURE_VERSION) > 0

    def test_label_version_is_string(self):
        assert isinstance(LABEL_VERSION, str) and len(LABEL_VERSION) > 0

    def test_feature_version_matches_feature_count(self):
        """FEATURE_VERSION description should mention v4 (FEATURE_COUNT=20)."""
        assert "v4" in FEATURE_VERSION

    def test_versions_stored_in_fitted_ensemble(self):
        """After fit(), VoxEnsemble should carry version stamps."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, FEATURE_COUNT))
        y = rng.integers(0, 2, size=60)
        ens = VoxEnsemble(shadow_lab_enabled=False)
        ens.fit(X, y)
        assert getattr(ens, "_model_version",   None) == MODEL_VERSION
        assert getattr(ens, "_feature_version", None) == FEATURE_VERSION
        assert getattr(ens, "_label_version",   None) == LABEL_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Voting weights: LR excluded from active voting
# ─────────────────────────────────────────────────────────────────────────────

class TestVotingWeights:
    """LR (index 0) must have weight 0 in CLASSIFIER_WEIGHTS."""

    def test_lr_weight_is_zero(self):
        """CLASSIFIER_WEIGHTS[0] (LR) must be 0.0."""
        assert CLASSIFIER_WEIGHTS[0] == 0.0, (
            f"LR voting weight should be 0.0, got {CLASSIFIER_WEIGHTS[0]}"
        )

    def test_hgbc_has_highest_weight(self):
        """HGBC (index 1) should have the highest weight."""
        assert CLASSIFIER_WEIGHTS[1] == max(CLASSIFIER_WEIGHTS), (
            "HGBC should have the largest classifier weight"
        )

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0 (ensures proper normalisation)."""
        assert abs(sum(CLASSIFIER_WEIGHTS) - 1.0) < 1e-9, (
            f"Weights should sum to 1.0, got {sum(CLASSIFIER_WEIGHTS)}"
        )

    def test_four_weights(self):
        """There must be exactly 4 weights (lr, hgbc, et, rf)."""
        assert len(CLASSIFIER_WEIGHTS) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — RiskManager rolling window
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerRollingWindow:
    """rolling_sl_count must correctly count SL exits in the trailing window."""

    def _make_rm(self):
        return RiskManager(
            max_daily_sl=5, cooldown_mins=30, sl_cooldown_mins=60,
            max_dd_pct=0.15, cash_buffer=0.95,
        )

    def test_initial_rolling_count_is_zero(self):
        rm = self._make_rm()
        now = datetime(2024, 1, 1, 12, 0, 0)
        assert rm.rolling_sl_count(now, window_hours=2) == 0

    def test_sl_exit_increments_rolling(self):
        rm = self._make_rm()
        sym = MagicMock()
        t = datetime(2024, 1, 1, 12, 0, 0)
        rm.record_exit(sym, is_sl=True, exit_time=t)
        assert rm.rolling_sl_count(t + timedelta(minutes=5), window_hours=2) == 1

    def test_non_sl_exit_does_not_increment_rolling(self):
        """EXIT_BE (is_sl=False) must not increment rolling risk count."""
        rm = self._make_rm()
        sym = MagicMock()
        t = datetime(2024, 1, 1, 12, 0, 0)
        rm.record_exit(sym, is_sl=False, exit_time=t)  # breakeven
        assert rm.rolling_sl_count(t + timedelta(minutes=5), window_hours=2) == 0

    def test_old_sl_exits_outside_window_not_counted(self):
        rm = self._make_rm()
        sym = MagicMock()
        t_old = datetime(2024, 1, 1, 9, 0, 0)   # 3 hours ago
        t_now = datetime(2024, 1, 1, 12, 0, 0)
        rm.record_exit(sym, is_sl=True, exit_time=t_old)
        assert rm.rolling_sl_count(t_now, window_hours=2) == 0  # outside 2h window

    def test_multiple_sl_exits_counted_correctly(self):
        rm = self._make_rm()
        sym = MagicMock()
        base = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(3):
            rm.record_exit(sym, is_sl=True, exit_time=base + timedelta(minutes=i*20))
        assert rm.rolling_sl_count(base + timedelta(hours=1), window_hours=2) == 3

    def test_risk_exit_log_attribute_exists(self):
        rm = self._make_rm()
        assert hasattr(rm, "_risk_exit_log")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Position-count size discount helper
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionCountSizeMultiplier:
    def test_zero_positions_returns_one(self):
        assert position_count_size_multiplier(0) == pytest.approx(1.0, abs=1e-9)

    def test_one_position_returns_approx_0_707(self):
        assert position_count_size_multiplier(1) == pytest.approx(1 / math.sqrt(2), abs=1e-9)

    def test_three_positions_returns_approx_0_5(self):
        assert position_count_size_multiplier(3) == pytest.approx(0.5, abs=1e-9)

    def test_returns_between_0_and_1(self):
        for n in range(0, 10):
            m = position_count_size_multiplier(n)
            assert 0 < m <= 1.0

    def test_strictly_decreasing(self):
        mults = [position_count_size_multiplier(n) for n in range(5)]
        for a, b in zip(mults, mults[1:]):
            assert a > b

    def test_negative_treated_as_zero(self):
        """Negative open_positions should not crash; treated as 0."""
        m = position_count_size_multiplier(-1)
        assert m == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Kraken backtest guard
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchKrakenTop20UsdBacktestGuard:
    """fetch_kraken_top20_usd must return static list in non-live mode."""

    def _make_algo(self, live_mode):
        algo = MagicMock()
        algo.live_mode = live_mode
        algo.log = MagicMock()
        return algo

    def test_returns_static_list_in_backtest(self):
        algo = self._make_algo(live_mode=False)
        result = fetch_kraken_top20_usd(algo)
        assert result == list(KRAKEN_PAIRS), (
            "Must return static KRAKEN_PAIRS in backtest to avoid look-ahead bias"
        )

    def test_logs_warning_in_backtest(self):
        algo = self._make_algo(live_mode=False)
        fetch_kraken_top20_usd(algo)
        algo.log.assert_called()   # must log something

    def test_no_download_call_in_backtest(self):
        """Must NOT call algorithm.download() in backtest mode."""
        algo = self._make_algo(live_mode=False)
        fetch_kraken_top20_usd(algo)
        algo.download.assert_not_called()

    def test_live_mode_attempts_api_call(self):
        """In live mode, algorithm.download() should be called."""
        algo = self._make_algo(live_mode=True)
        # Simulate a network failure so we fall back to static list
        algo.download.side_effect = Exception("network error")
        result = fetch_kraken_top20_usd(algo)
        algo.download.assert_called()
        # Fall back to static list on error
        assert result == list(KRAKEN_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — VoxEnsemble load_state version checks
# ─────────────────────────────────────────────────────────────────────────────

class TestVoxEnsembleVersionChecks:
    def _fitted_ensemble(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, FEATURE_COUNT))
        y = rng.integers(0, 2, size=60)
        ens = VoxEnsemble(shadow_lab_enabled=False)
        ens.fit(X, y)
        return ens

    def test_load_state_accepts_same_version(self):
        ens = self._fitted_ensemble()
        fresh = VoxEnsemble(shadow_lab_enabled=False)
        fresh.load_state(ens)
        assert fresh.is_fitted

    def test_load_state_rejects_wrong_feature_count(self):
        ens = self._fitted_ensemble()
        ens._feature_count = 10   # simulate old model
        fresh = VoxEnsemble(shadow_lab_enabled=False)
        fresh.load_state(ens)
        assert not fresh.is_fitted   # should be rejected

    def test_load_state_rejects_wrong_model_version(self):
        ens = self._fitted_ensemble()
        ens._model_version = "v0.0"   # simulate incompatible old version
        fresh = VoxEnsemble(shadow_lab_enabled=False)
        fresh.load_state(ens)
        assert not fresh.is_fitted   # should be rejected

    def test_load_state_warns_on_feature_version_mismatch(self):
        ens = self._fitted_ensemble()
        ens._feature_version = "v0.0"   # simulate soft mismatch
        messages = []
        fresh = VoxEnsemble(shadow_lab_enabled=False)
        fresh._logger = messages.append
        fresh.load_state(ens)
        # Soft mismatch: accepted but warned
        assert fresh.is_fitted
        assert any("FEATURE_VERSION" in m for m in messages)
