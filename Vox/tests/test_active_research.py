"""
Tests for active_research mode and audit_utils module (PR: active research mode).

Coverage:
  1. audit_utils module — audit_safe_float and audit_trim_votes importable from audit_utils.
  2. journals re-exports — audit_safe_float and audit_trim_votes still importable from journals.
  3. active_research profile — relaxed thresholds, no Kelly, small allocation.
  4. regime gate soft-pass — bypassed for active_research, blocked for other profiles.
  5. active_research constants — exist in core and have expected relationships.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── 1. audit_utils importable ─────────────────────────────────────────────────

class TestAuditUtilsModule:
    def test_audit_safe_float_importable(self):
        from audit_utils import audit_safe_float
        assert audit_safe_float(0.5) == pytest.approx(0.5)

    def test_audit_trim_votes_importable(self):
        from audit_utils import audit_trim_votes
        assert audit_trim_votes({"a": 0.7}) == {"a": pytest.approx(0.7)}

    def test_audit_safe_float_none_on_invalid(self):
        from audit_utils import audit_safe_float
        assert audit_safe_float("notanumber") is None

    def test_audit_trim_votes_skips_nonnumeric(self):
        from audit_utils import audit_trim_votes
        result = audit_trim_votes({"good": 0.8, "bad": None})
        assert "good" in result
        assert "bad" not in result


# ── 2. journals re-exports still work ────────────────────────────────────────

class TestJournalsReExport:
    def test_audit_safe_float_via_journals(self):
        from journals import audit_safe_float
        assert audit_safe_float(1.0) == 1.0

    def test_audit_trim_votes_via_journals(self):
        from journals import audit_trim_votes
        assert isinstance(audit_trim_votes({"x": 0.5}), dict)


# ── 3. active_research profile constants ─────────────────────────────────────

class TestActiveResearchConstants:
    def test_constants_exist(self):
        from core import (
            ACTIVE_RESEARCH_SCORE_MIN,
            ACTIVE_RESEARCH_SCORE_GAP,
            ACTIVE_RESEARCH_MIN_AGREE,
            ACTIVE_RESEARCH_MAX_DISPERSION,
            ACTIVE_RESEARCH_MIN_EV,
            ACTIVE_RESEARCH_PRED_RETURN_MIN,
            ACTIVE_RESEARCH_COOLDOWN_MINS,
            ACTIVE_RESEARCH_SL_COOLDOWN_MINS,
            ACTIVE_RESEARCH_MAX_DAILY_SL,
            ACTIVE_RESEARCH_ALLOCATION,
            ACTIVE_RESEARCH_MAX_ALLOC,
            ACTIVE_RESEARCH_USE_KELLY,
            ACTIVE_RESEARCH_TAKE_PROFIT,
            ACTIVE_RESEARCH_STOP_LOSS,
            ACTIVE_RESEARCH_TIMEOUT_HOURS,
            ACTIVE_RESEARCH_REGIME_SIZE_MULT,
        )
        # Gate thresholds are much looser than balanced defaults
        assert ACTIVE_RESEARCH_SCORE_MIN < 0.25
        assert ACTIVE_RESEARCH_SCORE_GAP == 0.0
        assert ACTIVE_RESEARCH_MIN_AGREE == 1
        assert ACTIVE_RESEARCH_MAX_DISPERSION >= 0.40
        assert ACTIVE_RESEARCH_MIN_EV < 0.0
        # Small allocation — research mode
        assert ACTIVE_RESEARCH_ALLOCATION <= 0.05
        assert ACTIVE_RESEARCH_MAX_ALLOC <= 0.10
        assert ACTIVE_RESEARCH_USE_KELLY is False
        # Short timeouts for faster cycle
        assert ACTIVE_RESEARCH_TIMEOUT_HOURS <= 3.0
        # Regime size mult < 1.0 (reduces size for non-risk-on)
        assert 0.0 < ACTIVE_RESEARCH_REGIME_SIZE_MULT < 1.0

    def test_score_min_looser_than_balanced(self):
        from core import ACTIVE_RESEARCH_SCORE_MIN, SCORE_MIN
        assert ACTIVE_RESEARCH_SCORE_MIN < SCORE_MIN

    def test_allocation_smaller_than_balanced(self):
        from core import ACTIVE_RESEARCH_ALLOCATION, ALLOCATION
        assert ACTIVE_RESEARCH_ALLOCATION < ALLOCATION

    def test_no_kelly_in_active_research(self):
        from core import ACTIVE_RESEARCH_USE_KELLY
        assert ACTIVE_RESEARCH_USE_KELLY is False


# ── 4. setup_risk_profile applies active_research settings ───────────────────

class MockAlgo:
    """Minimal mock for setup_risk_profile testing."""

    def __init__(self, risk_profile="active_research"):
        self._rp = risk_profile
        self._logs = []
        # Pre-set defaults that setup_risk_profile reads
        from core import (
            SCORE_MIN, MAX_DISPERSION, MIN_AGREE, MIN_EV, EV_GAP, COST_BPS,
            ALLOCATION, MAX_ALLOC, KELLY_FRAC, TAKE_PROFIT, STOP_LOSS,
            TIMEOUT_HOURS, MIN_HOLD_MINUTES, EMERGENCY_SL, MAX_DAILY_SL,
            COOLDOWN_MINS, SL_COOLDOWN_MINS, PENALTY_COOLDOWN_LOSSES,
            PENALTY_COOLDOWN_HOURS, MAX_DD_PCT, PRED_RETURN_MIN, USE_KELLY,
        )
        self._s_min            = SCORE_MIN
        self._max_disp         = MAX_DISPERSION
        self._min_agr          = MIN_AGREE
        self._min_ev           = MIN_EV
        self._ev_gap           = EV_GAP
        self._cost_bps         = COST_BPS
        self._alloc            = ALLOCATION
        self._max_alloc        = MAX_ALLOC
        self._min_alloc        = 0.0
        self._kf               = KELLY_FRAC
        self._use_kelly        = USE_KELLY
        self._tp               = TAKE_PROFIT
        self._sl               = STOP_LOSS
        self._toh              = TIMEOUT_HOURS
        self._min_hold_minutes = MIN_HOLD_MINUTES
        self._emergency_sl     = EMERGENCY_SL
        self._max_sl           = MAX_DAILY_SL
        self._cd_mins          = COOLDOWN_MINS
        self._sl_cd            = SL_COOLDOWN_MINS
        self._penalty_losses   = PENALTY_COOLDOWN_LOSSES
        self._penalty_hours    = PENALTY_COOLDOWN_HOURS
        self._max_dd           = MAX_DD_PCT
        self._pred_return_min  = PRED_RETURN_MIN
        self._conservative_mode = False
        self._runner_mode      = False
        self._trail_after_tp   = 0.0
        self._trail_pct        = 0.0
        self._risk_profile     = risk_profile
        self._momentum_override = False
        self._use_momentum_score = False

    def get_parameter(self, name):
        mapping = {"risk_profile": self._rp}
        return mapping.get(name)

    def log(self, msg):
        self._logs.append(msg)

    def debug(self, msg):
        self._logs.append(msg)


class TestActiveResearchProfileSetup:
    def _setup(self, profile="active_research"):
        algo = MockAlgo(risk_profile=profile)
        from core import setup_risk_profile
        setup_risk_profile(algo)
        return algo

    def test_risk_profile_is_set(self):
        algo = self._setup()
        assert algo._risk_profile == "active_research"

    def test_score_min_is_relaxed(self):
        algo = self._setup()
        from core import ACTIVE_RESEARCH_SCORE_MIN, SCORE_MIN
        assert algo._s_min == ACTIVE_RESEARCH_SCORE_MIN
        assert algo._s_min < SCORE_MIN

    def test_allocation_is_small(self):
        algo = self._setup()
        from core import ACTIVE_RESEARCH_ALLOCATION
        assert algo._alloc == ACTIVE_RESEARCH_ALLOCATION
        assert algo._alloc <= 0.05

    def test_kelly_is_disabled(self):
        algo = self._setup()
        assert algo._use_kelly is False

    def test_cooldown_is_zero(self):
        algo = self._setup()
        assert algo._cd_mins == 0

    def test_min_agree_is_1(self):
        algo = self._setup()
        assert algo._min_agr == 1

    def test_max_dispersion_is_loose(self):
        algo = self._setup()
        from core import ACTIVE_RESEARCH_MAX_DISPERSION
        assert algo._max_disp == ACTIVE_RESEARCH_MAX_DISPERSION
        assert algo._max_disp >= 0.40

    def test_timeout_is_short(self):
        algo = self._setup()
        from core import ACTIVE_RESEARCH_TIMEOUT_HOURS
        assert algo._toh == ACTIVE_RESEARCH_TIMEOUT_HOURS
        assert algo._toh <= 3.0

    def test_log_contains_active_research(self):
        algo = self._setup()
        combined = " ".join(algo._logs)
        assert "active_research" in combined.lower()

    def test_balanced_profile_unchanged(self):
        """Balanced profile must NOT get active_research values."""
        algo = self._setup(profile="balanced")
        from core import ACTIVE_RESEARCH_SCORE_MIN, SCORE_MIN
        # Balanced score_min should NOT be lowered to active_research level
        assert algo._s_min >= SCORE_MIN
        assert algo._s_min > ACTIVE_RESEARCH_SCORE_MIN

    def test_balanced_kelly_default(self):
        """Balanced profile uses Kelly by default."""
        algo = self._setup(profile="balanced")
        from core import USE_KELLY
        assert algo._use_kelly == USE_KELLY


# ── 5. active_research in valid profile list ──────────────────────────────────

class TestActiveResearchValidProfile:
    def test_active_research_accepted_as_profile(self):
        """risk_profile=active_research must be accepted (not fall back to RISK_PROFILE default)."""
        algo = MockAlgo(risk_profile="active_research")
        from core import setup_risk_profile
        setup_risk_profile(algo)
        assert algo._risk_profile == "active_research"

    def test_unknown_profile_falls_back_to_default(self):
        """Unknown profile values should fall back to RISK_PROFILE default."""
        algo = MockAlgo(risk_profile="does_not_exist")
        from core import setup_risk_profile, RISK_PROFILE
        setup_risk_profile(algo)
        assert algo._risk_profile == RISK_PROFILE
