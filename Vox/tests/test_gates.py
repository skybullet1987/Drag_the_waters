"""
Unit tests for the Vox confidence-gate logic.

No QuantConnect dependency — runs with plain pytest + numpy + scikit-learn.
"""
import sys
import os

import numpy as np
import pytest

# Allow importing from the Vox package without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import VoxEnsemble  # type: ignore  # noqa: E402

# Mirror constants from main.py (cannot import main.py without AlgorithmImports).
SCORE_MIN       = 0.25
SCORE_MIN_FLOOR = 0.15
MIN_AGREE       = 1


def _score_min_eff(positive_rate, s_min_floor=SCORE_MIN_FLOOR, s_min=SCORE_MIN):
    """Mirror of the runtime formula in VoxAlgorithm._try_enter()."""
    return float(np.clip(max(s_min_floor, 3.0 * positive_rate), s_min_floor, s_min))


# ─────────────────────────────────────────────────────────────────────────────
# agree_threshold tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgreeThreshold:
    def _ens(self, positive_rate):
        ens = VoxEnsemble()
        ens._positive_rate = positive_rate
        return ens

    def test_low_positive_rate_gives_floor(self):
        """With positive_rate=0.03, agree_thr should floor at 0.15."""
        assert self._ens(0.03)._agree_threshold() == pytest.approx(0.15)

    def test_medium_positive_rate(self):
        """With positive_rate=0.15, agree_thr = clip(0.30, 0.15, 0.55) = 0.30."""
        assert self._ens(0.15)._agree_threshold() == pytest.approx(0.30)

    def test_high_positive_rate_clipped(self):
        """With positive_rate=0.40, agree_thr is capped at 0.55."""
        assert self._ens(0.40)._agree_threshold() == pytest.approx(0.55)

    def test_always_within_bounds(self):
        """agree_thr must always stay within [0.15, 0.55] for any positive_rate."""
        for pr in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
            thr = self._ens(pr)._agree_threshold()
            assert 0.15 <= thr <= 0.55, f"agree_thr={thr} out of bounds for pr={pr}"

    def test_prob_vector_around_020_passes_agree_gate(self):
        """
        A 5-model ensemble with probas ~0.20 must NOT be rejected by the agree
        gate when positive_rate=0.03 (reproduces the reported zero-trade bug).
        """
        ens = VoxEnsemble()
        ens._positive_rate = 0.03
        agree_thr = ens._agree_threshold()

        proba_vector = [0.18, 0.22, 0.19, 0.21, 0.17]
        n_agree = sum(1 for p in proba_vector if p >= agree_thr)

        assert agree_thr <= 0.20, (
            f"agree_thr={agree_thr} should be <= 0.20 for positive_rate=0.03"
        )
        assert n_agree >= MIN_AGREE, (
            f"n_agree={n_agree} < MIN_AGREE={MIN_AGREE} for agree_thr={agree_thr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# score_min_eff tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMinEff:
    def test_low_positive_rate_gives_floor(self):
        """With positive_rate=0.03, effective score threshold floors at 0.15."""
        assert _score_min_eff(0.03) == pytest.approx(SCORE_MIN_FLOOR)

    def test_medium_positive_rate(self):
        """With positive_rate=0.06, effective score = clip(max(0.15, 0.18), 0.15, 0.25) = 0.18."""
        assert _score_min_eff(0.06) == pytest.approx(0.18)

    def test_high_positive_rate_clipped_to_score_min(self):
        """With positive_rate=0.30, effective score is capped at SCORE_MIN=0.25."""
        assert _score_min_eff(0.30) == pytest.approx(SCORE_MIN)

    def test_always_within_bounds(self):
        """score_min_eff must always stay within [SCORE_MIN_FLOOR, SCORE_MIN]."""
        for pr in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
            eff = _score_min_eff(pr)
            assert SCORE_MIN_FLOOR <= eff <= SCORE_MIN, (
                f"score_min_eff={eff} out of [{SCORE_MIN_FLOOR}, {SCORE_MIN}] for pr={pr}"
            )

    def test_score_min_constant_remains_upper_clamp(self):
        """SCORE_MIN=0.25 is preserved as the upper bound, never exceeded."""
        assert _score_min_eff(1.0) == pytest.approx(SCORE_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# positive_rate persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveRatePersistence:
    def test_positive_rate_initialises_to_zero(self):
        ens = VoxEnsemble()
        assert ens._positive_rate == 0.0

    def test_load_state_restores_positive_rate(self):
        """load_state must copy _positive_rate from the serialised ensemble."""
        src = VoxEnsemble()
        src._positive_rate = 0.042

        dst = VoxEnsemble()
        dst.load_state(src)

        assert dst._positive_rate == pytest.approx(0.042)

    def test_load_state_defaults_missing_positive_rate(self):
        """Loading an old pickle without _positive_rate must default to 0.0."""
        src = VoxEnsemble()
        # Simulate an old pickle that lacks the attribute.
        if hasattr(src, "_positive_rate"):
            del src.__dict__["_positive_rate"]

        dst = VoxEnsemble()
        dst.load_state(src)

        assert dst._positive_rate == 0.0
