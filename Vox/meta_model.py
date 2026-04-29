"""
Lightweight meta-filter / veto model for Vox ruthless entries.

Uses a rules-based meta-score to veto low-conviction entry signals
before committing large ruthless allocations.
"""
import numpy as np


class MetaFilter:
    """Rules-based meta-filter that vets entry candidates.

    Given base model signals and contextual features, computes a
    meta-score in [0, 1] and vetoes entries that fall below a threshold.

    The meta-score combines:
      - Model confidence (class_proba, n_agree, std_proba)
      - Expected value (ev_score)
      - Short-term momentum (ret_4, ret_16)
      - Volume confirmation (volume_ratio)
      - Market mode alignment
    """

    def __init__(self, min_proba=0.55, enabled=True):
        self.min_proba = min_proba
        self.enabled   = enabled

    def compute_score(
        self,
        class_proba,
        ev_score,
        n_agree,
        std_proba,
        pred_return,
        feat,
        market_mode=None,
        ruthless_allowed_modes=None,
    ):
        """Compute meta-score in [0, 1].

        Parameters
        ----------
        class_proba   : float — weighted ensemble probability
        ev_score      : float — expected value after costs
        n_agree       : int   — number of agreeing models
        std_proba     : float — standard deviation of model probabilities
        pred_return   : float — regressor ensemble return prediction
        feat          : array-like — feature vector (at least 7 elements)
        market_mode   : str or None — current detected market mode
        ruthless_allowed_modes : list[str] or None

        Returns
        -------
        float  — meta-score in [0, 1]
        """
        score = 0.0

        # Confidence component (0–0.35)
        score += min(0.35, class_proba * 0.35 / 0.65)

        # Model agreement bonus (0–0.20)
        if n_agree >= 3:
            score += 0.20
        elif n_agree >= 2:
            score += 0.10

        # Dispersion penalty
        score -= min(0.15, std_proba * 0.5)

        # EV component (0–0.20)
        score += min(0.20, max(0.0, ev_score * 20.0))

        # Momentum confirmation (0–0.15) from feat
        if feat is not None and len(feat) >= 7:
            ret4  = float(feat[1])
            ret16 = float(feat[3])
            vol_r = float(feat[6])
            if ret4 > 0.01 and ret16 > 0.02:
                score += 0.10
            elif ret4 > 0.005:
                score += 0.05
            if vol_r > 1.5:
                score += 0.05

        # Market mode alignment (0–0.10)
        if market_mode is not None:
            allowed = ruthless_allowed_modes or ["risk_on_trend", "pump"]
            if market_mode in allowed:
                score += 0.10
            elif market_mode in ("chop", "selloff"):
                score -= 0.10

        return float(np.clip(score, 0.0, 1.0))

    def approve(
        self,
        class_proba,
        ev_score,
        n_agree,
        std_proba,
        pred_return,
        feat,
        market_mode=None,
        ruthless_allowed_modes=None,
    ):
        """Returns (approved: bool, meta_score: float).

        When disabled, always returns (True, 1.0).
        """
        if not self.enabled:
            return True, 1.0
        score = self.compute_score(
            class_proba, ev_score, n_agree, std_proba,
            pred_return, feat, market_mode, ruthless_allowed_modes,
        )
        return score >= self.min_proba, score
