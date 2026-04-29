"""Momentum override and scoring helpers for Vox aggressive/ruthless mode."""
import numpy as np


def check_momentum_override_conditions(feat, ret4_min, ret16_min, vol_min, btc_rel_min):
    """Return True if momentum breakout override conditions are met.

    Feature layout: [ret_1, ret_4, ret_8, ret_16, rsi_14, atr_n,
                     vol_r, btc_rel, hour, ...]
    """
    return (
        float(feat[1]) >= ret4_min
        and float(feat[3]) >= ret16_min
        and float(feat[6]) >= vol_min
        and float(feat[7]) >= btc_rel_min
    )


def compute_momentum_score(feat):
    """Compute bounded momentum score for aggressive/ruthless ranking.

    Combines ret_4, ret_16, normalised volume excess and btc_rel.
    Volume excess is normalised to [0, 1] and capped to prevent explosion.
    Returns a float clipped to [-0.05, 0.10].
    """
    vol_excess = min(max(float(feat[6]) - 1.0, 0.0), 4.0) / 4.0  # [0, 1]
    raw = (
        0.40 * float(feat[1])    # ret_4
        + 0.30 * float(feat[3])  # ret_16
        + 0.20 * vol_excess      # normalised volume spike
        + 0.10 * float(feat[7])  # btc_rel
    )
    return float(np.clip(raw, -0.05, 0.10))
