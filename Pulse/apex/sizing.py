"""Pulse.apex.sizing — probability → position size.

Translation chain:
    prob (0..1) ──► edge_pct ──► raw_kelly_fraction ──► position_usd

Where:
    edge_pct        = expected_return_if_correct * (2*prob - 1)
    raw_kelly_frac  = edge_pct / variance_estimate
    safe_kelly_frac = raw_kelly_frac × APEX_KELLY_FRACTION (deratings)
    position_usd    = available_equity × safe_kelly_frac × regime_mult
                       capped at tier_max_pos_usd
                       floored at APEX_MIN_POSITION_USD (else skip entry)

The Kelly derating (default 0.25) is the SAFETY MULTIPLIER. Even at
"full Kelly" the expected log growth is maximized — but variance kills
you. Quarter-Kelly is conventional retail wisdom and survives heavy DD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from Pulse.apex.config import (
    APEX_KELLY_FRACTION,
    APEX_MIN_POSITION_USD,
    APEX_ENTRY_THRESHOLD,
)


# ─── Tunables (overridable per-call) ────────────────────────────────────────

DEFAULT_TARGET_RETURN = 0.025      # +2.5% expected for "correct" prediction
DEFAULT_VARIANCE_EST  = 0.04 ** 2  # daily variance proxy — 4% annualized vol-ish


# ─── Sizing primitives ──────────────────────────────────────────────────────


def edge_pct_from_prob(prob: float, target_return: float = DEFAULT_TARGET_RETURN
                       ) -> float:
    """Expected % return from the probability of a correct call.

    Convention: model says P(forward_return > +1.5%). When prob is the
    odds of upside, the symmetric expected return is:

        E[r] = prob * (+target) - (1 - prob) * (target)
             = target * (2 * prob - 1)

    Returns 0 when prob == 0.5 (no edge).
    """
    return target_return * (2.0 * float(prob) - 1.0)


def raw_kelly_fraction(prob: float, *,
                       target_return: float = DEFAULT_TARGET_RETURN,
                       variance: float = DEFAULT_VARIANCE_EST) -> float:
    """Kelly-optimal fraction = edge / variance.

    Returns 0 when prob ≤ 0.5 (no edge).
    """
    if prob <= 0.5:
        return 0.0
    edge = edge_pct_from_prob(prob, target_return)
    if variance <= 0:
        return 0.0
    return max(0.0, edge / variance)


def safe_kelly_fraction(prob: float, *,
                         target_return: float = DEFAULT_TARGET_RETURN,
                         variance: float = DEFAULT_VARIANCE_EST,
                         derate: float = APEX_KELLY_FRACTION) -> float:
    """Quarter-Kelly (or whatever derate) — capped at 1.0 to prevent leverage."""
    raw = raw_kelly_fraction(prob, target_return=target_return,
                              variance=variance)
    return min(1.0, raw * derate)


# ─── Final size composition ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SizeDecision:
    """Output of compute_position_usd. None of the fields are clamped to
    integer lots — that's the execution layer's job."""
    position_usd:  float
    safe_kelly:    float
    raw_kelly:     float
    edge_pct:      float
    regime_mult:   float
    tier_cap_usd:  float
    skip_reason:   Optional[str]   # set when sizing returned 0


def compute_position_usd(
    prob: float,
    *,
    available_equity:       float,
    tier_max_pos_usd:       float,
    regime_size_mult:       float = 1.0,
    target_return:          float = DEFAULT_TARGET_RETURN,
    variance:               float = DEFAULT_VARIANCE_EST,
    derate:                 float = APEX_KELLY_FRACTION,
    min_position_usd:       float = APEX_MIN_POSITION_USD,
    entry_threshold:        float = APEX_ENTRY_THRESHOLD,
) -> SizeDecision:
    """Compose probability + capital + regime into a USD position size."""
    if prob < entry_threshold:
        return SizeDecision(0.0, 0.0, 0.0, 0.0,
                            regime_size_mult, tier_max_pos_usd,
                            skip_reason=f"prob<{entry_threshold}")
    raw_k = raw_kelly_fraction(prob, target_return=target_return,
                                variance=variance)
    safe_k = min(1.0, raw_k * derate)
    if safe_k <= 0:
        return SizeDecision(0.0, safe_k, raw_k, 0.0,
                            regime_size_mult, tier_max_pos_usd,
                            skip_reason="zero_kelly")
    proposed = available_equity * safe_k * float(regime_size_mult)
    capped = min(proposed, tier_max_pos_usd)
    if capped < min_position_usd:
        return SizeDecision(0.0, safe_k, raw_k,
                            edge_pct_from_prob(prob, target_return),
                            regime_size_mult, tier_max_pos_usd,
                            skip_reason=f"below_min_position_usd({min_position_usd})")
    return SizeDecision(
        position_usd=capped,
        safe_kelly=safe_k, raw_kelly=raw_k,
        edge_pct=edge_pct_from_prob(prob, target_return),
        regime_mult=float(regime_size_mult),
        tier_cap_usd=tier_max_pos_usd,
        skip_reason=None,
    )
