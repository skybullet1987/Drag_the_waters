"""optimal_execution — Almgren-Chriss order-slicing for larger entries.

When a high-conviction entry exceeds a tier's per-order capacity (e.g. a
$5K BTC position that would move price 15-30bp if dumped at once), this
module computes a slice schedule that minimizes:

    cost = E[total_slippage] + risk_aversion × Var[total_slippage]

The classic Almgren-Chriss (2000) closed-form solution gives an
exponentially-decaying schedule when the trader has positive risk
aversion. With risk_aversion=0, it degenerates to TWAP (equal slices).

Inputs Pulse needs:
  - target_quantity: total qty to acquire
  - n_slices: number of child orders (typically 3-10)
  - bar_volume_estimate: rolling avg of recent bar volumes (units, not USD)
  - risk_aversion: λ in the original paper (0 = TWAP; higher = front-load)
  - permanent_impact, temporary_impact: linear-impact coefficients

Output:
  Slice plan: list of (slice_idx, quantity, expected_slip_bps).

Pure-Python — no scipy/numpy hard dependency. Trivially testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_N_SLICES                 = 5
DEFAULT_RISK_AVERSION            = 1e-4
DEFAULT_PERMANENT_IMPACT_COEFF   = 0.10        # gamma — fraction of bar volume
DEFAULT_TEMPORARY_IMPACT_COEFF   = 0.30        # eta — instantaneous impact
DEFAULT_MIN_SLICE_QTY_FRAC       = 0.05        # floor each slice at 5% of total
DEFAULT_MAX_SLICE_QTY_FRAC       = 0.50        # ceiling each slice at 50%
DEFAULT_LARGE_ORDER_THRESHOLD_BPS = 25         # only slice when expected slip ≥ 25bp


# ─── Output dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SliceStep:
    """One step in the slice plan."""
    idx:                int
    quantity:           float
    qty_fraction:       float          # quantity / total
    expected_slip_bps:  float


@dataclass
class SlicePlan:
    """Full slice schedule + diagnostics."""
    total_quantity:        float
    n_slices:              int
    steps:                 list[SliceStep] = field(default_factory=list)
    expected_total_slip_bps: float = 0.0
    twap_slip_bps:         float = 0.0   # what TWAP would have cost
    saved_bps:             float = 0.0   # twap - actual
    skipped_reason:        str | None = None    # set when no slicing applied

    def to_dict(self) -> dict:
        return {
            "total_quantity":        self.total_quantity,
            "n_slices":              self.n_slices,
            "expected_total_slip_bps": round(self.expected_total_slip_bps, 2),
            "twap_slip_bps":         round(self.twap_slip_bps, 2),
            "saved_bps":             round(self.saved_bps, 2),
            "skipped_reason":        self.skipped_reason,
            "steps": [
                {"idx": s.idx, "qty": s.quantity,
                 "frac": round(s.qty_fraction, 4),
                 "slip_bps": round(s.expected_slip_bps, 2)}
                for s in self.steps
            ],
        }


# ─── Core math ──────────────────────────────────────────────────────────────

def _slip_bps_for_slice(qty: float, bar_volume: float,
                        eta: float, gamma: float) -> float:
    """Expected slippage in bps for one child order against given bar volume.

        slip_pct = eta × (qty / bar_volume) + gamma × (qty / bar_volume)^2

    Returns slip in bps.
    """
    if bar_volume <= 0 or qty <= 0:
        return 0.0
    participation = qty / bar_volume
    slip_pct = eta * participation + gamma * (participation ** 2)
    return slip_pct * 10_000


def twap_total_slip_bps(total_qty: float, n_slices: int,
                        bar_volume: float, eta: float, gamma: float) -> float:
    """Total slippage if we just do equal TWAP slices."""
    if n_slices <= 0:
        return 0.0
    per_slice = total_qty / n_slices
    return _slip_bps_for_slice(per_slice, bar_volume, eta, gamma) * n_slices


def almgren_chriss_schedule(
    total_qty: float,
    n_slices: int,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    eta: float = DEFAULT_TEMPORARY_IMPACT_COEFF,
    gamma: float = DEFAULT_PERMANENT_IMPACT_COEFF,
    sigma: float = 0.005,    # per-slice price volatility (decimal)
) -> list[float]:
    """Compute the Almgren-Chriss optimal liquidation/acquisition schedule.

    The closed-form solution gives an exponentially-decaying remaining
    inventory:

        x_k = X * sinh(κ(N-k)/N) / sinh(κ)

    where κ = sqrt(λ × σ² / η) is the urgency parameter.

    Returns list of N child-order quantities summing to total_qty.

    Special case: risk_aversion → 0 → linear schedule (TWAP).
    """
    if n_slices <= 0:
        return []
    if n_slices == 1 or risk_aversion <= 0 or eta <= 0:
        # TWAP (equal slices)
        return [total_qty / n_slices] * n_slices

    # Compute kappa
    try:
        kappa = math.sqrt(risk_aversion * (sigma ** 2) / eta)
    except (ValueError, ZeroDivisionError):
        return [total_qty / n_slices] * n_slices

    if kappa <= 0:
        return [total_qty / n_slices] * n_slices

    # Inventory at each step k = 0..N: x_k = X × sinh(κ(N-k)/N) / sinh(κ)
    # Slice quantity = x_{k-1} - x_k for k=1..N
    sinh_kappa = math.sinh(kappa)
    if sinh_kappa <= 0:
        return [total_qty / n_slices] * n_slices

    inventory = []
    for k in range(n_slices + 1):
        x_k = total_qty * math.sinh(kappa * (n_slices - k) / n_slices) / sinh_kappa
        inventory.append(x_k)

    slices = [inventory[k] - inventory[k + 1] for k in range(n_slices)]
    # Numerical correction: ensure sum equals total_qty
    actual_sum = sum(slices)
    if actual_sum > 0:
        scale = total_qty / actual_sum
        slices = [s * scale for s in slices]
    return slices


def build_slice_plan(
    total_quantity: float,
    bar_volume_estimate: float,
    *,
    n_slices: int = DEFAULT_N_SLICES,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    eta: float = DEFAULT_TEMPORARY_IMPACT_COEFF,
    gamma: float = DEFAULT_PERMANENT_IMPACT_COEFF,
    sigma: float = 0.005,
    min_slice_frac: float = DEFAULT_MIN_SLICE_QTY_FRAC,
    max_slice_frac: float = DEFAULT_MAX_SLICE_QTY_FRAC,
    large_order_threshold_bps: float = DEFAULT_LARGE_ORDER_THRESHOLD_BPS,
    min_qty_per_slice: float = 0.0,
) -> SlicePlan:
    """Top-level: returns a SlicePlan with quantities + expected slippage.

    Skip slicing entirely when:
      - total_quantity ≤ 0 or n_slices ≤ 1
      - bar_volume_estimate ≤ 0 (no impact estimate available)
      - expected one-shot slippage < large_order_threshold_bps
      - total_quantity < n_slices × min_qty_per_slice  (would create
        sub-minimum child orders that the exchange rejects — the bug
        observed in QC backtest where 5 BCH slices were all invalid)
    """
    if total_quantity <= 0:
        return SlicePlan(total_quantity=total_quantity, n_slices=0,
                         skipped_reason="zero_or_negative_quantity")
    if n_slices <= 1:
        return SlicePlan(total_quantity=total_quantity, n_slices=1,
                         steps=[SliceStep(0, total_quantity, 1.0,
                                          _slip_bps_for_slice(
                                              total_quantity, bar_volume_estimate,
                                              eta, gamma))],
                         skipped_reason="n_slices<=1")
    if bar_volume_estimate <= 0:
        return SlicePlan(total_quantity=total_quantity, n_slices=1,
                         steps=[SliceStep(0, total_quantity, 1.0, 0.0)],
                         skipped_reason="no_volume_estimate")

    # Min-qty-per-slice guard: if slicing would create child orders below
    # the exchange minimum, fall back to a single one-shot order so we
    # don't get N invalid rejections.
    if min_qty_per_slice > 0 and total_quantity < n_slices * min_qty_per_slice:
        one_shot_slip = _slip_bps_for_slice(
            total_quantity, bar_volume_estimate, eta, gamma,
        )
        return SlicePlan(total_quantity=total_quantity, n_slices=1,
                         steps=[SliceStep(0, total_quantity, 1.0, one_shot_slip)],
                         expected_total_slip_bps=one_shot_slip,
                         twap_slip_bps=one_shot_slip,
                         saved_bps=0.0,
                         skipped_reason="below_min_qty_per_slice")

    one_shot_slip = _slip_bps_for_slice(
        total_quantity, bar_volume_estimate, eta, gamma,
    )
    if one_shot_slip < large_order_threshold_bps:
        return SlicePlan(total_quantity=total_quantity, n_slices=1,
                         steps=[SliceStep(0, total_quantity, 1.0, one_shot_slip)],
                         expected_total_slip_bps=one_shot_slip,
                         twap_slip_bps=one_shot_slip,
                         saved_bps=0.0,
                         skipped_reason="below_large_order_threshold")

    # Compute schedule
    raw_qtys = almgren_chriss_schedule(
        total_quantity, n_slices,
        risk_aversion=risk_aversion, eta=eta, gamma=gamma, sigma=sigma,
    )

    # Apply min/max envelope per slice (iterative cap+redistribute so the
    # cap actually holds — naive renormalize would push capped slices back
    # above the ceiling).
    min_q = total_quantity * min_slice_frac
    max_q = total_quantity * max_slice_frac
    capped = list(raw_qtys)
    for _ in range(10):
        # Step 1: clip each
        clipped = [max(min_q, min(max_q, q)) for q in capped]
        deficit = total_quantity - sum(clipped)
        if abs(deficit) < 1e-9:
            capped = clipped
            break
        # Step 2: distribute the deficit only to slices that have headroom
        if deficit > 0:
            # Need MORE qty — give only to non-ceiling slices
            room = [(i, max_q - q) for i, q in enumerate(clipped) if q < max_q - 1e-9]
            total_room = sum(r for _, r in room)
            if total_room <= 0:
                capped = clipped
                break
            for i, r in room:
                clipped[i] += deficit * (r / total_room)
        else:
            # Need LESS qty — take only from non-floor slices
            slack = [(i, q - min_q) for i, q in enumerate(clipped) if q > min_q + 1e-9]
            total_slack = sum(s for _, s in slack)
            if total_slack <= 0:
                capped = clipped
                break
            for i, sl in slack:
                clipped[i] += deficit * (sl / total_slack)   # deficit < 0
        capped = clipped

    steps = []
    total_slip = 0.0
    for i, q in enumerate(capped):
        slip = _slip_bps_for_slice(q, bar_volume_estimate, eta, gamma)
        total_slip += slip
        steps.append(SliceStep(
            idx=i, quantity=q, qty_fraction=q / total_quantity,
            expected_slip_bps=slip,
        ))

    twap_slip = twap_total_slip_bps(
        total_quantity, n_slices, bar_volume_estimate, eta, gamma,
    )
    return SlicePlan(
        total_quantity=total_quantity,
        n_slices=n_slices,
        steps=steps,
        expected_total_slip_bps=total_slip,
        twap_slip_bps=twap_slip,
        saved_bps=twap_slip - total_slip,
    )
