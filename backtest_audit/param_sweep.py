"""param_sweep — Latin hypercube parameter sampler for Phase 3 walk-forward.

PLAN.md §5.1 — generates a near-uniform sample of the 12-dimensional
Pulse parameter space without exhaustive grid search (would be 3^12
= 531,441 combos; LHS gets us comparable coverage in 50-200 samples).

The 12 swept parameters and their ranges (from PLAN.md §5.1):

| Param | Low | Mid | High |
|---|---|---|---|
| entry_threshold              | 0.50  | 0.55 | 0.60 |
| high_conviction_threshold    | 0.65  | 0.70 | 0.75 |
| quick_take_profit            | 0.08  | 0.12 | 0.18 |
| tight_stop_loss              | 0.025 | 0.035| 0.05 |
| atr_tp_mult                  | 3.0   | 4.0  | 5.0 |
| atr_sl_mult                  | 1.5   | 2.0  | 2.5 |
| trail_activation             | 0.025 | 0.04 | 0.06 |
| trail_stop_pct               | 0.020 | 0.025| 0.035 |
| time_stop_hours              | 2     | 3    | 4 |
| max_positions                | 4     | 6    | 8 |
| max_position_usd             | 250   | 500  | 1000 |
| target_position_ann_vol      | 0.25  | 0.35 | 0.45 |

Output is a list of dicts with these keys; each dict can be used as
parameter overrides for a Pulse backtest run. Pure-Python — no numpy
or scipy dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


# ─── Default Pulse 12-dim parameter space ──────────────────────────────────

DEFAULT_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "entry_threshold":           (0.50,  0.60),
    "high_conviction_threshold": (0.65,  0.75),
    "quick_take_profit":         (0.08,  0.18),
    "tight_stop_loss":           (0.025, 0.05),
    "atr_tp_mult":               (3.0,   5.0),
    "atr_sl_mult":               (1.5,   2.5),
    "trail_activation":          (0.025, 0.06),
    "trail_stop_pct":            (0.020, 0.035),
    "time_stop_hours":           (2.0,   4.0),
    "max_positions":             (4.0,   8.0),
    "max_position_usd":          (250.0, 1000.0),
    "target_position_ann_vol":   (0.25,  0.45),
}

# Parameters that should be cast to int for QC parameter passing
INT_PARAMS = ("max_positions", "time_stop_hours")


# ─── Latin hypercube ───────────────────────────────────────────────────────

def latin_hypercube(
    n_samples: int,
    n_dims: int,
    seed: int | None = 42,
) -> list[list[float]]:
    """Generate ``n_samples`` points in the unit hypercube [0,1]^n_dims using
    Latin Hypercube Sampling.

    Each dimension is divided into n_samples equal bins; each bin contains
    exactly one sample point per dimension (orthogonal projection has full
    coverage). Within each bin, the sample is placed uniformly at random.
    Different dimensions are then permuted independently to break correlation.

    Returns a list of n_samples points, each a list of n_dims floats in [0, 1].
    """
    if n_samples < 1 or n_dims < 1:
        raise ValueError("n_samples and n_dims must be >= 1")
    rng = random.Random(seed)

    # Build bin centers + jitter per dimension
    bin_size = 1.0 / n_samples
    samples = [[0.0] * n_dims for _ in range(n_samples)]
    for d in range(n_dims):
        # Get a permutation of bin indices
        perm = list(range(n_samples))
        rng.shuffle(perm)
        for i in range(n_samples):
            jitter = rng.random()
            samples[i][d] = (perm[i] + jitter) * bin_size
    return samples


def map_to_param_space(
    unit_samples: list[list[float]],
    param_ranges: dict[str, tuple[float, float]] = DEFAULT_PARAM_RANGES,
    int_params: tuple[str, ...] = INT_PARAMS,
) -> list[dict]:
    """Map [0,1]^n unit samples into named parameter dicts.

    Order of dimensions follows the param_ranges dict insertion order.
    Integer-typed params are cast to int after scaling.
    """
    names = list(param_ranges.keys())
    if not unit_samples or len(unit_samples[0]) != len(names):
        raise ValueError(
            f"unit_samples has {len(unit_samples[0]) if unit_samples else 0} "
            f"dims but param_ranges has {len(names)}"
        )
    out: list[dict] = []
    for u in unit_samples:
        params: dict[str, float | int] = {}
        for i, name in enumerate(names):
            lo, hi = param_ranges[name]
            val = lo + u[i] * (hi - lo)
            if name in int_params:
                val = int(round(val))
            else:
                # Round float params to 6 decimals for cleaner QC parameter passing
                val = round(val, 6)
            params[name] = val
        out.append(params)
    return out


def generate_param_sweep(
    n_samples: int = 50,
    param_ranges: dict[str, tuple[float, float]] = DEFAULT_PARAM_RANGES,
    seed: int | None = 42,
) -> list[tuple[str, dict]]:
    """Top-level helper: generates `n_samples` named parameter sets ready
    for ``backtest_audit.regime_runner.sweep()``.

    Each entry is a tuple ``(param_id, params_dict)`` where param_id is
    the string ``"sweep_{i:03d}"`` and params_dict is the parameter override.
    """
    samples = latin_hypercube(n_samples=n_samples, n_dims=len(param_ranges),
                              seed=seed)
    mapped = map_to_param_space(samples, param_ranges=param_ranges)
    return [(f"sweep_{i:03d}", p) for i, p in enumerate(mapped)]


# ─── Coverage diagnostics ──────────────────────────────────────────────────

def lhs_coverage_metric(unit_samples: list[list[float]],
                         n_bins: int | None = None) -> dict:
    """Diagnostics: how well do the samples cover each dimension?

    Returns:
        {
          'n_samples': N,
          'n_dims':   D,
          'mean_bin_count_per_dim': float,
          'min_bin_count_per_dim':  int,    # 1 if perfect LHS
          'max_bin_count_per_dim':  int,    # 1 if perfect LHS
          'unique_pairs_pct':        float, # closer to 1.0 = better spread
        }
    """
    if not unit_samples:
        return {"n_samples": 0, "n_dims": 0}
    n = len(unit_samples)
    d = len(unit_samples[0])
    bins = n_bins or n
    bin_size = 1.0 / bins

    bin_counts = [[0] * bins for _ in range(d)]
    for s in unit_samples:
        for j in range(d):
            b = min(int(s[j] / bin_size), bins - 1)
            bin_counts[j][b] += 1

    flat_counts = [c for col in bin_counts for c in col]
    return {
        "n_samples":             n,
        "n_dims":                d,
        "mean_bin_count_per_dim": sum(flat_counts) / max(len(flat_counts), 1),
        "min_bin_count_per_dim": min(flat_counts),
        "max_bin_count_per_dim": max(flat_counts),
    }


# ─── Convenience: Pulse-specific helpers ───────────────────────────────────

def pulse_param_sweep(n_samples: int = 50,
                      seed: int | None = 42) -> list[tuple[str, dict]]:
    """Convenience: standard 12-dim Pulse parameter sweep with defaults."""
    return generate_param_sweep(n_samples=n_samples, seed=seed)
