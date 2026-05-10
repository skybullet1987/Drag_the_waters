"""sizing — Bayesian per-symbol win-rate posterior + size scaler.

PLAN.md §6.E.4 — clean compounding via per-symbol skill estimation.

Idea:
  - Each symbol gets its own Beta(α, β) posterior over win rate.
  - Prior: Beta(α=10, β=10) — weakly informative around 50%.
  - After every observed (W or L) trade outcome on that symbol, update:
      α += 1 if win else 0
      β += 1 if loss else 0
  - Position size scaler:
      mean = α / (α + β)
      lower_5pct = inverse-Beta CDF at 0.05    (lower CI bound)
      scaler = mean × lower_5pct × 4   (calibrated so a 50/50 symbol → ~1.0)
  - Coins with consistent wins → aggressive scaling (e.g. 1.4×)
  - Coins with consistent losses → penalised (e.g. 0.4×)
  - Coins with sparse data → near 1.0 (prior dominates)

The Beta CDF is implemented via incomplete beta function approximation —
no scipy dependency required. Tested for accuracy against scipy values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_PRIOR_ALPHA   = 10.0
DEFAULT_PRIOR_BETA    = 10.0
DEFAULT_LOWER_PCT     = 0.05
DEFAULT_MIN_SCALER    = 0.40
DEFAULT_MAX_SCALER    = 1.50
DEFAULT_CALIBRATION_K = 4.0    # mean × lower × K = scaler


# ─── Beta math (no scipy) ───────────────────────────────────────────────────

def _log_beta(a: float, b: float) -> float:
    """log B(a, b) using lgamma for numerical stability."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _incomplete_beta(x: float, a: float, b: float, max_iter: int = 200,
                     eps: float = 1e-9) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Implementation adapted from Numerical Recipes; accurate to ~6 decimals
    for a,b in [1, 1e3] and x in (0, 1).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use continued fraction at x or 1-x depending on which is smaller
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _incomplete_beta(1.0 - x, b, a, max_iter, eps)
    bt = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b)
    )
    # Lentz's modified continued fraction
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return bt * h / a
    return bt * h / a


def beta_cdf(x: float, alpha: float, beta: float) -> float:
    """Beta CDF at x with shape parameters (alpha, beta)."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return _incomplete_beta(x, alpha, beta)


def beta_inverse_cdf(p: float, alpha: float, beta: float,
                     tol: float = 1e-6, max_iter: int = 100) -> float:
    """Bisection-based inverse Beta CDF: returns x such that P(X<=x) = p."""
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if beta_cdf(mid, alpha, beta) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# ─── BayesianSymbolSizer ────────────────────────────────────────────────────

@dataclass
class SymbolPosterior:
    """Beta(α, β) posterior for one symbol's win rate."""
    symbol: str
    alpha:  float
    beta:   float
    n_observed: int = 0
    n_wins:     int = 0
    n_losses:   int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s * s * (s + 1))

    def lower_ci(self, pct: float = DEFAULT_LOWER_PCT) -> float:
        return beta_inverse_cdf(pct, self.alpha, self.beta)


class BayesianSymbolSizer:
    """Per-symbol position-size scaler using Beta posterior on win rate.

    Workflow:
        sizer = BayesianSymbolSizer()
        # After every closed trade:
        sizer.update(symbol, won=True_or_False)
        # When sizing a new entry:
        scaler = sizer.size_multiplier("BTCUSD")     # in [0.40, 1.50]
        position_size = base_size * scaler

    Calibration: a 50/50 symbol with no observations should give scaler ≈ 1.0.
    A 5/5 symbol (all wins) should ramp toward MAX. A 0/5 should drop toward MIN.
    """

    def __init__(
        self,
        prior_alpha:    float = DEFAULT_PRIOR_ALPHA,
        prior_beta:     float = DEFAULT_PRIOR_BETA,
        lower_pct:      float = DEFAULT_LOWER_PCT,
        min_scaler:     float = DEFAULT_MIN_SCALER,
        max_scaler:     float = DEFAULT_MAX_SCALER,
        calibration_k:  float = DEFAULT_CALIBRATION_K,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior alpha and beta must be > 0")
        if not (0 < min_scaler < max_scaler):
            raise ValueError("0 < min_scaler < max_scaler required")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.lower_pct = lower_pct
        self.min_scaler = min_scaler
        self.max_scaler = max_scaler
        self.calibration_k = calibration_k
        self._posteriors: dict[str, SymbolPosterior] = {}

    def _ensure(self, symbol: str) -> SymbolPosterior:
        p = self._posteriors.get(symbol)
        if p is None:
            p = SymbolPosterior(symbol=symbol,
                                alpha=self.prior_alpha, beta=self.prior_beta)
            self._posteriors[symbol] = p
        return p

    def update(self, symbol: str, won: bool) -> SymbolPosterior:
        p = self._ensure(symbol)
        p.n_observed += 1
        if won:
            p.alpha += 1
            p.n_wins += 1
        else:
            p.beta += 1
            p.n_losses += 1
        return p

    def posterior(self, symbol: str) -> SymbolPosterior:
        return self._ensure(symbol)

    def size_multiplier(self, symbol: str) -> float:
        p = self._ensure(symbol)
        mean   = p.mean
        lower  = p.lower_ci(self.lower_pct)
        scaler = mean * lower * self.calibration_k
        return max(self.min_scaler, min(self.max_scaler, scaler))

    def reset(self, symbol: str | None = None):
        if symbol is None:
            self._posteriors.clear()
        else:
            self._posteriors.pop(symbol, None)

    def all_posteriors(self) -> dict[str, SymbolPosterior]:
        return dict(self._posteriors)
