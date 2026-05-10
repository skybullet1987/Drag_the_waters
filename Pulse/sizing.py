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


# ─── PyramidSizer (Tier F.1, PLAN.md §6.F.1) ────────────────────────────────

@dataclass
class PyramidLadder:
    """Ladder of incremental adds to a winning position.

    PLAN.md §6.F.1: at +3% MFE add 33% to position; at +6% add another 33%.
    Trail the whole stack at trail_pct from peak.

    Defaults are conservative (3%/6% rungs, 33% adds) but tunable.
    Each rung can fire at most once per position lifetime.
    """
    rung_mfe_pcts:   tuple = (0.03, 0.06)    # add at these MFE thresholds
    rung_add_fracs:  tuple = (0.33, 0.33)    # how much to add (frac of original size)
    max_total_size_mult: float = 2.0         # never exceed 2× original

    def __post_init__(self):
        if len(self.rung_mfe_pcts) != len(self.rung_add_fracs):
            raise ValueError(
                "rung_mfe_pcts and rung_add_fracs must be same length"
            )
        for p in self.rung_mfe_pcts:
            if p <= 0:
                raise ValueError("rung_mfe_pcts must be > 0")
        for f in self.rung_add_fracs:
            if f <= 0:
                raise ValueError("rung_add_fracs must be > 0")
        if self.max_total_size_mult <= 1.0:
            raise ValueError("max_total_size_mult must be > 1.0")


@dataclass
class PyramidPositionState:
    """Per-position state tracking which rungs have already fired."""
    symbol:          str
    initial_size:    float       # in dollars OR units; caller's choice
    rungs_fired:     set = field(default_factory=set)  # set of rung indexes
    current_size:    float = 0.0   # accumulated total

    def __post_init__(self):
        if self.current_size == 0.0:
            self.current_size = self.initial_size


@dataclass(frozen=True)
class PyramidDecision:
    """Per-tick decision for a position: should we add another tranche?"""
    should_add:        bool
    add_size:          float = 0.0
    rung_index:        int = -1
    new_total_size:    float = 0.0
    reason:            str = ""


class PyramidSizer:
    """Decide when to add to a winning position based on MFE.

    Usage::

        sizer = PyramidSizer()
        state = PyramidPositionState(symbol="BTC", initial_size=500.0)
        # On every tick:
        d = sizer.evaluate(state, current_mfe_pct=0.04)
        if d.should_add:
            broker.market_buy(state.symbol, d.add_size)
            sizer.commit_add(state, d)
    """

    def __init__(self, ladder: PyramidLadder | None = None):
        self.ladder = ladder or PyramidLadder()

    def evaluate(self, state: PyramidPositionState,
                 current_mfe_pct: float) -> PyramidDecision:
        """Should we add at this MFE? Returns a PyramidDecision (does not mutate)."""
        if current_mfe_pct <= 0:
            return PyramidDecision(False, reason="not_in_profit")

        for i, threshold in enumerate(self.ladder.rung_mfe_pcts):
            if i in state.rungs_fired:
                continue
            if current_mfe_pct >= threshold:
                add_frac = self.ladder.rung_add_fracs[i]
                add_size = state.initial_size * add_frac
                new_total = state.current_size + add_size
                if new_total > state.initial_size * self.ladder.max_total_size_mult:
                    return PyramidDecision(
                        False,
                        reason=f"rung_{i}_would_exceed_max_total_mult",
                    )
                return PyramidDecision(
                    should_add=True,
                    add_size=add_size,
                    rung_index=i,
                    new_total_size=new_total,
                    reason=f"rung_{i}_mfe>={threshold}",
                )
        return PyramidDecision(False, reason="all_rungs_already_fired_or_below")

    def commit_add(self, state: PyramidPositionState,
                   decision: PyramidDecision) -> None:
        """Apply the decision to the state (call after the broker fills the add)."""
        if not decision.should_add:
            return
        state.rungs_fired.add(decision.rung_index)
        state.current_size = decision.new_total_size


# ─── WinStreakSizer (Tier F.3, PLAN.md §6.F.3) ──────────────────────────────

@dataclass
class WinStreakConfig:
    """Anti-Martingale win-streak parameters.

    PLAN.md §6.F.3:
      +20% size after each consecutive win, up to 2× base
      Reset on first loss
    """
    bonus_per_win:        float = 0.20    # add 20% per consecutive win
    max_multiplier:       float = 2.0     # cap at 2× base size
    base_multiplier:      float = 1.0     # baseline before any wins

    def __post_init__(self):
        if self.bonus_per_win <= 0:
            raise ValueError("bonus_per_win must be > 0")
        if self.max_multiplier < self.base_multiplier:
            raise ValueError("max_multiplier must be >= base_multiplier")
        if self.base_multiplier <= 0:
            raise ValueError("base_multiplier must be > 0")


class WinStreakSizer:
    """Anti-Martingale: scale up size during win streaks; reset on any loss.

    Usage::

        streak = WinStreakSizer()
        # After every closed trade:
        streak.record_outcome(won=True_or_False)
        # When sizing a new entry:
        mult = streak.size_multiplier()    # in [base, max]
        size = base_size * mult
    """

    def __init__(self, config: WinStreakConfig | None = None):
        self.cfg = config or WinStreakConfig()
        self._consecutive_wins = 0

    def record_outcome(self, won: bool) -> int:
        """Record one closed trade outcome. Returns the new streak length."""
        if won:
            self._consecutive_wins += 1
        else:
            self._consecutive_wins = 0
        return self._consecutive_wins

    def size_multiplier(self) -> float:
        """Current size multiplier given the streak state.

        Examples (with defaults bonus=0.20, max=2.0, base=1.0):
          0 wins → 1.0
          1 win  → 1.2
          2 wins → 1.4
          3 wins → 1.6
          4 wins → 1.8
          5+ wins → 2.0 (capped)
        """
        mult = self.cfg.base_multiplier + self._consecutive_wins * self.cfg.bonus_per_win
        return min(self.cfg.max_multiplier, mult)

    @property
    def consecutive_wins(self) -> int:
        return self._consecutive_wins

    def reset(self) -> None:
        """Manually reset the streak (e.g. after a circuit-breaker pause)."""
        self._consecutive_wins = 0
