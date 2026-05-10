"""regime_runner — multi-window walk-forward + survivability scoring.

Stops us from "best of N backtests" selection bias by forcing a parameter
set to demonstrate consistency across multiple historical regime windows.

The 6 windows (PLAN.md §2.2.3):
    1. 2022-01-01 → 2022-06-30  Crypto winter / LUNA collapse
    2. 2022-07-01 → 2023-06-30  Sideways grind
    3. 2023-07-01 → 2024-06-30  BTC recovery, alts lag
    4. 2024-07-01 → 2024-12-31  Chop
    5. 2025-01-01 → 2025-12-31  Alt season
    6. 2026-01-01 → 2026-05-10  YTD

Survivability score (favors consistency over peak):
    score = mean(net_return)
            − consistency_penalty * std(net_return)
            − worst_case_penalty * max(drawdown_per_window)

A "blessed" parameter set has positive `score`, no window catastrophic
(net < -15% or DD > 25%), and at least 4 of 6 windows positive.

Two modes of use:
    1. From QC backtest results — pass a list of (window, statistics_dict).
    2. From a custom backtest function — provide a callable that accepts
       a (start_date, end_date, params) tuple and returns a stats dict.

This module does NOT submit backtests to QC itself (that requires API
credentials + cloud time and is opt-in from the audit dashboard).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Callable, Iterable, Sequence


# ─── Window definitions ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeWindow:
    name: str
    start: date
    end: date
    description: str

    @property
    def label(self) -> str:
        return f"{self.start}→{self.end}"


REGIME_WINDOWS: tuple[RegimeWindow, ...] = (
    RegimeWindow("crypto_winter_2022h1", date(2022, 1, 1), date(2022, 6, 30),
                 "Crypto winter / LUNA collapse"),
    RegimeWindow("sideways_2022h2_2023h1", date(2022, 7, 1), date(2023, 6, 30),
                 "Sideways grind"),
    RegimeWindow("btc_recovery_2023h2_2024h1", date(2023, 7, 1), date(2024, 6, 30),
                 "BTC recovery, alts lag"),
    RegimeWindow("chop_2024h2", date(2024, 7, 1), date(2024, 12, 31),
                 "Chop"),
    RegimeWindow("alt_season_2025", date(2025, 1, 1), date(2025, 12, 31),
                 "Alt season"),
    RegimeWindow("ytd_2026", date(2026, 1, 1), date(2026, 5, 10),
                 "YTD 2026"),
)


# ─── Per-window result + multi-window summary ────────────────────────────────

@dataclass
class WindowResult:
    """Outcome of running one parameter set on one window."""
    window: str
    net_return_pct: float       # 0-100 (e.g. +50 means +50%)
    drawdown_pct: float         # positive number (e.g. 25.0 means 25% MDD)
    sharpe: float
    win_rate_pct: float
    trades: int
    catastrophic: bool = False  # True if net < -15% OR DD > 25%

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SurvivabilityResult:
    """Aggregate result for one parameter set across all 6 windows."""
    param_id: str
    windows: list[WindowResult] = field(default_factory=list)
    mean_return_pct: float = 0.0
    std_return_pct: float = 0.0
    worst_window_return_pct: float = 0.0
    worst_window_dd_pct: float = 0.0
    n_positive_windows: int = 0
    n_catastrophic_windows: int = 0
    survivability_score: float = 0.0
    blessed: bool = False

    def to_dict(self) -> dict:
        d = {
            "param_id": self.param_id,
            "mean_return_pct":         round(self.mean_return_pct, 3),
            "std_return_pct":          round(self.std_return_pct, 3),
            "worst_window_return_pct": round(self.worst_window_return_pct, 3),
            "worst_window_dd_pct":     round(self.worst_window_dd_pct, 3),
            "n_positive_windows":      self.n_positive_windows,
            "n_catastrophic_windows":  self.n_catastrophic_windows,
            "survivability_score":     round(self.survivability_score, 3),
            "blessed":                 self.blessed,
            "windows": [w.to_dict() for w in self.windows],
        }
        return d

    def summary_text(self) -> str:
        lines = [
            f"  param_id={self.param_id}  blessed={self.blessed}",
            f"    mean_return={self.mean_return_pct:+.2f}%  "
            f"std={self.std_return_pct:.2f}%  "
            f"score={self.survivability_score:+.2f}",
            f"    worst_window: ret={self.worst_window_return_pct:+.2f}%  "
            f"dd={self.worst_window_dd_pct:.2f}%",
            f"    positive_windows={self.n_positive_windows}/{len(self.windows)}  "
            f"catastrophic={self.n_catastrophic_windows}",
        ]
        for w in self.windows:
            lines.append(
                f"      {w.window:<32} ret={w.net_return_pct:+7.2f}%  "
                f"dd={w.drawdown_pct:5.2f}%  sharpe={w.sharpe:+5.2f}  "
                f"trades={w.trades:5d}  WR={w.win_rate_pct:.1f}%"
                + ("  [CATASTROPHIC]" if w.catastrophic else "")
            )
        return "\n".join(lines)


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _stats(seq: Sequence[float]) -> tuple[float, float]:
    if not seq:
        return 0.0, 0.0
    mean = sum(seq) / len(seq)
    var = sum((x - mean) ** 2 for x in seq) / len(seq)
    return mean, var ** 0.5


def score_param_set(
    param_id: str,
    window_results: Sequence[WindowResult],
    *,
    catastrophic_return_threshold_pct: float = -15.0,
    catastrophic_dd_threshold_pct:     float = 25.0,
    consistency_penalty:                float = 1.0,
    worst_case_penalty:                 float = 0.5,
    min_positive_windows:               int   = 4,
) -> SurvivabilityResult:
    """Compute survivability score + blessed flag.

    blessed = True iff:
      - score > 0
      - no catastrophic window
      - >= min_positive_windows are positive
    """
    # Mark catastrophic
    marked = []
    for w in window_results:
        cat = (
            w.net_return_pct < catastrophic_return_threshold_pct
            or w.drawdown_pct > catastrophic_dd_threshold_pct
        )
        marked.append(WindowResult(**{**asdict(w), "catastrophic": cat}))

    rets = [w.net_return_pct for w in marked]
    dds  = [w.drawdown_pct   for w in marked]
    mean_ret, std_ret = _stats(rets)
    worst_ret = min(rets) if rets else 0.0
    worst_dd  = max(dds)  if dds  else 0.0

    score = (
        mean_ret
        - consistency_penalty * std_ret
        - worst_case_penalty  * worst_dd
    )

    n_pos = sum(1 for w in marked if w.net_return_pct > 0)
    n_cat = sum(1 for w in marked if w.catastrophic)

    blessed = (
        score > 0
        and n_cat == 0
        and n_pos >= min_positive_windows
    )

    return SurvivabilityResult(
        param_id=param_id,
        windows=marked,
        mean_return_pct=mean_ret,
        std_return_pct=std_ret,
        worst_window_return_pct=worst_ret,
        worst_window_dd_pct=worst_dd,
        n_positive_windows=n_pos,
        n_catastrophic_windows=n_cat,
        survivability_score=score,
        blessed=blessed,
    )


# ─── Sweep + ranking ─────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """All parameter sets ranked by survivability."""
    results: list[SurvivabilityResult]

    @property
    def ranked(self) -> list[SurvivabilityResult]:
        return sorted(self.results,
                      key=lambda r: r.survivability_score, reverse=True)

    @property
    def blessed(self) -> list[SurvivabilityResult]:
        return [r for r in self.ranked if r.blessed]

    def top_n(self, n: int = 5) -> list[SurvivabilityResult]:
        return self.ranked[:n]

    def summary_text(self) -> str:
        lines = [
            f"─── Walk-forward sweep ({len(self.results)} param sets, "
            f"{len(self.blessed)} blessed) ───",
            "",
            "Top 10 by survivability score:",
        ]
        for r in self.ranked[:10]:
            lines.append(
                f"  {r.survivability_score:+8.2f}  "
                f"mean={r.mean_return_pct:+7.2f}%  "
                f"std={r.std_return_pct:5.2f}%  "
                f"worst={r.worst_window_return_pct:+7.2f}% "
                f"(dd={r.worst_window_dd_pct:5.2f}%)  "
                f"pos={r.n_positive_windows}/{len(r.windows)}  "
                f"cat={r.n_catastrophic_windows}  "
                f"blessed={r.blessed}  {r.param_id}"
            )
        return "\n".join(lines)


def sweep(
    param_sets: Iterable[tuple[str, dict]],
    backtest_fn: Callable[[RegimeWindow, dict], WindowResult],
    windows: Sequence[RegimeWindow] = REGIME_WINDOWS,
    **score_kwargs,
) -> SweepResult:
    """Run all (param_id, params) pairs across all windows.

    Args:
        param_sets:   iterable of (param_id, params_dict)
        backtest_fn:  callable (window, params) → WindowResult
        windows:      defaults to REGIME_WINDOWS
        score_kwargs: forwarded to score_param_set

    Caller's `backtest_fn` is responsible for whatever execution path
    makes sense (local pure-Python sim, QC API submission, harsh-sim
    rerun, etc.). This module just orchestrates.
    """
    out: list[SurvivabilityResult] = []
    for pid, params in param_sets:
        wrs: list[WindowResult] = []
        for w in windows:
            try:
                wrs.append(backtest_fn(w, params))
            except Exception as e:
                # Treat backtest failure as a catastrophic 0-trade window
                wrs.append(WindowResult(
                    window=w.name, net_return_pct=-100.0, drawdown_pct=100.0,
                    sharpe=-10.0, win_rate_pct=0.0, trades=0, catastrophic=True,
                ))
        out.append(score_param_set(pid, wrs, **score_kwargs))
    return SweepResult(results=out)
