# ── Ruthless V2 — meta scoring, split-exit helper, and utilities ─────────────

from .cfg import (
    RUTHLESS_V2_PARTIAL_TP_FRACTION,
    RUTHLESS_V2_SCALP_TP,
    RUTHLESS_V2_CONTINUATION_TP,
    RUTHLESS_V2_RUNNER_INITIAL_TP,
    RUTHLESS_V2_RUNNER_TRAIL_PCT,
    RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT,
)


def compute_meta_entry_score(
    vote_score=0.0,
    dynamic_vote_score=0.0,
    model_disagreement=0.0,
    regime_score=0.5,
    volume_expansion_score=0.5,
    relative_strength_score=0.5,
    breakout_score=0.5,
    exhaustion_score=0.0,
    recent_win_rate=0.5,
    pump_continuation_score=0.0,
):
    """Compute a lightweight meta-entry score from composite signals.

    Returns
    -------
    float — meta_entry_score in [0, 1]
    """
    model_q = 0.4 * vote_score + 0.3 * dynamic_vote_score
    model_q *= max(0.5, 1.0 - model_disagreement)

    market_q = 0.3 * regime_score + 0.3 * volume_expansion_score + 0.4 * relative_strength_score

    opp_q = 0.4 * breakout_score + 0.3 * pump_continuation_score + 0.3 * recent_win_rate

    exhaustion_pen = exhaustion_score * 0.5

    raw = (
        0.35 * model_q
        + 0.30 * market_q
        + 0.35 * opp_q
        - exhaustion_pen
    )
    return max(0.0, min(raw, 1.0))


class SplitExitHelper:
    """Compute safe partial and runner exit quantities for V2 split exits.

    Ensures no over-selling: total of partial + runner ≤ current qty.

    Usage::

        helper = SplitExitHelper()
        partial_qty, runner_qty = helper.compute_split_quantities(
            current_qty=500.0, partial_fraction=0.50, min_runner_qty=0.01
        )
    """

    def __init__(
        self,
        partial_tp_fraction=RUTHLESS_V2_PARTIAL_TP_FRACTION,
    ):
        self.partial_tp_fraction = partial_tp_fraction

    def compute_split_quantities(self, current_qty, partial_fraction=None, min_runner_qty=0.0):
        """Compute (partial_qty, runner_qty) for a split exit."""
        if partial_fraction is None:
            partial_fraction = self.partial_tp_fraction

        partial_fraction = max(0.01, min(0.99, partial_fraction))
        if current_qty <= 0.0:
            return 0.0, 0.0

        partial_qty = current_qty * partial_fraction
        runner_qty  = current_qty - partial_qty

        if runner_qty < min_runner_qty:
            runner_qty  = 0.0
            partial_qty = current_qty

        partial_qty = min(partial_qty, current_qty)
        runner_qty  = max(0.0, current_qty - partial_qty)

        return partial_qty, runner_qty

    def get_lane_tp(self, lane):
        """Return take-profit target for a given lane name."""
        _map = {
            "scalp":        RUTHLESS_V2_SCALP_TP,
            "continuation": RUTHLESS_V2_CONTINUATION_TP,
            "runner":       RUTHLESS_V2_RUNNER_INITIAL_TP,
        }
        return _map.get(lane, RUTHLESS_V2_CONTINUATION_TP)

    def get_lane_trail_pct(self, lane, pump_mode=False):
        """Return trailing-stop pct for runner remainder by lane."""
        if pump_mode:
            return RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT
        if lane == "runner":
            return RUTHLESS_V2_RUNNER_TRAIL_PCT
        return RUTHLESS_V2_RUNNER_TRAIL_PCT


def rank_candidates_v2(candidates):
    """Sort a list of candidate dicts by v2_opportunity_score descending.

    Each candidate dict should contain a 'v2_opportunity_score' key.
    Returns the sorted list (highest score first).
    """
    return sorted(candidates, key=lambda c: c.get("v2_opportunity_score", 0.0), reverse=True)
