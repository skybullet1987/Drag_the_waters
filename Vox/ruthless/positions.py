# ── Ruthless V2 — position management ────────────────────────────────────────
#
# MultiPositionManager  — track open positions, enforce limits
# DynamicVoterWeighting — contextual bandit-style rolling payoff tracker
# _date_key             — datetime helper
# ─────────────────────────────────────────────────────────────────────────────

from .cfg import (
    RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
    RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
    RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY,
    RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
    RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
    RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
    RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER,
    RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST,
    RUTHLESS_V2_DECAY_FACTOR,
    RUTHLESS_V2_BASE_WEIGHTS,
)


def _date_key(dt):
    """Convert datetime to YYYY-MM-DD string (timezone-naive safe)."""
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(dt)[:10]


class MultiPositionManager:
    """Track open V2 positions and enforce multi-position limits.

    Enforces:
      - max_concurrent_positions: total simultaneous open positions
      - max_symbol_allocation:    per-symbol capital fraction
      - max_total_exposure:       sum of all allocations (can exceed 1.0 for leverage sim)
      - max_new_entries_per_day:  entries taken on current calendar day
      - max_entries_per_symbol_per_day: per-symbol daily limit
      - reentry_cooldown_min:     minutes between exits and re-entry for same symbol

    Usage::

        mgr = MultiPositionManager()
        can_enter, reason = mgr.can_enter(sym, alloc, current_time)
        if can_enter:
            mgr.open_position(sym, alloc, trade_id, current_time)
        ...
        mgr.close_position(sym, trade_id, current_time)
    """

    def __init__(
        self,
        max_concurrent=RUTHLESS_V2_MAX_CONCURRENT_POSITIONS,
        max_new_per_day=RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY,
        max_per_symbol_per_day=RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY,
        max_symbol_alloc=RUTHLESS_V2_MAX_SYMBOL_ALLOCATION,
        min_symbol_alloc=RUTHLESS_V2_MIN_SYMBOL_ALLOCATION,
        max_total_exposure=RUTHLESS_V2_MAX_TOTAL_EXPOSURE,
        reentry_cooldown_min=RUTHLESS_V2_REENTRY_COOLDOWN_MIN,
    ):
        self.max_concurrent        = max_concurrent
        self.max_new_per_day       = max_new_per_day
        self.max_per_symbol_per_day = max_per_symbol_per_day
        self.max_symbol_alloc      = max_symbol_alloc
        self.min_symbol_alloc      = min_symbol_alloc
        self.max_total_exposure    = max_total_exposure
        self.reentry_cooldown_min  = reentry_cooldown_min

        self._open_positions = {}
        self._daily_counts   = {}
        self._last_exit_time = {}

    def open_position_count(self):
        """Number of currently open V2 positions."""
        return len(self._open_positions)

    def symbol_position_count(self, symbol):
        """Number of open positions for a given symbol."""
        return sum(
            1 for p in self._open_positions.values()
            if p["symbol"] == symbol
        )

    def total_exposure(self):
        """Sum of allocations across all open positions."""
        return sum(p["allocation"] for p in self._open_positions.values())

    def symbol_exposure(self, symbol):
        """Sum of allocations for a given symbol."""
        return sum(
            p["allocation"] for p in self._open_positions.values()
            if p["symbol"] == symbol
        )

    def can_enter(self, symbol, allocation, current_time):
        """Check if a new position can be opened for symbol.

        Parameters
        ----------
        symbol       : str
        allocation   : float  — fraction of portfolio (e.g. 0.20)
        current_time : datetime-like — algo.time

        Returns
        -------
        (allowed: bool, reason: str)
        """
        allocation = max(allocation, self.min_symbol_alloc)
        allocation = min(allocation, self.max_symbol_alloc)

        if len(self._open_positions) >= self.max_concurrent:
            return False, f"max_concurrent={self.max_concurrent} reached"

        sym_exp = self.symbol_exposure(symbol)
        if sym_exp + allocation > self.max_symbol_alloc:
            return False, f"symbol_exposure {sym_exp+allocation:.2f} > max={self.max_symbol_alloc}"

        total_exp = self.total_exposure()
        if total_exp + allocation > self.max_total_exposure:
            return False, f"total_exposure {total_exp+allocation:.2f} > max={self.max_total_exposure}"

        date_str = _date_key(current_time)
        day = self._daily_counts.get(date_str, {})
        if day.get("total", 0) >= self.max_new_per_day:
            return False, f"max_new_per_day={self.max_new_per_day} reached"
        if day.get(symbol, 0) >= self.max_per_symbol_per_day:
            return False, f"max_per_symbol_per_day={self.max_per_symbol_per_day} reached for {symbol}"

        last_exit = self._last_exit_time.get(symbol)
        if last_exit is not None:
            try:
                elapsed_min = (current_time - last_exit).total_seconds() / 60.0
            except Exception:
                elapsed_min = 99999.0
            if elapsed_min < self.reentry_cooldown_min:
                return (
                    False,
                    f"reentry_cooldown {elapsed_min:.1f}min < {self.reentry_cooldown_min}min for {symbol}",
                )

        return True, "ok"

    def open_position(self, symbol, allocation, trade_id, current_time):
        """Register a new open position."""
        allocation = max(self.min_symbol_alloc, min(self.max_symbol_alloc, allocation))
        self._open_positions[trade_id] = {
            "symbol":     symbol,
            "allocation": allocation,
            "open_time":  current_time,
        }
        date_str = _date_key(current_time)
        day = self._daily_counts.setdefault(date_str, {})
        day["total"] = day.get("total", 0) + 1
        day[symbol]  = day.get(symbol, 0) + 1

    def close_position(self, trade_id, current_time):
        """Deregister an open position and record exit time."""
        pos = self._open_positions.pop(trade_id, None)
        if pos is not None:
            symbol = pos["symbol"]
            self._last_exit_time[symbol] = current_time
        return pos

    def get_open_positions(self):
        """Return a snapshot of open positions dict."""
        return dict(self._open_positions)

    def get_daily_counts(self, current_time):
        """Return today's entry counts."""
        date_str = _date_key(current_time)
        return dict(self._daily_counts.get(date_str, {}))


class DynamicVoterWeighting:
    """Contextual bandit-style rolling model payoff tracker.

    Tracks per-model yes-vote outcomes from *selected* trades only.
    After each trade closes, call update() for each model that was active
    at entry.  Weights are adjusted exponentially and capped.

    Usage::

        dv = DynamicVoterWeighting()
        dv.initialize(base_weights)

        # at entry, snapshot active votes
        snapshot = dv.snapshot_entry_votes(conf["active_votes"])

        # at exit
        dv.update(snapshot, realized_return, winner=realized_return > 0)

        # get effective weight for scoring
        eff_w = dv.effective_weight("rf")
    """

    def __init__(
        self,
        base_weights=None,
        max_multiplier=RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER,
        min_multiplier=RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER,
        min_obs=RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST,
        decay=RUTHLESS_V2_DECAY_FACTOR,
        vote_threshold=0.50,
    ):
        self.max_multiplier  = max_multiplier
        self.min_multiplier  = min_multiplier
        self.min_obs         = min_obs
        self.decay           = decay
        self.vote_threshold  = vote_threshold
        self._state = {}
        self.initialize(base_weights or RUTHLESS_V2_BASE_WEIGHTS)

    def initialize(self, base_weights):
        """Set base weights and initialize performance state."""
        for mid, bw in base_weights.items():
            if mid not in self._state:
                self._state[mid] = {
                    "base_weight":         float(bw),
                    "perf_score":          1.0,
                    "yes_count":           0,
                    "yes_wins":            0,
                    "yes_rets_sum":        0.0,
                    "yes_losses":          0,
                    "yes_losses_sum":      0.0,
                    "recent_decay_score":  1.0,
                    "last_updated":        None,
                }

    def snapshot_entry_votes(self, active_votes):
        """Return {model_id: voted_yes(bool)} snapshot for a given active_votes dict."""
        if not active_votes:
            return {}
        return {
            mid: float(proba) >= self.vote_threshold
            for mid, proba in active_votes.items()
        }

    def update(self, entry_vote_snapshot, realized_return, winner=None, current_time=None):
        """Update dynamic weights after a trade closes."""
        if winner is None:
            winner = realized_return > 0.0

        for mid, voted_yes in entry_vote_snapshot.items():
            if not voted_yes:
                continue
            s = self._state.setdefault(mid, {
                "base_weight": RUTHLESS_V2_BASE_WEIGHTS.get(mid, 1.0),
                "perf_score": 1.0, "yes_count": 0, "yes_wins": 0,
                "yes_rets_sum": 0.0, "yes_losses": 0,
                "yes_losses_sum": 0.0, "recent_decay_score": 1.0,
                "last_updated": None,
            })

            s["recent_decay_score"] = (
                self.decay * s["recent_decay_score"] + (1.0 - self.decay) * (1.0 if winner else 0.0)
            )

            s["yes_count"] += 1
            if winner:
                s["yes_wins"] += 1
                s["yes_rets_sum"] += realized_return
            else:
                s["yes_losses"] += 1
                s["yes_losses_sum"] += abs(realized_return)

            gross_profit = s["yes_rets_sum"]
            gross_loss   = s["yes_losses_sum"]
            if gross_loss > 0:
                s["yes_profit_factor"] = gross_profit / gross_loss
            else:
                s["yes_profit_factor"] = min(3.0, gross_profit * 100 + 1.0)

            if s["yes_count"] >= self.min_obs:
                win_rate = s["yes_wins"] / s["yes_count"]
                raw_mult = 0.5 * (win_rate / 0.5) + 0.5 * s["recent_decay_score"]
                s["perf_score"] = max(
                    self.min_multiplier,
                    min(self.max_multiplier, raw_mult),
                )
            s["last_updated"] = current_time

    def effective_weight(self, model_id):
        """Return base_weight * performance_multiplier for model_id."""
        s = self._state.get(model_id)
        if s is None:
            return RUTHLESS_V2_BASE_WEIGHTS.get(model_id, 1.0)
        return s["base_weight"] * s["perf_score"]

    def get_all_effective_weights(self):
        """Return {model_id: effective_weight} for all tracked models."""
        return {mid: self.effective_weight(mid) for mid in self._state}

    def get_state_summary(self):
        """Return a compact {model_id: {...}} summary suitable for logging/persist."""
        result = {}
        for mid, s in self._state.items():
            result[mid] = {
                "base_weight":        s["base_weight"],
                "perf_score":         round(s["perf_score"], 4),
                "effective_weight":   round(self.effective_weight(mid), 4),
                "yes_count":          s["yes_count"],
                "yes_wins":           s["yes_wins"],
                "yes_win_rate":       round(s["yes_wins"] / s["yes_count"], 3)
                                      if s["yes_count"] else None,
                "yes_avg_return":     round(s["yes_rets_sum"] / s["yes_wins"], 4)
                                      if s["yes_wins"] else None,
                "yes_profit_factor":  round(s.get("yes_profit_factor", 1.0), 3),
                "recent_decay_score": round(s["recent_decay_score"], 4),
            }
        return result
