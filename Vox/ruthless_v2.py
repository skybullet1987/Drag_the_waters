# ── Ruthless V2 Engine ────────────────────────────────────────────────────────
#
# Aggressive multi-position opportunity engine for Vox.
#
# This module is ONLY active when risk_profile=ruthless_v2 OR
# (risk_profile=ruthless AND ruthless_v2_mode=true).
# All existing V1 profiles are unaffected.
#
# Components:
#   - V2 config defaults
#   - MultiPositionManager  — track open positions, enforce per-symbol and
#                              total-exposure limits
#   - DynamicVoterWeighting — contextual bandit-style rolling payoff tracking
#   - compute_multihorizon_scores() — scalp / continuation / runner lane scores
#   - compute_v2_opportunity_score() — cross-sectional ranking formula
#   - compute_pump_scores() — pump continuation and exhaustion signals
#   - compute_meta_entry_score() — lightweight meta-gate
#   - SplitExitHelper — partial TP sizing + runner remainder
# ─────────────────────────────────────────────────────────────────────────────

import math

# ── V2 config defaults ────────────────────────────────────────────────────────

RUTHLESS_V2_MODE                       = False  # global default off; activated by profile/param

RUTHLESS_V2_MAX_CONCURRENT_POSITIONS   = 4
RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY    = 8
RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY = 2
RUTHLESS_V2_MAX_SYMBOL_ALLOCATION      = 0.30
RUTHLESS_V2_MIN_SYMBOL_ALLOCATION      = 0.08
RUTHLESS_V2_MAX_TOTAL_EXPOSURE         = 1.25
RUTHLESS_V2_DECISION_INTERVAL_MIN      = 15
RUTHLESS_V2_MIN_SCORE_TO_TRADE         = 0.005
RUTHLESS_V2_REENTRY_COOLDOWN_MIN       = 30

# Partial TP / split-exit defaults
RUTHLESS_V2_PARTIAL_TP_ENABLED         = True
RUTHLESS_V2_PARTIAL_TP_FRACTION        = 0.50
RUTHLESS_V2_SCALP_TP                   = 0.015
RUTHLESS_V2_CONTINUATION_TP            = 0.04
RUTHLESS_V2_RUNNER_INITIAL_TP          = 0.06
RUTHLESS_V2_RUNNER_TRAIL_PCT           = 0.04
RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT      = 0.06

# V2 active voter pool (per problem statement)
RUTHLESS_V2_ACTIVE_MODELS    = ["rf", "et", "hgbc_l2", "lgbm_bal", "gbc", "ada"]
RUTHLESS_V2_OPTIONAL_MODELS  = ["xgb_bal", "catboost_bal"]  # active if available & validated
RUTHLESS_V2_DIAGNOSTIC_MODELS = [
    "gnb", "lr", "lr_bal", "cal_et", "cal_rf",
    "et_shallow", "rf_shallow",
    "markov_regime", "hmm_regime", "kmeans_regime",
    "isoforest_risk",
]

# Base weights for V2 active models (used as starting point for dynamic weighting)
RUTHLESS_V2_BASE_WEIGHTS = {
    "rf":           1.35,
    "hgbc_l2":      1.10,
    "lgbm_bal":     1.00,
    "et":           0.80,
    "gbc":          0.85,
    "ada":          0.70,
    "xgb_bal":      0.75,
    "catboost_bal": 0.75,
}

# Dynamic weight caps
RUTHLESS_V2_MAX_WEIGHT_MULTIPLIER  = 2.0
RUTHLESS_V2_MIN_WEIGHT_MULTIPLIER  = 0.25
RUTHLESS_V2_MIN_OBS_BEFORE_ADJUST  = 5    # min trades before weight changes
RUTHLESS_V2_DECAY_FACTOR           = 0.85  # exponential decay weight for historical performance (lower = more emphasis on recent trades)


# ── Multi-position manager ────────────────────────────────────────────────────

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

        # open_positions: trade_id -> {"symbol", "allocation", "open_time"}
        self._open_positions = {}
        # daily entry tracking: date_str -> {"total": int, symbol: int, ...}
        self._daily_counts   = {}
        # last exit time per symbol
        self._last_exit_time = {}

    # ── Capacity checks ───────────────────────────────────────────────────────

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

        # Concurrent position cap
        if len(self._open_positions) >= self.max_concurrent:
            return False, f"max_concurrent={self.max_concurrent} reached"

        # Per-symbol exposure cap (no duplicate same-symbol by default)
        sym_exp = self.symbol_exposure(symbol)
        if sym_exp + allocation > self.max_symbol_alloc:
            return False, f"symbol_exposure {sym_exp+allocation:.2f} > max={self.max_symbol_alloc}"

        # Total exposure cap
        total_exp = self.total_exposure()
        if total_exp + allocation > self.max_total_exposure:
            return False, f"total_exposure {total_exp+allocation:.2f} > max={self.max_total_exposure}"

        # Daily entry cap
        date_str = _date_key(current_time)
        day = self._daily_counts.get(date_str, {})
        if day.get("total", 0) >= self.max_new_per_day:
            return False, f"max_new_per_day={self.max_new_per_day} reached"
        if day.get(symbol, 0) >= self.max_per_symbol_per_day:
            return False, f"max_per_symbol_per_day={self.max_per_symbol_per_day} reached for {symbol}"

        # Reentry cooldown
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


def _date_key(dt):
    """Convert datetime to YYYY-MM-DD string (timezone-naive safe)."""
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(dt)[:10]


# ── Dynamic voter weighting ───────────────────────────────────────────────────

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

        # model_id -> {"base_weight", "perf_score", "yes_count", "yes_wins",
        #              "yes_rets_sum", "yes_losses", "yes_profit_factor",
        #              "recent_decay_score", "last_updated"}
        self._state = {}
        self.initialize(base_weights or RUTHLESS_V2_BASE_WEIGHTS)

    def initialize(self, base_weights):
        """Set base weights and initialize performance state."""
        for mid, bw in base_weights.items():
            if mid not in self._state:
                self._state[mid] = {
                    "base_weight":         float(bw),
                    "perf_score":          1.0,   # starts neutral
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
        """Update dynamic weights after a trade closes.

        Parameters
        ----------
        entry_vote_snapshot : dict[str, bool] — from snapshot_entry_votes()
        realized_return     : float
        winner              : bool or None — if None, inferred from realized_return > 0
        current_time        : datetime-like or None
        """
        if winner is None:
            winner = realized_return > 0.0

        for mid, voted_yes in entry_vote_snapshot.items():
            if not voted_yes:
                continue  # only penalise / reward yes-votes
            s = self._state.setdefault(mid, {
                "base_weight": RUTHLESS_V2_BASE_WEIGHTS.get(mid, 1.0),
                "perf_score": 1.0, "yes_count": 0, "yes_wins": 0,
                "yes_rets_sum": 0.0, "yes_losses": 0,
                "yes_losses_sum": 0.0, "recent_decay_score": 1.0,
                "last_updated": None,
            })

            # Decay existing score toward neutral
            s["recent_decay_score"] = (
                self.decay * s["recent_decay_score"] + (1.0 - self.decay) * (1.0 if winner else 0.0)
            )

            # Accumulate stats
            s["yes_count"] += 1
            if winner:
                s["yes_wins"] += 1
                s["yes_rets_sum"] += realized_return
            else:
                s["yes_losses"] += 1
                s["yes_losses_sum"] += abs(realized_return)

            # Compute profit factor (guard against zero denominator)
            gross_profit = s["yes_rets_sum"]
            gross_loss   = s["yes_losses_sum"]
            if gross_loss > 0:
                s["yes_profit_factor"] = gross_profit / gross_loss
            else:
                s["yes_profit_factor"] = min(3.0, gross_profit * 100 + 1.0)

            # Compute performance multiplier
            if s["yes_count"] >= self.min_obs:
                win_rate = s["yes_wins"] / s["yes_count"]
                # Blend normalized win_rate [0,2] and recent_decay_score [0,1] for multiplier.
                # win_rate/0.5 normalizes a 50% win_rate to 1.0 (neutral); above/below shifts weight.
                raw_mult = 0.5 * (win_rate / 0.5) + 0.5 * s["recent_decay_score"]
                s["perf_score"] = max(
                    self.min_multiplier,
                    min(self.max_multiplier, raw_mult),
                )
            # Before min_obs, keep perf_score = 1.0 (neutral)
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


# ── Multi-horizon scoring ─────────────────────────────────────────────────────

def compute_multihorizon_scores(feat, conf, ev_score, pred_return, market_mode=None):
    """Compute scalp / continuation / runner lane scores.

    Uses existing features and model signals — no separate model training needed.
    Scores are heuristic approximations:
      - scalp_score        : fast short-term momentum + low dispersion
      - continuation_score : medium-term momentum + volume expansion + model agreement
      - runner_score       : strong breakout potential + high EV + regime alignment

    Parameters
    ----------
    feat        : list/array — feature vector from build_features()
                  [ret_1, ret_4, ret_8, ret_16, ret_32, vol_price, vol_r,
                   btc_rel, rsi, bb_pos, bb_width, ...]
    conf        : dict — predict_with_confidence output
    ev_score    : float — ev_after_costs
    pred_return : float — predicted return from regressor
    market_mode : str or None

    Returns
    -------
    dict with keys: scalp_score, continuation_score, runner_score, lane_selected
    """
    if feat is None or len(feat) < 12:
        return {
            "scalp_score":        0.0,
            "continuation_score": 0.0,
            "runner_score":       0.0,
            "lane_selected":      "scalp",
        }

    try:
        ret_1  = float(feat[0])
        ret_4  = float(feat[1])
        ret_8  = float(feat[2])
        ret_16 = float(feat[3])
        vol_r  = float(feat[6])
        rsi    = float(feat[8])   if len(feat) > 8  else 50.0
        bb_pos = float(feat[9])   if len(feat) > 9  else 0.0
        bb_w   = float(feat[10])  if len(feat) > 10 else 0.02
    except (TypeError, IndexError, ValueError):
        return {
            "scalp_score":        0.0,
            "continuation_score": 0.0,
            "runner_score":       0.0,
            "lane_selected":      "scalp",
        }

    # Pull model agreement info
    active_count  = conf.get("active_model_count", 0)
    vote_score    = conf.get("vote_score", 0.0)
    top3_mean     = conf.get("top3_mean", 0.0)
    n_agree       = conf.get("active_n_agree", conf.get("n_agree", 0))
    class_proba   = conf.get("class_proba", 0.0)

    # ── Scalp lane (30–90 min): fast momentum, tight BB, low dispersion ──────
    # Good scalp: short burst (ret_4 > 0), RSI not overbought, tight spread
    scalp_momentum = max(0.0, min(ret_4 * 10.0, 1.0))
    scalp_rsi_ok   = 1.0 - max(0.0, min((rsi - 70.0) / 30.0, 1.0))
    scalp_vol_ok   = max(0.0, min((vol_r - 1.0) / 2.0, 1.0))
    scalp_bb_ok    = max(0.0, min(bb_pos, 1.0))  # lower in BB band is better entry
    scalp_score    = (
        0.40 * scalp_momentum
        + 0.25 * scalp_rsi_ok
        + 0.20 * scalp_vol_ok
        + 0.15 * scalp_bb_ok
    )
    scalp_score = max(0.0, min(scalp_score, 1.0))

    # ── Continuation lane (2–8h): sustained momentum + volume + model ────────
    cont_momentum  = max(0.0, min((ret_16 * 5.0), 1.0))
    cont_vol       = max(0.0, min((vol_r - 1.0) / 3.0, 1.0))
    cont_model     = max(0.0, min((top3_mean - 0.5) * 2.0, 1.0))
    cont_ev        = max(0.0, min(ev_score * 50.0, 1.0))
    continuation_score = (
        0.35 * cont_momentum
        + 0.25 * cont_vol
        + 0.25 * cont_model
        + 0.15 * cont_ev
    )
    continuation_score = max(0.0, min(continuation_score, 1.0))

    # ── Runner lane (12–48h): breakout potential + strong confluence ──────────
    runner_momentum = max(0.0, min((ret_16 + ret_8 * 0.5) * 4.0, 1.0))
    runner_vol      = max(0.0, min((vol_r - 1.5) / 3.0, 1.0))
    runner_model    = max(0.0, min((vote_score - 0.5) * 2.0, 1.0))
    runner_ev       = max(0.0, min(ev_score * 40.0, 1.0))
    # Regime bonus: pump / risk_on_trend mode helps runner
    regime_bonus    = 0.15 if market_mode in ("pump", "risk_on_trend") else 0.0
    runner_score    = (
        0.30 * runner_momentum
        + 0.20 * runner_vol
        + 0.25 * runner_model
        + 0.15 * runner_ev
        + 0.10 * max(0.0, min((class_proba - 0.5) * 2.0, 1.0))
        + regime_bonus
    )
    runner_score = max(0.0, min(runner_score, 1.0))

    # Select lane by max score
    scores = {
        "scalp":        scalp_score,
        "continuation": continuation_score,
        "runner":       runner_score,
    }
    lane_selected = max(scores, key=scores.__getitem__)

    return {
        "scalp_score":        round(scalp_score, 4),
        "continuation_score": round(continuation_score, 4),
        "runner_score":       round(runner_score, 4),
        "lane_selected":      lane_selected,
    }


# ── Cross-sectional opportunity ranking ──────────────────────────────────────

def compute_v2_opportunity_score(
    dynamic_vote_score,
    continuation_score,
    runner_score,
    breakout_score,
    volume_expansion_score,
    regime_score,
    cost_penalty=0.0,
    exhaustion_penalty=0.0,
    relative_strength_score=0.0,
):
    """Compute V2 cross-sectional opportunity score.

    Formula (per problem statement):
        ruthless_v2_score = (
            0.25 * dynamic_vote_score
            + 0.20 * continuation_score
            + 0.20 * runner_score
            + 0.15 * breakout_score
            + 0.10 * volume_expansion_score
            + 0.10 * regime_score
            - cost_penalty
            - exhaustion_penalty
        )

    All input scores should be in [0, 1]. Result may be negative if penalties are large.
    """
    raw = (
        0.25 * dynamic_vote_score
        + 0.20 * continuation_score
        + 0.20 * runner_score
        + 0.15 * breakout_score
        + 0.10 * volume_expansion_score
        + 0.10 * regime_score
        - cost_penalty
        - exhaustion_penalty
    )
    return raw


def compute_breakout_score(feat):
    """Compute a simple breakout/breakout-potential score from features.

    High score = strong upward momentum + volume expansion + Bollinger breakout.
    """
    if feat is None or len(feat) < 11:
        return 0.0
    try:
        ret_4  = float(feat[1])
        ret_16 = float(feat[3])
        vol_r  = float(feat[6])
        bb_pos = float(feat[9])   if len(feat) > 9  else 0.0
        bb_w   = float(feat[10])  if len(feat) > 10 else 0.02
    except (TypeError, IndexError, ValueError):
        return 0.0

    mom_score   = max(0.0, min((ret_4 * 8.0 + ret_16 * 4.0), 1.0))
    vol_score   = max(0.0, min((vol_r - 1.0) / 3.0, 1.0))
    bb_score    = max(0.0, min(bb_pos * 1.5, 1.0))    # high bb_pos = near upper band
    width_score = max(0.0, min(bb_w / 0.10, 1.0))     # wide band = volatility expanding
    return max(0.0, min(
        0.40 * mom_score + 0.30 * vol_score + 0.20 * bb_score + 0.10 * width_score,
        1.0,
    ))


def compute_volume_expansion_score(feat):
    """Return a [0, 1] score for current volume expansion."""
    if feat is None or len(feat) < 7:
        return 0.0
    try:
        vol_r = float(feat[6])
    except (TypeError, IndexError, ValueError):
        return 0.0
    return max(0.0, min((vol_r - 1.0) / 4.0, 1.0))


def compute_regime_score(market_mode):
    """Convert market_mode string to a numeric [0, 1] regime quality score."""
    if market_mode is None:
        return 0.4
    mm = str(market_mode).lower()
    if "pump" in mm:
        return 1.0
    if "risk_on_trend" in mm:
        return 0.85
    if "trend" in mm:
        return 0.65
    if "chop" in mm:
        return 0.30
    if "selloff" in mm or "bear" in mm:
        return 0.10
    return 0.40


def compute_relative_strength_scores(candidates_feat_map):
    """Compute cross-sectional relative strength scores.

    Parameters
    ----------
    candidates_feat_map : dict[str, list/array]
        symbol -> feature vector

    Returns
    -------
    dict[str, float] — symbol -> relative_strength_score in [0, 1]
    dict[str, int]   — symbol -> rank (1 = strongest)
    """
    if not candidates_feat_map:
        return {}, {}

    sym_ret = {}
    for sym, feat in candidates_feat_map.items():
        if feat is None or len(feat) < 4:
            sym_ret[sym] = 0.0
        else:
            try:
                sym_ret[sym] = float(feat[3])  # ret_16 as RS proxy
            except (TypeError, IndexError, ValueError):
                sym_ret[sym] = 0.0

    vals = list(sym_ret.values())
    if not vals:
        return {}, {}

    v_min = min(vals)
    v_max = max(vals)
    v_range = v_max - v_min

    # Normalize to [0, 1]
    rs_scores = {}
    for sym, v in sym_ret.items():
        rs_scores[sym] = (v - v_min) / v_range if v_range > 1e-9 else 0.5

    # Rank (1 = highest score)
    sorted_syms = sorted(sym_ret, key=sym_ret.__getitem__, reverse=True)
    ranks = {sym: i + 1 for i, sym in enumerate(sorted_syms)}

    return rs_scores, ranks


# ── Pump continuation and exhaustion ─────────────────────────────────────────

def compute_pump_scores(
    symbol,
    same_symbol_entries_2h=0,
    same_symbol_trail_wins_2h=0,
    same_symbol_realized_return_2h=0.0,
    minutes_since_last_exit=None,
    prior_exit_reason=None,
    feat=None,
    conf=None,
):
    """Compute pump continuation and exhaustion scores for a symbol.

    Parameters
    ----------
    symbol                          : str
    same_symbol_entries_2h          : int — entries in the same symbol in last 2h
    same_symbol_trail_wins_2h       : int — trail wins in same symbol in last 2h
    same_symbol_realized_return_2h  : float — realized PnL from same symbol in 2h
    minutes_since_last_exit         : float or None
    prior_exit_reason               : str or None — e.g. "EXIT_TRAIL", "EXIT_SL"
    feat                            : feature vector or None
    conf                            : model confidence dict or None

    Returns
    -------
    dict with keys:
        pump_continuation_score : float [0, 1] — higher = stronger pump, re-entry ok
        pump_exhaustion_score   : float [0, 1] — higher = late/exhausted, avoid entry
    """
    # --- Continuation signals (favor re-entry) --------------------------------
    cont_vol        = 0.0
    cont_momentum   = 0.0
    cont_model      = 0.0
    if feat is not None and len(feat) >= 7:
        try:
            ret_4 = float(feat[1])
            vol_r = float(feat[6])
            cont_vol      = max(0.0, min((vol_r - 1.5) / 3.0, 1.0))
            cont_momentum = max(0.0, min(ret_4 * 10.0, 1.0))
        except (TypeError, IndexError, ValueError):
            pass
    if conf is not None:
        top3_mean = conf.get("top3_mean", 0.0)
        cont_model = max(0.0, min((top3_mean - 0.50) * 2.5, 1.0))

    # Recent series of trail wins suggests pump is ongoing
    trail_wins_bonus = min(same_symbol_trail_wins_2h * 0.15, 0.30)

    pump_continuation_score = max(0.0, min(
        0.35 * cont_momentum
        + 0.30 * cont_vol
        + 0.25 * cont_model
        + trail_wins_bonus,
        1.0,
    ))

    # --- Exhaustion signals (avoid re-entry) -----------------------------------
    # Repeated entries in 2h window
    entry_exhaustion = min(same_symbol_entries_2h * 0.20, 0.60)
    # Many trail wins → pump may be extended
    win_exhaustion   = min(same_symbol_trail_wins_2h * 0.15, 0.45)
    # Very fast re-entry after exit
    time_exhaustion  = 0.0
    if minutes_since_last_exit is not None:
        time_exhaustion = max(0.0, min((10.0 - minutes_since_last_exit) / 10.0, 0.40))
    # Prior SL in same symbol → bearish exhaustion signal
    prior_sl_penalty = 0.20 if prior_exit_reason and "SL" in str(prior_exit_reason) else 0.0
    # Fading volume
    vol_fade = 0.0
    if feat is not None and len(feat) >= 7:
        try:
            vol_r = float(feat[6])
            if vol_r < 0.8:
                vol_fade = min((0.8 - vol_r) * 2.0, 0.30)
        except (TypeError, IndexError, ValueError):
            pass

    pump_exhaustion_score = max(0.0, min(
        entry_exhaustion + win_exhaustion + time_exhaustion
        + prior_sl_penalty + vol_fade,
        1.0,
    ))

    return {
        "pump_continuation_score": round(pump_continuation_score, 4),
        "pump_exhaustion_score":   round(pump_exhaustion_score, 4),
    }


def exhaustion_override_allowed(pump_continuation_score, pump_exhaustion_score,
                                continuation_threshold=0.55, exhaustion_threshold=0.60):
    """Return True if continuation is strong enough to override simple cooldown.

    Allows re-entry even when reentry_cooldown is active if:
      - continuation score is high (real pump strength confirmed)
      - exhaustion score is below threshold
    """
    return (
        pump_continuation_score >= continuation_threshold
        and pump_exhaustion_score < exhaustion_threshold
    )


# ── Meta-entry score ──────────────────────────────────────────────────────────

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

    This is a transparent heuristic meta-score (not a trained meta-model).
    Features match those described in the problem statement.

    Returns
    -------
    float — meta_entry_score in [0, 1]
    """
    # Model quality signals
    model_q = 0.4 * vote_score + 0.3 * dynamic_vote_score
    # Reduce for high disagreement
    model_q *= max(0.5, 1.0 - model_disagreement)

    # Market context
    market_q = 0.3 * regime_score + 0.3 * volume_expansion_score + 0.4 * relative_strength_score

    # Opportunity quality
    opp_q = 0.4 * breakout_score + 0.3 * pump_continuation_score + 0.3 * recent_win_rate

    # Exhaustion penalty
    exhaustion_pen = exhaustion_score * 0.5

    raw = (
        0.35 * model_q
        + 0.30 * market_q
        + 0.35 * opp_q
        - exhaustion_pen
    )
    return max(0.0, min(raw, 1.0))


# ── Split-exit sizing helper ──────────────────────────────────────────────────

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
        """Compute (partial_qty, runner_qty) for a split exit.

        Parameters
        ----------
        current_qty     : float — current open quantity (must be positive)
        partial_fraction: float — fraction to sell at partial TP (default: self.partial_tp_fraction)
        min_runner_qty  : float — minimum quantity to keep as runner (prevent dust)

        Returns
        -------
        (partial_qty: float, runner_qty: float)
        partial_qty + runner_qty ≤ current_qty
        """
        if partial_fraction is None:
            partial_fraction = self.partial_tp_fraction

        partial_fraction = max(0.01, min(0.99, partial_fraction))
        if current_qty <= 0.0:
            return 0.0, 0.0

        partial_qty = current_qty * partial_fraction
        runner_qty  = current_qty - partial_qty

        # Ensure runner_qty meets minimum (avoid dust)
        if runner_qty < min_runner_qty:
            runner_qty  = 0.0
            partial_qty = current_qty  # sell all if runner would be dust

        # Safety: never exceed current_qty
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


# ── V2 opportunity list sorter ────────────────────────────────────────────────

def rank_candidates_v2(candidates):
    """Sort a list of candidate dicts by v2_opportunity_score descending.

    Each candidate dict should contain a 'v2_opportunity_score' key.
    Returns the sorted list (highest score first).
    """
    return sorted(candidates, key=lambda c: c.get("v2_opportunity_score", 0.0), reverse=True)


# ── Startup log helper ────────────────────────────────────────────────────────

def format_v2_startup_log(
    risk_profile, v2_mode, max_positions, active_models,
    dynamic_weights=None,
):
    """Format V2 startup audit log lines (returns list of strings)."""
    lines = [
        f"[profile] risk_profile={risk_profile} v2={v2_mode}"
        f" max_positions={max_positions}"
        f" active_models={','.join(active_models)}",
    ]
    if dynamic_weights:
        w_str = " ".join(f"{m}={w:.3f}" for m, w in sorted(dynamic_weights.items()))
        lines.append(f"[v2_weights] {w_str}")
    return lines
