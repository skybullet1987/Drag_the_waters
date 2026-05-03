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
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  APEX PREDATOR regime  (see config.py for all tunable APEX_* constants)    ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  Goal: massively more active trading, targeting 10x/20x compounding.        ║
# ║  Evidence: 538 entry_attempts vs 24 orders — signals were suppressed.       ║
# ║                                                                              ║
# ║  Weighted apex_score (per-bar):                                              ║
# ║    apex_score = 0.35 * vote_lr_bal        (PF ~8  at >=0.50 — strongest)    ║
# ║               + 0.25 * vote_hgbc_l2       (PF ~3.35 at >=0.55)              ║
# ║               + 0.15 * active_rf          (PF 2.76  at >=0.60)              ║
# ║               + 0.10 * active_hgbc_l2     (PF 3.38  at >=0.50)              ║
# ║               + 0.10 * active_lgbm_bal    (always-on confirmer)             ║
# ║               + 0.05 * vote_et            (diversifier)                     ║
# ║  Missing columns: weight redistributed pro-rata across present columns.     ║
# ║                                                                              ║
# ║  Entry fires on ANY of five trigger paths (Apex v2 — aggressive gates):     ║
# ║    1. apex_score >= APEX_SCORE_ENTRY  (0.50 — lowered from 0.55)           ║
# ║    2. vote_lr_bal >= 0.50             (proven PF ~8)                        ║
# ║    3. vote_hgbc_l2 >= 0.55 AND active_lgbm_bal >= 0.55                     ║
# ║    4. mean_proba >= 0.50 AND n_agree >= 1   (relaxed strong-ML backstop)    ║
# ║    5. active_lgbm_bal >= 0.50         (always-on confirmer direct gate)     ║
# ║  Technical overlay helpers (callable independently):                         ║
# ║    apex_breakout_signal()            — price × N-bar high + volume spike    ║
# ║    apex_pullback_signal()            — RSI < 35 in uptrend                  ║
# ║    apex_momentum_continuation_signal() — 3 consec. higher closes + vol     ║
# ║  Each rejected attempt is now logged via apex_rejected_entry_log().         ║
# ║                                                                              ║
# ║  confirm/market_mode is a SCORE BOOSTER, not a hard gate.                   ║
# ║                                                                              ║
# ║  Sizing (Kelly-lite + pyramiding):                                           ║
# ║    base_alloc = APEX_BASE_ALLOC (0.20)                                       ║
# ║    edge_mult  = clip((apex_score - 0.50) / 0.30, 0.0, 1.5)                 ║
# ║    conf_mult  = 1.0 + 0.5 * (n_agree >= 4)                                  ║
# ║    size_frac  = clip(base_alloc * (1 + edge_mult) * conf_mult, 0.05, 0.45) ║
# ║    Cap: total gross exposure <= APEX_MAX_GROSS (2.0x equity)                ║
# ║    Pyramid: if open position +1.5% unrealised and apex_score >= 0.55,       ║
# ║             add second tranche at 50% original size (max 2 adds)            ║
# ║                                                                              ║
# ║  Stops / TP / Trail / Time-stop / Breakeven:                                 ║
# ║    SL    = entry - APEX_ATR_SL_MULT * ATR(14); floor 0.8%, ceil 4%          ║
# ║    TP    = entry + APEX_ATR_TP_MULT * ATR(14); floor 2.5%, ceil 15%         ║
# ║    Trail = arms at +APEX_TRAIL_ARM_PCT (1%), trails max(0.8*ATR, 0.6%)      ║
# ║    Breakeven: once MFE >= APEX_BREAKEVEN_MFE (2%), stop → entry + 0.1%      ║
# ║    Time-stop: close after APEX_TIME_STOP_HRS (48h) if MFE < +1%             ║
# ║                                                                              ║
# ║  Concurrency: APEX_MAX_CONCURRENT=8 total, APEX_MAX_PER_SYMBOL=2,           ║
# ║               APEX_COOLDOWN_MIN=15 min reentry cooldown                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import math

# ── V2 config defaults ────────────────────────────────────────────────────────

RUTHLESS_V2_MODE                       = False  # global default off; activated by profile/param

RUTHLESS_V2_MAX_CONCURRENT_POSITIONS   = 5
RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY    = 12
RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY = 3
RUTHLESS_V2_MAX_SYMBOL_ALLOCATION      = 0.35
RUTHLESS_V2_MIN_SYMBOL_ALLOCATION      = 0.08
RUTHLESS_V2_MAX_TOTAL_EXPOSURE         = 1.50
RUTHLESS_V2_DECISION_INTERVAL_MIN      = 15
RUTHLESS_V2_MIN_SCORE_TO_TRADE         = -0.25
RUTHLESS_V2_REENTRY_COOLDOWN_MIN       = 30

# ── V2 machine-gun mode ───────────────────────────────────────────────────────
# When True, V2 acts as an aggressive "machine-gun" opportunity engine:
#   - Regime blocks become soft score penalties instead of hard rejects.
#   - Meta-filter rejections become score/allocation penalties instead of hard rejects.
#   - Top-N candidates are forced when slots are open, even at low scores.
#   - Smaller allocations by default; scales up only for high conviction.
RUTHLESS_V2_MACHINE_GUN_MODE          = True

# Force top-N entries when candidates exist and slots are available.
RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES = 2

# Regime blocks become score penalties (not hard rejects) in machine-gun mode.
# Only truly dangerous modes (risk_off_crash, dump, emergency) remain hard blocks.
RUTHLESS_V2_REGIME_HARD_BLOCK         = False

# Meta-filter becomes a score/allocation modifier instead of hard reject.
RUTHLESS_V2_META_HARD_FILTER          = False
RUTHLESS_V2_META_AS_SCORE_PENALTY     = True
RUTHLESS_V2_META_SCORE_WEIGHT         = 0.20
RUTHLESS_V2_LOW_META_ALLOC_MULT       = 0.50

# Allow entries in chop regime (capped at RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC).
RUTHLESS_V2_ALLOW_CHOP_SCALPS         = True
RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC      = 0.10

# Allocation tiers for machine-gun mode.
RUTHLESS_V2_BASE_ALLOCATION           = 0.12
RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION = 0.30

# Regime penalty applied to score in soft-block mode.
RUTHLESS_V2_SOFT_REGIME_PENALTY          = 0.12
RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER   = 1.5   # extra weight for selloff/bear regimes
RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT     = 0.5   # reduced penalty for mild weak regimes

# Dangerous market-mode strings that always hard-block even in machine-gun mode.
RUTHLESS_V2_HARD_BLOCK_MODES          = frozenset(["risk_off_crash", "dump", "emergency"])

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


# ── Machine-gun mode helpers ──────────────────────────────────────────────────

def apply_regime_soft_penalty(
    symbol,
    score,
    market_mode,
    machine_gun_mode=True,
    regime_hard_block=False,
    soft_penalty=None,
    hard_block_modes=None,
    _log_fn=None,
):
    """Apply regime filter in machine-gun mode.

    In machine-gun mode with regime_hard_block=False, normal regime blocks
    (e.g. chop) become soft score penalties instead of hard rejects.  Only
    truly dangerous modes (risk_off_crash, dump, emergency) remain hard blocks.

    Parameters
    ----------
    symbol         : str
    score          : float — current opportunity score
    market_mode    : str or None
    machine_gun_mode : bool
    regime_hard_block : bool — if True, always hard-block (V1 behavior)
    soft_penalty   : float or None — score deduction for soft block;
                     defaults to RUTHLESS_V2_SOFT_REGIME_PENALTY
    hard_block_modes : set or None — modes that always hard-block;
                     defaults to RUTHLESS_V2_HARD_BLOCK_MODES
    _log_fn        : callable(str) or None — optional compact log function

    Returns
    -------
    (score: float, blocked: bool, log_msg: str or None)
      blocked=True  → hard reject (do not trade)
      blocked=False → candidate still eligible (score may be penalized)
    """
    if soft_penalty is None:
        soft_penalty = RUTHLESS_V2_SOFT_REGIME_PENALTY
    if hard_block_modes is None:
        hard_block_modes = RUTHLESS_V2_HARD_BLOCK_MODES

    mm = str(market_mode).lower() if market_mode else ""

    # Always hard-block truly dangerous modes
    for danger in hard_block_modes:
        if danger in mm:
            msg = f"[v2_hard_regime] {symbol} mode={market_mode} hard_block=True"
            if _log_fn:
                _log_fn(msg)
            return score, True, msg

    # If not in machine-gun mode or hard block is configured, use V1 behavior
    if not machine_gun_mode or regime_hard_block:
        # V1 hard block for non-allowed modes
        if mm and mm not in ("risk_on_trend", "pump", "trend"):
            return score, True, None
        return score, False, None

    # Machine-gun soft penalty
    is_chop    = "chop" in mm
    is_selloff = "selloff" in mm or "bear" in mm
    is_weak    = mm and mm not in ("risk_on_trend", "pump", "trend")

    penalty = 0.0
    if is_selloff:
        penalty = soft_penalty * RUTHLESS_V2_SELLOFF_PENALTY_MULTIPLIER
    elif is_chop:
        penalty = soft_penalty
    elif is_weak:
        penalty = soft_penalty * RUTHLESS_V2_WEAK_REGIME_PENALTY_MULT

    if penalty > 0.0:
        score_before = score
        score        = score - penalty
        msg = (
            f"[v2_soft_regime] {symbol} mode={market_mode}"
            f" penalty={penalty:.2f}"
            f" score_before={score_before:.4f}"
            f" score_after={score:.4f}"
        )
        if _log_fn:
            _log_fn(msg)
        return score, False, msg

    return score, False, None


def apply_meta_soft_penalty(
    symbol,
    score,
    allocation,
    meta_score,
    machine_gun_mode=True,
    meta_hard_filter=False,
    meta_as_score_penalty=True,
    meta_score_weight=None,
    low_meta_alloc_mult=None,
    meta_score_floor=-1.0,
    _log_fn=None,
):
    """Apply meta-filter in machine-gun mode.

    In machine-gun mode with meta_hard_filter=False, meta-score modifies
    score and allocation instead of hard-rejecting the candidate.

    Parameters
    ----------
    symbol              : str
    score               : float — current opportunity score
    allocation          : float — current intended allocation
    meta_score          : float — meta-entry score (typically -1..1 or 0..1)
    machine_gun_mode    : bool
    meta_hard_filter    : bool — if True, use V1 hard-reject behavior
    meta_as_score_penalty : bool — blend meta_score into final score
    meta_score_weight   : float or None — weight of meta in score blend
    low_meta_alloc_mult : float or None — alloc multiplier when meta is low
    meta_score_floor    : float — scores below this are still hard-rejected
                          (emergency safety; default -1.0 = never hard-reject)
    _log_fn             : callable(str) or None

    Returns
    -------
    (score: float, allocation: float, blocked: bool, log_msg: str or None)
    """
    if meta_score_weight is None:
        meta_score_weight = RUTHLESS_V2_META_SCORE_WEIGHT
    if low_meta_alloc_mult is None:
        low_meta_alloc_mult = RUTHLESS_V2_LOW_META_ALLOC_MULT

    # Hard reject floor (emergency safety, applies even in machine-gun mode)
    if meta_score < meta_score_floor:
        msg = f"[v2_meta_emergency_reject] {symbol} meta={meta_score:.3f} < floor={meta_score_floor:.3f}"
        if _log_fn:
            _log_fn(msg)
        return score, allocation, True, msg

    # If not machine-gun mode or hard filter configured, use V1 behavior
    if not machine_gun_mode or meta_hard_filter:
        return score, allocation, False, None

    # Soft mode: blend meta score into opportunity score
    alloc_mult = 1.0
    if meta_as_score_penalty:
        score = score + meta_score_weight * meta_score

    # Low meta → reduce allocation
    if meta_score < 0.0:
        alloc_mult = low_meta_alloc_mult
        allocation = allocation * alloc_mult

    msg = (
        f"[v2_soft_meta] {symbol}"
        f" meta={meta_score:.3f}"
        f" score_after={score:.4f}"
        f" alloc_mult={alloc_mult:.2f}"
    )
    if _log_fn:
        _log_fn(msg)
    return score, allocation, False, msg


def select_top_n_machine_gun(
    ranked_candidates,
    open_slots,
    force_top_n=None,
    min_score=None,
    _log_fn=None,
):
    """Select up to force_top_n candidates from ranked list for machine-gun entries.

    Parameters
    ----------
    ranked_candidates : list of dict — sorted descending by v2_opportunity_score;
                        each dict must have 'v2_opportunity_score' key and optionally
                        'symbol', 'lane'.
    open_slots        : int — number of available concurrent position slots
    force_top_n       : int or None — max candidates to take; defaults to
                        RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES
    min_score         : float or None — minimum score floor; defaults to
                        RUTHLESS_V2_MIN_SCORE_TO_TRADE
    _log_fn           : callable(str) or None

    Returns
    -------
    list of dict — the selected candidate dicts (may be empty)
    """
    if force_top_n is None:
        force_top_n = RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES
    if min_score is None:
        min_score = RUTHLESS_V2_MIN_SCORE_TO_TRADE

    n_candidates = len(ranked_candidates)
    take          = min(force_top_n, open_slots, n_candidates)

    # Filter by score floor
    eligible = [
        c for c in ranked_candidates
        if c.get("v2_opportunity_score", 0.0) >= min_score
    ]

    selected = eligible[:take]

    msg = (
        f"[v2_rank] candidates={n_candidates}"
        f" eligible={len(eligible)}"
        f" slots={open_slots}"
        f" force_top_n={force_top_n}"
        f" taking={len(selected)}"
    )
    if _log_fn:
        _log_fn(msg)

    return selected


def compute_machine_gun_allocation(
    score,
    lane,
    market_mode=None,
    base_alloc=None,
    high_conviction_alloc=None,
    max_symbol_alloc=None,
    chop_scalp_max_alloc=None,
    allow_chop_scalps=True,
):
    """Compute allocation for machine-gun mode based on score and lane.

    Allocation tiers:
      low score scalp/chop      -> 0.08–0.10
      medium score              -> 0.12–0.18
      high score continuation   -> 0.20–0.25
      high conviction runner    -> up to high_conviction_alloc

    Parameters
    ----------
    score                : float — v2_opportunity_score (may be negative)
    lane                 : str — 'scalp', 'continuation', or 'runner'
    market_mode          : str or None
    base_alloc           : float or None — defaults to RUTHLESS_V2_BASE_ALLOCATION
    high_conviction_alloc: float or None — defaults to RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION
    max_symbol_alloc     : float or None — defaults to RUTHLESS_V2_MAX_SYMBOL_ALLOCATION
    chop_scalp_max_alloc : float or None — cap for chop scalps;
                           defaults to RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC
    allow_chop_scalps    : bool

    Returns
    -------
    float — allocation fraction in [RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, max_symbol_alloc]
    """
    if base_alloc is None:
        base_alloc = RUTHLESS_V2_BASE_ALLOCATION
    if high_conviction_alloc is None:
        high_conviction_alloc = RUTHLESS_V2_HIGH_CONVICTION_ALLOCATION
    if max_symbol_alloc is None:
        max_symbol_alloc = RUTHLESS_V2_MAX_SYMBOL_ALLOCATION
    if chop_scalp_max_alloc is None:
        chop_scalp_max_alloc = RUTHLESS_V2_CHOP_SCALP_MAX_ALLOC

    mm = str(market_mode).lower() if market_mode else ""
    is_chop = "chop" in mm

    # Runner lane at high conviction
    if lane == "runner" and score >= 0.50:
        alloc = min(high_conviction_alloc + (score - 0.50) * 0.10, max_symbol_alloc)
    elif lane == "runner" and score >= 0.30:
        alloc = 0.20 + (score - 0.30) * 0.25
    elif lane == "continuation" and score >= 0.35:
        alloc = 0.14 + (score - 0.35) * 0.22
    elif score >= 0.20:
        alloc = base_alloc + (score - 0.20) * 0.10
    elif score >= 0.05:
        alloc = base_alloc
    else:
        # Low score / scalp
        alloc = max(RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, 0.08 + max(score, 0.0) * 0.20)

    # Chop cap
    if is_chop and allow_chop_scalps:
        alloc = min(alloc, chop_scalp_max_alloc)

    # Clip to allowed range
    alloc = max(RUTHLESS_V2_MIN_SYMBOL_ALLOCATION, min(alloc, max_symbol_alloc))
    return round(alloc, 4)


# ── Startup log helper ────────────────────────────────────────────────────────

def format_v2_startup_log(
    risk_profile, v2_mode, max_positions, active_models,
    dynamic_weights=None,
    machine_gun_mode=None,
    regime_hard_block=None,
    meta_hard_filter=None,
    force_top_n=None,
    min_score_to_trade=None,
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

    # Machine-gun mode diagnostics
    mg = machine_gun_mode if machine_gun_mode is not None else RUTHLESS_V2_MACHINE_GUN_MODE
    rh = regime_hard_block if regime_hard_block is not None else RUTHLESS_V2_REGIME_HARD_BLOCK
    mh = meta_hard_filter  if meta_hard_filter  is not None else RUTHLESS_V2_META_HARD_FILTER
    fn = force_top_n       if force_top_n        is not None else RUTHLESS_V2_FORCE_TOP_N_WHEN_CANDIDATES
    ms = min_score_to_trade if min_score_to_trade is not None else RUTHLESS_V2_MIN_SCORE_TO_TRADE
    lines.append(
        f"[v2_machine_gun] enabled={mg}"
        f" regime_hard_block={rh}"
        f" meta_hard_filter={mh}"
        f" force_top_n={fn}"
        f" min_score={ms}"
        f" max_entries_day={RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY}"
        f" max_concurrent={RUTHLESS_V2_MAX_CONCURRENT_POSITIONS}"
        f" reentry_cooldown_min={RUTHLESS_V2_REENTRY_COOLDOWN_MIN}"
    )
    return lines


# ── APEX PREDATOR helpers ─────────────────────────────────────────────────────
#
# These functions implement the Apex Predator regime described at the top of
# this module.  All knobs are imported from config.APEX_* constants.
#
# Public API:
#   compute_apex_score()       — weighted ensemble vote with missing-col handling
#   apex_entry_decision()      — evaluates the four trigger paths
#   compute_apex_size()        — Kelly-lite allocation with pyramiding
#   compute_apex_atr_stops()   — ATR-based SL / TP / trail / breakeven / time-stop
# ─────────────────────────────────────────────────────────────────────────────

# Apex vote column weights (must sum to 1.0)
_APEX_WEIGHTS = {
    "vote_lr_bal":      0.35,
    "vote_hgbc_l2":     0.25,
    "active_rf":        0.15,
    "active_hgbc_l2":   0.10,
    "active_lgbm_bal":  0.10,
    "vote_et":          0.05,
}


def compute_apex_score(votes):
    """Compute the weighted Apex Predator ensemble vote score.

    Parameters
    ----------
    votes : dict[str, float]
        Per-column model probability/vote values.  Keys are drawn from
        {vote_lr_bal, vote_hgbc_l2, active_rf, active_hgbc_l2,
         active_lgbm_bal, vote_et}.
        Missing keys have their weight redistributed pro-rata across the
        present columns so the result always sums correctly.

    Returns
    -------
    apex_score : float
        Weighted ensemble vote score in [0.0, 1.0].
    weight_present : float
        Sum of weights for columns present in *votes* (1.0 when all columns
        are present; < 1.0 when some are missing).

    Notes
    -----
    Weights are defined in _APEX_WEIGHTS (module-level constant).
    If *no* columns are present the function returns (0.0, 0.0).
    """
    present = {col: w for col, w in _APEX_WEIGHTS.items() if col in votes}
    if not present:
        return 0.0, 0.0

    total_w = sum(present.values())
    if total_w <= 0.0:
        return 0.0, 0.0

    # Redistribute missing weight pro-rata: scale present weights to sum to 1
    score = sum(
        (w / total_w) * float(votes[col])
        for col, w in present.items()
    )
    return max(0.0, min(1.0, score)), total_w


def apex_entry_decision(
    votes,
    mean_proba=0.0,
    n_agree=0,
    apex_score_entry=None,
):
    """Evaluate the five Apex Predator entry trigger paths (v2 — aggressive gates).

    Entry fires when **ANY** of the following is True:
      1. apex_score >= apex_score_entry        (default 0.50 — lowered from 0.55)
      2. vote_lr_bal >= 0.50                   (proven PF ~8)
      3. vote_hgbc_l2 >= 0.55 AND active_lgbm_bal >= 0.55
      4. mean_proba >= APEX_ENTRY_PATH4_PROBA_MIN (0.50) AND
         n_agree >= APEX_ENTRY_PATH4_N_AGREE_MIN (1)  (relaxed strong-ML backstop)
      5. active_lgbm_bal >= APEX_ENTRY_LGBM_BAL_MIN (0.50)  (always-on confirmer)

    ``confirm`` / ``market_mode`` is intentionally NOT a hard gate here;
    callers may boost the apex_score externally but should not block based
    on it alone.

    Parameters
    ----------
    votes           : dict[str, float] — same dict passed to compute_apex_score()
    mean_proba      : float — ensemble mean probability (active models)
    n_agree         : int   — number of active models with proba >= threshold
    apex_score_entry: float or None — overrides config.APEX_SCORE_ENTRY

    Returns
    -------
    dict with keys:
      "triggered"   : bool — True if any path fires
      "apex_score"  : float
      "weight_present": float — fraction of total weight present in votes
      "path"        : str or None — name of the first matching path
      "path_detail" : dict — per-path bool results
      "reject_reason": str or None — first failed gate label when not triggered
    """
    # Import here to avoid circular issues; config is always available
    try:
        from config import (
            APEX_SCORE_ENTRY           as _cfg_entry,
            APEX_ENTRY_PATH4_PROBA_MIN as _p4_proba,
            APEX_ENTRY_PATH4_N_AGREE_MIN as _p4_agree,
            APEX_ENTRY_LGBM_BAL_MIN    as _p5_lgbm,
        )
    except ImportError:
        _cfg_entry = 0.50
        _p4_proba  = 0.50
        _p4_agree  = 1
        _p5_lgbm   = 0.50

    if apex_score_entry is None:
        apex_score_entry = _cfg_entry

    apex_score, weight_present = compute_apex_score(votes)

    # Individual vote values (default 0.0 if missing)
    lr_bal      = float(votes.get("vote_lr_bal",     0.0))
    hgbc_l2     = float(votes.get("vote_hgbc_l2",    0.0))
    lgbm_bal    = float(votes.get("active_lgbm_bal", 0.0))

    path1 = apex_score >= apex_score_entry
    path2 = lr_bal >= 0.50
    path3 = hgbc_l2 >= 0.55 and lgbm_bal >= 0.55
    path4 = float(mean_proba) >= _p4_proba and int(n_agree) >= _p4_agree
    path5 = lgbm_bal >= _p5_lgbm

    triggered = path1 or path2 or path3 or path4 or path5

    path_name = None
    if path1:
        path_name = "apex_score"
    elif path2:
        path_name = "vote_lr_bal"
    elif path3:
        path_name = "hgbc_l2_x_lgbm_bal"
    elif path4:
        path_name = "strong_ml_backstop"
    elif path5:
        path_name = "lgbm_bal_direct"

    # Build a human-readable rejection reason when no path fires
    reject_reason = None
    if not triggered:
        reject_reason = (
            f"apex_score={apex_score:.3f}<{apex_score_entry} | "
            f"vote_lr_bal={lr_bal:.3f}<0.50 | "
            f"hgbc_l2={hgbc_l2:.3f}<0.55 or lgbm_bal={lgbm_bal:.3f}<0.55 | "
            f"mean_proba={mean_proba:.3f}<{_p4_proba} or n_agree={n_agree}<{_p4_agree} | "
            f"lgbm_bal={lgbm_bal:.3f}<{_p5_lgbm}"
        )

    return {
        "triggered":       triggered,
        "apex_score":      apex_score,
        "weight_present":  weight_present,
        "path":            path_name,
        "reject_reason":   reject_reason,
        "path_detail": {
            "apex_score_gate":      path1,
            "vote_lr_bal_gate":     path2,
            "hgbc_lgbm_gate":       path3,
            "strong_ml_backstop":   path4,
            "lgbm_bal_gate":        path5,
        },
    }


def compute_apex_size(
    apex_score,
    n_agree=0,
    current_total_exposure=0.0,
    base_alloc=None,
    max_gross=None,
):
    """Compute Apex Predator position size (Kelly-lite with pyramiding support).

    Formula::

        edge_mult = clip((apex_score - 0.50) / 0.30, 0.0, 1.5)
        conf_mult = 1.0 + 0.5 * (n_agree >= 4)
        size_frac = clip(base_alloc * (1 + edge_mult) * conf_mult, 0.05, 0.45)

    The result is further clamped so that
    ``current_total_exposure + size_frac <= max_gross``.

    Parameters
    ----------
    apex_score             : float  — from compute_apex_score()
    n_agree                : int    — number of agreeing active models
    current_total_exposure : float  — sum of allocations already open
    base_alloc             : float  — baseline allocation; defaults to APEX_BASE_ALLOC (0.20)
    max_gross              : float  — maximum total gross; defaults to APEX_MAX_GROSS (2.0)

    Returns
    -------
    float — allocation fraction in [0.05, 0.45], further capped by remaining headroom.
    """
    try:
        from config import APEX_BASE_ALLOC as _ba, APEX_MAX_GROSS as _mg
    except ImportError:
        _ba, _mg = 0.20, 2.0

    if base_alloc is None:
        base_alloc = _ba
    if max_gross is None:
        max_gross = _mg

    edge_mult = max(0.0, min(1.5, (float(apex_score) - 0.50) / 0.30))
    conf_mult = 1.0 + 0.5 * (int(n_agree) >= 4)
    size_frac = base_alloc * (1.0 + edge_mult) * conf_mult

    # Hard clamp to [0.05, 0.45]
    size_frac = max(0.05, min(0.45, size_frac))

    # Respect total gross exposure cap
    remaining = float(max_gross) - float(current_total_exposure)
    size_frac = max(0.0, min(size_frac, remaining))

    return round(size_frac, 4)


def compute_apex_atr_stops(
    entry_price,
    atr,
    atr_sl_mult=None,
    atr_tp_mult=None,
    sl_floor_pct=0.008,
    sl_ceil_pct=0.040,
    tp_floor_pct=0.025,
    tp_ceil_pct=0.150,
):
    """Compute Apex Predator ATR-based stop-loss and take-profit levels.

    SL = entry - atr_sl_mult * ATR; clamped to [sl_floor_pct, sl_ceil_pct]
    TP = entry + atr_tp_mult * ATR; clamped to [tp_floor_pct, tp_ceil_pct]

    Parameters
    ----------
    entry_price  : float
    atr          : float — Average True Range (14-bar or similar)
    atr_sl_mult  : float or None — defaults to APEX_ATR_SL_MULT (1.25)
    atr_tp_mult  : float or None — defaults to APEX_ATR_TP_MULT (4.0)
    sl_floor_pct : float — minimum SL distance as fraction of entry (default 0.8%)
    sl_ceil_pct  : float — maximum SL distance as fraction of entry (default 4%)
    tp_floor_pct : float — minimum TP distance as fraction of entry (default 2.5%)
    tp_ceil_pct  : float — maximum TP distance as fraction of entry (default 15%)

    Returns
    -------
    dict with keys: "sl_price", "tp_price", "sl_pct", "tp_pct",
                    "trail_arm_pct", "trail_dist_pct", "breakeven_mfe_pct",
                    "time_stop_hrs"
    """
    try:
        from config import (
            APEX_ATR_SL_MULT   as _sl_m,
            APEX_ATR_TP_MULT   as _tp_m,
            APEX_TRAIL_ARM_PCT as _arm,
            APEX_TRAIL_ATR_MULT as _trail_m,
            APEX_BREAKEVEN_MFE as _be,
            APEX_TIME_STOP_HRS as _tsh,
        )
    except ImportError:
        _sl_m, _tp_m, _arm, _trail_m, _be, _tsh = 1.25, 4.0, 0.010, 0.8, 0.02, 48

    if atr_sl_mult is None:
        atr_sl_mult = _sl_m
    if atr_tp_mult is None:
        atr_tp_mult = _tp_m

    ep = float(entry_price)
    if ep <= 0.0:
        return {
            "sl_price": ep, "tp_price": ep,
            "sl_pct": sl_floor_pct, "tp_pct": tp_floor_pct,
            "trail_arm_pct": _arm, "trail_dist_pct": 0.006,
            "breakeven_mfe_pct": _be, "time_stop_hrs": _tsh,
        }

    atr_v = float(atr) if atr and float(atr) > 0 else ep * 0.01  # fallback 1%

    # SL distance
    raw_sl_pct = atr_sl_mult * atr_v / ep
    sl_pct     = max(sl_floor_pct, min(sl_ceil_pct, raw_sl_pct))
    sl_price   = ep * (1.0 - sl_pct)

    # TP distance
    raw_tp_pct = atr_tp_mult * atr_v / ep
    tp_pct     = max(tp_floor_pct, min(tp_ceil_pct, raw_tp_pct))
    tp_price   = ep * (1.0 + tp_pct)

    # Trail distance = max(APEX_TRAIL_ATR_MULT * ATR, 0.6%)
    raw_trail_pct = _trail_m * atr_v / ep
    trail_pct     = max(0.006, raw_trail_pct)

    return {
        "sl_price":          round(sl_price, 8),
        "tp_price":          round(tp_price, 8),
        "sl_pct":            round(sl_pct,   6),
        "tp_pct":            round(tp_pct,   6),
        "trail_arm_pct":     _arm,
        "trail_dist_pct":    round(trail_pct, 6),
        "breakeven_mfe_pct": _be,
        "time_stop_hrs":     _tsh,
    }


# ── Apex Predator: technical overlay entry helpers ────────────────────────────


def apex_breakout_signal(closes, volumes, n_bars=None, vol_mult=None):
    """Return True when price breaks above the rolling N-bar high with a volume spike.

    A breakout is confirmed when:
      1. ``closes[-1]`` > ``max(closes[-n_bars-1 : -1])``   (crosses prior high)
      2. ``volumes[-1]`` >= ``vol_mult × mean(volumes[-n_bars-1 : -1])``

    Parameters
    ----------
    closes   : sequence of float — recent close prices (needs at least n_bars+1 values)
    volumes  : sequence of float — matching volume series (same length as closes)
    n_bars   : int or None — rolling look-back window; defaults to APEX_BREAKOUT_NBARS (20)
    vol_mult : float or None — volume confirmation multiplier; defaults to
               APEX_BREAKOUT_VOL_MULT (1.5)

    Returns
    -------
    bool — True when both breakout conditions are met, False otherwise.
    """
    try:
        from config import (
            APEX_BREAKOUT_NBARS    as _nb,
            APEX_BREAKOUT_VOL_MULT as _vm,
        )
    except ImportError:
        _nb, _vm = 20, 1.5

    if n_bars is None:
        n_bars = _nb
    if vol_mult is None:
        vol_mult = _vm

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed or len(volumes) < needed:
        return False

    try:
        prior_highs = [float(c) for c in closes[-(n_bars + 1):-1]]
        current_close = float(closes[-1])
        prior_vols = [float(v) for v in volumes[-(n_bars + 1):-1]]
        current_vol = float(volumes[-1])
    except (TypeError, ValueError, IndexError):
        return False

    if not prior_highs or not prior_vols:
        return False

    rolling_high = max(prior_highs)
    avg_vol = sum(prior_vols) / len(prior_vols)

    price_breakout = current_close > rolling_high
    vol_confirmed  = avg_vol > 0 and current_vol >= vol_mult * avg_vol

    return price_breakout and vol_confirmed


def apex_pullback_signal(closes, rsi, n_bars=None, rsi_max=None):
    """Return True when RSI is oversold inside a confirmed uptrend (pullback long).

    Conditions:
      1. ``rsi`` <= ``rsi_max``                           (oversold / pullback)
      2. ``closes[-1]`` > ``closes[-n_bars-1]``           (price higher than n_bars ago)

    Parameters
    ----------
    closes  : sequence of float — recent close prices (needs at least n_bars+1 values)
    rsi     : float — current RSI value (0–100)
    n_bars  : int or None — trend confirmation look-back; defaults to
              APEX_PULLBACK_TREND_BARS (10)
    rsi_max : float or None — maximum RSI to qualify as a pullback; defaults to
              APEX_PULLBACK_RSI_MAX (35)

    Returns
    -------
    bool — True when pullback conditions are met, False otherwise.
    """
    try:
        from config import (
            APEX_PULLBACK_TREND_BARS as _nb,
            APEX_PULLBACK_RSI_MAX    as _rmax,
        )
    except ImportError:
        _nb, _rmax = 10, 35

    if n_bars is None:
        n_bars = _nb
    if rsi_max is None:
        rsi_max = _rmax

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed:
        return False

    try:
        current_close = float(closes[-1])
        past_close    = float(closes[-(n_bars + 1)])
        rsi_val       = float(rsi)
    except (TypeError, ValueError, IndexError):
        return False

    oversold      = rsi_val <= rsi_max
    in_uptrend    = current_close > past_close

    return oversold and in_uptrend


def apex_momentum_continuation_signal(closes, volumes, n_bars=None, vol_mult=None):
    """Return True on momentum continuation: N consecutive higher closes + volume spike.

    Conditions:
      1. Each of the last ``n_bars`` closes is strictly higher than the one before it.
      2. ``volumes[-1]`` >= ``vol_mult × mean(volumes[-n_bars-1 : -1])``

    Parameters
    ----------
    closes   : sequence of float — recent close prices (needs at least n_bars+1 values)
    volumes  : sequence of float — matching volume series (same length as closes)
    n_bars   : int or None — number of consecutive higher closes required; defaults to
               APEX_MOMENTUM_CONT_BARS (3)
    vol_mult : float or None — volume confirmation multiplier; defaults to
               APEX_MOMENTUM_CONT_VOL_MULT (1.5)

    Returns
    -------
    bool — True when momentum continuation conditions are met, False otherwise.
    """
    try:
        from config import (
            APEX_MOMENTUM_CONT_BARS     as _nb,
            APEX_MOMENTUM_CONT_VOL_MULT as _vm,
        )
    except ImportError:
        _nb, _vm = 3, 1.5

    if n_bars is None:
        n_bars = _nb
    if vol_mult is None:
        vol_mult = _vm

    n_bars = max(1, int(n_bars))
    needed = n_bars + 1
    if len(closes) < needed or len(volumes) < needed:
        return False

    try:
        recent_closes = [float(c) for c in closes[-(n_bars + 1):]]
        prior_vols    = [float(v) for v in volumes[-(n_bars + 1):-1]]
        current_vol   = float(volumes[-1])
    except (TypeError, ValueError, IndexError):
        return False

    if len(recent_closes) < n_bars + 1 or not prior_vols:
        return False

    # All consecutive pairs must be strictly increasing
    consecutive_higher = all(
        recent_closes[i] > recent_closes[i - 1]
        for i in range(1, len(recent_closes))
    )

    avg_vol       = sum(prior_vols) / len(prior_vols)
    vol_confirmed = avg_vol > 0 and current_vol >= vol_mult * avg_vol

    return consecutive_higher and vol_confirmed


def apex_rejected_entry_log(votes, mean_proba, n_agree, decision):
    """Build a structured log entry for a rejected Apex Predator entry attempt.

    Returns a dict with the specific gate(s) that caused the rejection so that
    callers can emit it to the algorithm log for diagnostics.

    Parameters
    ----------
    votes      : dict[str, float] — vote dict passed to apex_entry_decision()
    mean_proba : float — ensemble mean probability
    n_agree    : int   — number of agreeing models
    decision   : dict  — return value of apex_entry_decision()

    Returns
    -------
    dict with keys:
      "triggered"     : bool
      "reject_reason" : str or None
      "path_detail"   : dict — per-path bool results
      "apex_score"    : float
      "mean_proba"    : float
      "n_agree"       : int
      "votes_snapshot": dict — copy of the votes dict
    """
    return {
        "triggered":      decision.get("triggered", False),
        "reject_reason":  decision.get("reject_reason"),
        "path_detail":    decision.get("path_detail", {}),
        "apex_score":     decision.get("apex_score", 0.0),
        "mean_proba":     float(mean_proba),
        "n_agree":        int(n_agree),
        "votes_snapshot": {k: float(v) for k, v in votes.items()},
    }
