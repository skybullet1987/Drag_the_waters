# ── strategy.py: aggressive_config + apex_voting + risk + profit_voting + execution + shadow_lab ──
#
# === APEX PREDATOR PARAMETERS ===
#
# This module contains all tunable strategy constants for the "apex predator /
# gatling gun" high-frequency aggressive trading profile. Key settings:
#
#   Entry gates (drastically loosened to fire on ~95% of signals):
#     APEX_GATE_MIN_CLASS_PROBA  = 0.45   (was ~0.55+)
#     APEX_GATE_MIN_N_AGREE      = 1      (was 2-3)
#     APEX_GATE_MIN_FINAL_SCORE  = 0.0    (no veto)
#     APEX_GATE_COOLDOWN_MIN     = 15     (15 min, was 60+)
#     APEX_GATE_MAX_CONCURRENT   = 12     (was 1-3)
#     APEX_WEIGHTED_YES_THRESHOLD = 0.45  (was 0.60)
#
#   Exits (tighter for faster turnover):
#     APEX_SL_ATR_MULT           = 1.5    (1.5x ATR stop, was 3x)
#     APEX_SL_PCT_FLOOR          = 0.025  (2.5% max SL)
#     APEX_TIME_STOP_DAYS        = 30     (exit stale positions)
#
#   Conviction-weighted sizing (boost on high-conviction signals):
#     APEX_SIZE_BASE_ALLOC       = 0.10   (10% base per trade)
#     APEX_SIZE_MAX_FRAC         = 0.25   (25% max)
#     APEX_SIZE_CONV_K           = 4.0    (conviction scaler)
#     APEX_MAX_LEVERAGE          = 3.0    (up to 3x notional)
#
# These parameters target ≥200 orders per backtest period, ≥10 symbols, and
# are tuned based on model_accuracy_summary showing lgbm_bal PF 1.7-3.4,
# hgbc_l2 PF 3.35, lr_bal PF 7.99 at the 0.50 threshold.
#
import numpy as np
from collections import deque
from datetime import timedelta
try:
    from AlgorithmImports import *
except ImportError:
    pass


# ===============================================================================
# aggressive_config
# ===============================================================================

# ── Apex Predator — aggressive strategy configuration ────────────────────────
#
# Named constants for the "apex predator / gatling gun" strategy profile.
# Evidence source: combined_export_20260503_132454.txt model_accuracy_summary
#   - vote_lgbm_bal, shadow_lgbm_bal, active_lgbm_bal: PF 1.7–3.4 at thr 0.50
#   - hgbc_l2: PF 3.35 at thr 0.55–0.60
#   - lr_bal:  PF 7.99 at thr 0.50 (small N but highest discriminating power)
#   - vote_gnb: always near 1.0 → useless as discriminator → weight 0
#   - vote_lr, vote_xgb_bal, vote_cal_*: near-zero medians → weight 0 / 0.5
#
# Import from here in strategy code; do NOT hardcode thresholds inline.
# ─────────────────────────────────────────────────────────────────────────────

# ── Section D: weighted vote aggregator weights ───────────────────────────────
# Derived from profit_factor_yes_50 column of model_accuracy_summary_hotfix.
# Models with evidence of positive expectancy get higher weights.
# Dead voters (gnb, lr) are zeroed; weak voters (xgb_bal, cal_*) kept low.
APEX_WEIGHTED_VOTE_WEIGHTS = {
    "lgbm_bal":    2.5,   # PF 1.7–3.4 at thr 0.50 — consistent performer
    "hgbc_l2":     2.0,   # PF 3.35 at thr 0.55–0.60
    "rf":          1.5,   # PF 2.76 at thr 0.60
    "et":          1.0,   # diversifier
    "lr_bal":      1.5,   # PF 7.99 at thr 0.50 (small N, high signal)
    "lgbm":        1.0,   # base lgbm (non-balanced)
    "gnb":         0.0,   # constant ~1.0 → no discrimination power
    "lr":          0.0,   # near-zero votes → dead voter
    "xgb_bal":     0.5,   # weak signal, occasional contribution
    "cal_et":      0.5,   # calibrated ET — modest contribution
    "cal_rf":      0.5,   # calibrated RF — modest contribution
    "rf_shallow":  0.3,   # near-zero median → minimal weight
    "et_shallow":  0.3,   # near-zero median → minimal weight
}

# Weighted yes-fraction threshold: trade fires if weighted_yes_fraction >= this
APEX_WEIGHTED_YES_THRESHOLD = 0.45   # lower than majority vote → more trades

# Alternative fire conditions (OR logic with threshold):
#   momentum_override = True                         → always fire
#   hgbc_l2 >= APEX_COMBO_HGBC_MIN AND lgbm_bal >= APEX_COMBO_LGBM_MIN → fire
APEX_COMBO_HGBC_MIN  = 0.55   # hgbc_l2 proba floor for combo trigger
APEX_COMBO_LGBM_MIN  = 0.55   # lgbm_bal proba floor for combo trigger

# ── Section A: gating / fire-rate constants ───────────────────────────────────
# Lower these vs. defaults to allow far more orders (target ≥ 200 per 17 months)
APEX_GATE_MIN_CLASS_PROBA  = 0.45   # was ~0.55+; lower floor fires more signals
APEX_GATE_MIN_FINAL_SCORE  = 0.0    # no minimum final-score veto
APEX_GATE_MIN_N_AGREE      = 1      # at least 1 model must agree (was 2–3)
APEX_GATE_COOLDOWN_MIN     = 15     # per-symbol reentry cooldown (min)
APEX_GATE_MAX_CONCURRENT   = 12     # concurrent open positions
APEX_GATE_MAX_PER_SYMBOL   = 3      # max simultaneous positions per symbol

# ── Section B: conviction-weighted sizing ────────────────────────────────────
# size = base_alloc * (1 + k * (final_score - 0.5)), clipped to [0.05, 0.25]
APEX_SIZE_BASE_ALLOC   = 0.10   # baseline per-trade allocation (10% equity)
APEX_SIZE_CONV_K       = 4.0    # conviction scaling factor
APEX_SIZE_MIN_FRAC     = 0.05   # minimum position size (5% equity)
APEX_SIZE_MAX_FRAC     = 0.25   # maximum position size (25% equity)
APEX_USE_LEVERAGE      = True   # allow up to 3x total notional
APEX_MAX_LEVERAGE      = 3.0    # total notional leverage cap

# ── Section C: stop-loss / trailing / time-stop ───────────────────────────────
APEX_SL_ATR_MULT       = 1.5    # SL = entry - 1.5 * ATR (tighter than prev)
APEX_SL_PCT_FLOOR      = 0.025  # 2.5% min SL distance (whichever is tighter)
APEX_TP_USE_FIXED      = False  # no fixed TP — let trail run
APEX_TRAIL_MULT        = 4.0    # ATR trail distance multiplier
APEX_TIME_STOP_DAYS    = 30     # close if open > 30 days without sufficient MFE
APEX_PYRAMID_MFE_ATR   = 1.0    # add 50% tranche after +1 ATR unrealised
APEX_PYRAMID_ADD_FRAC  = 0.50   # size of each pyramid add vs original
APEX_PYRAMID_MAX_ADDS  = 2      # maximum pyramid add-ons per position


# ===============================================================================
# apex_voting
# ===============================================================================

# ── Apex Voting — weighted ensemble vote aggregator ──────────────────────────
#
# Implements the Section D smarter ensemble vote from the apex-predator spec.
#
# Evidence (diagnostics export combined_export_20260503_132454.txt):
#   - vote_lr, vote_gnb, vote_xgb_bal, vote_cal_* are near-zero or constant
#     → dead voters dragging the ensemble vote fraction down.
#   - lgbm_bal, hgbc_l2, lr_bal have the highest profit-factor evidence.
#   - Previous simple majority vote was vetoed by dead-weight voters.
#
# New rule (OR logic):
#   FIRE if:
#     weighted_yes_fraction >= APEX_WEIGHTED_YES_THRESHOLD (0.45)
#     OR momentum_override is True
#     OR (hgbc_l2 >= APEX_COMBO_HGBC_MIN AND lgbm_bal >= APEX_COMBO_LGBM_MIN)
#
# All weights/thresholds are named constants in aggressive_config.py.
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted_yes_fraction(
    model_votes,
    vote_threshold=0.50,
    weights=None,
):
    """Compute the weighted yes-fraction from a dict of per-model vote probabilities."""
    if weights is None:
        weights = APEX_WEIGHTED_VOTE_WEIGHTS

    yes_w   = 0.0
    total_w = 0.0
    yes_models        = []
    no_models         = []
    zero_weight_models = []

    for model_id, proba in model_votes.items():
        w = float(weights.get(model_id, 1.0))  # unknown models default to 1.0
        if w == 0.0:
            zero_weight_models.append(model_id)
            # Still accumulate to total so 0-weight models don't inflate fraction
            total_w += 0.0
            continue

        total_w += w
        if float(proba) >= vote_threshold:
            yes_w += w
            yes_models.append(model_id)
        else:
            no_models.append(model_id)

    frac = yes_w / total_w if total_w > 0.0 else 0.0

    return {
        "weighted_yes_fraction": round(frac, 6),
        "yes_weight":            round(yes_w, 6),
        "total_weight":          round(total_w, 6),
        "yes_models":            yes_models,
        "no_models":             no_models,
        "zero_weight_models":    zero_weight_models,
    }


def apex_voting_decision(
    model_votes,
    momentum_override=False,
    vote_threshold=0.50,
    yes_threshold=None,
    combo_hgbc_min=None,
    combo_lgbm_min=None,
    weights=None,
):
    """Evaluate the apex voting decision using the weighted ensemble rule."""
    if yes_threshold is None:
        yes_threshold = APEX_WEIGHTED_YES_THRESHOLD
    if combo_hgbc_min is None:
        combo_hgbc_min = APEX_COMBO_HGBC_MIN
    if combo_lgbm_min is None:
        combo_lgbm_min = APEX_COMBO_LGBM_MIN

    vote_result = compute_weighted_yes_fraction(
        model_votes,
        vote_threshold=vote_threshold,
        weights=weights,
    )
    frac = vote_result["weighted_yes_fraction"]

    hgbc_l2  = float(model_votes.get("hgbc_l2",  0.0))
    lgbm_bal = float(model_votes.get("lgbm_bal", 0.0))

    path1 = frac >= yes_threshold
    path2 = bool(momentum_override)
    path3 = hgbc_l2 >= combo_hgbc_min and lgbm_bal >= combo_lgbm_min

    triggered = path1 or path2 or path3

    trigger_path = None
    if path1:
        trigger_path = "weighted_yes_fraction"
    elif path2:
        trigger_path = "momentum_override"
    elif path3:
        trigger_path = "hgbc_lgbm_combo"

    reject_reason = None
    if not triggered:
        reject_reason = (
            f"weighted_yes_fraction={frac:.3f}<{yes_threshold} | "
            f"momentum_override={momentum_override} | "
            f"hgbc_l2={hgbc_l2:.3f}<{combo_hgbc_min} or "
            f"lgbm_bal={lgbm_bal:.3f}<{combo_lgbm_min}"
        )

    return {
        "triggered":             triggered,
        "trigger_path":          trigger_path,
        "weighted_yes_fraction": frac,
        "yes_threshold":         yes_threshold,
        "momentum_override":     bool(momentum_override),
        "combo_fired":           path3,
        "yes_models":            vote_result["yes_models"],
        "no_models":             vote_result["no_models"],
        "zero_weight_models":    vote_result["zero_weight_models"],
        "reject_reason":         reject_reason,
    }


# ===============================================================================
# risk
# ===============================================================================

# ── Vox Risk ──────────────────────────────────────────────────────────────────
#
# Consolidated module for all pre-trade guards:
#   • RegimeFilter  — 4h BTC SMA(20) + slope gate
#   • kelly_fraction / compute_qty — fractional-Kelly position sizing
#   • RiskManager   — per-coin cooldown, daily SL cap, drawdown circuit-breaker
# Previously split across regime.py, sizing.py, and risk.py.
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeFilter:
    """Bitcoin-based macro regime filter."""

    _SMA_PERIOD   = 20
    _SLOPE_PERIOD = 5

    def __init__(self):
        self._closes = deque(maxlen=self._SMA_PERIOD + self._SLOPE_PERIOD)

    def update_btc(self, algorithm, btc_sym):
        """
        Register a 4-hour consolidator on *btc_sym* that feeds this filter.
        Call once from ``algorithm.initialize()``.
        """
        algorithm.consolidate(
            btc_sym,
            timedelta(hours=4),
            self._on_4h_bar,
        )

    def _on_4h_bar(self, bar):
        """Receive a closed 4-hour bar and append the close price."""
        self._closes.append(float(bar.close))

    def is_risk_on(self, btc_sym, sym=None):
        """
        Return True if the macro regime permits a long entry.

        Parameters
        ----------
        btc_sym : Symbol — BTC symbol (exempt from the regime gate).
        sym     : Symbol or None — symbol being evaluated.

        Returns
        -------
        bool
        """
        if sym is not None and sym == btc_sym:
            return True

        closes = list(self._closes)
        if len(closes) < self._SMA_PERIOD:
            return True   # insufficient history → allow trading

        sma20  = float(np.mean(closes[-self._SMA_PERIOD:]))
        latest = closes[-1]
        if latest < sma20:
            return False

        if len(closes) >= self._SLOPE_PERIOD and self._SLOPE_PERIOD > 1:
            window = closes[-self._SLOPE_PERIOD:]
            slope  = (window[-1] - window[0]) / (self._SLOPE_PERIOD - 1)
            if slope < 0.0:
                return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING  (fractional Kelly)
# ═══════════════════════════════════════════════════════════════════════════════

def kelly_fraction(p, tp, sl, kelly_frac=0.25, max_alloc=0.80):
    """Compute the fractional-Kelly allocation for a long trade."""
    if sl <= 0:
        return 0.0
    b      = tp / sl
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, min(f_full * kelly_frac, max_alloc))


def compute_qty(
    mean_proba,
    tp,
    sl,
    price,
    portfolio_value,
    kelly_frac,
    max_alloc,
    cash_buffer,
    use_kelly,
    allocation,
    min_alloc=0.0,
):
    """Compute the quantity (in coin units) to purchase."""
    if use_kelly:
        alloc = kelly_fraction(mean_proba, tp, sl, kelly_frac, max_alloc)
        if alloc <= 0.0:
            alloc = allocation
        elif min_alloc > 0.0:
            alloc = max(alloc, min_alloc)
    else:
        alloc = allocation

    alloc        = min(alloc, max_alloc)   # honor hard ceiling after any floor
    dollar_value = portfolio_value * alloc * cash_buffer
    qty          = dollar_value / price if price > 0 else 0.0
    return qty, alloc


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER


class RiskManager:
    """Pre-trade risk guardrails for the Vox strategy."""

    def __init__(
        self,
        max_daily_sl,
        cooldown_mins,
        sl_cooldown_mins,
        max_dd_pct,
        cash_buffer,
    ):
        self._max_daily_sl     = max_daily_sl
        self._cooldown         = timedelta(minutes=cooldown_mins)
        self._sl_cooldown      = timedelta(minutes=sl_cooldown_mins)
        self._max_dd_pct       = max_dd_pct
        self._cash_buffer      = cash_buffer

        # Mutable state
        self._daily_sl         = 0          # stop-loss hits today
        self._last_exit_time   = None       # time of most recent exit (any)
        self._sl_exit_times    = {}         # sym -> time of last SL exit
        self._rolling_high     = None       # rolling portfolio high-water mark

    # ── State updates ─────────────────────────────────────────────────────────

    def record_exit(self, sym, is_sl, exit_time):
        """
        Record that a position in *sym* was exited at *exit_time*.

        Parameters
        ----------
        sym       : Symbol — the exited symbol.
        is_sl     : bool   — True if the exit was triggered by stop-loss.
        exit_time : datetime — the exit timestamp.
        """
        self._last_exit_time = exit_time
        if is_sl:
            self._daily_sl += 1
            self._sl_exit_times[sym] = exit_time

    def record_sl(self):
        """Increment the daily stop-loss counter (alternative to record_exit)."""
        self._daily_sl += 1

    def reset_daily(self):
        """Reset daily counters.  Call at midnight via scheduled event."""
        self._daily_sl       = 0

    def update_rolling_high(self, portfolio_value):
        """
        Update the 30-day rolling high-water mark.

        Parameters
        ----------
        portfolio_value : float — current total portfolio value.
        """
        if self._rolling_high is None or portfolio_value > self._rolling_high:
            self._rolling_high = portfolio_value

    # ── Checks ────────────────────────────────────────────────────────────────

    def check_drawdown(self, portfolio_value):
        """
        Return True (circuit-breaker tripped) if equity has dropped more than
        *max_dd_pct* from the rolling high-water mark.

        Parameters
        ----------
        portfolio_value : float

        Returns
        -------
        bool
            True  → halt trading.
            False → trading permitted.
        """
        if self._rolling_high is None or self._rolling_high == 0:
            return False
        dd = (self._rolling_high - portfolio_value) / self._rolling_high
        return dd > self._max_dd_pct

    def can_enter(self, sym, current_time, portfolio_value, rolling_high=None):
        """Evaluate all pre-trade risk checks for a potential long entry."""
        # Update rolling high with caller-provided value if given
        if rolling_high is not None:
            self.update_rolling_high(rolling_high)
        self.update_rolling_high(portfolio_value)

        # ── Daily SL cap ──────────────────────────────────────────────────────
        if self._daily_sl >= self._max_daily_sl:
            return False, f"daily_sl_cap ({self._daily_sl}/{self._max_daily_sl})"

        # ── Drawdown circuit-breaker ───────────────────────────────────────────
        if self.check_drawdown(portfolio_value):
            return False, "drawdown_circuit_breaker"

        # ── Global post-exit cooldown ─────────────────────────────────────────
        if self._last_exit_time is not None:
            elapsed = current_time - self._last_exit_time
            if elapsed < self._cooldown:
                remaining = int((self._cooldown - elapsed).total_seconds() / 60)
                return False, f"global_cooldown ({remaining}m remaining)"

        # ── Per-coin SL cooldown ──────────────────────────────────────────────
        if sym in self._sl_exit_times:
            elapsed = current_time - self._sl_exit_times[sym]
            if elapsed < self._sl_cooldown:
                remaining = int((self._sl_cooldown - elapsed).total_seconds() / 60)
                return False, f"sl_cooldown_{sym.value} ({remaining}m remaining)"

        return True, "ok"


# ===============================================================================
# profit_voting
# ===============================================================================

# ── Vox Profit-Voting Gate ────────────────────────────────────────────────────
#
# Active vote-score / ranking system for ruthless profit-voting mode.
#
# This module provides:
#   - compute_vote_score()  : calculate vote_score, vote_yes_fraction, top3_mean
#   - check_profit_voting_gate() : entry-gate check for ruthless profit-voting mode
#   - chop supermajority requirements
#
# None of these functions affect balanced/conservative/aggressive profiles.
# ─────────────────────────────────────────────────────────────────────────────


# ── Vote score constants (defaults; override via config) ──────────────────────
# Bootstrap defaults — intentionally relaxed to restore trading and gather data.
# Tighten after observing candidate journal / reject diagnostics.
DEFAULT_VOTE_THRESHOLD          = 0.50   # probability floor to count as "yes" vote
DEFAULT_VOTE_YES_FRACTION_MIN   = 0.34   # min fraction of active models voting yes
DEFAULT_TOP3_MEAN_MIN           = 0.55   # min mean of top-3 active probabilities
DEFAULT_VOTE_EV_FLOOR           = 0.001  # minimum EV for all profit-voting entries

# Chop market mode — stricter supermajority (still stricter than trend, but relaxed)
DEFAULT_CHOP_VOTE_YES_FRAC_MIN  = 0.50   # yes-fraction requirement in chop
DEFAULT_CHOP_TOP3_MEAN_MIN      = 0.60   # top-3-mean requirement in chop
DEFAULT_CHOP_PRED_RETURN_MIN    = 0.000  # pred_return floor in chop (disabled)
DEFAULT_CHOP_EV_MIN             = 0.002  # EV floor required in chop

# Weight allocation for vote_score composite
_W_ACTIVE_MEAN   = 0.40
_W_YES_FRAC      = 0.30
_W_TOP3_MEAN     = 0.30


def compute_vote_score(active_votes, vote_thr=DEFAULT_VOTE_THRESHOLD):
    """Compute profit-voting score fields from active-model votes."""
    if not active_votes:
        return {
            "active_model_count": 0,
            "vote_yes_fraction":  0.0,
            "top3_mean":          0.0,
            "vote_score":         0.0,
        }

    vals = sorted(active_votes.values(), reverse=True)
    n    = len(vals)
    am   = float(np.mean(vals))
    yes_frac = sum(1 for v in vals if v >= vote_thr) / n
    top3_mean = float(np.mean(vals[:3])) if vals else 0.0
    vote_score = _W_ACTIVE_MEAN * am + _W_YES_FRAC * yes_frac + _W_TOP3_MEAN * top3_mean

    return {
        "active_model_count": n,
        "vote_yes_fraction":  yes_frac,
        "top3_mean":          top3_mean,
        "vote_score":         vote_score,
    }


def check_profit_voting_gate(
    conf,
    market_mode,
    vote_thr=DEFAULT_VOTE_THRESHOLD,
    vote_yes_frac_min=DEFAULT_VOTE_YES_FRACTION_MIN,
    top3_mean_min=DEFAULT_TOP3_MEAN_MIN,
    chop_vote_yes_frac_min=DEFAULT_CHOP_VOTE_YES_FRAC_MIN,
    chop_top3_mean_min=DEFAULT_CHOP_TOP3_MEAN_MIN,
    chop_pred_return_min=DEFAULT_CHOP_PRED_RETURN_MIN,
    chop_ev_min=DEFAULT_CHOP_EV_MIN,
    ev_score=0.0,
    ev_floor=0.0,
    require_min_active_models=3,
):
    """Check profit-voting entry gate for ruthless profit-voting mode."""
    active_votes     = conf.get("active_votes", {})
    active_count     = conf.get("active_model_count", len(active_votes))
    vote_yes_frac    = conf.get("vote_yes_fraction", 0.0)
    top3_mean        = conf.get("top3_mean", 0.0)
    pred_return      = conf.get("pred_return", 0.0)

    # Recompute if fields not pre-populated
    if "vote_yes_fraction" not in conf and active_votes:
        vs = compute_vote_score(active_votes, vote_thr)
        active_count  = vs["active_model_count"]
        vote_yes_frac = vs["vote_yes_fraction"]
        top3_mean     = vs["top3_mean"]

    # Require minimum active model count
    if active_count < require_min_active_models:
        return False, f"active_count={active_count} < {require_min_active_models}"

    # EV floor — applies to all market modes
    if ev_floor > 0.0 and ev_score < ev_floor:
        return False, f"ev_floor={ev_score:.4f} < {ev_floor}"

    is_chop = (market_mode is not None and "chop" in str(market_mode).lower())

    if is_chop:
        # Supermajority required in chop
        if vote_yes_frac < chop_vote_yes_frac_min:
            return (
                False,
                f"chop: vote_yes_frac={vote_yes_frac:.2f} < {chop_vote_yes_frac_min}"
            )
        if top3_mean < chop_top3_mean_min:
            return (
                False,
                f"chop: top3_mean={top3_mean:.3f} < {chop_top3_mean_min}"
            )
        if pred_return < chop_pred_return_min:
            return (
                False,
                f"chop: pred_return={pred_return:.4f} < {chop_pred_return_min}"
            )
        if ev_score < chop_ev_min:
            return (
                False,
                f"chop: ev_score={ev_score:.4f} < {chop_ev_min}"
            )
        return True, "chop_supermajority"

    # Trend / pump / unknown market mode
    if vote_yes_frac < vote_yes_frac_min:
        return (
            False,
            f"vote_yes_frac={vote_yes_frac:.2f} < {vote_yes_frac_min}"
        )
    if top3_mean < top3_mean_min:
        return (
            False,
            f"top3_mean={top3_mean:.3f} < {top3_mean_min}"
        )
    return True, "profit_vote_pass"


def format_profit_vote_log(sym_str, conf, vote_score_fields, market_mode=None, approved=True, reason=""):
    """Format a compact log line for profit-voting decisions.

    Parameters
    ----------
    sym_str           : str
    conf              : dict — predict_with_confidence output
    vote_score_fields : dict — from compute_vote_score
    market_mode       : str or None
    approved          : bool
    reason            : str

    Returns
    -------
    str
    """
    am  = conf.get("active_mean", conf.get("class_proba", 0.0))
    n   = vote_score_fields.get("active_model_count", 0)
    yf  = vote_score_fields.get("vote_yes_fraction", 0.0)
    t3  = vote_score_fields.get("top3_mean", 0.0)
    vs  = vote_score_fields.get("vote_score", 0.0)
    mode_str = f" mode={market_mode}" if market_mode else ""
    status   = "OK" if approved else f"BLOCKED:{reason}"
    return (
        f"[profit_vote] {sym_str}{mode_str}"
        f" active_count={n} active_mean={am:.3f}"
        f" yes_frac={yf:.2f} top3={t3:.3f}"
        f" vote_score={vs:.4f}"
        f" status={status}"
    )


# ── Ruthless active-model promotion ──────────────────────────────────────────
# In ruthless profit-voting mode, models listed in RUTHLESS_ACTIVE_MODELS that
# are currently in shadow_votes are promoted into active_votes for the purpose
# of vote-score / gate computation.  Backward-compat fields (class_proba,
# std_proba, n_agree) are updated to reflect the promoted active pool.
# Models in RUTHLESS_DIAGNOSTIC_MODELS are never promoted.

def apply_ruthless_active_promotion(conf, active_models, diagnostic_models=None):
    """Promote shadow models to active pool in ruthless profit-voting mode."""
    if not active_models:
        return 0

    diag_set    = set(diagnostic_models or [])
    shadow      = conf.get("shadow_votes", {})
    current_act = conf.get("active_votes", {})

    # Build promoted set: existing active + shadow models in active_models list
    promoted = {}
    for mid, proba in current_act.items():
        if mid not in diag_set:
            promoted[mid] = proba

    added = 0
    for mid in active_models:
        if mid in diag_set:
            continue
        if mid in promoted:
            continue  # already active
        if mid in shadow:
            promoted[mid] = shadow[mid]
            added += 1

    if added == 0:
        return 0  # nothing to do

    # Recompute vote-score fields from promoted active pool
    vs = compute_vote_score(promoted)

    conf["active_votes"]         = promoted
    conf["active_model_count"]   = vs["active_model_count"]
    conf["vote_yes_fraction"]    = vs["vote_yes_fraction"]
    conf["top3_mean"]            = vs["top3_mean"]
    conf["vote_score"]           = vs["vote_score"]

    # Update backward-compat fields to reflect promoted active pool
    if promoted:
        vals = list(promoted.values())
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        nagree = int(sum(1 for v in vals if v >= 0.5))
        conf["class_proba"]    = mean_v
        conf["mean_proba"]     = mean_v
        conf["std_proba"]      = std_v
        conf["n_agree"]        = nagree
        conf["active_mean"]    = mean_v
        conf["active_std"]     = std_v
        conf["active_n_agree"] = nagree

    return added


# ── Profit-voting reject counters ─────────────────────────────────────────────

def make_pv_counters():
    """Return a fresh profit-voting reject counter dict."""
    return {
        "candidates":         0,
        "pass":               0,
        "fail_active_count":  0,
        "fail_ev_floor":      0,
        "fail_yes_frac":      0,
        "fail_top3":          0,
        "fail_chop_yes_frac": 0,
        "fail_chop_top3":     0,
        "fail_chop_pred":     0,
        "fail_chop_ev":       0,
        "no_active_votes":    0,
    }


def increment_pv_counter(pv_counters, reason):
    """Increment the appropriate reject counter from a gate reject reason string."""
    if "active_count" in reason:
        pv_counters["fail_active_count"] += 1
    elif "ev_floor" in reason:
        pv_counters["fail_ev_floor"] += 1
    elif "chop: vote_yes_frac" in reason:
        pv_counters["fail_chop_yes_frac"] += 1
    elif "chop: top3_mean" in reason:
        pv_counters["fail_chop_top3"] += 1
    elif "chop: pred_return" in reason:
        pv_counters["fail_chop_pred"] += 1
    elif "chop: ev_score" in reason:
        pv_counters["fail_chop_ev"] += 1
    elif "vote_yes_frac" in reason:
        pv_counters["fail_yes_frac"] += 1
    elif "top3_mean" in reason:
        pv_counters["fail_top3"] += 1
    elif "no_active" in reason:
        pv_counters["no_active_votes"] += 1


def format_pv_reject_log(sym_str, conf, market_mode, reason):
    """Format a compact per-candidate profit-voting reject log line.

    Example:
        [pv_reject] ADAUSD mode=chop reason=top3_mean yes_frac=0.50 top3=0.54 ev=0.003 active_n=5
    """
    yf  = conf.get("vote_yes_fraction", 0.0)
    t3  = conf.get("top3_mean", 0.0)
    n   = conf.get("active_model_count", len(conf.get("active_votes", {})))
    ev  = 0.0  # caller may inject via conf if needed
    pr  = conf.get("pred_return", 0.0)
    mode_str = f" mode={market_mode}" if market_mode else ""
    return (
        f"[pv_reject] {sym_str}{mode_str}"
        f" reason={reason}"
        f" yes_frac={yf:.2f} top3={t3:.3f}"
        f" pred={pr:.4f}"
        f" active_n={n}"
    )


def format_pv_summary_log(pv_counters):
    """Format a one-line profit-voting cycle summary log.

    Example:
        [pv_summary] candidates=18 pass=0 fail_active=2 fail_yes_frac=7 fail_top3=5 fail_chop=4
    """
    c   = pv_counters
    return (
        f"[pv_summary] candidates={c['candidates']}"
        f" pass={c['pass']}"
        f" fail_active_count={c['fail_active_count']}"
        f" fail_ev_floor={c['fail_ev_floor']}"
        f" fail_yes_frac={c['fail_yes_frac']}"
        f" fail_top3={c['fail_top3']}"
        f" fail_chop_yes_frac={c['fail_chop_yes_frac']}"
        f" fail_chop_top3={c['fail_chop_top3']}"
        f" fail_chop_pred={c['fail_chop_pred']}"
        f" fail_chop_ev={c['fail_chop_ev']}"
        f" no_active={c['no_active_votes']}"
    )


# ===============================================================================
# execution
# ===============================================================================

"""Exit and entry execution helpers for Vox ruthless mode."""


# ── Breakeven stop ──────────────────────────────────────────────────────────

def apply_breakeven(ret, max_return_seen, breakeven_after, breakeven_buffer):
    """Return True if breakeven stop should trigger.

    Triggers when the position has reached +breakeven_after and then pulled
    back to at or below +breakeven_buffer (protecting most of the gain).
    """
    if max_return_seen >= breakeven_after:
        return ret <= breakeven_buffer
    return False


def is_breakeven_active(max_return_seen, breakeven_after):
    """Return True if the breakeven protection has been activated."""
    return max_return_seen >= breakeven_after


# ── Momentum-failure exit ────────────────────────────────────────────────────

def should_exit_momentum_fail(elapsed_minutes, ret, feat,
                               min_hold_minutes=30, fail_loss=-0.012):
    """Return True if the trade should be cut early due to momentum failure.

    Conditions:
      - held >= min_hold_minutes
      - current return <= fail_loss
      - ret_4 (feat[1]) < 0
      - ret_16 (feat[3]) < 0
    """
    if elapsed_minutes < min_hold_minutes:
        return False
    if ret > fail_loss:
        return False
    if feat is None or len(feat) < 4:
        return False
    return float(feat[1]) < 0.0 and float(feat[3]) < 0.0


# ── Timeout extension ────────────────────────────────────────────────────────

def evaluate_timeout(
    elapsed_hours, ret, feat, toh,
    timeout_min_profit=0.03,
    timeout_extend_hours=12,
    max_timeout_hours=48,
    extension_hours_used=0.0,
):
    """Evaluate timeout logic for ruthless runner mode.

    Returns one of: 'exit', 'extend', 'hold'
    """
    if elapsed_hours < toh:
        return 'hold'

    if ret >= timeout_min_profit:
        return 'exit'

    if (
        extension_hours_used < (max_timeout_hours - toh)
        and feat is not None
        and len(feat) >= 4
        and ret > -0.01
        and float(feat[1]) > 0.0
    ):
        return 'extend'

    return 'exit'


# ── Entry limit order TTL tracker ────────────────────────────────────────────

class LimitOrderTracker:
    """Track a pending entry limit order and manage TTL expiry."""

    def __init__(self):
        self._order_id    = None
        self._submit_time = None
        self._ttl_minutes = 3

    def start(self, order_id, submit_time, ttl_minutes=3):
        self._order_id    = order_id
        self._submit_time = submit_time
        self._ttl_minutes = ttl_minutes

    def is_pending(self):
        return self._order_id is not None

    def is_expired(self, current_time):
        if self._submit_time is None:
            return False
        elapsed = (current_time - self._submit_time).total_seconds() / 60.0
        return elapsed >= self._ttl_minutes

    def cancel_and_clear(self, algo):
        """Cancel the limit order (if possible) and reset state."""
        if self._order_id is not None:
            try:
                algo.transactions.cancel_order(self._order_id)
            except Exception:
                pass
        self._order_id    = None
        self._submit_time = None

    def clear(self):
        self._order_id    = None
        self._submit_time = None

    @property
    def order_id(self):
        return self._order_id


# ── Candidate scoring helper ─────────────────────────────────────────────────

def evaluate_candidate(
    sym, feat, conf, price, atr,
    risk_profile, tp_base, sl_base, atr_tp_mult, atr_sl_mult,
    cost_fraction,
    momentum_override_enabled, momentum_ret4_min, momentum_ret16_min,
    momentum_volume_min, momentum_btc_rel_min, momentum_override_min_ev,
    ruthless_confirm_ev_min, ruthless_confirm_proba_min, ruthless_confirm_agree_min,
    ruthless_confirm_ret4_min, ruthless_confirm_ret16_min, ruthless_confirm_volr_min,
    use_momentum_score, reg_fitted,
    score_min_eff, max_disp, min_agr, min_ev, pred_return_min,
    compute_momentum_score_fn,
    counters,
    market_mode=None,
    ruthless_allowed_modes=None,
    ruthless_good_mode_relaxation=True,
    ruthless_good_mode_ev_min=0.004,
    ruthless_good_mode_volr_min=1.3,
    # Profit-voting mode parameters (ruthless only)
    ruthless_profit_voting_mode=False,
    pv_vote_threshold=0.50,
    pv_vote_yes_frac_min=0.34,
    pv_top3_mean_min=0.55,
    pv_vote_ev_floor=DEFAULT_VOTE_EV_FLOOR,
    pv_chop_yes_frac_min=0.50,
    pv_chop_top3_mean_min=0.60,
    pv_chop_pred_return_min=0.000,
    pv_chop_ev_min=0.002,
    # Ruthless active model promotion (PV mode)
    ruthless_active_models=None,
    ruthless_diagnostic_models=None,
):
    """Evaluate a single candidate for entry. Returns a result dict or None if filtered."""
    class_proba = conf["class_proba"]
    std_proba   = conf["std_proba"]
    n_agree     = conf["n_agree"]
    pred_return = conf["pred_return"]

    passed_disp  = std_proba  <= max_disp
    passed_agree = n_agree    >= min_agr
    passed_score = class_proba >= score_min_eff
    if passed_disp:  counters["n_pass_disp"]  += 1
    if passed_agree: counters["n_pass_agree"] += 1
    if passed_score: counters["n_pass_score"] += 1

    ml_gates_pass = passed_disp and passed_agree and passed_score
    entry_path = "ml"

    if not ml_gates_pass:
        if not momentum_override_enabled:
            return None
        if not (
            float(feat[1]) >= momentum_ret4_min
            and float(feat[3]) >= momentum_ret16_min
            and float(feat[6]) >= momentum_volume_min
            and float(feat[7]) >= momentum_btc_rel_min
        ):
            return None
        entry_path = "momentum_override"

    # ATR-based TP/SL
    if atr > 0:
        tp_use = (atr * atr_tp_mult) / price
        sl_use = (atr * atr_sl_mult) / price
    else:
        tp_use = tp_base
        sl_use = sl_base

    # Ruthless TP/SL floors
    tp_floor_applied = False
    sl_floor_applied = False
    if risk_profile == "ruthless":
        if tp_use < tp_base:
            tp_use = tp_base
            tp_floor_applied = True
        if sl_use < sl_base:
            sl_use = sl_base
            sl_floor_applied = True

    # EV
    ev_after_costs = (
        class_proba * tp_use
        - (1.0 - class_proba) * sl_use
        - cost_fraction
    )

    if entry_path == "momentum_override":
        if ev_after_costs < momentum_override_min_ev:
            return None
        counters["n_momentum_override"] += 1
        counters["n_pass_ev"] += 1
        counters["n_pass_pred_ret"] += 1
    else:
        if ev_after_costs <= min_ev:
            return None
        counters["n_pass_ev"] += 1
        if reg_fitted and pred_return < pred_return_min:
            return None
        counters["n_pass_pred_ret"] += 1

    # ── Good-market-mode relaxation (ruthless only) ───────────────────────────
    # In pump/risk_on_trend modes, slightly relax confirmation thresholds to
    # increase sample size without touching chop/selloff.
    _eff_confirm_ev_min   = ruthless_confirm_ev_min
    _eff_confirm_volr_min = ruthless_confirm_volr_min
    _good_mode_active     = False
    if (
        risk_profile == "ruthless"
        and ruthless_good_mode_relaxation
        and market_mode is not None
        and market_mode in (ruthless_allowed_modes or ["risk_on_trend", "pump"])
    ):
        _eff_confirm_ev_min   = min(ruthless_confirm_ev_min,   ruthless_good_mode_ev_min)
        _eff_confirm_volr_min = min(ruthless_confirm_volr_min, ruthless_good_mode_volr_min)
        _good_mode_active = True

    # Ruthless confirmation gate — priority order:
    #   1. momentum_override (entry was already a momentum breakout)
    #   2. strong_ml         (high EV + high class_proba + multi-model agreement)
    #   3. trend_momentum    (short-term price momentum + volume expansion)
    #   4. market_mode       (BTC regime aligned — lowest priority, fallback only)
    # All four paths are checked; first match wins.  If none match, entry is skipped.
    confirm_reason = None
    if risk_profile == "ruthless":
        if entry_path == "momentum_override":
            confirm_reason = "momentum_override"
        elif (
            ev_after_costs >= _eff_confirm_ev_min
            and class_proba >= ruthless_confirm_proba_min
            and n_agree    >= ruthless_confirm_agree_min
        ):
            confirm_reason = "strong_ml" + ("_relax" if _good_mode_active else "")
        elif (
            float(feat[1]) >= ruthless_confirm_ret4_min
            and float(feat[3]) >= ruthless_confirm_ret16_min
            and float(feat[6]) >= _eff_confirm_volr_min
        ):
            confirm_reason = "trend_momentum" + ("_relax" if _good_mode_active else "")
        # market_mode is the lowest-priority confirmation path: only tried when
        # momentum_override, strong_ml, and trend_momentum all fail.
        if confirm_reason is None and market_mode is not None:
            allowed = ruthless_allowed_modes or ["risk_on_trend", "pump"]
            if market_mode in allowed:
                confirm_reason = "market_mode"
        if confirm_reason is None:
            return None

    # ── Profit-voting gate (ruthless profit-voting mode) ──────────────────────
    # In profit-voting mode, entries additionally require a minimum vote-score
    # ranking (vote_yes_fraction + top3_mean).  Chop entries require a
    # stricter supermajority.  This gate is applied AFTER the standard ruthless
    # confirmation gate so existing behavior is preserved in non-PV mode.
    #
    # First, promote shadow models listed in RUTHLESS_ACTIVE_MODELS into the
    # active voting pool (modifies conf in-place).
    pv_reject_reason = None
    if risk_profile == "ruthless" and ruthless_profit_voting_mode:
        # Active-model promotion — makes RUTHLESS_ACTIVE_MODELS real, not informational
        if ruthless_active_models:
            apply_ruthless_active_promotion(conf, ruthless_active_models, ruthless_diagnostic_models)

        # Warn if active pool is still empty after promotion
        if not conf.get("active_votes"):
            pv_reject_reason = "no_active_votes"
            return None

        _pv_approved, _pv_reason = check_profit_voting_gate(
            conf=conf,
            market_mode=market_mode,
            vote_thr=pv_vote_threshold,
            vote_yes_frac_min=pv_vote_yes_frac_min,
            top3_mean_min=pv_top3_mean_min,
            chop_vote_yes_frac_min=pv_chop_yes_frac_min,
            chop_top3_mean_min=pv_chop_top3_mean_min,
            chop_pred_return_min=pv_chop_pred_return_min,
            chop_ev_min=pv_chop_ev_min,
            ev_score=ev_after_costs,
            ev_floor=pv_vote_ev_floor,
        )
        if not _pv_approved:
            pv_reject_reason = _pv_reason
            return None

    # Final score
    if use_momentum_score:
        momentum_score = compute_momentum_score_fn(feat)
        if reg_fitted and pred_return != 0.0:
            final_score = (
                0.50 * ev_after_costs
                + 0.25 * pred_return
                + 0.25 * momentum_score
            )
        else:
            confidence_adj = max(0.0, 1.0 - std_proba)
            final_score = (
                0.75 * ev_after_costs * confidence_adj
                + 0.25 * momentum_score
            )
    else:
        if reg_fitted and pred_return != 0.0:
            final_score = 0.6 * ev_after_costs + 0.4 * pred_return
        else:
            confidence_adj = max(0.0, 1.0 - std_proba)
            final_score = ev_after_costs * confidence_adj

    # Incorporate vote_score as a tiebreaker in final_score for ruthless PV mode.
    # Weights: 85% base EV/momentum score + 15% vote quality score.
    _PV_BASE_SCORE_WEIGHT = 0.85
    _PV_VOTE_SCORE_WEIGHT = 0.15
    if risk_profile == "ruthless" and ruthless_profit_voting_mode:
        vote_sc = conf.get("vote_score", 0.0)
        final_score = _PV_BASE_SCORE_WEIGHT * final_score + _PV_VOTE_SCORE_WEIGHT * vote_sc

    return {
        "sym":               sym,
        "final_score":       final_score,
        "ev":                ev_after_costs,
        "tp_use":            tp_use,
        "sl_use":            sl_use,
        "atr":               atr,
        "price":             price,
        "tp_floor_applied":  tp_floor_applied,
        "sl_floor_applied":  sl_floor_applied,
        "entry_path":        entry_path,
        "confirm_reason":    confirm_reason,
        "vote_score":        conf.get("vote_score", 0.0),
        "vote_yes_fraction": conf.get("vote_yes_fraction", 0.0),
        "top3_mean":         conf.get("top3_mean", 0.0),
    }


# ===============================================================================
# shadow_lab
# ===============================================================================

# ── Vox Shadow Lab — Extended Shadow and Diagnostic Models ───────────────────
#
# This module provides additional shadow and diagnostic models that are loaded
# by _make_shadow_estimators in models.py.
#
# Shadow models (gbc, ada):
#   - Predicted and logged, never affect trading decisions.
#   - Provides buy-probability score for post-hoc attribution.
#
# Diagnostic/regime models (markov_regime, hmm_regime, kmeans_regime,
#                            isoforest_risk):
#   - Produce regime state / risk overlay scores.
#   - Persisted in diagnostic_scores / regime_model_state fields.
#   - Never used as direct buy probability.
#
# All models are optional and fail silently if sklearn dependencies are missing.
# ─────────────────────────────────────────────────────────────────────────────


# ── Optional HMM import ───────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

ROLE_SHADOW     = "shadow"
ROLE_DIAGNOSTIC = "diagnostic"


# ── Buy-probability shadow models ─────────────────────────────────────────────

def _make_gbc(logger=None):
    """Build a compact GradientBoostingClassifier shadow model."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8,
            random_state=42,
        )
    except Exception as exc:
        if logger:
            logger(f"[shadow_lab] gbc init failed: {exc}")
        return None


def _make_ada(logger=None):
    """Build a compact AdaBoostClassifier shadow model."""
    try:
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(
            n_estimators=80, learning_rate=0.05,
            random_state=42,
        )
    except Exception as exc:
        if logger:
            logger(f"[shadow_lab] ada init failed: {exc}")
        return None


# ── Regime diagnostic models ───────────────────────────────────────────────────
#
# These are sklearn wrappers that output a probability-like score or cluster
# state rather than a buy probability. They are tagged ROLE_DIAGNOSTIC and
# never affect active trading decisions.

class MarkovRegimeDiagnostic:
    """Lightweight Markov-inspired regime diagnostic."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._clf = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42,
        )
        self._fitted = False
        # Feature indices used from the full feature vector:
        # [0]=ret_1, [1]=ret_4, [2]=ret_8, [3]=ret_16, [4]=rsi_14,
        # [5]=atr_pct, [6]=vol_ratio, [7]=btc_rel,  …
        self._feat_idx = [1, 3, 5, 6, 7]  # ret_4, ret_16, atr_pct, vol_ratio, btc_rel

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def _make_labels(self, X, y_class):
        """Build regime labels from features heuristically."""
        X2 = self._extract(X)
        ret4   = X2[:, 0]
        ret16  = X2[:, 1]
        vol_r  = X2[:, 3] if X2.shape[1] > 3 else np.zeros(len(X2))
        # 0 = uptrend, 1 = chop, 2 = downtrend
        labels = np.ones(len(X2), dtype=int)  # default chop
        labels[(ret4 > 0.005) & (ret16 > 0.010)] = 0   # uptrend
        labels[(ret4 < -0.005) & (ret16 < -0.010)] = 2  # downtrend
        return labels

    def fit(self, X, y_class):
        try:
            labels = self._make_labels(X, y_class)
            self._clf.fit(self._extract(X), labels)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        """Return P(uptrend) as a scalar in a 2-column array (col 1 = P(uptrend))."""
        if not self._fitted:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        try:
            probs = self._clf.predict_proba(self._extract(X))
            # class 0 = uptrend → col 1 carries the uptrend probability
            n_classes = probs.shape[1]
            up_col = 0  # class 0 = uptrend
            up_prob = probs[:, up_col] if n_classes > 0 else np.full(len(probs), 0.5)
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class HMMRegimeDiagnostic:
    """Optional HMM-based regime diagnostic (requires hmmlearn).

    If hmmlearn is not installed, falls back to MarkovRegimeDiagnostic.
    Tagged ROLE_DIAGNOSTIC; outputs probability of being in the "up" state.
    """

    def __init__(self, n_components=3):
        self._n = n_components
        self._hmm = None
        self._fitted = False
        self._fallback = MarkovRegimeDiagnostic()
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        if not HMM_AVAILABLE:
            self._fallback.fit(X, y_class)
            return
        try:
            Xr = self._extract(X).astype(float)
            self._hmm = GaussianHMM(
                n_components=self._n, covariance_type="diag",
                n_iter=50, random_state=42,
            )
            self._hmm.fit(Xr)
            self._fitted = True
        except Exception:
            self._fallback.fit(X, y_class)
            self._fitted = False

    def predict_proba(self, X):
        if not (HMM_AVAILABLE and self._fitted and self._hmm is not None):
            return self._fallback.predict_proba(X)
        try:
            Xr = self._extract(X).astype(float)
            # Use log-probability of state 0 as a risk-on proxy
            log_p = self._hmm.predict_proba(Xr)
            up_col = 0
            up_prob = log_p[:, up_col] if log_p.shape[1] > 0 else np.full(len(Xr), 0.5)
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            return self._fallback.predict_proba(X)


class KMeansRegimeDiagnostic:
    """KMeans-based regime clustering diagnostic.

    Clusters the feature space into N regimes and assigns a regime label.
    The probability output is based on distance to the "risk-on" cluster
    (identified heuristically as the cluster with highest mean ret_4).
    """

    def __init__(self, n_clusters=4):
        self._n = n_clusters
        self._km = None
        self._risk_on_cluster = 0
        self._fitted = False
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        try:
            from sklearn.cluster import KMeans
            Xr = self._extract(X).astype(float)
            self._km = KMeans(
                n_clusters=self._n, random_state=42, n_init=5,
            )
            labels = self._km.fit_predict(Xr)
            # Identify "risk-on" cluster: highest mean ret_4 among clusters
            ret4 = Xr[:, 0]
            cluster_means = [
                np.mean(ret4[labels == k]) if np.any(labels == k) else 0.0
                for k in range(self._n)
            ]
            self._risk_on_cluster = int(np.argmax(cluster_means))
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        if not self._fitted or self._km is None:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        try:
            Xr = self._extract(X).astype(float)
            dists = self._km.transform(Xr)  # shape (n, n_clusters)
            # Softmax inverse-distance to risk-on cluster
            inv_d = 1.0 / (dists + 1e-6)
            inv_sum = inv_d.sum(axis=1, keepdims=True)
            probs = inv_d / inv_sum
            up_prob = probs[:, self._risk_on_cluster]
            return np.column_stack([1.0 - up_prob, up_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class IsoForestRiskDiagnostic:
    """IsolationForest anomaly/risk diagnostic.

    Flags unusual/outlier market setups via anomaly score.
    Higher score (closer to 1.0) = more anomalous = higher risk.

    Output in col-1 position is the anomaly probability (re-scaled from
    IsolationForest decision_function → [0, 1]).
    """

    def __init__(self):
        self._iso = None
        self._fitted = False
        self._feat_idx = [1, 3, 5, 6, 7]

    def _extract(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] <= max(self._feat_idx):
            return X
        return X[:, self._feat_idx]

    def fit(self, X, y_class):
        try:
            from sklearn.ensemble import IsolationForest
            Xr = self._extract(X).astype(float)
            self._iso = IsolationForest(
                n_estimators=80, contamination=0.05, random_state=42, n_jobs=1,
            )
            self._iso.fit(Xr)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict_proba(self, X):
        """Return anomaly probability: col-1 = P(anomaly) in [0, 1]."""
        if not self._fitted or self._iso is None:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])
        try:
            Xr = self._extract(X).astype(float)
            # decision_function: negative = anomaly, positive = normal
            scores = self._iso.decision_function(Xr)
            # Re-scale to [0, 1]: anomaly probability increases as score goes negative.
            # Scale factor controls steepness of the sigmoid mapping.
            _ANOMALY_SIGMOID_SCALE = 10.0
            anom_prob = 1.0 / (1.0 + np.exp(_ANOMALY_SIGMOID_SCALE * scores))
            return np.column_stack([1.0 - anom_prob, anom_prob])
        except Exception:
            n = len(np.atleast_2d(X))
            return np.column_stack([np.full(n, 0.9), np.full(n, 0.1)])


# ── Public API: extend existing shadow list ────────────────────────────────────

def extend_shadow_estimators(existing, max_count=12, logger=None):
    """Extend an existing shadow estimator list with gbc, ada, and regime diagnostics.

    Parameters
    ----------
    existing  : list[(id, estimator, role)]
    max_count : int — hard cap on total models returned
    logger    : callable or None

    Returns
    -------
    list[(id, estimator, role)] — extended list, capped at max_count
    """
    shadows = list(existing)

    def _try_add(model_id, factory_fn, role):
        if len(shadows) >= max_count:
            return
        est = factory_fn(logger)
        if est is not None:
            shadows.append((model_id, est, role))

    # ── Buy-probability shadow models ─────────────────────────────────────────
    _try_add("gbc", _make_gbc, ROLE_SHADOW)
    _try_add("ada", _make_ada, ROLE_SHADOW)

    # ── Regime/risk diagnostic models ─────────────────────────────────────────
    if len(shadows) < max_count:
        try:
            shadows.append(("markov_regime", MarkovRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] markov_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("hmm_regime", HMMRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] hmm_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("kmeans_regime", KMeansRegimeDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] kmeans_regime init failed: {exc}")

    if len(shadows) < max_count:
        try:
            shadows.append(("isoforest_risk", IsoForestRiskDiagnostic(), ROLE_DIAGNOSTIC))
        except Exception as exc:
            if logger:
                logger(f"[shadow_lab] isoforest_risk init failed: {exc}")

    return shadows[:max_count]
