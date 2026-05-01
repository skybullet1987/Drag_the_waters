# ── Vox Model Health Diagnostics ──────────────────────────────────────────────
#
# Tracks rolling per-model probability statistics to detect degenerate models.
#
# Flags emitted:
#   degenerate_bullish  — model votes >= extreme_proba on >= degenerate_frac of obs
#   degenerate_bearish  — model votes <= (1 - extreme_proba) on >= degenerate_frac
#   low_variance        — rolling std of model probabilities < low_std threshold
#
# Usage:
#   tracker = ModelHealthTracker()
#   tracker.update_batch({"hgbc": 0.62, "lr": 0.01, "gnb": 1.0})  # per-prediction
#   flags = tracker.get_all_flags()
#   summary = tracker.format_log_summary(roles_dict={"gnb": "diagnostic"})
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque


# Default thresholds (overridable via ModelHealthTracker constructor)
DEFAULT_MIN_OBS         = 20
DEFAULT_EXTREME_PROBA   = 0.95
DEFAULT_DEGENERATE_FRAC = 0.90
DEFAULT_LOW_STD         = 0.01


class ModelHealthTracker:
    """Tracks rolling probability statistics per model for health diagnostics.

    Parameters
    ----------
    min_obs          : int   — minimum observations before emitting flags
    extreme_proba    : float — threshold defining "extreme" probability (e.g. 0.95)
    degenerate_frac  : float — fraction of obs above/below extreme that triggers flag
    low_std          : float — rolling std below this → low_variance flag
    window           : int   — rolling window size (0 = unbounded)
    """

    def __init__(
        self,
        min_obs         = DEFAULT_MIN_OBS,
        extreme_proba   = DEFAULT_EXTREME_PROBA,
        degenerate_frac = DEFAULT_DEGENERATE_FRAC,
        low_std         = DEFAULT_LOW_STD,
        window          = 200,
    ):
        self._min_obs         = int(min_obs)
        self._extreme_proba   = float(extreme_proba)
        self._degenerate_frac = float(degenerate_frac)
        self._low_std         = float(low_std)
        self._window          = int(window)
        # model_id -> deque of float probabilities
        self._history = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, model_id, proba):
        """Record a single probability prediction for a model.

        Parameters
        ----------
        model_id : str
        proba    : float in [0, 1]
        """
        if model_id not in self._history:
            maxlen = self._window if self._window > 0 else None
            self._history[model_id] = deque(maxlen=maxlen)
        try:
            self._history[model_id].append(float(proba))
        except (ValueError, TypeError):
            pass

    def update_batch(self, votes_dict):
        """Record predictions for multiple models at once.

        Parameters
        ----------
        votes_dict : dict[str, float] — model_id -> probability
        """
        for mid, proba in votes_dict.items():
            self.update(mid, proba)

    # ── Flag computation ──────────────────────────────────────────────────────

    def get_flags(self, model_id):
        """Return health flags for a single model.

        Parameters
        ----------
        model_id : str

        Returns
        -------
        dict with keys:
            n_obs           — int:   number of observations recorded
            mean_proba      — float: rolling mean probability
            std_proba       — float: rolling std probability
            pct_above_thr   — float: fraction of obs >= extreme_proba
            pct_below_thr   — float: fraction of obs <= (1 - extreme_proba)
            degenerate_bullish  — bool
            degenerate_bearish  — bool
            low_variance        — bool
            flags           — list[str]: active flag names
        """
        hist = list(self._history.get(model_id, []))
        n = len(hist)
        if n == 0:
            return {
                "n_obs": 0, "mean_proba": None, "std_proba": None,
                "pct_above_thr": None, "pct_below_thr": None,
                "degenerate_bullish": False, "degenerate_bearish": False,
                "low_variance": False, "flags": [],
            }

        mean_p = sum(hist) / n
        std_p  = _std(hist)
        low_thr = 1.0 - self._extreme_proba
        pct_above = sum(1 for p in hist if p >= self._extreme_proba) / n
        pct_below = sum(1 for p in hist if p <= low_thr) / n

        deg_bull = (n >= self._min_obs) and (pct_above >= self._degenerate_frac)
        deg_bear = (n >= self._min_obs) and (pct_below >= self._degenerate_frac)
        low_var  = (n >= self._min_obs) and (std_p < self._low_std)

        active_flags = []
        if deg_bull:
            active_flags.append("degenerate_bullish")
        if deg_bear:
            active_flags.append("degenerate_bearish")
        if low_var:
            active_flags.append("low_variance")

        return {
            "n_obs":               n,
            "mean_proba":          mean_p,
            "std_proba":           std_p,
            "pct_above_thr":       pct_above,
            "pct_below_thr":       pct_below,
            "degenerate_bullish":  deg_bull,
            "degenerate_bearish":  deg_bear,
            "low_variance":        low_var,
            "flags":               active_flags,
        }

    def get_all_flags(self):
        """Return health flags for all tracked models.

        Returns
        -------
        dict[str, dict] — model_id -> flags dict (same shape as get_flags)
        """
        return {mid: self.get_flags(mid) for mid in self._history}

    # ── Log formatting ────────────────────────────────────────────────────────

    def format_log_summary(self, roles_dict=None):
        """Format a compact multi-line health summary for logging.

        Parameters
        ----------
        roles_dict : dict[str, str] or None — model_id -> role string

        Returns
        -------
        str — newline-joined log lines, one per model
        """
        lines = []
        for mid in sorted(self._history.keys()):
            f = self.get_flags(mid)
            if f["n_obs"] == 0:
                continue
            role = (roles_dict or {}).get(mid, "?")
            mean_s = f"{f['mean_proba']:.3f}" if f["mean_proba"] is not None else "?"
            std_s  = f"{f['std_proba']:.3f}"  if f["std_proba"]  is not None else "?"
            pct_s  = f"{f['pct_above_thr']:.0%}" if f["pct_above_thr"] is not None else "?"
            flag_s = f["flags"][0] if f["flags"] else "ok"
            lines.append(
                f"[model_health] {mid} role={role}"
                f" n={f['n_obs']} mean={mean_s} std={std_s} yes={pct_s}"
                f" flag={flag_s}"
            )
        return "\n".join(lines) if lines else "[model_health] no data"

    # ── State management ──────────────────────────────────────────────────────

    def model_ids(self):
        """Return list of tracked model IDs."""
        return list(self._history.keys())

    def reset(self, model_id=None):
        """Clear history for one model or all models.

        Parameters
        ----------
        model_id : str or None — if None, clears all
        """
        if model_id is None:
            self._history.clear()
        else:
            self._history.pop(model_id, None)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _std(vals):
    """Population std-dev of a list of floats (pure-Python, no numpy)."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
