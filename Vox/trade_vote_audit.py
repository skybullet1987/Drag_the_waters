# ── Vox Trade Vote Audit ─────────────────────────────────────────────────────
#
# True selected-trade vote audit with unique trade_id.
#
# Unlike the trade_journal (which reconstructs attempts), this module records
# only *selected and filled* entries/exits so model accuracy can be assessed
# without duplicate/attempt pollution.
#
# Persistence: ObjectStore key "vox/trade_vote_audit.jsonl"
# Format: newline-delimited JSON (one record per line, append-mode).
#
# Entry record fields (entry_type="entry"):
#   trade_id, entry_order_id, symbol, entry_time, entry_price, entry_qty,
#   allocation, risk_profile, ruthless_v2_mode, market_mode, confirm,
#   entry_path, vote_score, dynamic_vote_score, core_vote_score,
#   vote_yes_fraction, top3_mean, active_model_count, active_mean, active_std,
#   active_n_agree, pred_return, ev_score, final_score,
#   scalp_score, continuation_score, runner_score, breakout_score,
#   relative_strength_score, volume_expansion_score, exhaustion_score,
#   lane_selected, meta_entry_score,
#   active_votes, shadow_votes, diagnostic_votes,
#   effective_model_weights
#
# Exit record fields (entry_type="exit"):
#   trade_id, exit_order_id, exit_time, exit_price, exit_reason, hold_minutes,
#   realized_return, realized_pnl, fees, mfe, mae, winner,
#   partial_exit_qty, partial_exit_price, runner_exit_qty, runner_exit_price
#
# Research snippet: see README.md "Trade Vote Audit — Research Analysis"
# ─────────────────────────────────────────────────────────────────────────────

import json
import uuid


# ObjectStore key for the audit log
AUDIT_STORE_KEY = "vox/trade_vote_audit.jsonl"

# All required entry fields (for validation / shape assertions)
ENTRY_RECORD_REQUIRED_FIELDS = [
    "trade_id",
    "entry_type",
    "symbol",
    "entry_time",
    "entry_price",
    "entry_qty",
    "allocation",
    "risk_profile",
    "ruthless_v2_mode",
    "market_mode",
    "vote_score",
    "active_model_count",
    "active_votes",
    "shadow_votes",
    "diagnostic_votes",
    "effective_model_weights",
]

EXIT_RECORD_REQUIRED_FIELDS = [
    "trade_id",
    "entry_type",
    "exit_time",
    "exit_price",
    "exit_reason",
    "hold_minutes",
    "realized_return",
    "winner",
]


def _make_trade_id():
    """Generate a unique trade_id (short UUID4 hex)."""
    return uuid.uuid4().hex[:16]


class TradeVoteAudit:
    """True selected-trade vote audit.

    Records only confirmed filled entry/exit events (not attempts).
    Persists to ObjectStore as newline-delimited JSON for offline analysis.

    Usage::

        audit = TradeVoteAudit(logger=self.log)

        # At entry fill:
        trade_id = audit.record_entry(symbol, entry_snapshot)

        # At exit fill:
        audit.record_exit(trade_id, exit_outcome)

        # Persist to ObjectStore:
        audit.save(self.object_store)

        # Restore from ObjectStore:
        audit.load(self.object_store)
    """

    def __init__(self, logger=None, max_memory=2000):
        self._logger   = logger
        self._max_mem  = max_memory
        # In-memory records (entry + exit separately)
        self._records  = []
        # Open entries awaiting exit: trade_id -> entry record
        self._open     = {}

    # ── Record entry ──────────────────────────────────────────────────────────

    def record_entry(self, symbol, entry_snapshot, trade_id=None):
        """Record a confirmed filled entry.

        Parameters
        ----------
        symbol         : str
        entry_snapshot : dict — all available entry fields (votes, scores, etc.)
        trade_id       : str or None — if None, a new UUID is generated

        Returns
        -------
        str — trade_id (use to pair with exit)
        """
        if trade_id is None:
            trade_id = _make_trade_id()

        rec = {
            "trade_id":               trade_id,
            "entry_type":             "entry",
            "symbol":                 symbol,
            # Entry core fields
            "entry_order_id":         entry_snapshot.get("entry_order_id"),
            "entry_time":             _safe_str(entry_snapshot.get("entry_time")),
            "entry_price":            entry_snapshot.get("entry_price"),
            "entry_qty":              entry_snapshot.get("entry_qty"),
            "allocation":             entry_snapshot.get("allocation"),
            # Profile
            "risk_profile":           entry_snapshot.get("risk_profile", "unknown"),
            "ruthless_v2_mode":       entry_snapshot.get("ruthless_v2_mode", False),
            "market_mode":            entry_snapshot.get("market_mode"),
            "confirm":                entry_snapshot.get("confirm"),
            "entry_path":             entry_snapshot.get("entry_path"),
            # Vote quality
            "vote_score":             entry_snapshot.get("vote_score", 0.0),
            "dynamic_vote_score":     entry_snapshot.get("dynamic_vote_score", 0.0),
            "core_vote_score":        entry_snapshot.get("core_vote_score", 0.0),
            "vote_yes_fraction":      entry_snapshot.get("vote_yes_fraction", 0.0),
            "top3_mean":              entry_snapshot.get("top3_mean", 0.0),
            "active_model_count":     entry_snapshot.get("active_model_count", 0),
            "active_mean":            entry_snapshot.get("active_mean", 0.0),
            "active_std":             entry_snapshot.get("active_std", 0.0),
            "active_n_agree":         entry_snapshot.get("active_n_agree", 0),
            # Prediction quality
            "pred_return":            entry_snapshot.get("pred_return", 0.0),
            "ev_score":               entry_snapshot.get("ev_score", 0.0),
            "final_score":            entry_snapshot.get("final_score", 0.0),
            # V2 lane scores
            "scalp_score":            entry_snapshot.get("scalp_score", 0.0),
            "continuation_score":     entry_snapshot.get("continuation_score", 0.0),
            "runner_score":           entry_snapshot.get("runner_score", 0.0),
            "breakout_score":         entry_snapshot.get("breakout_score", 0.0),
            "relative_strength_score":entry_snapshot.get("relative_strength_score", 0.0),
            "volume_expansion_score": entry_snapshot.get("volume_expansion_score", 0.0),
            "exhaustion_score":       entry_snapshot.get("exhaustion_score", 0.0),
            "lane_selected":          entry_snapshot.get("lane_selected"),
            "meta_entry_score":       entry_snapshot.get("meta_entry_score", 0.0),
            # Pump signals
            "pump_continuation_score":entry_snapshot.get("pump_continuation_score", 0.0),
            "pump_exhaustion_score":  entry_snapshot.get("pump_exhaustion_score", 0.0),
            # V2 ranking
            "v2_opportunity_score":   entry_snapshot.get("v2_opportunity_score", 0.0),
            "relative_strength_rank": entry_snapshot.get("relative_strength_rank"),
            # Votes (full dicts)
            "active_votes":           entry_snapshot.get("active_votes", {}),
            "shadow_votes":           entry_snapshot.get("shadow_votes", {}),
            "diagnostic_votes":       entry_snapshot.get("diagnostic_votes", {}),
            # Dynamic weights at time of entry
            "effective_model_weights":entry_snapshot.get("effective_model_weights", {}),
        }

        self._open[trade_id] = rec
        self._records.append(rec)
        self._trim()
        return trade_id

    # ── Record exit ───────────────────────────────────────────────────────────

    def record_exit(self, trade_id, exit_outcome):
        """Record a confirmed filled exit, pairing with its entry.

        Parameters
        ----------
        trade_id     : str — from record_entry()
        exit_outcome : dict — exit fields

        Returns
        -------
        dict — exit record appended to audit
        """
        entry_rec = self._open.pop(trade_id, None)

        exit_rec = {
            "trade_id":          trade_id,
            "entry_type":        "exit",
            "symbol":            exit_outcome.get("symbol",
                                  entry_rec["symbol"] if entry_rec else None),
            "exit_order_id":     exit_outcome.get("exit_order_id"),
            "exit_time":         _safe_str(exit_outcome.get("exit_time")),
            "exit_price":        exit_outcome.get("exit_price"),
            "exit_reason":       exit_outcome.get("exit_reason"),
            "hold_minutes":      exit_outcome.get("hold_minutes"),
            "realized_return":   exit_outcome.get("realized_return"),
            "realized_pnl":      exit_outcome.get("realized_pnl"),
            "fees":              exit_outcome.get("fees"),
            "mfe":               exit_outcome.get("mfe"),
            "mae":               exit_outcome.get("mae"),
            "winner":            exit_outcome.get("winner"),
            # Partial/runner exit details
            "partial_exit_qty":  exit_outcome.get("partial_exit_qty"),
            "partial_exit_price":exit_outcome.get("partial_exit_price"),
            "runner_exit_qty":   exit_outcome.get("runner_exit_qty"),
            "runner_exit_price": exit_outcome.get("runner_exit_price"),
            # Link back to entry
            "entry_time":        entry_rec["entry_time"] if entry_rec else None,
            "entry_price":       entry_rec["entry_price"] if entry_rec else None,
            "risk_profile":      entry_rec["risk_profile"] if entry_rec else None,
            "ruthless_v2_mode":  entry_rec["ruthless_v2_mode"] if entry_rec else None,
            "lane_selected":     entry_rec.get("lane_selected") if entry_rec else None,
            "market_mode":       entry_rec.get("market_mode") if entry_rec else None,
        }

        self._records.append(exit_rec)
        self._trim()
        return exit_rec

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_records(self, entry_type=None):
        """Return records, optionally filtered by entry_type ('entry' or 'exit')."""
        if entry_type is None:
            return list(self._records)
        return [r for r in self._records if r.get("entry_type") == entry_type]

    def get_open_trades(self):
        """Return dict of open (entry pending exit) records."""
        return dict(self._open)

    def record_count(self, entry_type=None):
        """Count records, optionally by entry_type."""
        return len(self.get_records(entry_type))

    def _trim(self):
        if len(self._records) > self._max_mem:
            self._records = self._records[-self._max_mem:]

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_jsonl(self):
        """Serialize all records to newline-delimited JSON string."""
        lines = []
        for rec in self._records:
            try:
                lines.append(json.dumps(rec, default=str))
            except Exception as exc:
                if self._logger:
                    self._logger(f"[trade_vote_audit] to_jsonl error: {exc}")
        return "\n".join(lines)

    def save(self, object_store):
        """Persist audit to ObjectStore.

        Parameters
        ----------
        object_store : QC ObjectStore-like object with .save(key, value)
        """
        try:
            jsonl = self.to_jsonl()
            object_store.save(AUDIT_STORE_KEY, jsonl)
            if self._logger:
                self._logger(
                    f"[trade_vote_audit] saved {self.record_count()} records"
                    f" to {AUDIT_STORE_KEY}"
                )
        except Exception as exc:
            if self._logger:
                self._logger(f"[trade_vote_audit] save failed: {exc}")

    def load(self, object_store):
        """Load audit from ObjectStore (merges with existing, deduplicates by trade_id+type).

        Parameters
        ----------
        object_store : QC ObjectStore-like object with .contains_key(key) / .read(key)
        """
        try:
            if not object_store.contains_key(AUDIT_STORE_KEY):
                return
            raw = object_store.read(AUDIT_STORE_KEY)
            if not raw:
                return
            seen_keys = set(
                (r.get("trade_id"), r.get("entry_type")) for r in self._records
            )
            loaded = 0
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        continue
                    key = (rec.get("trade_id"), rec.get("entry_type"))
                    if key not in seen_keys:
                        self._records.append(rec)
                        seen_keys.add(key)
                        loaded += 1
                except Exception:
                    continue
            self._trim()
            if self._logger:
                self._logger(
                    f"[trade_vote_audit] loaded {loaded} new records"
                    f" from {AUDIT_STORE_KEY}"
                )
        except Exception as exc:
            if self._logger:
                self._logger(f"[trade_vote_audit] load failed: {exc}")

    # ── Model attribution (selected trades only) ──────────────────────────────

    def compute_model_attribution(self, vote_threshold=0.50):
        """Compute per-model attribution from selected (filled) closed trades.

        Only uses entry+exit pairs — no attempt pollution.

        Returns
        -------
        dict[str, dict] — model_id -> attribution stats
        """
        # Pair entries with exits by trade_id
        entries = {r["trade_id"]: r for r in self._records if r.get("entry_type") == "entry"}
        exits   = {r["trade_id"]: r for r in self._records if r.get("entry_type") == "exit"}

        model_stats = {}

        for tid, exit_rec in exits.items():
            entry_rec = entries.get(tid)
            if entry_rec is None:
                continue
            ret = exit_rec.get("realized_return")
            if ret is None:
                continue
            try:
                ret = float(ret)
            except (TypeError, ValueError):
                continue
            is_win = ret > 0.0

            # Combine all vote sources for attribution
            all_votes = {}
            for vote_key in ("active_votes", "shadow_votes", "diagnostic_votes"):
                vdict = entry_rec.get(vote_key)
                if isinstance(vdict, dict):
                    all_votes.update(vdict)

            for mid, proba in all_votes.items():
                try:
                    voted_yes = float(proba) >= vote_threshold
                except (TypeError, ValueError):
                    continue
                s = model_stats.setdefault(mid, {
                    "yes_list": [], "no_list": [],
                })
                if voted_yes:
                    s["yes_list"].append((is_win, ret))
                else:
                    s["no_list"].append((is_win, ret))

        summary = {}
        for mid, s in model_stats.items():
            yes_list = s["yes_list"]
            no_list  = s["no_list"]
            yes_wins = [r for (w, r) in yes_list if w]
            yes_rets = [r for (_, r) in yes_list]
            no_rets  = [r for (_, r) in no_list]
            summary[mid] = {
                "vote_yes_count":      len(yes_list),
                "vote_no_count":       len(no_list),
                "win_rate_when_yes":   len(yes_wins) / len(yes_list) if yes_list else None,
                "avg_return_when_yes": sum(yes_rets) / len(yes_rets) if yes_rets else None,
                "avg_return_when_no":  sum(no_rets) / len(no_rets) if no_rets else None,
                "precision_proxy":     len(yes_wins) / len(yes_list) if yes_list else None,
            }
        return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_str(val):
    """Convert a value to string safely (handles datetime and None)."""
    if val is None:
        return None
    try:
        return str(val)
    except Exception:
        return None


def build_entry_snapshot(
    symbol,
    entry_order_id,
    entry_time,
    entry_price,
    entry_qty,
    allocation,
    risk_profile,
    ruthless_v2_mode,
    conf,
    ev_score=0.0,
    final_score=0.0,
    market_mode=None,
    confirm=None,
    entry_path=None,
    multihorizon_scores=None,
    pump_scores=None,
    v2_opportunity_score=0.0,
    relative_strength_score=0.0,
    relative_strength_rank=None,
    meta_entry_score=0.0,
    effective_model_weights=None,
    dynamic_vote_score=0.0,
):
    """Build a complete entry snapshot dict for trade_vote_audit.record_entry().

    Parameters
    ----------
    symbol, entry_order_id, entry_time, entry_price, entry_qty, allocation : basic trade info
    risk_profile : str
    ruthless_v2_mode : bool
    conf : dict — predict_with_confidence output
    ev_score, final_score : float
    market_mode, confirm, entry_path : str or None
    multihorizon_scores : dict or None — from compute_multihorizon_scores()
    pump_scores : dict or None — from compute_pump_scores()
    v2_opportunity_score, relative_strength_score : float
    relative_strength_rank : int or None
    meta_entry_score : float
    effective_model_weights : dict or None — {model_id: weight}
    dynamic_vote_score : float — weighted vote score from DynamicVoterWeighting

    Returns
    -------
    dict
    """
    mh = multihorizon_scores or {}
    ps = pump_scores or {}

    snap = {
        "entry_order_id":          entry_order_id,
        "entry_time":              entry_time,
        "entry_price":             entry_price,
        "entry_qty":               entry_qty,
        "allocation":              allocation,
        "risk_profile":            risk_profile,
        "ruthless_v2_mode":        ruthless_v2_mode,
        "market_mode":             market_mode,
        "confirm":                 confirm,
        "entry_path":              entry_path,
        # Vote quality
        "vote_score":              conf.get("vote_score", 0.0),
        "dynamic_vote_score":      dynamic_vote_score,
        "core_vote_score":         conf.get("vote_score", 0.0),
        "vote_yes_fraction":       conf.get("vote_yes_fraction", 0.0),
        "top3_mean":               conf.get("top3_mean", 0.0),
        "active_model_count":      conf.get("active_model_count", 0),
        "active_mean":             conf.get("active_mean", conf.get("class_proba", 0.0)),
        "active_std":              conf.get("active_std", conf.get("std_proba", 0.0)),
        "active_n_agree":          conf.get("active_n_agree", conf.get("n_agree", 0)),
        "pred_return":             conf.get("pred_return", 0.0),
        "ev_score":                ev_score,
        "final_score":             final_score,
        # Lane scores
        "scalp_score":             mh.get("scalp_score", 0.0),
        "continuation_score":      mh.get("continuation_score", 0.0),
        "runner_score":            mh.get("runner_score", 0.0),
        "lane_selected":           mh.get("lane_selected"),
        # Additional V2 scores
        "breakout_score":          mh.get("breakout_score", 0.0),
        "relative_strength_score": relative_strength_score,
        "volume_expansion_score":  mh.get("volume_expansion_score", 0.0),
        "exhaustion_score":        ps.get("pump_exhaustion_score", 0.0),
        "meta_entry_score":        meta_entry_score,
        "pump_continuation_score": ps.get("pump_continuation_score", 0.0),
        "pump_exhaustion_score":   ps.get("pump_exhaustion_score", 0.0),
        "v2_opportunity_score":    v2_opportunity_score,
        "relative_strength_rank":  relative_strength_rank,
        # Vote dicts
        "active_votes":            dict(conf.get("active_votes", {})),
        "shadow_votes":            dict(conf.get("shadow_votes", {})),
        "diagnostic_votes":        dict(conf.get("diagnostic_votes", {})),
        # Effective model weights at time of entry
        "effective_model_weights": dict(effective_model_weights or {}),
    }
    return snap


def build_exit_outcome(
    trade_id,
    symbol,
    exit_order_id,
    exit_time,
    exit_price,
    exit_reason,
    entry_price,
    entry_qty,
    hold_minutes=None,
    fees=None,
    mfe=None,
    mae=None,
    partial_exit_qty=None,
    partial_exit_price=None,
    runner_exit_qty=None,
    runner_exit_price=None,
):
    """Build a complete exit outcome dict for trade_vote_audit.record_exit().

    Returns
    -------
    dict
    """
    realized_return = None
    realized_pnl    = None
    winner          = None

    if entry_price and entry_price > 0 and exit_price:
        try:
            realized_return = (float(exit_price) - float(entry_price)) / float(entry_price)
            realized_pnl    = (float(exit_price) - float(entry_price)) * float(entry_qty or 0)
            winner          = realized_return > 0.0
        except (TypeError, ZeroDivisionError):
            pass

    return {
        "trade_id":          trade_id,
        "symbol":            symbol,
        "exit_order_id":     exit_order_id,
        "exit_time":         exit_time,
        "exit_price":        exit_price,
        "exit_reason":       exit_reason,
        "hold_minutes":      hold_minutes,
        "realized_return":   realized_return,
        "realized_pnl":      realized_pnl,
        "fees":              fees,
        "mfe":               mfe,
        "mae":               mae,
        "winner":            winner,
        "partial_exit_qty":  partial_exit_qty,
        "partial_exit_price":partial_exit_price,
        "runner_exit_qty":   runner_exit_qty,
        "runner_exit_price": runner_exit_price,
    }
