# ── journals.py: candidate_journal + trade_journal + trade_vote_audit + diagnostics + tuning ──
import json
import uuid


# ===============================================================================
# candidate_journal
# ===============================================================================

# ── Vox Candidate Journal ─────────────────────────────────────────────────────
#
# Records candidate ranking and skipped-candidate diagnostics for each
# decision cycle.  Allows post-hoc analysis of symbols that had strong votes
# but were skipped or rejected.
#
# Each cycle, the top N candidates (by vote_score / final_score) are journaled
# whether or not they were selected for entry.  This includes reject reasons,
# active/shadow/diagnostic votes, and the entry path that was evaluated.
#
# Records are kept in memory with a rolling cap and optionally persisted to
# ObjectStore via the save/load helpers in PersistenceManager.
# ─────────────────────────────────────────────────────────────────────────────


# Default cap on candidate records stored in memory.
CANDIDATE_JOURNAL_MAX_SIZE = 2000
# Default max candidates journaled per decision cycle.
CANDIDATE_JOURNAL_TOP_N    = 5


class CandidateJournal:
    """Journal for candidate ranking and skipped-candidate diagnostics."""

    def __init__(self, max_size=CANDIDATE_JOURNAL_MAX_SIZE, top_n=CANDIDATE_JOURNAL_TOP_N,
                 logger=None):
        self._max_size = max_size
        self._top_n    = top_n
        self._logger   = logger
        self._records  = []

    # ── Record a candidate decision cycle ─────────────────────────────────────

    def record_cycle(self, time, candidates):
        """Record the top-N candidates from a decision cycle."""
        if not candidates:
            return

        time_str = str(time)
        for c in candidates[: self._top_n]:
            record = {
                "time":              time_str,
                "symbol":            c.get("symbol", ""),
                "rank":              c.get("rank", 0),
                "selected":          c.get("selected", False),
                "reject_reason":     c.get("reject_reason"),
                "market_mode":       c.get("market_mode"),
                "confirm":           c.get("confirm"),
                "vote_score":        round(c.get("vote_score", 0.0), 6),
                "active_mean":       round(c.get("active_mean", 0.0), 4),
                "active_std":        round(c.get("active_std", 0.0), 4),
                "active_n_agree":    c.get("active_n_agree", 0),
                "vote_yes_fraction": round(c.get("vote_yes_fraction", 0.0), 4),
                "top3_mean":         round(c.get("top3_mean", 0.0), 4),
                "pred_return":       round(c.get("pred_return", 0.0), 6),
                "ev_score":          round(c.get("ev_score", 0.0), 6),
                "active_votes":      c.get("active_votes", {}),
                "shadow_votes":      c.get("shadow_votes", {}),
                "diagnostic_votes":  c.get("diagnostic_votes", {}),
                "entry_path":        c.get("entry_path", "ml"),
            }
            self._records.append(record)

        # Rolling cap — drop oldest when exceeded
        if len(self._records) > self._max_size:
            self._records = self._records[-self._max_size:]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_records(self):
        """Return all candidate records (list of dicts)."""
        return list(self._records)

    def get_skipped_records(self):
        """Return only skipped (not selected) candidate records."""
        return [r for r in self._records if not r.get("selected", False)]

    def get_selected_records(self):
        """Return only selected (entry taken) candidate records."""
        return [r for r in self._records if r.get("selected", False)]

    def __len__(self):
        return len(self._records)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self):
        """Return all records as a JSON string (pretty-printed)."""
        return json.dumps(self._records, indent=2, default=str)

    def from_json(self, json_str):
        """Load records from a JSON string (replaces current records)."""
        try:
            loaded = json.loads(json_str)
            if isinstance(loaded, list):
                self._records = loaded[-self._max_size:]
        except Exception as exc:
            if self._logger:
                self._logger(f"[candidate_journal] load_from_json failed: {exc}")

    def clear(self):
        """Clear all records."""
        self._records = []


def build_candidate_records(
    ranked_results,
    conf_data,
    ev_data,
    entry_path_data,
    scores,
    market_mode=None,
    confirm_reasons=None,
    selected_sym=None,
    rejected_reason=None,
):
    """Build candidate record list for a single decision cycle."""
    if confirm_reasons is None:
        confirm_reasons = {}
    records = []
    for rank, (sym, final_sc) in enumerate(ranked_results, start=1):
        conf = conf_data.get(sym, {})
        is_selected = (sym == selected_sym)
        reject_r = None
        if not is_selected:
            reject_r = rejected_reason if (rank == 1 and rejected_reason) else "lower_rank"
        records.append({
            "symbol":            getattr(sym, "value", str(sym)),
            "rank":              rank,
            "selected":          is_selected,
            "reject_reason":     reject_r,
            "market_mode":       market_mode,
            "confirm":           confirm_reasons.get(sym),
            "vote_score":        conf.get("vote_score", 0.0),
            "active_mean":       conf.get("active_mean", conf.get("class_proba", 0.0)),
            "active_std":        conf.get("active_std", conf.get("std_proba", 0.0)),
            "active_n_agree":    conf.get("active_n_agree", conf.get("n_agree", 0)),
            "vote_yes_fraction": conf.get("vote_yes_fraction", 0.0),
            "top3_mean":         conf.get("top3_mean", 0.0),
            "pred_return":       conf.get("pred_return", 0.0),
            "ev_score":          ev_data.get(sym, 0.0),
            "active_votes":      conf.get("active_votes", {}),
            "shadow_votes":      conf.get("shadow_votes", {}),
            "diagnostic_votes":  conf.get("diagnostic_votes", {}),
            "entry_path":        entry_path_data.get(sym, "ml"),
            "final_score":       round(final_sc, 6),
        })
    return records


def build_rejected_candidate_records(conf_dict, market_mode=None, top_n=5):
    """Build rejected candidate records for cycles where no trades pass all gates.

    Parameters
    ----------
    conf_dict   : dict[sym, conf] — all evaluated candidates (sym→conf from predict)
    market_mode : str or None
    top_n       : int — max records to build

    Returns
    -------
    list[dict] — ready for CandidateJournal.record_cycle(), all with selected=False
    """
    if not conf_dict:
        return []
    # Sort by vote_score descending so the best rejected candidate is rank=1
    ranked = sorted(
        conf_dict.items(),
        key=lambda kv: kv[1].get("vote_score", 0.0),
        reverse=True,
    )
    records = []
    for rank, (sym, conf) in enumerate(ranked[:top_n], start=1):
        records.append({
            "symbol":            getattr(sym, "value", str(sym)),
            "rank":              rank,
            "selected":          False,
            "reject_reason":     "pv_no_pass",
            "market_mode":       market_mode,
            "confirm":           None,
            "vote_score":        round(conf.get("vote_score", 0.0), 6),
            "active_mean":       round(conf.get("active_mean", conf.get("class_proba", 0.0)), 4),
            "active_std":        round(conf.get("active_std", conf.get("std_proba", 0.0)), 4),
            "active_n_agree":    conf.get("active_n_agree", conf.get("n_agree", 0)),
            "vote_yes_fraction": round(conf.get("vote_yes_fraction", 0.0), 4),
            "top3_mean":         round(conf.get("top3_mean", 0.0), 4),
            "pred_return":       round(conf.get("pred_return", 0.0), 6),
            "ev_score":          0.0,
            "active_votes":      conf.get("active_votes", {}),
            "shadow_votes":      conf.get("shadow_votes", {}),
            "diagnostic_votes":  conf.get("diagnostic_votes", {}),
            "entry_path":        "ml",
            "active_model_count": conf.get("active_model_count", 0),
        })
    return records


# ===============================================================================
# trade_journal
# ===============================================================================

# ── Vox Trade Journal ─────────────────────────────────────────────────────────
#
# Compact trade journal for model attribution analysis.
#
# Each completed trade produces a journal record containing:
#   - Entry snapshot: model votes, ensemble metrics, market conditions
#   - Exit outcome: realized return, exit reason, diagnostics
#
# Records are kept in memory with a rolling cap (default 500).
# Optional persistence to ObjectStore via the save/load helpers.
# ─────────────────────────────────────────────────────────────────────────────


# Maximum records kept in memory.  Oldest records are dropped when exceeded.
JOURNAL_MAX_SIZE = 500


class TradeJournal:
    """Compact trade journal for per-model attribution analysis.

    Usage
    -----
    At entry::

        journal.record_entry(symbol, entry_snapshot_dict)

    At exit::

        record = journal.record_exit(symbol, exit_outcome_dict)

    Retrieve records::

        records = journal.get_records()
        summary = journal.compute_model_attribution()
    """

    def __init__(self, max_size=JOURNAL_MAX_SIZE, logger=None):
        self._max_size    = max_size
        self._logger      = logger
        self._records     = []            # list of completed trade records
        self._open_trades = {}            # symbol (str) -> entry snapshot dict

    # ── Entry snapshot ────────────────────────────────────────────────────────

    def record_entry(self, symbol, entry_data):
        """Store entry snapshot keyed by symbol.

        Parameters
        ----------
        symbol     : str
        entry_data : dict — entry snapshot (model votes, ensemble metrics, etc.)
        """
        self._open_trades[symbol] = dict(entry_data)

    # ── Exit outcome ──────────────────────────────────────────────────────────

    def record_exit(self, symbol, exit_data):
        """Attach exit outcome to the open entry and save a completed record.

        Parameters
        ----------
        symbol    : str
        exit_data : dict — exit outcome fields

        Returns
        -------
        dict — completed trade record (entry + exit merged)
        """
        entry = self._open_trades.pop(symbol, {})
        record = {}
        record.update(entry)
        record.update(exit_data)
        record["symbol"] = symbol

        self._records.append(record)

        # Rolling cap — drop oldest when exceeded
        if len(self._records) > self._max_size:
            self._records = self._records[-self._max_size:]

        return record

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_records(self):
        """Return a copy of all completed trade records."""
        return list(self._records)

    def get_open_trades(self):
        """Return a copy of all open (entry-only) trade snapshots."""
        return dict(self._open_trades)

    def record_count(self):
        """Number of completed trade records."""
        return len(self._records)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self):
        """Serialise completed records to a JSON string."""
        return json.dumps(self._records, default=str)

    def load_json(self, json_str):
        """Load records from a JSON string (e.g. from ObjectStore).

        Merges with existing records; deduplicates by (symbol, entry_time).
        """
        try:
            loaded = json.loads(json_str)
            if not isinstance(loaded, list):
                return
            seen = set()
            for r in self._records:
                key = (r.get("symbol"), r.get("entry_time"))
                seen.add(key)
            for r in loaded:
                if not isinstance(r, dict):
                    continue
                key = (r.get("symbol"), r.get("entry_time"))
                if key not in seen:
                    self._records.append(r)
                    seen.add(key)
            # Re-apply cap
            if len(self._records) > self._max_size:
                self._records = self._records[-self._max_size:]
        except Exception as exc:
            if self._logger:
                self._logger(f"[trade_journal] load_json failed: {exc}")

    # ── Model attribution ─────────────────────────────────────────────────────

    def compute_model_attribution(self):
        """Compute per-model attribution metrics from completed journal records."""
        model_stats = {}

        for rec in self._records:
            votes = rec.get("model_votes")
            if not isinstance(votes, dict):
                continue
            ret = rec.get("realized_return")
            if ret is None:
                continue
            try:
                ret = float(ret)
            except (ValueError, TypeError):
                continue

            is_win = ret > 0.0
            vote_threshold = float(rec.get("vote_threshold", 0.5))

            for model_id, proba in votes.items():
                if model_id not in model_stats:
                    model_stats[model_id] = {
                        "vote_yes": [],
                        "vote_no":  [],
                    }
                try:
                    voted_yes = float(proba) >= vote_threshold
                except (ValueError, TypeError):
                    continue

                bucket = "vote_yes" if voted_yes else "vote_no"
                model_stats[model_id][bucket].append((is_win, ret))

        summary = {}
        for mid, s in model_stats.items():
            yes_list = s["vote_yes"]
            no_list  = s["vote_no"]

            yes_wins    = [r for (w, r) in yes_list if w]
            yes_rets    = [r for (_, r) in yes_list]
            no_rets     = [r for (_, r) in no_list]

            summary[mid] = {
                "vote_yes_count":      len(yes_list),
                "vote_no_count":       len(no_list),
                "win_rate_when_yes":   len(yes_wins) / len(yes_list) if yes_list else None,
                "avg_return_when_yes": sum(yes_rets) / len(yes_rets) if yes_rets else None,
                "avg_return_when_no":  sum(no_rets) / len(no_rets)   if no_rets  else None,
                "precision_proxy":     len(yes_wins) / len(yes_list) if yes_list else None,
            }

        return summary

    def format_attribution_summary(self):
        """Return a human-readable multi-line attribution summary string."""
        attr = self.compute_model_attribution()
        if not attr:
            return "[trade_journal] No attribution data (no completed trades with votes)."
        lines = [f"[trade_journal] Model attribution ({len(self._records)} trades):"]
        for mid, s in sorted(attr.items()):
            wr = s["win_rate_when_yes"]
            ar = s["avg_return_when_yes"]
            line = (
                f"  {mid:<12}"
                f" yes={s['vote_yes_count']:>3} no={s['vote_no_count']:>3}"
                + (f" wr_yes={wr:.1%}" if wr is not None else " wr_yes=n/a")
                + (f" avg_ret_yes={ar:.3%}" if ar is not None else "")
            )
            lines.append(line)
        return "\n".join(lines)


# ── Entry snapshot builder ────────────────────────────────────────────────────

def build_entry_snapshot(
    entry_time,
    symbol,
    entry_price,
    quantity,
    entry_path,
    market_mode,
    confirm_reason,
    class_proba,
    ensemble_std,
    n_agree,
    model_votes,
    vote_threshold,
    pred_return,
    ev_score,
    tp_use,
    sl_use,
    allocation,
    min_alloc,
    use_kelly,
    runner_mode,
    trail_after_tp,
    trail_pct,
    entry_order_type="market",
    meta_score=None,
    model_weights=None,
    weighted_mean=None,
    ret_4=None,
    ret_16=None,
    volume_ratio=None,
    btc_rel=None,
    limit_price=None,
    limit_offset=None,
    limit_submitted_time=None,
):
    """Build a compact JSON-serialisable entry snapshot dict.

    All fields are optional-safe (None allowed for any field).
    """
    snap = {
        "entry_time":       str(entry_time),
        "symbol":           str(symbol),
        "entry_price":      _safe_float(entry_price),
        "quantity":         _safe_float(quantity),
        "entry_path":       entry_path,
        "market_mode":      market_mode,
        "confirm_reason":   confirm_reason,
        "class_proba":      _safe_float(class_proba),
        "ensemble_std":     _safe_float(ensemble_std),
        "n_agree":          int(n_agree) if n_agree is not None else None,
        "model_votes":      dict(model_votes) if model_votes else {},
        "vote_threshold":   _safe_float(vote_threshold, default=0.5),
        "pred_return":      _safe_float(pred_return),
        "ev_score":         _safe_float(ev_score),
        "tp_use":           _safe_float(tp_use),
        "sl_use":           _safe_float(sl_use),
        "allocation":       _safe_float(allocation),
        "min_alloc":        _safe_float(min_alloc),
        "use_kelly":        use_kelly,
        "runner_mode":      runner_mode,
        "trail_after_tp":   _safe_float(trail_after_tp),
        "trail_pct":        _safe_float(trail_pct),
        "entry_order_type": entry_order_type,
    }
    if meta_score is not None:
        snap["meta_score"] = _safe_float(meta_score)
    if model_weights:
        snap["model_weights"] = {k: float(v) for k, v in model_weights.items()}
    if weighted_mean is not None:
        snap["weighted_mean"] = _safe_float(weighted_mean)
    if ret_4 is not None:
        snap["ret_4"] = _safe_float(ret_4)
    if ret_16 is not None:
        snap["ret_16"] = _safe_float(ret_16)
    if volume_ratio is not None:
        snap["volume_ratio"] = _safe_float(volume_ratio)
    if btc_rel is not None:
        snap["btc_rel"] = _safe_float(btc_rel)
    if limit_price is not None:
        snap["limit_price"] = _safe_float(limit_price)
    if limit_offset is not None:
        snap["limit_offset"] = _safe_float(limit_offset)
    if limit_submitted_time is not None:
        snap["limit_submitted_time"] = str(limit_submitted_time)
    return snap


# ── Exit outcome builder ──────────────────────────────────────────────────────

def build_exit_outcome(
    exit_time,
    exit_fill_price,
    exit_reason,
    realized_return,
    max_return_seen,
    hold_minutes,
    trail_active=False,
    trail_high_px=None,
    breakeven_active=False,
    timeout_extended=False,
    momentum_fail=False,
    stop_price=None,
    order_tag=None,
    fee_adjusted_return=None,
    limit_fill_time=None,
    limit_canceled_ttl=False,
):
    """Build a compact JSON-serialisable exit outcome dict."""
    out = {
        "exit_time":         str(exit_time),
        "exit_fill_price":   _safe_float(exit_fill_price),
        "exit_reason":       exit_reason,
        "realized_return":   _safe_float(realized_return),
        "max_return_seen":   _safe_float(max_return_seen),
        "hold_minutes":      _safe_float(hold_minutes),
        "trail_active":      trail_active,
        "breakeven_active":  breakeven_active,
        "timeout_extended":  timeout_extended,
        "momentum_fail":     momentum_fail,
    }
    if trail_high_px is not None:
        out["trail_high_px"] = _safe_float(trail_high_px)
    if stop_price is not None:
        out["stop_price"] = _safe_float(stop_price)
    if order_tag is not None:
        out["order_tag"] = order_tag
    if fee_adjusted_return is not None:
        out["fee_adjusted_return"] = _safe_float(fee_adjusted_return)
    if limit_fill_time is not None:
        out["limit_fill_time"] = str(limit_fill_time)
    if limit_canceled_ttl:
        out["limit_canceled_due_ttl"] = True
    return out


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(value, default=None):
    """Safely convert to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ===============================================================================
# trade_vote_audit
# ===============================================================================

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
    # Apex Predator fields (appended; default to 0/None for legacy records)
    "apex_score",
    "apex_path",
    "n_agree",
    "mean_proba",
    "cost_bps",
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
    """True selected-trade vote audit."""

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
            # Apex Predator fields
            "apex_score":             entry_snapshot.get("apex_score", 0.0),
            "apex_path":              entry_snapshot.get("apex_path"),
            "n_agree":                entry_snapshot.get("n_agree", 0),
            "mean_proba":             entry_snapshot.get("mean_proba", 0.0),
            "cost_bps":               entry_snapshot.get("cost_bps", 0.0),
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
    """Build a complete entry snapshot dict for trade_vote_audit.record_entry()."""
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


# ===============================================================================
# diagnostics
# ===============================================================================

# ── Vox Diagnostics ───────────────────────────────────────────────────────────
#
# Formatting helpers for vote logs and exit diagnostics.
#
# These are pure functions — no side effects, no QC dependencies.
# All formatters return strings suitable for self.log() or self.debug().
# ─────────────────────────────────────────────────────────────────────────────


# ── Feature diagnostics ───────────────────────────────────────────────────────

def _feature_diag_suffix(ft):
    """Return a compact r4/r16/volume-ratio suffix string from a feature vector.

    Safe against None, too-short vectors, and NumPy arrays (avoids ambiguous
    truth-value checks on multi-element arrays).

    Parameters
    ----------
    ft : array-like or None

    Returns
    -------
    str — e.g. " r4=0.0123 r16=0.0234 vr=1.45", or "" when unavailable.
    """
    try:
        if ft is None or len(ft) <= 6:
            return ""
        return f" r4={float(ft[1]):.4f} r16={float(ft[3]):.4f} vr={float(ft[6]):.2f}"
    except Exception:
        return ""


# ── Vote log formatting ───────────────────────────────────────────────────────

def format_vote_log(symbol, conf, meta_score=None, market_mode=None):
    """Format a compact per-model vote log line with role-separated vote groups."""
    # Prefer active-role statistics when available
    if "active_mean" in conf:
        mean    = conf["active_mean"]
        std     = conf["active_std"]
        n_agree = conf["active_n_agree"]
        total   = len(conf.get("active_votes", {}))
        line = (
            f"[vote] {symbol}"
            f" active_mean={mean:.2f} active_std={std:.2f}"
            f" agree={n_agree}/{total}"
        )
        if meta_score is not None:
            line += f" meta={meta_score:.2f}"
        if market_mode is not None:
            line += f" mode={market_mode}"
        av = conf.get("active_votes", {})
        if av:
            line += " active=" + ",".join(f"{m}:{v:.2f}" for m, v in av.items())
        sv = conf.get("shadow_votes", {})
        if sv:
            line += " shadow=" + ",".join(f"{m}:{v:.2f}" for m, v in sv.items())
        dv = conf.get("diagnostic_votes", {})
        if dv:
            line += " diag=" + ",".join(f"{m}:{v:.2f}" for m, v in dv.items())
        excl = conf.get("excluded_models", {})
        if excl:
            line += " excluded=" + ",".join(f"{m}:{r}" for m, r in excl.items())
        return line

    # Legacy format (no role fields)
    mean    = conf.get("class_proba", conf.get("mean_proba", 0.0))
    std     = conf.get("std_proba", 0.0)
    n_agree = conf.get("n_agree", 0)
    votes   = conf.get("per_model", {})
    total   = len(votes)

    line = (
        f"[vote] {symbol}"
        f" mean={mean:.2f} std={std:.2f}"
        f" agree={n_agree}/{total}"
    )
    if meta_score is not None:
        line += f" meta={meta_score:.2f}"
    if market_mode is not None:
        line += f" mode={market_mode}"
    if votes:
        vote_str = ",".join(f"{mid}:{v:.2f}" for mid, v in votes.items())
        line += f" votes={vote_str}"
    return line


def format_entry_tag(mean, n_agree, total, meta_score, market_mode):
    """Format a compact entry order tag suitable for order.tag field.

    Example::

        ENTRY|ml|mean=0.64|agree=5/6|meta=0.59|mode=pump
    """
    return (
        f"ENTRY|ml"
        f"|mean={mean:.2f}"
        f"|agree={n_agree}/{total}"
        f"|meta={meta_score:.2f}"
        f"|mode={market_mode or 'n/a'}"
    )


# ── Exit diagnostic formatting ────────────────────────────────────────────────

def format_exit_diagnostic(
    symbol,
    entry_price,
    exit_fill_price,
    exit_reason,
    realized_return,
    sl_use,
    tp_use,
    max_return_seen,
    elapsed_minutes,
    trail_active=False,
    trail_high_px=None,
    breakeven_active=False,
    stop_price=None,
):
    """Format a detailed exit diagnostic string."""
    # Classify why EXIT_SL may be firing on what looks like a flat/positive fill
    notes = []
    if exit_reason == "EXIT_SL":
        if realized_return >= 0.0:
            notes.append("warn:ret>=0_tagged_sl")
        configured_sl_price = entry_price * (1.0 - sl_use) if entry_price > 0 else None
        if (
            configured_sl_price is not None
            and exit_fill_price > 0
            and exit_fill_price < configured_sl_price - 0.001 * entry_price
        ):
            notes.append("warn:fill_below_configured_sl")
        if breakeven_active and realized_return >= -sl_use * 0.5:
            notes.append("info:breakeven_stop")
    elif exit_reason == "EXIT_TRAIL":
        if trail_active and trail_high_px is not None and trail_high_px > 0:
            drawdown_from_high = (exit_fill_price - trail_high_px) / trail_high_px
            notes.append(f"trail_drawdown={drawdown_from_high:.3%}")

    parts = [
        f"[exit_diag] {symbol}",
        f"entry={entry_price:.5f}",
        f"fill={exit_fill_price:.5f}",
        f"ret={realized_return:+.3%}",
        f"tag={exit_reason}",
        f"sl={sl_use:.4f}",
        f"tp={tp_use:.4f}",
        f"max_ret={max_return_seen:+.3%}",
        f"held={elapsed_minutes:.1f}m",
    ]
    if trail_active and trail_high_px is not None:
        parts.append(f"trail_high={trail_high_px:.5f}")
    if breakeven_active:
        parts.append("breakeven=active")
    if stop_price is not None:
        parts.append(f"stop_px={stop_price:.5f}")
    if notes:
        parts.append("(" + " ".join(notes) + ")")
    return " ".join(parts)


# ── Model attribution summary formatting ─────────────────────────────────────

def format_model_attribution_summary(attribution_dict, n_trades):
    """Format a multi-line per-model attribution summary for logging.

    Parameters
    ----------
    attribution_dict : dict — output of TradeJournal.compute_model_attribution()
    n_trades         : int  — total completed trades used

    Returns
    -------
    str
    """
    if not attribution_dict:
        return f"[model_attr] No attribution data ({n_trades} trades, no votes logged)."

    lines = [f"[model_attr] Per-model attribution ({n_trades} trades — small sample, noisy):"]
    for mid, s in sorted(attribution_dict.items()):
        wr  = s.get("win_rate_when_yes")
        ar  = s.get("avg_return_when_yes")
        arn = s.get("avg_return_when_no")
        line = (
            f"  {mid:<12}"
            f"  yes={s.get('vote_yes_count', 0):>3}"
            f"  no={s.get('vote_no_count', 0):>3}"
        )
        if wr is not None:
            line += f"  wr_yes={wr:.0%}"
        if ar is not None:
            line += f"  avg_ret_yes={ar:+.2%}"
        if arn is not None:
            line += f"  avg_ret_no={arn:+.2%}"
        lines.append(line)
    lines.append(
        "  WARNING: sample size < 50 makes these estimates unreliable."
        " Run multiple windows (2023/2024/2025/bull/chop/selloff)."
    )
    return "\n".join(lines)


# ── Startup configuration log ─────────────────────────────────────────────────

def format_limit_order_startup_log(
    use_entry_limit_orders,
    entry_limit_offset,
    entry_limit_ttl_minutes,
    entry_limit_chase=False,
    use_exit_limit_orders=False,
    exit_limit_offset=None,
    exit_limit_ttl_minutes=None,
):
    """Format a startup log line for entry/exit limit order configuration."""
    return (
        f"[limit_orders]"
        f" entry_limit={use_entry_limit_orders}"
        f" offset={entry_limit_offset}"
        f" ttl_min={entry_limit_ttl_minutes}"
        f" chase={entry_limit_chase}"
        f" exit_limit={use_exit_limit_orders}"
        + (
            f" exit_offset={exit_limit_offset}"
            f" exit_ttl={exit_limit_ttl_minutes}"
            if use_exit_limit_orders else ""
        )
    )


# ===============================================================================
# tuning
# ===============================================================================

# ── Vox Tuning — Good-Market-Mode Relaxation ─────────────────────────────────
#
# Controlled parameter relaxation for ruthless mode in favorable market modes.
#
# Problem: ruthless mode had only 11 round trips in ~16 months.
# Goal: carefully increase sample size only when market is in pump/risk_on_trend.
#
# This module provides helpers to compute slightly relaxed confirmation/filter
# thresholds when the market is in a good mode, without touching chop/selloff.
# ─────────────────────────────────────────────────────────────────────────────

# ── Default good-mode constants ───────────────────────────────────────────────
# These are applied only when risk_profile=ruthless AND market_mode is favorable.
# Overridable via config.py / QC parameters.

RUTHLESS_GOOD_MODES = ["risk_on_trend", "pump"]

# Slightly lower meta-filter threshold in good modes (0.55 -> 0.52)
RUTHLESS_GOOD_MODE_META_MIN_PROBA = 0.52

# Slightly lower minimum EV in good modes (0.006 -> 0.004 for confirm, 0.0 base)
RUTHLESS_GOOD_MODE_MIN_EV = 0.004

# Slightly lower volume-ratio requirement in good modes (1.5 -> 1.3)
RUTHLESS_GOOD_MODE_VOLUME_MIN = 1.3

# Master switch — set False to disable all good-mode relaxation
RUTHLESS_GOOD_MODE_RELAXATION = True


# ── Core relaxation helper ────────────────────────────────────────────────────

def get_relaxed_thresholds(
    market_mode,
    risk_profile,
    base_confirm_ev_min,
    base_confirm_volr_min,
    base_meta_min_proba,
    good_modes=None,
    relaxation_enabled=True,
    relaxed_ev_min=RUTHLESS_GOOD_MODE_MIN_EV,
    relaxed_volr_min=RUTHLESS_GOOD_MODE_VOLUME_MIN,
    relaxed_meta_min_proba=RUTHLESS_GOOD_MODE_META_MIN_PROBA,
):
    """Return (confirm_ev_min, confirm_volr_min, meta_min_proba) for the current bar."""
    if risk_profile != "ruthless" or not relaxation_enabled:
        return base_confirm_ev_min, base_confirm_volr_min, base_meta_min_proba

    allowed = good_modes if good_modes is not None else RUTHLESS_GOOD_MODES
    if market_mode in allowed:
        return (
            min(base_confirm_ev_min,   relaxed_ev_min),
            min(base_confirm_volr_min, relaxed_volr_min),
            min(base_meta_min_proba,   relaxed_meta_min_proba),
        )

    # Strict modes (chop, selloff, high_vol_reversal, None)
    return base_confirm_ev_min, base_confirm_volr_min, base_meta_min_proba


def is_good_mode(market_mode, good_modes=None):
    """Return True if market_mode is in the list of favorable modes.

    Parameters
    ----------
    market_mode : str or None
    good_modes  : list[str] or None — defaults to RUTHLESS_GOOD_MODES
    """
    allowed = good_modes if good_modes is not None else RUTHLESS_GOOD_MODES
    return market_mode in allowed


def format_relaxation_log(
    market_mode,
    base_ev,
    eff_ev,
    base_volr,
    eff_volr,
    base_meta,
    eff_meta,
):
    """Format a log line showing when good-mode relaxation is applied.

    Example::

        [relax] mode=pump ev=0.006->0.004 volr=1.5->1.3 meta=0.55->0.52
    """
    parts = [f"[relax] mode={market_mode}"]
    if eff_ev < base_ev:
        parts.append(f"ev={base_ev:.3f}->{eff_ev:.3f}")
    if eff_volr < base_volr:
        parts.append(f"volr={base_volr:.2f}->{eff_volr:.2f}")
    if eff_meta < base_meta:
        parts.append(f"meta={base_meta:.2f}->{eff_meta:.2f}")
    if len(parts) == 1:
        parts.append("(no change)")
    return " ".join(parts)


# ── Parameter resolution from config/algo ────────────────────────────────────

def resolve_good_mode_params(algo, config_module):
    """Resolve good-mode relaxation parameters from algo QC params or config.

    Returns
    -------
    dict with keys:
        enabled, good_modes, relaxed_ev_min, relaxed_volr_min, relaxed_meta_min_proba
    """
    enabled = getattr(config_module, "RUTHLESS_GOOD_MODE_RELAXATION", RUTHLESS_GOOD_MODE_RELAXATION)
    # Allow QC param override
    _raw = None
    try:
        _raw = algo.get_parameter("ruthless_good_mode_relaxation")
    except Exception:
        pass
    if _raw:
        enabled = str(_raw).lower() in ("true", "1", "yes")

    relaxed_ev   = getattr(config_module, "RUTHLESS_GOOD_MODE_MIN_EV",         RUTHLESS_GOOD_MODE_MIN_EV)
    relaxed_volr = getattr(config_module, "RUTHLESS_GOOD_MODE_VOLUME_MIN",     RUTHLESS_GOOD_MODE_VOLUME_MIN)
    relaxed_meta = getattr(config_module, "RUTHLESS_GOOD_MODE_META_MIN_PROBA", RUTHLESS_GOOD_MODE_META_MIN_PROBA)

    # Allow QC param overrides
    try:
        _v = algo.get_parameter("ruthless_good_mode_min_ev")
        if _v:
            relaxed_ev = float(_v)
    except Exception:
        pass
    try:
        _v = algo.get_parameter("ruthless_good_mode_volume_min")
        if _v:
            relaxed_volr = float(_v)
    except Exception:
        pass
    try:
        _v = algo.get_parameter("ruthless_good_mode_meta_min_proba")
        if _v:
            relaxed_meta = float(_v)
    except Exception:
        pass

    return {
        "enabled":                 enabled,
        "good_modes":              list(RUTHLESS_GOOD_MODES),
        "relaxed_ev_min":          relaxed_ev,
        "relaxed_volr_min":        relaxed_volr,
        "relaxed_meta_min_proba":  relaxed_meta,
    }
