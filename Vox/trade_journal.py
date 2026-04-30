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

import json

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
        """Compute per-model attribution metrics from completed journal records.

        Returns
        -------
        dict[str, dict] — model_id -> attribution metrics:
            vote_yes_count       — times model voted yes (proba >= threshold)
            vote_no_count        — times model voted no
            win_rate_when_yes    — win rate when model voted yes (None if no data)
            avg_return_when_yes  — avg realized return when model voted yes
            avg_return_when_no   — avg realized return when model voted no
            precision_proxy      — fraction of yes-votes that resulted in wins

        Notes
        -----
        With a small sample (< 50 trades), results are noisy.
        See README.md for offline multi-window analysis guidance.
        """
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
