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

import json

# Default cap on candidate records stored in memory.
CANDIDATE_JOURNAL_MAX_SIZE = 2000
# Default max candidates journaled per decision cycle.
CANDIDATE_JOURNAL_TOP_N    = 5


class CandidateJournal:
    """Journal for candidate ranking and skipped-candidate diagnostics.

    Usage
    -----
    At each decision cycle::

        journal.record_cycle(time, candidates)

    Retrieve records::

        records = journal.get_records()
        journal.to_json()

    Each candidate record contains::

        time, symbol, rank, selected, reject_reason,
        market_mode, confirm, vote_score, active_mean, active_std,
        active_n_agree, vote_yes_fraction, top3_mean, pred_return,
        ev_score, active_votes, shadow_votes, diagnostic_votes,
        entry_path
    """

    def __init__(self, max_size=CANDIDATE_JOURNAL_MAX_SIZE, top_n=CANDIDATE_JOURNAL_TOP_N,
                 logger=None):
        self._max_size = max_size
        self._top_n    = top_n
        self._logger   = logger
        self._records  = []

    # ── Record a candidate decision cycle ─────────────────────────────────────

    def record_cycle(self, time, candidates):
        """Record the top-N candidates from a decision cycle.

        Parameters
        ----------
        time       : datetime-like or str — decision timestamp
        candidates : list[dict]
            Each dict must contain at minimum:
              symbol (str), rank (int), selected (bool),
              reject_reason (str or None)
            Optional but journaled when present:
              market_mode, confirm, vote_score, active_mean, active_std,
              active_n_agree, vote_yes_fraction, top3_mean, pred_return,
              ev_score, active_votes, shadow_votes, diagnostic_votes,
              entry_path
        """
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
    """Build candidate record list for a single decision cycle.

    Parameters
    ----------
    ranked_results   : list[(sym, final_score)] — sorted descending by score
    conf_data        : dict[sym, conf_dict]
    ev_data          : dict[sym, float]
    entry_path_data  : dict[sym, str]
    scores           : dict[sym, float]
    market_mode      : str or None
    confirm_reasons  : dict[sym, str] or None
    selected_sym     : sym or None — the symbol chosen for entry
    rejected_reason  : str or None — why no entry was taken (meta-filter, regime, etc.)

    Returns
    -------
    list[dict] — ready for CandidateJournal.record_cycle()
    """
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

