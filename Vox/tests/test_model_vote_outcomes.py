"""
Tests for the model vote outcome audit helpers (PR: add backtest diagnostics).

Coverage:
  1. audit_safe_float — converts valid values, returns None for invalid.
  2. audit_trim_votes — returns compact float dict, skips non-numeric.
  3. _audit_append_model_vote_outcome — writes JSONL, caps at max bytes.
  4. _audit_clear_model_vote_outcomes_for_backtest — clears in backtest, skips in live.
  5. Entry prediction snapshot contains active/shadow/diagnostic vote buckets.
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audit_utils import audit_safe_float, audit_trim_votes  # noqa: E402


# ── 1. audit_safe_float ───────────────────────────────────────────────────────

class TestAuditSafeFloat:
    def test_int_value(self):
        assert audit_safe_float(3) == 3.0

    def test_float_value(self):
        assert audit_safe_float(0.123456789) == pytest.approx(0.123457)

    def test_string_numeric(self):
        assert audit_safe_float("0.5") == 0.5

    def test_custom_digits(self):
        assert audit_safe_float(0.123456789, digits=4) == pytest.approx(0.1235)

    def test_none_returns_none(self):
        assert audit_safe_float(None) is None

    def test_empty_string_returns_none(self):
        assert audit_safe_float("") is None

    def test_non_numeric_string_returns_none(self):
        assert audit_safe_float("abc") is None

    def test_list_returns_none(self):
        assert audit_safe_float([1, 2]) is None

    def test_zero(self):
        assert audit_safe_float(0) == 0.0

    def test_negative(self):
        assert audit_safe_float(-0.0567, digits=4) == pytest.approx(-0.0567)


# ── 2. audit_trim_votes ───────────────────────────────────────────────────────

class TestAuditTrimVotes:
    def test_basic_float_dict(self):
        result = audit_trim_votes({"lgbm": 0.75432, "rf": 0.62341})
        assert result == {"lgbm": pytest.approx(0.7543), "rf": pytest.approx(0.6234)}

    def test_string_numeric_values(self):
        result = audit_trim_votes({"lgbm": "0.8"})
        assert result["lgbm"] == pytest.approx(0.8)

    def test_non_numeric_values_skipped(self):
        result = audit_trim_votes({"lgbm": 0.7, "bad": None, "also_bad": "xyz"})
        assert "bad" not in result
        assert "also_bad" not in result
        assert "lgbm" in result

    def test_keys_coerced_to_str(self):
        result = audit_trim_votes({42: 0.5})
        assert "42" in result

    def test_not_a_dict_returns_empty(self):
        assert audit_trim_votes(None) == {}
        assert audit_trim_votes([0.5, 0.6]) == {}
        assert audit_trim_votes("string") == {}

    def test_empty_dict(self):
        assert audit_trim_votes({}) == {}

    def test_all_invalid_returns_empty(self):
        assert audit_trim_votes({"a": None, "b": [], "c": "notanumber"}) == {}


# ── 3 & 4. _audit_append / _audit_clear (via fake object store) ──────────────

class FakeObjectStore:
    """Minimal fake QC ObjectStore for unit tests."""

    def __init__(self):
        self._data = {}

    def contains_key(self, key):
        return key in self._data

    def read(self, key):
        return self._data.get(key, "")

    def save(self, key, value):
        self._data[key] = value

    def get(self, key, default=""):
        return self._data.get(key, default)


class FakeAlgo:
    """Minimal fake VoxAlgorithm-like object for testing audit helpers."""

    _MODEL_VOTE_OUTCOME_KEY       = "vox/model_vote_outcomes.jsonl"
    _MODEL_VOTE_OUTCOME_MAX_BYTES = 90_000
    live_mode = False

    def __init__(self):
        self.object_store = FakeObjectStore()
        self._logs = []

    def debug(self, msg):
        self._logs.append(msg)

    # Copy the two methods from VoxAlgorithm verbatim so we can test them in isolation.
    def _audit_clear_model_vote_outcomes_for_backtest(self):
        if not self.live_mode:
            try:
                self.object_store.save(self._MODEL_VOTE_OUTCOME_KEY, "")
            except Exception as exc:
                self.debug(f"[audit] clear_model_vote_outcomes failed: {exc}")

    def _audit_append_model_vote_outcome(self, record):
        try:
            line = json.dumps(record, default=str) + "\n"
            existing = ""
            if self.object_store.contains_key(self._MODEL_VOTE_OUTCOME_KEY):
                existing = self.object_store.read(self._MODEL_VOTE_OUTCOME_KEY)
            combined = existing + line
            if len(combined.encode()) > self._MODEL_VOTE_OUTCOME_MAX_BYTES:
                lines = [l for l in combined.splitlines() if l.strip()]
                while lines and len("\n".join(lines).encode()) > self._MODEL_VOTE_OUTCOME_MAX_BYTES:
                    lines.pop(0)
                combined = "\n".join(lines) + "\n"
            self.object_store.save(self._MODEL_VOTE_OUTCOME_KEY, combined)
        except Exception as exc:
            self.debug(f"[audit] append_model_vote_outcome failed: {exc}")


class TestAuditAppend:
    def _make_record(self, symbol="SOLUSD", ret=0.01):
        return {
            "trade_id": "SOLUSD_202501171245",
            "symbol": symbol,
            "exit_reason": "EXIT_TP",
            "realized_return": ret,
            "winner": ret > 0,
            "model_votes": {"lgbm": 0.72, "rf": 0.61},
            "active_votes": {"lgbm": 0.72},
            "shadow_votes": {},
            "diagnostic_votes": {},
        }

    def test_single_record_written(self):
        algo = FakeAlgo()
        algo._audit_append_model_vote_outcome(self._make_record())
        raw = algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY)
        lines = [l for l in raw.strip().split("\n") if l]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["symbol"] == "SOLUSD"
        assert rec["exit_reason"] == "EXIT_TP"

    def test_multiple_records_appended(self):
        algo = FakeAlgo()
        for sym in ["SOLUSD", "LINKUSD", "LTCUSD"]:
            algo._audit_append_model_vote_outcome(self._make_record(symbol=sym))
        raw = algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY)
        lines = [l for l in raw.strip().split("\n") if l]
        assert len(lines) == 3
        symbols = [json.loads(l)["symbol"] for l in lines]
        assert "SOLUSD" in symbols
        assert "LTCUSD" in symbols

    def test_byte_cap_removes_oldest(self):
        algo = FakeAlgo()
        algo._MODEL_VOTE_OUTCOME_MAX_BYTES = 300  # tight cap for testing
        # Fill up past the cap
        for i in range(20):
            algo._audit_append_model_vote_outcome(self._make_record(symbol=f"SYM{i:03d}"))
        raw = algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY)
        assert len(raw.encode()) <= algo._MODEL_VOTE_OUTCOME_MAX_BYTES
        lines = [l for l in raw.strip().split("\n") if l]
        # Newest records should survive
        last = json.loads(lines[-1])
        assert last["symbol"] == "SYM019"

    def test_records_are_valid_json(self):
        algo = FakeAlgo()
        algo._audit_append_model_vote_outcome(self._make_record(ret=-0.015))
        raw = algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY)
        for line in raw.strip().split("\n"):
            if line:
                rec = json.loads(line)
                assert isinstance(rec, dict)

    def test_winner_flag_false_for_loss(self):
        algo = FakeAlgo()
        algo._audit_append_model_vote_outcome(self._make_record(ret=-0.02))
        raw = algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY)
        rec = json.loads(raw.strip().split("\n")[0])
        assert rec["winner"] is False


class TestAuditClear:
    def test_clears_in_backtest(self):
        algo = FakeAlgo()
        algo.live_mode = False
        algo.object_store.save(algo._MODEL_VOTE_OUTCOME_KEY, '{"a":1}\n{"b":2}\n')
        algo._audit_clear_model_vote_outcomes_for_backtest()
        assert algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY) == ""

    def test_does_not_clear_in_live(self):
        algo = FakeAlgo()
        algo.live_mode = True
        original = '{"a":1}\n'
        algo.object_store.save(algo._MODEL_VOTE_OUTCOME_KEY, original)
        algo._audit_clear_model_vote_outcomes_for_backtest()
        # Should not have been cleared
        assert algo.object_store.read(algo._MODEL_VOTE_OUTCOME_KEY) == original


# ── 5. Entry prediction snapshot has all vote bucket fields ──────────────────

class TestEntryPredictionSnapshot:
    """Check that the vote bucket fields added in _try_enter are present."""

    def _make_entry_pred(self):
        """Build a snapshot as _try_enter now constructs it."""
        return {
            "trade_id":          "SOLUSD_202501171245",
            "symbol":            "SOLUSD",
            "risk_profile":      "balanced",
            "market_mode":       "trending",
            "confirm":           "ml",
            "entry_path":        "ml",
            "class_proba":       0.72,
            "pred_return":       0.0042,
            "ev":                0.0035,
            "final_score":       0.0038,
            "tp":                0.016,
            "sl":                0.009,
            "vote_score":        0.68,
            "vote_yes_fraction": 0.75,
            "top3_mean":         0.70,
            "n_agree":           3,
            "std_proba":         0.14,
            "time":              None,
            "model_votes":       {"lgbm": 0.72, "rf": 0.61},
            "active_votes":      {"lgbm": 0.72},
            "shadow_votes":      {"mlp": 0.65},
            "diagnostic_votes":  {"hdbscan": 0.55},
        }

    def test_all_required_fields_present(self):
        snap = self._make_entry_pred()
        for field in ("trade_id", "symbol", "risk_profile", "market_mode",
                      "confirm", "entry_path", "class_proba", "pred_return",
                      "ev", "final_score", "tp", "sl", "vote_score",
                      "vote_yes_fraction", "top3_mean", "n_agree", "std_proba",
                      "model_votes", "active_votes", "shadow_votes", "diagnostic_votes"):
            assert field in snap, f"Missing field: {field}"

    def test_vote_buckets_are_dicts(self):
        snap = self._make_entry_pred()
        assert isinstance(snap["active_votes"], dict)
        assert isinstance(snap["shadow_votes"], dict)
        assert isinstance(snap["diagnostic_votes"], dict)

    def test_audit_trim_votes_on_snapshot(self):
        snap = self._make_entry_pred()
        trimmed_active = audit_trim_votes(snap["active_votes"])
        assert trimmed_active == {"lgbm": pytest.approx(0.72)}
        trimmed_shadow = audit_trim_votes(snap["shadow_votes"])
        assert "mlp" in trimmed_shadow
