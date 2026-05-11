"""Tests for apex_data fetchers (no network — uses fixture data)."""

from __future__ import annotations

import os
from datetime import date

import pytest

from apex_data.fetch_etf_flows import (
    parse_source_csv as parse_etf_csv,
    to_apex_csv as etf_to_apex_csv,
    write_apex_csv as etf_write,
)
from apex_data.fetch_token_unlocks import (
    KNOWN_UNLOCK_SCHEDULES,
    project_schedule, project_all,
    to_csv as unlocks_to_csv,
    write_csv as unlocks_write,
    _eom, _next_month,
)


# ───────────────────────────────────────────────────────────────────────────────
# fetch_etf_flows.parse_source_csv
# ───────────────────────────────────────────────────────────────────────────────

ETF_FIXTURE = """date,IBIT,FBTC,GBTC,total
2024-01-11,111.7,227.0,-95.1,655.3
2024-01-12,386.0,195.3,-484.1,203.0
2024-01-15,-,-,-,-
2024-01-16,212.7,102.0,-594.4,-52.7
malformed,line,here
"""


def test_etf_parse_extracts_date_and_total():
    rows = parse_etf_csv(ETF_FIXTURE)
    assert rows == [
        ("2024-01-11", 655.3),
        ("2024-01-12", 203.0),
        ("2024-01-16", -52.7),
    ]


def test_etf_parse_returns_empty_on_no_header():
    assert parse_etf_csv("") == []
    assert parse_etf_csv("just,one,line") == []


def test_etf_parse_returns_empty_when_no_total_col():
    s = "date,IBIT,FBTC\n2024-01-11,100,200\n"
    assert parse_etf_csv(s) == []


def test_etf_parse_handles_dollar_signs():
    s = "date,total\n2024-01-11,$500.50\n"
    assert parse_etf_csv(s) == [("2024-01-11", 500.5)]


def test_etf_to_apex_csv_round_trip():
    rendered = etf_to_apex_csv([("2024-01-11", 655.3), ("2024-01-12", -200.0)])
    assert "Date,Inflow_USD_M" in rendered
    assert "2024-01-11,655.30" in rendered
    assert "2024-01-12,-200.00" in rendered


def test_etf_write_atomic_no_partial_file_on_clean_exit(tmp_path):
    out = tmp_path / "etf.csv"
    etf_write([("2024-01-11", 100.0)], str(out))
    assert out.exists()
    # tmp file should be cleaned up
    assert not (tmp_path / "etf.csv.tmp").exists()
    assert "Date,Inflow_USD_M" in out.read_text()


def test_etf_apex_csv_consumable_by_pulse_loader(tmp_path):
    """Round-trip: render → write → load via Pulse.apex.data.etf_flows."""
    from Pulse.apex.data.etf_flows import ETFFlowStore
    out = tmp_path / "etf.csv"
    etf_write([("2024-01-11", 100.0), ("2024-01-12", -50.0),
               ("2024-01-13", 200.0)], str(out))
    store = ETFFlowStore(keep_days=10)
    store.load_csv(out.read_text())
    assert store.get() == [100.0, -50.0, 200.0]


# ───────────────────────────────────────────────────────────────────────────────
# fetch_token_unlocks helpers
# ───────────────────────────────────────────────────────────────────────────────

def test_eom_returns_last_day_of_month():
    assert _eom(2024, 2) == date(2024, 2, 29)   # leap
    assert _eom(2025, 2) == date(2025, 2, 28)
    assert _eom(2024, 12) == date(2024, 12, 31)


def test_next_month_handles_year_rollover():
    assert _next_month(date(2025, 12, 16)) == date(2026, 1, 16)


def test_next_month_clamps_to_month_length():
    assert _next_month(date(2025, 1, 31)) == date(2025, 2, 28)


# ───────────────────────────────────────────────────────────────────────────────
# fetch_token_unlocks.project_schedule
# ───────────────────────────────────────────────────────────────────────────────

def _arb_schedule() -> dict:
    return {
        "symbol":          "ARB",
        "first_unlock":    date(2024, 3, 16),
        "cadence":         "monthly_day",
        "day_of_month":    16,
        "tokens_per_event": 100_000_000,
        "circulating_at_first_unlock": 5_000_000_000,
        "stop_after":      date(2024, 6, 16),   # 4 events
        "source":          "test",
    }


def _op_schedule() -> dict:
    return {
        "symbol":          "OP",
        "first_unlock":    date(2024, 1, 31),
        "cadence":         "monthly_eom",
        "day_of_month":    None,
        "tokens_per_event": 24_000_000,
        "circulating_at_first_unlock": 1_100_000_000,
        "stop_after":      date(2024, 4, 30),   # 4 events
        "source":          "test",
    }


def test_project_schedule_arb_emits_correct_dates():
    rows = project_schedule(_arb_schedule(), date(2024, 12, 31))
    assert [r[0] for r in rows] == [
        date(2024, 3, 16), date(2024, 4, 16),
        date(2024, 5, 16), date(2024, 6, 16),
    ]
    assert all(r[1] == "ARB" for r in rows)


def test_project_schedule_op_uses_eom():
    rows = project_schedule(_op_schedule(), date(2024, 12, 31))
    assert [r[0] for r in rows] == [
        date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31),
        date(2024, 4, 30),
    ]


def test_project_schedule_pct_decreases_as_supply_grows():
    """Each unlock dilutes against the new larger supply → next pct < prev."""
    rows = project_schedule(_arb_schedule(), date(2024, 12, 31))
    pcts = [r[3] for r in rows]
    for a, b in zip(pcts, pcts[1:]):
        assert b < a


def test_project_schedule_stops_at_end_date():
    rows = project_schedule(_arb_schedule(), date(2024, 4, 16))
    # stop_after is 2024-06-16 in fixture, but end_date narrows to 2024-04-16
    assert [r[0] for r in rows] == [
        date(2024, 3, 16), date(2024, 4, 16),
    ]


def test_project_schedule_unknown_cadence_raises():
    bad = dict(_arb_schedule())
    bad["cadence"] = "weekly"
    with pytest.raises(ValueError, match="unknown cadence"):
        project_schedule(bad, date(2024, 12, 31))


def test_project_all_merges_and_sorts():
    rows = project_all(
        [_arb_schedule(), _op_schedule()],
        date(2024, 12, 31),
    )
    # Sorted by date ascending (then symbol)
    dates = [r[0] for r in rows]
    assert dates == sorted(dates)
    # Both symbols present
    syms = {r[1] for r in rows}
    assert syms == {"ARB", "OP"}


def test_known_schedules_real_data_sanity():
    """Run the actual hardcoded schedules and ensure they produce the
    expected number of events for the next 12 months."""
    rows = project_all(KNOWN_UNLOCK_SCHEDULES, date(2026, 12, 31))
    assert len(rows) > 30      # at least many monthly events
    syms = {r[1] for r in rows}
    assert syms == {"ARB", "OP"}
    # All pcts within plausible range
    for r in rows:
        assert 0.001 < r[3] < 0.10   # 0.1%–10% of supply


# ───────────────────────────────────────────────────────────────────────────────
# unlocks CSV → apex loader round-trip
# ───────────────────────────────────────────────────────────────────────────────

def test_unlocks_csv_consumable_by_pulse_loader(tmp_path):
    from Pulse.apex.data.token_unlocks import (
        UnlockCalendarStore, upcoming_unlocks,
    )
    rows = project_all(
        [_arb_schedule(), _op_schedule()],
        date(2024, 12, 31),
    )
    out = tmp_path / "unlocks.csv"
    unlocks_write(rows, str(out))
    store = UnlockCalendarStore()
    store.load_csv(out.read_text())
    # Should have all 8 events (4 ARB + 4 OP)
    assert len(store.get()) == 8
    # Spot check: ARB unlock on 2024-03-16 should be picked up
    upcoming = upcoming_unlocks(
        store.get(), "ARBUSD",
        now=__import__("datetime").datetime(2024, 3, 10),
        look_ahead_days=7,
    )
    assert len(upcoming) == 1
    assert upcoming[0].symbol == "ARB"


def test_unlocks_to_csv_format_matches_loader_expectation():
    rows = [(date(2024, 3, 16), "ARB", 1e9, 0.025)]
    s = unlocks_to_csv(rows)
    assert "Date,Symbol,UnlockUSD,UnlockPctOfSupply" in s
    assert "2024-03-16,ARB,1000000000.00,0.025000" in s
