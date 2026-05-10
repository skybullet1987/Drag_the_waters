# backtest_audit — Backtest-vs-Live gap measurement harness

Tools to **measure**, **reproduce**, and **prevent** the backtest-vs-live
performance gap that killed every prior strategy in this repo.

> The whole purpose of this harness: stop trusting backtest numbers blindly.
> The MG36 paper-trade log (`fixtures/mg36_paper_2026-03-16.txt`) showed that
> a strategy with `Sharpe 134` in backtest produced `−5.24% in 5 days` live.
> This package quantifies that gap and reproduces it under simulation.

---

## Module map

| Module | Purpose |
|---|---|
| `log_parser.py`     | Parse QC `algorithm-log_*.txt` → typed dataclasses; pair round-trip trades |
| `qc_api.py`         | QC REST v2 wrapper — read + write methods (project, files, compile, backtest) |
| `qc_runner.py`      | One-command Pulse deploy + backtest |
| `qc_orders.py`      | Convert QC backtest order JSON → `CompletedTrade` pairs |
| `qc_sweep_runner.py`| Phase 3 (param × window) automation |
| `compare.py`        | Live ↔ backtest pair matching + gap attribution |
| `harsh_simulator.py`| Pessimistic QC overrides anchored to live evidence |
| `regime_runner.py`  | 6 regime windows + survivability scoring |
| `report.py`         | Self-contained HTML renderer |
| `param_sweep.py`    | Latin hypercube sampler for 12-dim Pulse parameter space |
| `run_mg36_audit.py` | Phase 0 entry point — produces `reports/mg36_audit.html` |
| `run_phase2_validation.py` | Phase 2 auto-harsh validation runner |

---

## Setup

```bash
export QC_USER_ID=<your_qc_user_id>
export QC_API_TOKEN=<your_qc_api_token>
```

The QC API client also accepts legacy aliases: `QC_UID` / `QC_TOKEN`.

---

## Common workflows

### Audit any past live deployment
```bash
python3 backtest_audit/run_mg36_audit.py
# → backtest_audit/reports/mg36_audit.html
```

To audit a different log, pass `--log path/to/log.txt`.

### Phase 2 — harsh-sim validation
```bash
python3 backtest_audit/run_phase2_validation.py --project-id <PID>
# → backtest_audit/reports/phase2_validation_<timestamp>.html
```

What's in the report:
- Side-by-side stats: standard backtest vs harsh backtest
- Per-trade matched pairs, top 50 by absolute return gap
- Mean / p90 / p99 fill-price gap in bps
- Gap attribution (slippage_excess, missed_profit_bt, bad_extra_live, residual)
- Classification: PASS (gap ratio ∈ [0.30, 0.70]) / FAIL / WARN

### Phase 3 — walk-forward sweep
```bash
# Cost-bounded first run:
python3 backtest_audit/qc_sweep_runner.py --project-id <PID> \
    --n-samples 20 --max-windows 4 --use-harsh-sim

# Full sweep (~50 hours cloud time):
python3 backtest_audit/qc_sweep_runner.py --project-id <PID> \
    --n-samples 50 --use-harsh-sim

# Dry run (no QC calls — smoke test only):
python3 backtest_audit/qc_sweep_runner.py --dry-run --n-samples 30
```

Output: `backtest_audit/reports/phase3_sweep.html` ranked by survivability.

### Single-shot deploy + backtest
```bash
python3 backtest_audit/qc_runner.py --project-id <PID> --backtest-name "x"
```
Or to create a brand-new project:
```bash
python3 backtest_audit/qc_runner.py --project-name "Pulse-2026"
```

---

## How `runtime_overrides.py` works

`Pulse/main.py._apply_runtime_overrides()` reads a small Python file at
the QC project root that looks like:

```python
# runtime_overrides.py — pushed by qc_sweep_runner.py per backtest
OVERRIDES = {
    "SCALP_ENTRY_THRESHOLD": 0.55,
    "QUICK_TAKE_PROFIT_PCT": 0.12,
    "PULSE_OVERRIDE_START_YEAR": 2024,
    "PULSE_OVERRIDE_END_YEAR":   2024,
    "use_harsh_sim": True,
}
```

Routing:
- Keys in `Pulse.main.SPECIAL_OVERRIDE_KEYS` (dates, `use_harsh_sim`) are
  read by `_param()` during Initialize, *before* QC's parameter panel.
- Other keys with matching `Pulse/config.py` constants are assigned into
  the config module so subsequent imports pick up the new values.

This sidesteps QC's awkward per-backtest parameter API while staying
transparent: every overridden value is logged at start of backtest.

---

## Survivability scoring (`regime_runner.score_param_set`)

```
score = mean(net_return)
      − consistency_penalty × std(net_return)
      − worst_case_penalty  × max(drawdown_per_window)
```

A param set is **blessed** if all hold:
- `score > 0`
- 0 windows are catastrophic (`net_return < -15%` OR `DD > 25%`)
- ≥ 4 of 6 windows are positive

The blessed parameter set with the highest score is the recommended
deployment for Phase 5 paper trading.

---

## Test fixtures

| Fixture | What it represents |
|---|---|
| `fixtures/mg36_paper_2026-03-16.txt` | Real losing-case live log (the smoking gun) |
| `fixtures/synthetic_winner_paper.txt` | Synthetic winning case (verifies harness on +EV side) |
| `fixtures/build_winning_fixture.py` | Re-generator for the synthetic fixture |

Both are loaded by `tests/test_log_parser.py` to verify the harness
handles both directions correctly.

---

## API stability notes

Every module here is pure-Python where possible (HAS_QC guard pattern).
The QC adapter classes (`HarshSlippageModel`, `HarshFeeModel`, etc.) are
gated by `try: from AlgorithmImports import * except: HAS_QC = False`,
so the entire harness imports cleanly outside QC for unit testing.

`qc_api.py` and `qc_runner.py` make actual HTTP calls and require valid
credentials. Tests using them are gated by `@NEEDS_LIVE` markers and
skip automatically when env vars are absent.
