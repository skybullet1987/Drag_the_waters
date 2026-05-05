# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

This is a **QuantConnect algorithmic crypto trading** codebase ("Vox"), not a web app.
It has two strategies:

- `main.py` (root) — simple rule-based Kraken top-coin rotation (benchmark).
- `Vox/` package — ML ensemble strategy with 20-feature engineering, triple-barrier labeling,
  heterogeneous soft-voting classifier, regression ensemble, and multiple risk profiles.

The algorithms are designed to run on QuantConnect's cloud (LEAN engine). They **cannot** be
executed locally without QuantConnect's `AlgorithmImports`. All local development work is
limited to running the test suite and linting.

### Running tests

```bash
python3 -m pytest Vox/tests/ -v
```

All 12 test files (700+ tests) run with pytest + numpy + scikit-learn. No QuantConnect
dependency is needed for tests. Tests cover the ML ensemble, gate logic, feature engineering,
model roles, shadow lab, apex predator, profit voting, and file-size constraints.

### Linting

No linter is configured in the repo. Use `ruff` for syntax/import checking:

```bash
ruff check Vox/ main.py --select E9,F63,F7,F82
```

### Key constraints

- **Every `.py` file must stay under 63,000 bytes** for QuantConnect compatibility.
  Check with `wc -c Vox/*.py`. The test `test_file_sizes.py` enforces this.
- `Vox/main.py` and `Vox/models.py` are near the limit (~62K chars each) — monitor closely.
- Optional dependencies (lightgbm, xgboost, catboost, hmmlearn) are guarded by try/except
  imports. Tests that need unavailable deps use `pytest.skip`.

### Module layout

| Directory/File | Purpose |
|---|---|
| `main.py` (root) | Baseline rule-based algorithm |
| `Vox/core.py` | Config constants + market mode + momentum + meta model |
| `Vox/models.py` | Feature engineering, VoxEnsemble, training pipeline |
| `Vox/infra.py` | Universe list, OrderHelper, PersistenceManager |
| `Vox/strategy.py` | Strategy logic |
| `Vox/strategy_ext.py` | Strategy extensions |
| `Vox/main.py` | VoxAlgorithm QC entry point |
| `Vox/tests/` | 12 pytest test files |

### Running backtests on QuantConnect Cloud

The algorithm can only be backtested on QuantConnect's cloud (requires `AlgorithmImports`).
Use `Vox/qc_backtest_runner.py` to automate the full pipeline:

```bash
# Requires QC_USER_ID and QC_API_TOKEN env vars (add as Cursor secrets)
python3 Vox/qc_backtest_runner.py
```

This creates a QC project, uploads files, sets parameters, compiles, runs the backtest,
and analyzes results including per-model accuracy assessment.

**Manual alternative:** Upload `Vox/*.py` to a QC project, set `risk_profile=gatling`
(or another profile) in the parameter panel, and run the backtest from the QC web UI.

### Risk profiles

| Profile | Activate via | Character |
|---|---|---|
| `balanced` | default | Tradable defaults, controlled drawdown |
| `conservative` | `risk_profile=conservative` | Strict gates, rarely trades |
| `aggressive` | `risk_profile=aggressive` | Looser gates, larger sizing |
| `ruthless` | `risk_profile=ruthless` | Maximum single-position aggression |
| `apex_predator` | `risk_profile=apex_predator` | Ultra-loose gates, multi-path entry |
| `gatling` | `risk_profile=gatling` | Gatling-gun: 5-min decisions, near-zero gates, all models active |
| `active_research` | `risk_profile=active_research` | Data collection with tiny allocation |

### Model assessment after backtest

After a backtest (especially gatling), analyze per-model predictive power:

```python
from model_assessment import compute_model_accuracy, rank_models, format_assessment_report
accuracy = compute_model_accuracy(trades, vote_threshold=0.50)
print(format_assessment_report(accuracy))
```

### Notes for cloud agents

- `python` is not on PATH; always use `python3`.
- The codebase has no `requirements.txt`, `pyproject.toml`, or `setup.py`. Dependencies are
  installed directly via pip (see update script).
- Tests use `sys.path.insert(0, ...)` to import from `Vox/` without installing the package.
- There is no build step — this is pure Python interpreted by QuantConnect.
- There is no dev server to start — the algorithm runs on QuantConnect's cloud only.
- `core.py` is near the 63KB limit (~62.9K). When adding new profile constants, create a
  separate `*_config.py` file (like `gatling_config.py`) and import as a module in `core.py`.
- The gatling profile is recognized as ruthless-like in `strategy.py` and `entry_logic.py`
  (uses `risk_profile in ("ruthless", "gatling")` checks). New ruthless-like profiles must
  be added to these checks too.
