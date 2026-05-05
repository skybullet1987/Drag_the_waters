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

### Notes for cloud agents

- `python` is not on PATH; always use `python3`.
- The codebase has no `requirements.txt`, `pyproject.toml`, or `setup.py`. Dependencies are
  installed directly via pip (see update script).
- Tests use `sys.path.insert(0, ...)` to import from `Vox/` without installing the package.
- There is no build step — this is pure Python interpreted by QuantConnect.
- There is no dev server to start — the algorithm runs on QuantConnect's cloud only.
