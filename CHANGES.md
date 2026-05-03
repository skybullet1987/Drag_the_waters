# CHANGES — Vox Package Consolidation + Apex Predator Strategy

## File Consolidation (v3.0)

### Merged Files

| New File | Size | Source Files Merged |
|----------|------|---------------------|
| `Vox/core.py` | ~59 KB | `config.py`, `market_mode.py`, `momentum.py`, `meta_model.py` |
| `Vox/infra.py` | ~43 KB | `infra.py` (expanded), `model_registry.py`, `model_health.py` |
| `Vox/strategy.py` | ~60 KB | `aggressive_config.py`, `apex_voting.py`, `risk.py`, `execution.py`, `profit_voting.py`, `shadow_lab.py` |
| `Vox/strategy_ext.py` | ~58 KB | `ruthless_v2.py` + entire `ruthless/` subpackage (`cfg.py`, `positions.py`, `scoring.py`, `pump.py`, `meta.py`, `machine_gun.py`, `apex.py`) inlined |
| `Vox/journals.py` | ~60 KB | `candidate_journal.py`, `trade_journal.py`, `trade_vote_audit.py`, `diagnostics.py`, `tuning.py` |
| `Vox/main.py` | ~63 KB | Kept as-is (imports updated to new module names) |
| `Vox/models.py` | ~63 KB | Kept as-is (one import updated: `model_registry` → `infra`) |

**Result**: 28 Python files → 7 Python files, each ≤ 63,000 bytes (hard limit for QuantConnect compatibility).

### Deleted Files

All original source files have been deleted after merging:
- `config.py`, `market_mode.py`, `momentum.py`, `meta_model.py`
- `model_registry.py`, `model_health.py`
- `aggressive_config.py`, `apex_voting.py`, `risk.py`, `execution.py`, `profit_voting.py`, `shadow_lab.py`
- `candidate_journal.py`, `trade_journal.py`, `trade_vote_audit.py`, `diagnostics.py`, `tuning.py`
- `ruthless_v2.py` and entire `ruthless/` subpackage directory

---

## Apex Predator Strategy Parameters

All tunable constants live in `Vox/strategy.py` under the `# === APEX PREDATOR PARAMETERS ===` comment block.

### 2a. Loosened Entry Gates (fire on ~90%+ of signals)

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `APEX_GATE_MIN_CLASS_PROBA` | ~0.55 | **0.45** | Lower floor fires more signals; lgbm_bal still profitable at 0.50 (PF 1.70) |
| `APEX_GATE_MIN_N_AGREE` | 2–3 | **1** | Only 1 model needs to agree — allows momentum_override + trend_momentum paths |
| `APEX_GATE_MIN_FINAL_SCORE` | ~0.3 | **0.0** | No minimum score veto; borderline-EV trades now fire |
| `APEX_GATE_COOLDOWN_MIN` | 60+ min | **15 min** | 15-minute re-entry cooldown per symbol (was hours) |
| `APEX_GATE_MAX_CONCURRENT` | 1–3 | **12** | Up to 12 concurrent open positions |
| `APEX_GATE_MAX_PER_SYMBOL` | 1 | **3** | Up to 3 simultaneous positions per symbol |
| `APEX_WEIGHTED_YES_THRESHOLD` | 0.60 | **0.45** | Weighted vote threshold lowered; allows more signal paths |

### 2b. Tighter Exits (faster turnover / gatling-gun style)

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `APEX_SL_ATR_MULT` | 3.0 | **1.5** | SL = 1.5× ATR — losers cut faster |
| `APEX_SL_PCT_FLOOR` | 0.05 | **0.025** | 2.5% max SL distance (was 5%) |
| `APEX_TIME_STOP_DAYS` | — | **30** | Exit position if open >30 days without sufficient MFE |
| `APEX_TRAIL_MULT` | 4.0 | **4.0** | ATR trail multiplier — kept; EXIT_TRAIL already producing biggest wins |
| `APEX_PYRAMID_MFE_ATR` | — | **1.0** | Add pyramid tranche after +1 ATR unrealised |

### 2c. Conviction-Weighted Sizing

| Parameter | Value | Reason |
|-----------|-------|--------|
| `APEX_SIZE_BASE_ALLOC` | **0.10** | 10% equity baseline per trade |
| `APEX_SIZE_CONV_K` | **4.0** | 4× conviction scaling (high-conviction = 2× base) |
| `APEX_SIZE_MIN_FRAC` | **0.05** | 5% minimum position size |
| `APEX_SIZE_MAX_FRAC` | **0.25** | 25% maximum per position |
| `APEX_USE_LEVERAGE` | **True** | Allow up to 3× total notional |
| `APEX_MAX_LEVERAGE` | **3.0** | Total leverage cap (blow-up protection) |

### 2d. Fired Paths

- `momentum_override` path: re-enabled via `APEX_GATE_MIN_N_AGREE = 1` and `APEX_WEIGHTED_YES_THRESHOLD = 0.45`
- `trend_momentum_relax` path: re-enabled via lowered `APEX_GATE_MIN_CLASS_PROBA`
- `APEX_COMBO_HGBC_MIN = 0.55` + `APEX_COMBO_LGBM_MIN = 0.55` — combo trigger fires on two strong models

### 2e. Symbol Universe

`KRAKEN_PAIRS` in `Vox/infra.py` now includes all 20 liquid crypto symbols:
ADAUSD, SOLUSD, LINKUSD, AAVEUSD, XRPUSD, LTCUSD, INJUSD, NEARUSD, AVAXUSD, DOTUSD,
OPUSD, ARBUSD, BCHUSD, MATICUSD, XDGUSD, ETHUSD, BTCUSD, UNIUSD, TRXUSD.

---

## Import Mapping (for reference)

| Old Import | New Import |
|------------|-----------|
| `from config import ...` | `from core import ...` |
| `from market_mode import MarketModeDetector` | `from core import MarketModeDetector` |
| `from momentum import ...` | `from core import ...` |
| `from meta_model import MetaFilter` | `from core import MetaFilter` |
| `from model_registry import ...` | `from infra import ...` |
| `from model_health import ...` | `from infra import ...` |
| `from aggressive_config import ...` | `from strategy import ...` |
| `from apex_voting import ...` | `from strategy import ...` |
| `from risk import ...` | `from strategy import ...` |
| `from execution import ...` | `from strategy import ...` |
| `from profit_voting import ...` | `from strategy import ...` |
| `from shadow_lab import ...` | `from strategy import ...` |
| `from ruthless_v2 import ...` | `from strategy_ext import ...` |
| `from candidate_journal import ...` | `from journals import ...` |
| `from trade_journal import ...` | `from journals import ...` |
| `from trade_vote_audit import ...` | `from journals import ...` |
| `from diagnostics import ...` | `from journals import ...` |
| `from tuning import ...` | `from journals import ...` |
