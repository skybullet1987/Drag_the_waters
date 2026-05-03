# Drag the Waters — Execution-First Kraken Top-Coin Rotation

A single-file QuantConnect algorithm that rotates into the best-scoring Kraken
USD coin every 15 minutes.  **This version is execution-first**: it fixes
invalid-order and state-drift bugs before reintroducing ML logic.

## Why this version exists

Earlier backtests produced many `ENTRY` orders with `Price = 0` and
`Status = Invalid`, and exit logic attempted to liquidate quantities larger
than what was actually held.  The root causes were:

1. Position state was set optimistically — immediately after calling
   `set_holdings`, before any fill arrived.
2. Exit quantities were derived from a target allocation rather than the
   real portfolio holding.
3. There was no order-event callback to drive state transitions.
4. Stale internal state was never reconciled against actual holdings.

This version fixes all four issues and acts as a stable execution template
before ML scoring is layered back in.

## Files

| File | Purpose |
|---|---|
| `main.py` | Core algorithm — initialization, data handling, order state machine |
| `config.py` | Strategy constants and `setup_risk_profile()` |
| `models.py` | Feature engineering (20 features), VoxEnsemble, training pipeline |
| `execution.py` | Breakeven, momentum-fail, timeout extension, `evaluate_candidate()` |
| `market_mode.py` | Rules-based BTC market mode detection |
| `meta_model.py` | Meta-filter veto for low-conviction ruthless entries |
| `infra.py` | Universe management, order helpers, persistence |
| `risk.py` | Regime filter, risk manager, Kelly sizing |
| `momentum.py` | Momentum override conditions and score |
| `README.md` | This document |

## Strategy overview

Every 15 minutes `KrakenTopCoinAlgorithm` scores each of the 10 fixed Kraken
USD coins using a composite signal and enters the top-ranked coin long when the
signal is strong and clearly separates from the rest.

### Universe

Ten fixed Kraken spot pairs, all quoted in USD:

```
BTCUSD  ETHUSD  AVAXUSD  XRPUSD  ADAUSD
LTCUSD  LINKUSD DOTUSD   TRXUSD  SOLUSD
```

### Scoring (per coin, every 15 minutes)

| Weight | Signal | Detail |
|---|---|---|
| 50 % | Momentum | 4-bar (~1 h) return, normalised so 1 % = 1.0 |
| 30 % | RSI(14)  | Reward 55–74 (uptrend); penalise ≥ 75 (overbought) or < 40 (oversold) |
| 20 % | Volume spike | Current bar vs prior 15-bar mean, capped at 100 % above average |

The composite score is mapped to **[0, 1]**.  Entry requires:

- `score ≥ SCORE_MIN` (default **0.60**)
- gap between #1 and #2 coin `≥ SCORE_GAP` (default **0.04**)

### Position management

- **Sizing**: explicit `qty = (ALLOCATION × portfolio_value) / price` —
  no blind reliance on `set_holdings`.
- **Allocation**: 80 % of portfolio per trade, one open position at a time.
- **Take-profit**: +1.2 % from confirmed fill price.
- **Stop-loss**: −0.7 % from confirmed fill price.
- **Time-stop**: close after 2 hours if neither TP nor SL triggered.

### Risk controls

- **Cooldown**: 15 minutes after any exit before a new entry is allowed.
- **Daily SL cap**: after 2 stop-losses in one calendar day, no new entries
  until the next day.

## Execution correctness

| Fix | How it works |
|---|---|
| Fill-driven activation | `_pos_sym` / `_entry_px` are set only inside `on_order_event` on `FILLED` status |
| Actual-qty exits | `_check_exit` reads `portfolio[sym].quantity` before placing the sell order |
| Pre-trade validation | `_try_enter` checks `price > 0`, available cash, and computed `qty > 0` before sending any order |
| Duplicate-exit guard | `_exiting` flag is set when an exit order is in flight; `_check_exit` skips while it is True |
| State reconciliation | `_reconcile()` runs every tick; clears `_pos_sym` when portfolio qty has gone to zero |

## Parameters

All constants can be overridden via the QC parameter panel without editing code:

| Parameter | Default | Description |
|---|---|---|
| `take_profit` | `0.012` | Fraction gain that closes the trade (+1.2 %) |
| `stop_loss` | `0.007` | Fraction loss that closes the trade (−0.7 %) |
| `timeout_hours` | `2.0` | Maximum hours to hold a position |
| `score_min` | `0.60` | Minimum composite score to enter |
| `score_gap` | `0.04` | Minimum score lead of #1 over #2 coin |
| `allocation` | `0.80` | Portfolio fraction per trade |
| `max_daily_sl` | `2` | Stop new entries after this many daily SL hits |
| `cooldown_mins` | `15` | Minutes to wait after any exit |

## Quick Start

1. Create a new [QuantConnect](https://www.quantconnect.com/) project.
2. Copy `main.py` into the project root.
3. Set the algorithm class to `KrakenTopCoinAlgorithm`.
4. Run a backtest (default period: 2024-01-01 to 2025-12-31, $5 000 starting capital).

## Design notes

- **Single file**: all logic — scoring, entry, exit, risk controls — lives in
  `main.py`.  No helper modules are needed.
- **Execution-first**: the priority of this version is clean, valid orders and
  accurate state tracking.  The scoring heuristic is intentionally simple so
  that execution problems are easy to isolate in backtest logs.
- **QC-first**: the algorithm uses only standard QuantConnect APIs
  (`add_crypto`, `consolidate`, `market_order`, `schedule`) so it deploys
  without any external dependencies beyond `numpy`.
- **Upgrade path**: once execution is confirmed clean, replace `_score()` with
  an ensemble of trained classifiers (logistic regression, gradient boosting,
  random forest) built in QC Research.

## Ruthless v4 improvements

The `ruthless` risk profile activates a suite of aggressive trade-management
features targeting high-conviction breakout setups.

### New in v4

| Feature | Config constant | Default | Description |
|---|---|---|---|
| Breakeven stop | `RUTHLESS_BREAKEVEN_AFTER` | `0.03` | Move stop to entry+buffer once return ≥ 3% |
| Momentum-fail exit | `RUTHLESS_MOM_FAIL_ENABLED` | `True` | Cut early if return ≤ −1.2% with broken momentum |
| Timeout extension | `RUTHLESS_TIMEOUT_EXTEND_HOURS` | `12h` | Extend hold by 12h when in profit at timeout |
| Market mode gate | `MARKET_MODE_ENABLED` | `True` | Only enter in `risk_on_trend` / `pump` regimes |
| Meta-filter veto | `RUTHLESS_META_FILTER_ENABLED` | `True` | Block low-conviction signals via meta-score |
| 20-feature model | `FEATURE_COUNT = 20` in `models.py` | — | Extended feature set for better signal separation |
| Delayed trail | `RUTHLESS_TRAIL_AFTER_TP` | `0.07` | Trailing stop activates at +7% (was +4%) |
| Wider trail | `RUTHLESS_TRAIL_PCT` | `0.03` | Trail 3% from high-water mark (was 2.5%) |

### New modules

- **`execution.py`** — `evaluate_candidate()` (replaces inline scoring loop),
  `apply_breakeven()`, `should_exit_momentum_fail()`, `evaluate_timeout()`,
  `LimitOrderTracker`
- **`market_mode.py`** — `MarketModeDetector`: rules-based BTC regime
  classifier feeding off 4h bars (selloff / high_vol_reversal / pump /
  risk_on_trend / chop)
- **`meta_model.py`** — `MetaFilter`: rules-based meta-score that vetos
  low-conviction entries (disabled by default; enable with
  `RUTHLESS_META_FILTER_ENABLED = True`)

### New features in `build_features` (indices 10–19)

| Index | Name | Description |
|---|---|---|
| 10 | `range_eff` | 16-bar range efficiency (trend purity) |
| 11 | `sma_fast_slope` | 4-bar SMA slope |
| 12 | `price_vs_sma_fast` | Price relative to 4-bar SMA |
| 13 | `price_vs_sma_slow` | Price relative to 8-bar SMA |
| 14 | `recent_high_breakout` | Distance above 16-bar prior high |
| 15 | `vol_zscore` | Volume z-score over 16 bars |
| 16 | `reversal_frac` | Fraction of sign changes in last 8 bars |
| 17 | `green_bar_ratio` | Fraction of up-bars in last 8 bars |
| 18 | `atr_expansion` | Current ATR vs lagged ATR ratio |
| 19 | `btc_ret_1` | 1-bar BTC return |

> **Migration note**: saved models trained on 10 features will be automatically
> discarded on load (via `load_state` feature-count mismatch check) and a full
> retrain will be scheduled.


---

## Shadow Model Lab & Active-Vote Health Controls

This section documents the model role infrastructure added to make it safe to
test new ML models without risking degenerate signals affecting live trading.

### Why this was needed

Analysis of 11 completed trades (Jan 2025–Apr 2026) showed:

| Model | Observation | Implication |
|---|---|---|
| `gnb` | `vote_gnb = 1.0` on every trade | Degenerate/always-bullish; inflates `n_agree` |
| `lr`  | `vote_lr ≈ 0.006–0.023` on every trade | Always-bearish; never contributes a bullish signal |

The profitable behavior was driven by market mode, momentum, and exit logic —
not clean ML consensus.  Adding more models blindly would make this worse.

### Model roles

Each model has one of four roles:

| Role | Training | Prediction | Affects trading |
|---|---|---|---|
| `active` | ✅ | ✅ | ✅ Yes — counted in `active_mean`, `n_agree` |
| `shadow` | ✅ | ✅ | ❌ No — logged but never affects trades |
| `diagnostic` | ✅ | ✅ | ❌ No — logged for risk/debug only |
| `disabled` | ❌ | ❌ | ❌ Not trained or predicted |

### Default model roles

| Model | Default role | Reason |
|---|---|---|
| `hgbc` | `active` | Core booster, well-calibrated |
| `et` | `active` | Diverse random trees |
| `rf` | `active` | Bagged trees, stable |
| `lr` | `diagnostic` | Always-bearish in observed data |
| `gnb` | `diagnostic` | Always-bullish (vote_gnb=1.0); degenerate |
| `lgbm` | `shadow` | Optional external; tested in shadow first |
| `xgb` | `shadow` | Optional external; tested in shadow first |

Configure via `config.py`:

```python
MODEL_ROLE_LR   = "diagnostic"   # was always-bearish
MODEL_ROLE_GNB  = "diagnostic"   # was always-bullish
MODEL_ROLE_HGBC = "active"
MODEL_ROLE_ET   = "active"
MODEL_ROLE_RF   = "active"
MODEL_ROLE_LGBM = "shadow"
MODEL_ROLE_XGB  = "shadow"
```

### Backward-compatible fields

`class_proba`, `std_proba`, and `n_agree` in the prediction output now map to
**active-role models only**.  This means degenerate diagnostic models (GNB, LR)
no longer inflate these values.

### Shadow model lab

A set of compact shadow models is trained alongside the active ensemble.  They
never affect trading.  Shadow models include:

| ID | Type | Purpose |
|---|---|---|
| `et_shallow` | ExtraTrees, max_depth=3 | Less overfit variant |
| `rf_shallow` | RandomForest, max_depth=3 | Less overfit variant |
| `hgbc_l2` | HGBC, stronger L2 | Regularised booster |
| `cal_et` | Calibrated ExtraTrees (cv=3) | Alternative calibration |
| `cal_rf` | Calibrated RandomForest (cv=3) | Alternative calibration |
| `lr_bal` | LogisticRegression, balanced | Diagnostic linear baseline |
| `lgbm_bal` | LightGBM, balanced (if installed) | External shadow candidate |
| `xgb_bal` | XGBoost, balanced (if installed) | External shadow candidate |

Control via `config.py`:

```python
ENABLE_SHADOW_MODEL_LAB  = True
SHADOW_MODEL_MAX_COUNT   = 12
```

### Model health diagnostics

The `ModelHealthTracker` (in `model_health.py`) records rolling per-model
probability statistics and emits health flags:

| Flag | Condition |
|---|---|
| `degenerate_bullish` | ≥ 90% of predictions ≥ 0.95 |
| `degenerate_bearish` | ≥ 90% of predictions ≤ 0.05 |
| `low_variance` | rolling std < 0.01 |

Example log output:

```
[model_health] gnb role=diagnostic n=50 mean=1.000 std=0.000 yes=100% flag=degenerate_bullish
[model_health] lr  role=diagnostic n=50 mean=0.012 std=0.006 yes=0%   flag=degenerate_bearish
[model_health] hgbc role=active    n=50 mean=0.620 std=0.085 yes=72%  flag=ok
```

Configure via `config.py`:

```python
MODEL_HEALTH_ENABLED        = True
MODEL_HEALTH_MIN_OBS        = 20
MODEL_HEALTH_EXTREME_PROBA  = 0.95
MODEL_HEALTH_DEGENERATE_FRAC = 0.90
MODEL_HEALTH_LOW_STD        = 0.01
```

### Optional active std gate

A gate that blocks high-disagreement entries is available but **disabled by
default** because trade count is already low:

```python
RUTHLESS_USE_ACTIVE_STD_GATE   = False   # enable with caution
RUTHLESS_MAX_ACTIVE_STD_PROBA  = 0.30
```

### How to interpret model attribution data

Enable vote logging to see per-trade role breakdowns:

```python
LOG_MODEL_VOTES = True
```

Log line example:

```
[vote] ADAUSD active_mean=0.62 active_std=0.05 agree=3/3 mode=pump
       active=hgbc:0.70,et:0.67,rf:0.58
       shadow=et_shallow:0.64,cal_et:0.65 diag=lr:0.01
```

Pull the trade log from ObjectStore (in a Research notebook):

```python
from QuantConnect.Research import QuantBook
import json, pandas as pd

qb = QuantBook()
raw = qb.object_store.read("vox/trade_log.jsonl")
rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
df = pd.DataFrame(rows)

entries = df[df["event"] == "entry_attempt"].copy()
# shadow_votes and active_votes fields are present for each entry
```

### How to promote a shadow model to active

**Only promote after testing across multiple market periods**:

- `2023`, `2024`, `2025`, `2026 YTD`
- Bull periods, chop periods, selloffs

Promotion criteria (per period):

| Criterion | Threshold |
|---|---|
| `vote_yes_count` | ≥ 30 |
| `avg_return_when_yes` | > 0 |
| `win_rate_when_yes` | > strategy baseline |
| Not degenerate | `degenerate_bullish` = False, `degenerate_bearish` = False |
| Not redundant | < 0.95 correlation with existing active models |

Then update `config.py`:

```python
MODEL_ROLE_LGBM = "active"   # promote from shadow to active
```

### Important: 11 trades is not enough

The user's Jan 2025–Apr 2026 backtest had 22 orders.  At this sample size:

- Win-rate confidence intervals span ±20–30 percentage points
- Per-model attribution is highly noisy
- Apparent model quality differences may be pure chance

**Run multiple windows before drawing conclusions:**

```
2023 | 2024 | 2025 | 2026 YTD | bull periods | chop periods | selloffs
```

### QuantConnect file-size rule

Every Python file deployed to QuantConnect must remain **under 63,000 characters**.
If a module grows too large, split helper logic into a new file.

## Apex Predator

The **Apex Predator** regime replaces the over-tight gates that suppressed 538
entry signals down to 24 orders in the Jan 2025 – May 2026 backtest.  It is
implemented as a set of pure-Python helpers in `Vox/ruthless_v2.py` (functions
`compute_apex_score`, `apex_entry_decision`, `compute_apex_size`,
`compute_apex_atr_stops`) and is configured via `APEX_*` constants in
`Vox/config.py`.

### Weighted apex score

Every bar, a single composite score is computed from the six most informative
model columns, weighted by their empirical profit-factor from the research
diagnostics:

| Column | Weight | Diagnostic PF |
|--------|--------|---------------|
| `vote_lr_bal` | 0.35 | ~8.0 at ≥ 0.50 |
| `vote_hgbc_l2` | 0.25 | ~3.35 at ≥ 0.55 |
| `active_rf` | 0.15 | ~2.76 at ≥ 0.60 |
| `active_hgbc_l2` | 0.10 | ~3.38 at ≥ 0.50 |
| `active_lgbm_bal` | 0.10 | ~1.70 (always-on confirmer) |
| `vote_et` | 0.05 | diversifier |

Missing columns have their weight redistributed pro-rata so the score remains
correctly calibrated even when a model is unavailable.

### Four entry trigger paths

Entry fires when **any** of the following is true:

1. `apex_score >= APEX_SCORE_ENTRY` (0.55)
2. `vote_lr_bal >= 0.50` — proven PF ≈ 8 edge
3. `vote_hgbc_l2 >= 0.55 AND active_lgbm_bal >= 0.55`
4. `mean_proba >= 0.60 AND n_agree >= 3` — legacy strong-ML backstop

`confirm` / `market_mode` is a **score booster**, not a hard gate.

### Sizing (Kelly-lite + pyramiding)

```python
edge_mult = clip((apex_score - 0.50) / 0.30, 0.0, 1.5)
conf_mult = 1.0 + 0.5 * (n_agree >= 4)
size_frac = clip(APEX_BASE_ALLOC * (1 + edge_mult) * conf_mult, 0.05, 0.45)
```

Total gross exposure is capped at `APEX_MAX_GROSS` (2.0×).
Pyramiding: add a second tranche at 50 % of original size when unrealised PnL
≥ +1.5 % and `apex_score >= APEX_SCORE_PYRAMID` (max 2 adds per position).

### Stops / TP / Trail / Breakeven / Time-stop

| Parameter | Formula / Default |
|-----------|-------------------|
| Stop-loss | `entry − APEX_ATR_SL_MULT × ATR(14)`; floor 0.8%, ceil 4% |
| Take-profit | `entry + APEX_ATR_TP_MULT × ATR(14)`; floor 2.5%, ceil 15% |
| Trail arms | once unrealised PnL ≥ `APEX_TRAIL_ARM_PCT` (1.0%) |
| Trail distance | `max(APEX_TRAIL_ATR_MULT × ATR, 0.6%)` |
| Breakeven move | once MFE ≥ `APEX_BREAKEVEN_MFE` (2%), stop → entry + 0.1% |
| Time-stop | close after `APEX_TIME_STOP_HRS` (48 h) if MFE < +1% |

### Concurrency

`APEX_MAX_CONCURRENT = 8` simultaneous positions,
`APEX_MAX_PER_SYMBOL = 2` per symbol,
`APEX_COOLDOWN_MIN = 15` minute reentry cooldown.

### Tunable constants (Vox/config.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `APEX_SCORE_ENTRY` | 0.55 | Minimum apex_score to trigger entry |
| `APEX_SCORE_PYRAMID` | 0.55 | Minimum apex_score to pyramid |
| `APEX_BASE_ALLOC` | 0.20 | Baseline position size (20 % of equity) |
| `APEX_MAX_GROSS` | 2.0 | Maximum total gross exposure |
| `APEX_MAX_CONCURRENT` | 8 | Maximum simultaneous positions |
| `APEX_MAX_PER_SYMBOL` | 2 | Maximum positions per symbol |
| `APEX_COOLDOWN_MIN` | 15 | Reentry cooldown in minutes |
| `APEX_TIME_STOP_HRS` | 48 | Time-stop horizon (hours) |
| `APEX_ATR_SL_MULT` | 1.25 | ATR multiplier for stop-loss |
| `APEX_ATR_TP_MULT` | 4.0 | ATR multiplier for take-profit |
| `APEX_TRAIL_ARM_PCT` | 0.010 | PnL level to arm trailing stop |
| `APEX_TRAIL_ATR_MULT` | 0.8 | ATR multiplier for trail distance |
| `APEX_BREAKEVEN_MFE` | 0.02 | MFE level to trigger breakeven move |
