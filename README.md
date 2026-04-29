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

