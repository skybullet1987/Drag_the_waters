# Drag the Waters — Top-Coin Rotation on Kraken

A compact QuantConnect algorithm that rotates into the best-scoring Kraken USD
coin every 15 minutes, with rule-based take-profit / stop-loss / timeout exits.

## Files

| File | Purpose |
|---|---|
| `main.py` | Complete algorithm — all strategy logic in one file |
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

- **Allocation**: 80 % of portfolio per trade, one open position at a time.
- **Take-profit**: +1.2 % from entry.
- **Stop-loss**: −0.7 % from entry.
- **Time-stop**: close after 2 hours if neither TP nor SL triggered.

### Risk controls

- **Cooldown**: 15 minutes after any exit before a new entry is allowed.
- **Daily SL cap**: after 2 stop-losses in one calendar day, no new entries
  until the next day.

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
- **Rule-based scoring**: the composite score is an interpretable blend of
  momentum, RSI, and volume rather than a black-box ML model.  This makes it
  easy to tune and extend.
- **QC-first**: the algorithm uses only standard QuantConnect APIs
  (`add_crypto`, `consolidate`, `set_holdings`, `liquidate`, `schedule`) so it
  deploys without any external dependencies beyond `numpy`.
- **Upgrade path**: replace `_score()` with an ensemble of trained classifiers
  (logistic regression, gradient boosting, random forest) once a labelled
  dataset and walk-forward validation have been built in QC Research.
