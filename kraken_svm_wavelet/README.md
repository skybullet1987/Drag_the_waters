# Kraken SVM + Wavelet Algorithm

Multi-asset SVM + Wavelet daily forecaster for the top 50 Kraken USD pairs.

## Overview

This QuantConnect algorithm is a direct generalization of the [QC HandsOnAITradingBook FX SVM Wavelet Forecasting sample](https://github.com/QuantConnect/HandsOnAITradingBook) to a top-N Kraken crypto universe.

The forecasting model (`svmwavelet.py`) is **unchanged** from the textbook implementation: Discrete Wavelet Transform (Symlet-10) for decomposition and denoising, followed by GridSearchCV SVR per wavelet band, and inverse DWT reconstruction.

## Adaptations vs the Textbook

| Feature | Textbook | This Algorithm |
|---|---|---|
| Universe | Single FX pair | Top 50 Kraken USD pairs by volume |
| Execution | `SetHoldings` (market) | Limit orders at bid/ask (maker preference) |
| Model refit | Every bar | Cached, every `refit_every_bars` bars (default: 7 = weekly) |
| Position cap | None | `max_per_asset_weight` (default: 10%) |
| Short selling | Allowed | Disabled (Kraken cash account) |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `period` | 152 | Rolling window length (days) for wavelet decomposition |
| `weight_threshold` | 0.005 | Minimum predicted return (0.5%) to open a position |
| `max_universe_size` | 10 (set to 50 for full top-Kraken run) | Maximum number of Kraken USD pairs to trade |
| `refit_every_bars` | 7 | Refit SVMWavelet every N daily bars (1 = full textbook fidelity) |
| `max_per_asset_weight` | 0.10 | Maximum portfolio weight per asset (10%) |

## How to Deploy on QuantConnect

1. Create a new QC project.
2. Copy all files from this folder into the project root.
3. Set the algorithm class to `KrakenSvmWaveletAlgorithm`.
4. Run a backtest or deploy to live trading.

## Performance Note

With 50 assets × full textbook GridSearchCV (30 candidate models × 5-fold CV × ~4 wavelet bands = ~600 SVR fits per asset per refit), even with a weekly cache (`refit_every_bars=7`) the backtest will take noticeably longer than a single-asset run.

**Recommendation:** Start with `max_universe_size=10` for a fast first run, then scale up.

## What Was Deleted

The previous sophisticated rule-based algorithm at the repository root was retired after losing 21% in backtests. The barebones SVM+Wavelet approach replaced it as the primary strategy. All legacy files (`main.py`, `events.py`, `execution.py`, `scoring.py`, `circuit_breaker.py`, `realistic_slippage.py`, `alt-data.py`, standalone tools, and tests) have been removed from the repository.
