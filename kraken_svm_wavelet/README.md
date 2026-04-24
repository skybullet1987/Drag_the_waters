# Kraken SVM + Wavelet Algorithm

Multi-asset SVM + Wavelet daily forecaster for the top 10 Kraken USD pairs.

## Overview

This QuantConnect algorithm is a direct generalization of the [QC HandsOnAITradingBook FX SVM Wavelet Forecasting sample](https://github.com/QuantConnect/HandsOnAITradingBook) to a top-N Kraken crypto universe.

The forecasting model (`svmwavelet.py`) is **unchanged** from the textbook implementation: Discrete Wavelet Transform (Symlet-10) for decomposition and denoising, followed by GridSearchCV SVR per wavelet band, and inverse DWT reconstruction.

## Adaptations vs the Textbook

| Feature | Textbook | This Algorithm |
|---|---|---|
| Universe | Single FX pair | Top 10 Kraken USD pairs by volume |
| Execution | `SetHoldings` (market) | `set_holdings` (atomic list form, market orders, taker fees) — limit-order execution will be revisited as a separate experiment once the multi-asset signal is validated |
| Model refit | Every bar | Cached, every `refit_every_bars` bars (default: 7 = weekly) |
| Position cap | None | `max_per_asset_weight` (default: 5%) |
| Short selling | Allowed | Disabled (Kraken cash account) |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `period` | 152 | Rolling window length (days) for wavelet decomposition |
| `weight_threshold` | 0.005 | Minimum predicted return (0.5%) to open a position |
| `max_universe_size` | 10 (set to 50 for full top-Kraken run) | Maximum number of Kraken USD pairs to trade |
| `refit_every_bars` | 7 | Refit SVMWavelet every N daily bars (1 = full textbook fidelity) |
| `max_per_asset_weight` | 0.05 | Maximum portfolio weight per asset (5%) |
| `min_order_notional_usd` | 50.0 | Minimum dollar value for a buy target; targets below this are skipped to avoid Kraken minimum-order-size rejections |

## Why $50,000 Starting Capital

The default starting capital is **$50,000**. Here's the math:

- $50,000 × 5% per-asset cap = **~$2,500 per position** (across up to 10 assets)

Kraken enforces minimum order sizes for major pairs (typically $10–$25 per order, but the notional must comfortably clear fees and rounding). At $5,000 with a 10-asset portfolio and a 5% cap, each position would be ~$25 — right at or below the minimum, causing `NotSupported` minimum-order-size rejections. At $50,000 each position is ~$2,500, comfortably above Kraken's minimums for all major pairs.

**If you want to run the single-asset BTC variant** you can override with `set_cash(5000)` and `max_universe_size=1` — a single $5,000 position clears Kraken's minimum easily.

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
