# Drag the Waters — SVM + Wavelet Crypto Trading on Kraken

This repository contains a QuantConnect algorithm that trades the top Kraken USD cryptocurrency pairs using a Support Vector Machine + Discrete Wavelet Transform forecasting model.

## Strategy

The algorithm is located in [`kraken_svm_wavelet/`](./kraken_svm_wavelet/). It:

- Selects the top 50 Kraken USD pairs by 24-hour dollar volume.
- Forecasts next-day price using SVM regression on wavelet-decomposed price series.
- Executes trades using limit orders (maker-fee preference) on a daily schedule.

See [`kraken_svm_wavelet/README.md`](./kraken_svm_wavelet/README.md) for full documentation, parameters, and deployment instructions.

## Quick Start

1. Create a new [QuantConnect](https://www.quantconnect.com/) project.
2. Copy all files from `kraken_svm_wavelet/` into the project root.
3. Set the algorithm class to `KrakenSvmWaveletAlgorithm`.
4. Run a backtest. Use `max_universe_size=10` for a fast first run.

## History

This repository previously contained a sophisticated rule-based algorithm with partial take-profits, ATR stops, Fear & Greed filtering, and SVM entry gates. That algorithm was retired after underperforming the simpler ML-only approach in backtests (−21% vs. +6.6% over the same period). The SVM+Wavelet barebones strategy now serves as the primary approach.
