# BTC SVM Wavelet Barebone

Barebones port of QuantConnect's HandsOnAITradingBook "FX SVM Wavelet Forecasting" sample to BTCUSD on Kraken. Code is intentionally identical to the book — only the universe, brokerage, and leverage default were swapped.

Original book example:
https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/05%20FX%20SVM%20Wavelet%20Forecasting

## How to run

1. Open QuantConnect Cloud (or LEAN CLI).
2. Create a new Python algorithm project.
3. Replace the project's `main.py` with the contents of `btc_svm_wavelet_barebone/main.py`.
4. Add `btc_svm_wavelet_barebone/svmwavelet.py` to the project as a sibling file.
5. Backtest. Default period: 2019-01-01 → 2024-04-01.

## Parameters

- `period` (default 152) — rolling window length (minimum for sym10 with 3 decomposition levels)
- `leverage` (default 1) — leave at 1 for Kraken cash account
- `weight_threshold` (default 0.005) — minimum |forecast/price - 1| to trigger a rebalance

## What's NOT in here

- No fee model overrides
- No slippage model
- No risk management
- No partial TPs

By design, this is a clean baseline to compare against the main `Drag_the_waters` algorithm.

## Relationship to the rest of this repo

This folder is **independent** of the root `main.py`, `events.py`, `execution.py`, etc. Editing files here does NOT affect the main algorithm.
