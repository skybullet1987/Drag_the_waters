# Drag the Waters

## SVM + Wavelet Kraken strategy

This repository now includes an SVM + wavelet crypto strategy module inspired by QuantConnect's
"FX SVM Wavelet Forecasting" example:
https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/05%20FX%20SVM%20Wavelet%20Forecasting

Differences vs the QC notebook:
- Causal wavelet features only (strict trailing window, no future leakage)
- Cost-aware labels (thresholded by round-trip trading cost)
- Kraken spot data + realistic fee/slippage-aware gating

### Train

```bash
python svm_wavelet_cli.py train --config configs/svm_wavelet_kraken.yaml --pair XBT/USD
```

### Run (paper/live loop scaffold)

```bash
python svm_wavelet_cli.py run --config configs/svm_wavelet_kraken.yaml --paper
```

> Disclaimer: research code only, not financial advice.

## Improving an existing strategy

### 1) Diagnose an orders CSV quickly

```bash
python backtest_report.py "Alert Blue Seahorse_orders.csv" --starting-equity 5000
```

Key diagnostics include required break-even win rate:
`required_wr = avg_loss / (avg_win + avg_loss)`, edge gap, fee load as % of equity, and exit-tag PnL breakdown.

### 2) Enable SVM+Wavelet as an entry filter

Use algorithm parameters:
- `entry_filter=svm_wavelet`
- `svm_confidence=0.55`

When enabled, entries are only allowed when model direction/confidence/edge pass cost gating (`2 × (fees + slippage)`).

### 3) Refresh the Kraken universe by volume

```bash
python alt-data.py --top 10 --min-volume 50000000
```

This keeps only high-volume USD pairs, applies a minimum-volume floor, and excludes known high-fee meme symbols by default.
If fee-as-%-of-equity is elevated, reduce turnover and prefer post-only/maker flow where possible.
