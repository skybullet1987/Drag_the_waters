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
