"""Pulse.apex.ml — ML decision layer (XGBoost + LightGBM + LogReg ensemble).

Three layers:
  - dataset.py    Build (X, y) from historical signals + forward returns
  - train.py      Walk-forward train + soft-vote ensemble + joblib dump
  - predict.py    Load joblib + score a feature vector at inference

Optional XGBoost / LightGBM are guarded by try/except — sklearn LogReg
is the always-available fallback.
"""
