#!/usr/bin/env python
import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kraken_client import TokenBucket, fetch_ohlcv
from svm_wavelet import (
    ModelRegistry,
    SvmSignalModel,
    SvmWaveletStrategy,
    WalkForwardConfig,
    build_ta_features,
    causal_wavelet_features,
    make_cost_aware_labels,
)

try:
    from realistic_slippage import RealisticCryptoSlippage
except Exception:
    RealisticCryptoSlippage = None

try:
    from circuit_breaker import DrawdownCircuitBreaker
except Exception:
    DrawdownCircuitBreaker = None

try:
    import execution
except Exception:
    execution = None


EPSILON = 1e-12


class PaperAlgo:
    def __init__(self):
        self.orders = []

    def MarketOrder(self, symbol, quantity, tag="Paper"):
        self.orders.append({"time": datetime.now(timezone.utc), "symbol": symbol, "quantity": quantity, "tag": tag})
        print(f"ORDER {symbol}: qty={quantity:.6f} tag={tag}")


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _max_drawdown(returns):
    eq = (1 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def train_command(args):
    cfg = _load_config(args.config)
    interval = int(cfg.get("interval", 60))
    wave_cfg = cfg.get("wavelet", {})
    svm_cfg = cfg.get("svm", {})
    wf_cfg = cfg.get("walk_forward", {})
    k = float(cfg.get("k_threshold", 2.0))
    horizon = int(cfg.get("label_horizon", 1))
    fee_taker = float(cfg.get("fees", {}).get("taker", 0.0026))
    round_trip_cost = 2 * fee_taker
    assumed_slippage = float(cfg.get("assumed_slippage", 0.0015))

    limiter = TokenBucket(rate_per_sec=1.0, capacity=1.0)
    try:
        ohlcv = fetch_ohlcv(args.pair, interval=interval, session=None, rate_limiter=limiter)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Kraken OHLCV for {args.pair}: {exc}") from exc
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV data returned for {args.pair}")

    wave = causal_wavelet_features(
        ohlcv["close"],
        wavelet=wave_cfg.get("wavelet", "db4"),
        level=int(wave_cfg.get("level", 3)),
        window=int(wave_cfg.get("window", 256)),
        verify_causality=True,
    )
    ta = build_ta_features(ohlcv)
    X = pd.concat([wave, ta], axis=1)

    y = make_cost_aware_labels(
        ohlcv["close"],
        horizon=horizon,
        k=k,
        round_trip_cost=round_trip_cost,
    )

    valid = (~X.isna().any(axis=1)) & (~y.isna())
    Xv = X.loc[valid]
    yv = y.loc[valid]

    model = SvmSignalModel(
        C=float(svm_cfg.get("C", 1.0)),
        gamma=svm_cfg.get("gamma", "scale"),
        neutral_threshold=float(svm_cfg.get("neutral_threshold", 0.55)),
    )
    wf = WalkForwardConfig(
        mode=wf_cfg.get("mode", "expanding"),
        min_train_size=int(wf_cfg.get("min_train_size", 500)),
        train_window=int(wf_cfg.get("train_window", 2000)),
        retrain_every=int(wf_cfg.get("retrain_every", 24)),
        drop_flat=bool(wf_cfg.get("drop_flat", True)),
    )

    oos = model.walk_forward_predict(Xv, yv, config=wf)
    mask = oos["signal"].notna()
    y_true = yv.loc[mask].astype(int)
    y_pred = oos.loc[mask, "signal"].fillna(0).astype(int)

    print("=== OOS Classification ===")
    print(f"accuracy: {accuracy_score(y_true, y_pred):.4f}")
    non_zero = y_pred != 0
    if non_zero.any():
        hit_rate = (y_true[non_zero] == y_pred[non_zero]).mean()
    else:
        hit_rate = 0.0
    print(f"hit_rate_non_zero: {hit_rate:.4f}")
    print(classification_report(y_true, y_pred, labels=[-1, 0, 1], zero_division=0))

    fwd_ret = ohlcv["close"].shift(-horizon) / ohlcv["close"] - 1.0
    strat_ret = y_pred * fwd_ret.loc[y_pred.index]
    gross = strat_ret.fillna(0)
    cost_per_trade = round_trip_cost + assumed_slippage
    net = gross - (y_pred.abs() * cost_per_trade)

    bars_per_year = int((24 * 365 * 60) / interval)
    sharpe = float(np.sqrt(bars_per_year) * net.mean() / (net.std() + EPSILON))
    turnover = float(y_pred.diff().abs().fillna(0).sum())

    print("=== Cost-Adjusted PnL Summary ===")
    print(f"gross_pnl: {gross.sum():.6f}")
    print(f"net_pnl: {net.sum():.6f}")
    print(f"sharpe: {sharpe:.4f}")
    print(f"max_drawdown: {_max_drawdown(net):.4f}")
    print(f"turnover: {turnover:.4f}")

    model.fit(Xv, yv, drop_flat=wf.drop_flat)
    registry = ModelRegistry(root="artifacts")
    artifact = registry.save(model, args.pair, config=cfg)
    print(f"Saved model artifact: {artifact}")


def run_command(args):
    cfg = _load_config(args.config)
    pairs = cfg.get("pairs", ["XBT/USD", "ETH/USD", "SOL/USD"])
    interval_minutes = int(cfg.get("interval", 60))

    fee_taker = float(cfg.get("fees", {}).get("taker", 0.0026))
    round_trip_cost = float(cfg.get("round_trip_cost", 2 * fee_taker))

    registry = ModelRegistry(root="artifacts")
    limiter = TokenBucket(rate_per_sec=1.0, capacity=1.0)

    slippage_model = RealisticCryptoSlippage() if RealisticCryptoSlippage else None
    breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.15) if DrawdownCircuitBreaker else None

    strategies = {}
    for pair in pairs:
        path = registry.load_latest(pair)
        model = SvmSignalModel.load(path)
        strat_cfg = {
            "wavelet": cfg.get("wavelet", {}).get("wavelet", "db4"),
            "level": int(cfg.get("wavelet", {}).get("level", 3)),
            "window": int(cfg.get("wavelet", {}).get("window", 256)),
            "round_trip_cost": round_trip_cost,
            "risk_per_trade": float(cfg.get("risk_per_trade", 0.02)),
            "max_daily_turnover": float(cfg.get("max_daily_turnover", 3.0)),
            "min_holding_period_bars": int(cfg.get("min_holding_period_bars", 3)),
        }
        strategies[pair] = SvmWaveletStrategy(model=model, config=strat_cfg, slippage_model=slippage_model, circuit_breaker=breaker)

    algo = PaperAlgo()
    positions = {pair: 0 for pair in pairs}

    while True:
        for pair in pairs:
            bars = fetch_ohlcv(pair=pair, interval=interval_minutes, session=None, rate_limiter=limiter)
            if bars.empty:
                continue
            last = bars.iloc[-1]
            bar = {
                "timestamp": bars.index[-1],
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(last["close"]),
                "volume": float(last["volume"]),
            }
            decision = strategies[pair].on_bar(bar, context={"timestamp": bars.index[-1], "current_position": positions[pair]})
            target = int(decision["target_position"])
            if target == positions[pair]:
                continue

            qty = float(target - positions[pair])
            if execution is not None:
                if qty > 0:
                    execution.execute_buy(algo, pair, qty)
                else:
                    execution.execute_sell(algo, pair, abs(qty))
            else:
                algo.MarketOrder(pair, qty, tag="Paper fallback")
            positions[pair] = target

        time.sleep(max(interval_minutes, 1) * 60)


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--pair", required=True)
    train.set_defaults(func=train_command)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--paper", action="store_true")
    run.set_defaults(func=run_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
