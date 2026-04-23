#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import yaml
from datetime import datetime, timezone

from data.kraken import fetch_ohlcv, TokenBucket
from models.svm_signal import SvmSignalModel
from models.registry import ModelRegistry
from strategy.svm_wavelet import SvmWaveletStrategy

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


class PaperAlgo:
    def __init__(self):
        self.orders = []

    def MarketOrder(self, symbol, quantity, tag="Paper"):
        self.orders.append({"time": datetime.now(timezone.utc), "symbol": symbol, "qty": quantity, "tag": tag})
        print(f"ORDER {symbol}: qty={quantity:.6f} tag={tag}")


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    pairs = cfg.get("pairs", ["XBT/USD", "ETH/USD", "SOL/USD"])
    interval = int(cfg.get("interval", 60))

    fee_taker = float(cfg.get("fees", {}).get("taker", 0.0026))
    round_trip_cost = float(cfg.get("round_trip_cost", 2 * fee_taker))

    registry = ModelRegistry(root="models/artifacts")
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
            bars = fetch_ohlcv(pair=pair, interval=interval, session=None, rate_limiter=limiter)
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

        time.sleep(max(interval, 1) * 60)


if __name__ == "__main__":
    main()
