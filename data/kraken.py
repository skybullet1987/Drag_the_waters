import time
from dataclasses import dataclass
import requests
import pandas as pd

KRAKEN_BASE_URL = "https://api.kraken.com"


@dataclass
class TokenBucket:
    rate_per_sec: float = 1.0
    capacity: float = 1.0

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def acquire(self, tokens=1.0):
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - self.last_refill)
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            wait_s = (tokens - self.tokens) / max(self.rate_per_sec, 1e-9)
            time.sleep(wait_s)


def _public_get(endpoint, params=None, session=None, rate_limiter=None, timeout=20):
    session = session or requests.Session()
    if rate_limiter is not None:
        rate_limiter.acquire()
    url = f"{KRAKEN_BASE_URL}{endpoint}"
    response = session.get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken API error: {payload['error']}")
    return payload["result"]


def fetch_ohlcv(pair, interval=60, since=None, session=None, rate_limiter=None):
    params = {"pair": pair, "interval": int(interval)}
    if since is not None:
        params["since"] = int(since)

    result = _public_get("/0/public/OHLC", params=params, session=session, rate_limiter=rate_limiter)
    keys = [k for k in result.keys() if k != "last"]
    if not keys:
        return pd.DataFrame(columns=["open", "high", "low", "close", "vwap", "volume", "count"])

    rows = result[keys[0]]
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0).astype(int)
    return frame.set_index("timestamp").sort_index()


def get_ticker(pair, session=None, rate_limiter=None):
    result = _public_get("/0/public/Ticker", params={"pair": pair}, session=session, rate_limiter=rate_limiter)
    key = next(iter(result.keys()))
    return result[key]


def get_usd_pairs(min_volume_usd=1_000_000, session=None, rate_limiter=None):
    session = session or requests.Session()
    rate_limiter = rate_limiter or TokenBucket(rate_per_sec=1.0, capacity=1.0)
    result = _public_get("/0/public/AssetPairs", session=session, rate_limiter=rate_limiter)

    usd_pairs = []
    for pair_data in result.values():
        wsname = pair_data.get("wsname")
        if not wsname or not wsname.endswith("/USD"):
            continue
        if ".d" in wsname:
            continue

        ticker = get_ticker(wsname, session=session, rate_limiter=rate_limiter)
        try:
            last = float(ticker["c"][0])
            vol_24h = float(ticker["v"][1])
            volume_usd = last * vol_24h
        except Exception:
            continue

        if volume_usd >= float(min_volume_usd):
            usd_pairs.append((wsname, volume_usd))

    usd_pairs.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in usd_pairs]
