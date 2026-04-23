from unittest.mock import Mock, patch

import pandas as pd

from data.kraken import TokenBucket, fetch_ohlcv


class MockResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


def test_fetch_ohlcv_parses_response():
    session = Mock()
    session.get.return_value = MockResponse(
        {
            "error": [],
            "result": {
                "XXBTZUSD": [
                    [1700000000, "100", "110", "90", "105", "103", "12.3", "42"],
                    [1700003600, "105", "112", "101", "108", "107", "10.1", "35"],
                ],
                "last": "1700003600",
            },
        }
    )

    df = fetch_ohlcv("XBT/USD", interval=60, session=session)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "vwap", "volume", "count"]
    assert float(df.iloc[-1]["close"]) == 108.0


def test_token_bucket_backoff():
    bucket = TokenBucket(rate_per_sec=1.0, capacity=1.0)
    with patch("data.kraken.time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.2, 1.2, 1.2]), patch("data.kraken.time.sleep") as sleep:
        bucket.last_refill = 0.0
        bucket.tokens = 1.0
        bucket.acquire()
        bucket.acquire()
        assert sleep.called
