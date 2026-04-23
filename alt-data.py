import argparse
from datetime import datetime, timedelta

try:
    from AlgorithmImports import *  # type: ignore
except Exception:  # pragma: no cover - local CLI/testing mode
    class PythonData:  # minimal stub for CLI/test environments
        pass

from kraken_client import fetch_24h_volume

class FearGreedData(PythonData):
    """
    Custom data feed for the Alternative.me Fear & Greed Index.
    Daily updates, free API, no key required.
    Value: 0-100 integer (0=Extreme Fear, 100=Extreme Greed)
    """

    def GetSource(self, config, date, isLiveMode):
        # limit=0 gets all history (necessary for historical backtesting).
        # format=csv forces line-by-line data instead of multiline JSON.
        limit = "2" if isLiveMode else "0"
        url = f"https://api.alternative.me/fng/?limit={limit}&format=csv"
        
        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        # Skip empty lines and the CSV header
        if not line or line.strip() == "" or "value" in line.lower() or "timestamp" in line.lower():
            return None
            
        try:
            parts = line.split(',')
            
            timestamp = None
            value = None
            
            # Simple heuristic: The FNG value is 0-100. The Unix Timestamp is > 1 billion.
            for p in parts:
                p = p.strip()
                if not p: continue
                try:
                    num = float(p)
                    if num > 1000000000:
                        timestamp = int(num)
                    elif 0 <= num <= 100 and value is None:
                        value = num
                except ValueError:
                    pass
                    
            # If we couldn't parse the row successfully, skip it
            if timestamp is None or value is None:
                return None

            result = FearGreedData()
            result.Symbol = config.Symbol
            result.Time = datetime.utcfromtimestamp(timestamp)
            result.Value = value
            result.EndTime = result.Time + timedelta(days=1)
            
            return result
            
        except Exception:
            return None


MEME_BLOCKLIST = {"TRUMPUSD", "MELANIAUSD", "FWOGUSD", "XCNUSD", "GIGAUSD", "TURBOUSD", "FTMUSD"}


def select_top_universe(client, *, top_n=10, min_volume_usd=50_000_000, quote="USD", exclude_meme=True):
    quote = str(quote).upper()
    volume_map = client.fetch_24h_volume()
    pairs = []
    for pair, volume_usd in volume_map.items():
        pair_u = str(pair).replace("/", "").upper()
        if not pair_u.endswith(quote):
            continue
        if float(volume_usd) < float(min_volume_usd):
            continue
        if exclude_meme and pair_u in MEME_BLOCKLIST:
            continue
        pairs.append((pair_u, float(volume_usd)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in pairs[: int(top_n)]]


class _KrakenClientAdapter:
    @staticmethod
    def fetch_24h_volume():
        return fetch_24h_volume()


def main():
    parser = argparse.ArgumentParser(description="Select top Kraken USD universe by 24h volume.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=50_000_000)
    parser.add_argument("--quote", type=str, default="USD")
    parser.add_argument("--include-meme", action="store_true")
    args = parser.parse_args()
    symbols = select_top_universe(
        _KrakenClientAdapter(),
        top_n=args.top,
        min_volume_usd=args.min_volume,
        quote=args.quote,
        exclude_meme=not args.include_meme,
    )
    for symbol in symbols:
        print(symbol)


if __name__ == "__main__":
    main()
