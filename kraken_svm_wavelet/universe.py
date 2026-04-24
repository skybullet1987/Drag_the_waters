# region imports
from AlgorithmImports import *
# endregion


# Symbols permanently excluded — meme-coin / illiquid / news-driven
# (carried over from the old algorithm's MEME_BLOCKLIST + SYMBOL_BLACKLIST)
SYMBOL_BLOCKLIST = frozenset({
    "TRUMPUSD", "MELANIAUSD", "FWOGUSD", "XCNUSD", "GIGAUSD",
    "TURBOUSD", "FTMUSD", "JUPUSD", "ONDOUSD",
    # Structural exclusions from old SYMBOL_BLACKLIST
    "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
    "DAIUSD", "TUSDUSD", "WETHUSD", "WBTCUSD", "WAXLUSD",
    "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
    "XNYUSD",
    "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
    "BONDUSD", "KEEPUSD", "ORNUSD",
    "MUSD", "ICNTUSD",
    "EPTUSD", "LMWRUSD",
    "CPOOLUSD",
    "ARCUSD", "PAXGUSD",
    "PARTIUSD", "RAREUSD", "BANANAS31USD",
    # Backtest-derived exclusions
    "KTAUSD", "NANOUSD", "STRKUSD",
})

# Stablecoins / fiat — never trade these "as crypto"
FIAT_AND_STABLES = frozenset({
    "USDT", "USDC", "DAI", "EUR", "GBP", "CAD", "JPY", "AUD",
    "CHF", "USDP", "TUSD", "BUSD", "PYUSD", "EURT",
    # From old KNOWN_FIAT_CURRENCIES
    "NZD", "CNY", "HKD", "SGD", "SEK", "NOK", "DKK", "KRW",
    "TRY", "ZAR", "MXN", "INR", "BRL", "PLN", "THB",
})


def select_top_kraken_usd(coarse, max_count=50):
    """
    Universe filter for QC's CryptoUniverse.Kraken().
    Picks the top `max_count` Kraken USD pairs by 24h dollar volume,
    excluding the blocklist and any quote-fiat pairs.
    """
    candidates = []
    for crypto in coarse:
        ticker = crypto.Symbol.Value
        if not ticker.endswith("USD"):
            continue
        base = ticker[:-3]
        if base in FIAT_AND_STABLES:
            continue
        if ticker in SYMBOL_BLOCKLIST:
            continue
        if crypto.VolumeInUsd is None or crypto.VolumeInUsd <= 0:
            continue
        candidates.append(crypto)
    candidates.sort(key=lambda c: c.VolumeInUsd, reverse=True)
    return [c.Symbol for c in candidates[:max_count]]
