# ── Vox Universe ──────────────────────────────────────────────────────────────
#
# Static curated list of 20 Kraken USD pairs and helpers to register them with
# a QCAlgorithm instance.
# ─────────────────────────────────────────────────────────────────────────────

import json
from AlgorithmImports import *

# ── Static universe ───────────────────────────────────────────────────────────

KRAKEN_PAIRS = [
    "BTCUSD", "ETHUSD", "SOLUSD",  "XRPUSD",  "DOGEUSD",
    "ADAUSD", "AVAXUSD","LINKUSD", "DOTUSD",   "LTCUSD",
    "TRXUSD", "BCHUSD", "MATICUSD","ATOMUSD",  "UNIUSD",
    "AAVEUSD","ARBUSD", "OPUSD",   "INJUSD",   "NEARUSD",
]


def add_universe(algorithm):
    """
    Register every ticker in KRAKEN_PAIRS with *algorithm* at minute resolution
    on the Kraken market.

    Any pair that Kraken / QuantConnect does not support (raises on add_crypto)
    is logged and silently skipped so the backtest can still run with the
    remaining symbols.

    Parameters
    ----------
    algorithm : QCAlgorithm
        The running algorithm instance.

    Returns
    -------
    list[Symbol]
        The list of Symbol objects that were successfully added.
    """
    symbols = []
    for ticker in KRAKEN_PAIRS:
        try:
            sym = algorithm.add_crypto(
                ticker, Resolution.MINUTE, Market.KRAKEN
            ).symbol
            symbols.append(sym)
        except Exception as exc:
            algorithm.log(
                f"[universe] Skipping {ticker} — not supported: {exc}"
            )
    return symbols


def fetch_kraken_top20_usd(algorithm):
    """
    Query the Kraken public Ticker endpoint and return the top-20 USD trading
    pairs ranked by 24-hour quote volume.

    .. warning::
        **LIVE USE ONLY.**  Calling this during a backtest introduces
        look-ahead bias because the live Kraken snapshot reflects the current
        universe composition, not the composition at the backtest date.

    Parameters
    ----------
    algorithm : QCAlgorithm
        The running algorithm instance (used for ``algorithm.download``).

    Returns
    -------
    list[str]
        Up to 20 ticker strings (e.g. ``["XBTUSD", "ETHUSD", ...]``).
        Falls back to the static ``KRAKEN_PAIRS`` list on any error.
    """
    url = "https://api.kraken.com/0/public/Ticker"
    try:
        raw = algorithm.download(url)
        data = json.loads(raw)
        if data.get("error"):
            algorithm.log(f"[universe] Kraken Ticker error: {data['error']}")
            return list(KRAKEN_PAIRS)

        pairs = []
        for pair, info in data.get("result", {}).items():
            # Keep only USD-quoted pairs; skip leveraged/dark markers
            if not pair.endswith("USD"):
                continue
            if any(ch in pair for ch in (".", "_")):
                continue
            try:
                volume_24h = float(info["v"][1])   # 24-h rolling volume
                pairs.append((pair, volume_24h))
            except (KeyError, ValueError, IndexError):
                continue

        pairs.sort(key=lambda x: x[1], reverse=True)
        top20 = [p[0] for p in pairs[:20]]
        algorithm.log(f"[universe] Kraken live top-20: {top20}")
        return top20 if top20 else list(KRAKEN_PAIRS)

    except Exception as exc:
        algorithm.log(
            f"[universe] fetch_kraken_top20_usd failed ({exc}); "
            "falling back to static list"
        )
        return list(KRAKEN_PAIRS)
