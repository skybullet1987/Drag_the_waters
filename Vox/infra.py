# ── Vox Infrastructure ────────────────────────────────────────────────────────
#
# Consolidated module for universe management, order-execution helpers, and
# ObjectStore persistence.  Previously split across universe.py, execution.py,
# and persistence.py.
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import pickle
from AlgorithmImports import *

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

KRAKEN_PAIRS = [
    "BTCUSD",  "ETHUSD",   "SOLUSD",   "XRPUSD",  "XDGUSD",
    "ADAUSD",  "AVAXUSD",  "LINKUSD",  "DOTUSD",  "LTCUSD",
    "TRXUSD",  "BCHUSD",   "MATICUSD", "ATOMUSD", "UNIUSD",
    "AAVEUSD", "ARBUSD",   "OPUSD",    "INJUSD",  "NEARUSD",
]


def add_universe(algorithm):
    """
    Register every ticker in KRAKEN_PAIRS with *algorithm* at minute resolution
    on the Kraken market.

    Any pair that Kraken / QuantConnect does not support is logged and silently
    skipped so the backtest can still run with the remaining symbols.

    Parameters
    ----------
    algorithm : QCAlgorithm

    Returns
    -------
    list[Symbol]
        Successfully added Symbol objects.
    """
    symbols = []
    for ticker in KRAKEN_PAIRS:
        try:
            sym = algorithm.add_crypto(
                ticker, Resolution.MINUTE, Market.KRAKEN
            ).symbol
            symbols.append(sym)
        except Exception as exc:
            algorithm.log(f"[universe] Skipping {ticker} — not supported: {exc}")
    return symbols


def fetch_kraken_top20_usd(algorithm):
    """
    Query the Kraken public Ticker endpoint and return the top-20 USD pairs
    ranked by 24-hour quote volume.

    .. warning::
        **LIVE USE ONLY.**  Calling this during a backtest introduces
        look-ahead bias because the live Kraken snapshot reflects the current
        universe composition, not the composition at the backtest date.

    Parameters
    ----------
    algorithm : QCAlgorithm

    Returns
    -------
    list[str]
        Up to 20 ticker strings.  Falls back to the static KRAKEN_PAIRS list
        on any error.
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
            if not pair.endswith("USD"):
                continue
            if any(ch in pair for ch in (".", "_")):
                continue
            try:
                volume_24h = float(info["v"][1])
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


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class OrderHelper:
    """
    Static helpers for exchange lot-size arithmetic and order validation.

    All methods are stateless and may be called without instantiation.
    """

    @staticmethod
    def get_lot_size(security):
        """
        Read the lot size from *security.symbol_properties*.

        Returns
        -------
        float
            Lot size (e.g. 0.001 for BTC on Kraken), or 1e-8 as a safe minimum.
        """
        try:
            return float(security.symbol_properties.lot_size)
        except Exception:
            return 1e-8

    @staticmethod
    def get_min_order_size(security):
        """
        Read the minimum order size from *security.symbol_properties*.

        Returns
        -------
        float
            Minimum order size, or 0.0 if not available.
        """
        try:
            return float(security.symbol_properties.minimum_order_size)
        except Exception:
            return 0.0

    @staticmethod
    def round_qty(qty, lot_size):
        """
        Floor *qty* to the nearest multiple of *lot_size*.

        Uses integer arithmetic to avoid floating-point rounding surprises.
        """
        if lot_size <= 0:
            return qty
        return math.floor(qty / lot_size) * lot_size

    @staticmethod
    def validate_qty(qty, min_order_size):
        """Return False if *qty* is below the exchange minimum order size."""
        return qty >= min_order_size


class PartialFillTracker:
    """
    Accumulate partial fills for a single open order.

    The Vox strategy uses market orders which should fill immediately on
    Kraken, but partial fills can occur during low-liquidity windows.
    This tracker accumulates filled quantities and marks the order complete
    only when the cumulative fill meets or exceeds the target.

    Usage
    -----
    >>> tracker = PartialFillTracker()
    >>> tracker.start_order(order_id=101, target_qty=0.5)
    >>> tracker.on_fill(order_id=101, filled_qty=0.3)
    >>> tracker.on_fill(order_id=101, filled_qty=0.2)
    >>> tracker.is_complete(101)
    True
    """

    def __init__(self):
        self._orders = {}   # order_id -> {"target": float, "filled": float}

    def start_order(self, order_id, target_qty):
        """Begin tracking a new order."""
        self._orders[order_id] = {"target": target_qty, "filled": 0.0}

    def on_fill(self, order_id, filled_qty):
        """Accumulate a (partial) fill."""
        if order_id in self._orders:
            self._orders[order_id]["filled"] += abs(filled_qty)

    def is_complete(self, order_id):
        """Return True if the cumulative fill meets or exceeds the target."""
        if order_id not in self._orders:
            return True   # unknown → assume closed
        rec = self._orders[order_id]
        return rec["filled"] >= rec["target"]

    def get_filled(self, order_id):
        """Return the cumulative filled quantity for *order_id* (0.0 if not tracked)."""
        if order_id not in self._orders:
            return 0.0
        return self._orders[order_id]["filled"]

    def clear(self, order_id):
        """Remove *order_id* from tracking."""
        self._orders.pop(order_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE  (ObjectStore model, trade log, kill switch)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """
    Persist ML models and trade logs to the QuantConnect ObjectStore.

    All write/read operations are wrapped in try/except so that any storage
    failure degrades gracefully (logs a warning) rather than crashing the algo.

    Kill switch
    -----------
    If the key ``kill_key`` exists in the ObjectStore, the strategy will refuse
    to open new positions.  To activate from the QC Research environment::

        qb.object_store.save("vox/kill_switch", "1")

    To deactivate::

        qb.object_store.delete("vox/kill_switch")

    Parameters
    ----------
    algorithm  : QCAlgorithm
    model_key  : str — ObjectStore key for the pickled model.
    log_key    : str — ObjectStore key for the JSONL trade log.
    kill_key   : str — ObjectStore key for the kill switch flag.
    """

    def __init__(
        self,
        algorithm,
        model_key  = "vox/model.pkl",
        log_key    = "vox/trade_log.jsonl",
        kill_key   = "vox/kill_switch",
        flush_every = 50,
    ):
        self._algo        = algorithm
        self._model_key   = model_key
        self._log_key     = log_key
        self._kill_key    = kill_key
        self._flush_every = flush_every
        self._buffer      = []   # list[str] of pre-serialised JSON lines

    # ── Model serialisation ───────────────────────────────────────────────────

    def save_model(self, model):
        """Pickle *model* and write the bytes to the ObjectStore."""
        try:
            # Defensive: strip any non-picklable logger before serialisation.
            if hasattr(model, "set_logger"):
                try:
                    model.set_logger(None)
                except Exception as set_exc:
                    self._algo.log(f"[persistence] set_logger(None) failed: {set_exc}")
            data = pickle.dumps(model)
            self._algo.object_store.save_bytes(self._model_key, data)
            self._algo.log(
                f"[persistence] Model saved to ObjectStore key={self._model_key}"
            )
            # Re-attach logger so subsequent in-memory use still logs.
            if hasattr(model, "set_logger"):
                try:
                    model.set_logger(self._algo.log)
                except Exception as set_exc:
                    self._algo.log(f"[persistence] set_logger(log) failed: {set_exc}")
        except Exception as exc:
            self._algo.log(f"[persistence] save_model failed: {exc}")

    def load_model(self):
        """
        Load and unpickle the model from the ObjectStore.

        Returns
        -------
        object or None
        """
        try:
            if not self._algo.object_store.contains_key(self._model_key):
                return None
            data = self._algo.object_store.read_bytes(self._model_key)
            model = pickle.loads(data)
            self._algo.log(
                f"[persistence] Model loaded from ObjectStore key={self._model_key}"
            )
            return model
        except Exception as exc:
            self._algo.log(f"[persistence] load_model failed: {exc}")
            return None

    # ── Trade logging ─────────────────────────────────────────────────────────

    def log_trade(self, entry_dict):
        """
        Buffer *entry_dict* as a JSON line and flush to ObjectStore when the
        buffer reaches *flush_every* entries.
        """
        try:
            self._buffer.append(json.dumps(entry_dict, default=str))
            if len(self._buffer) >= self._flush_every:
                self.flush_trade_log()
        except Exception as exc:
            self._algo.log(f"[persistence] log_trade buffer failed: {exc}")

    def flush_trade_log(self):
        """Write all buffered trade lines to the ObjectStore and clear the buffer."""
        if not self._buffer:
            return
        try:
            existing = ""
            if self._algo.object_store.contains_key(self._log_key):
                existing = self._algo.object_store.read(self._log_key)
            payload = existing + "\n".join(self._buffer) + "\n"
            self._algo.object_store.save(self._log_key, payload)
            self._buffer.clear()
        except Exception as exc:
            self._algo.log(f"[persistence] flush_trade_log failed: {exc}")

    # ── Kill switch ───────────────────────────────────────────────────────────

    def is_kill_switch_active(self):
        """Return True if the kill switch key exists in the ObjectStore."""
        try:
            return self._algo.object_store.contains_key(self._kill_key)
        except Exception as exc:
            self._algo.log(
                f"[persistence] is_kill_switch_active check failed: {exc}"
            )
            return False
