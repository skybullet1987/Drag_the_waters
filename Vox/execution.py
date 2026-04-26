# ── Vox Execution Helpers ─────────────────────────────────────────────────────
#
# Lot-size arithmetic and partial-fill tracking.
# ─────────────────────────────────────────────────────────────────────────────

import math


class OrderHelper:
    """
    Static helpers for exchange lot-size arithmetic and order validation.

    All methods are stateless and may be called without instantiation.
    """

    @staticmethod
    def get_lot_size(security):
        """
        Read the lot size from *security.symbol_properties*.

        Parameters
        ----------
        security : Security — QC Security object.

        Returns
        -------
        float
            Lot size (e.g. 0.001 for BTC on Kraken), or 1e-8 as a safe
            minimum if the property is missing.
        """
        try:
            return float(security.symbol_properties.lot_size)
        except Exception:
            return 1e-8

    @staticmethod
    def get_min_order_size(security):
        """
        Read the minimum order size from *security.symbol_properties*.

        Parameters
        ----------
        security : Security — QC Security object.

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

        Parameters
        ----------
        qty      : float — desired quantity in coin units.
        lot_size : float — exchange lot size.

        Returns
        -------
        float
        """
        if lot_size <= 0:
            return qty
        return math.floor(qty / lot_size) * lot_size

    @staticmethod
    def validate_qty(qty, min_order_size):
        """
        Return False if *qty* is below the exchange minimum order size.

        Parameters
        ----------
        qty            : float
        min_order_size : float

        Returns
        -------
        bool
        """
        return qty >= min_order_size


# ── Partial Fill Tracker ──────────────────────────────────────────────────────

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
        """
        Begin tracking a new order.

        Parameters
        ----------
        order_id   : int   — QC order ID.
        target_qty : float — expected total fill quantity.
        """
        self._orders[order_id] = {"target": target_qty, "filled": 0.0}

    def on_fill(self, order_id, filled_qty):
        """
        Accumulate a (partial) fill.

        Parameters
        ----------
        order_id   : int   — QC order ID.
        filled_qty : float — quantity filled in this event (positive).
        """
        if order_id in self._orders:
            self._orders[order_id]["filled"] += abs(filled_qty)

    def is_complete(self, order_id):
        """
        Return True if the cumulative fill meets or exceeds the target.

        Parameters
        ----------
        order_id : int

        Returns
        -------
        bool
        """
        if order_id not in self._orders:
            return True   # unknown → assume closed
        rec = self._orders[order_id]
        return rec["filled"] >= rec["target"]

    def get_filled(self, order_id):
        """
        Return the cumulative filled quantity for *order_id*.

        Parameters
        ----------
        order_id : int

        Returns
        -------
        float  — 0.0 if not tracked.
        """
        if order_id not in self._orders:
            return 0.0
        return self._orders[order_id]["filled"]

    def clear(self, order_id):
        """
        Remove *order_id* from tracking.

        Parameters
        ----------
        order_id : int
        """
        self._orders.pop(order_id, None)
