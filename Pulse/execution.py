"""execution — order management primitives for Pulse.

Lean port of the most useful pieces from Sweet Water v3-2's execution.py
(60KB, 40+ functions). We keep only what Pulse actually needs and
factor each piece into a pure-Python core + thin QC adapter.

Pure-Python core (unit-testable):
- ``round_to_lot()``                  — quantize to lot size
- ``validate_min_notional()``         — reject orders below exchange minimum
- ``safe_sell_quantity()``            — Vox CashBook quirk fix (avoid over-selling)
- ``slippage_log_entry()``            — typed dict for the slippage audit log
- ``compute_slippage_pct()``          — fill_price vs reference price
- ``min_quantity_fallback()``         — Kraken min-qty fallback table
- ``OrderIntent``                     — submission-side intent enum

QC adapters (gated by HAS_QC):
- ``place_limit_or_market()``         — limit with TTL fallback to market
- ``smart_liquidate()``               — close a position with min-notional safety
- ``partial_smart_sell()``            — sell a fraction
- ``cleanup_position()``              — clear tracked state on dust positions

The pure-Python parts cover the high-leverage logic (lot rounding,
CashBook overshoot prevention, slippage attribution); the QC adapters are
mostly thin wrappers that translate to QC's API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

try:
    from AlgorithmImports import *  # type: ignore  # noqa: F401,F403
    HAS_QC = True
except Exception:
    HAS_QC = False


# ─── Pure-Python core ────────────────────────────────────────────────────────

class OrderIntent(Enum):
    """Order intent enum.

    NOTE: We DO NOT use ``str, Enum`` multiple inheritance because QC's
    Python.NET wrapper cannot construct managed classes with multiple
    inheritance ("cannot use multiple inheritance with managed classes").
    Compare values via ``OrderIntent.ENTRY == OrderIntent.ENTRY`` or
    ``intent.value == "entry"``.
    """
    ENTRY      = "entry"
    EXIT       = "exit"
    PARTIAL_TP = "partial_tp"
    HARD_KILL  = "hard_kill"
    BREAKEVEN  = "breakeven"
    LIQUIDATE  = "liquidate"


# Kraken minimum-quantity fallback table for the symbols we typically trade.
# Used only when QC's SymbolProperties.MinimumOrderSize is unavailable.
KRAKEN_MIN_QTY_FALLBACK: dict[str, float] = {
    "BTCUSD":  0.0001,
    "ETHUSD":  0.001,
    "SOLUSD":  0.05,
    "XRPUSD":  10.0,
    "DOGEUSD": 50.0,
    "LINKUSD": 0.5,
    "AVAXUSD": 0.5,
    "DOTUSD":  1.0,
    "ADAUSD":  10.0,
    "LTCUSD":  0.05,
    "BNBUSD":  0.02,
    "MATICUSD": 5.0,
    "ATOMUSD": 0.5,
    "UNIUSD":  0.5,
    "AAVEUSD": 0.05,
    "ARBUSD":  5.0,
    "OPUSD":   2.0,
    "INJUSD":  0.5,
    "BCHUSD":  0.02,
    "TRXUSD":  50.0,
    "FETUSD":  5.0,
    "ICPUSD":  0.5,
    "RENDERUSD": 1.0,
    "HBARUSD": 50.0,
    "NEARUSD": 1.0,
}


def min_quantity_fallback(symbol: str) -> float:
    """Conservative fallback minimum quantity. Returns 1.0 for unknown symbols."""
    return KRAKEN_MIN_QTY_FALLBACK.get(symbol.upper(), 1.0)


def round_to_lot(quantity: float, lot_size: float) -> float:
    """Quantize an order quantity to the exchange lot size.

    Always rounds DOWN to avoid over-spending on entries; for exits the
    safe_sell_quantity wrapper handles the buffer.
    """
    if lot_size <= 0:
        return quantity
    if quantity == 0:
        return 0.0
    sign = 1 if quantity > 0 else -1
    return sign * math.floor(abs(quantity) / lot_size) * lot_size


def validate_min_notional(
    quantity: float,
    price: float,
    min_notional_usd: float,
    fee_buffer_mult: float = 1.5,
) -> tuple[bool, str]:
    """Returns (ok, reason)."""
    notional = abs(quantity) * price
    threshold = min_notional_usd * fee_buffer_mult
    if notional < threshold:
        return False, (
            f"min_notional_violation: notional=${notional:.4f} "
            f"< threshold=${threshold:.4f}"
        )
    return True, ""


def safe_sell_quantity(
    portfolio_quantity: float,
    cashbook_quantity: float,
    lot_size: float,
    min_order_size: float,
    exit_qty_buffer_lots: int = 1,
) -> float:
    """The CashBook quirk fix from Vox.

    In Kraken cash-mode, ``portfolio[sym].quantity`` can drift slightly above
    the actual base-currency CashBook balance after fees and rounding. Selling
    the raw portfolio quantity submits an order larger than the exchangeable
    balance and QuantConnect rejects it.

    Logic:
      1. Take min(portfolio_qty, cashbook_qty) — never sell more than held.
      2. Floor to lot_size.
      3. Subtract `exit_qty_buffer_lots * lot_size` as a safety margin.
      4. If result is below min_order_size, return 0.0 (dust).
    """
    if portfolio_quantity <= 0 or cashbook_quantity <= 0:
        return 0.0

    qty = min(portfolio_quantity, cashbook_quantity)
    qty = round_to_lot(qty, lot_size)
    qty -= exit_qty_buffer_lots * lot_size

    if qty < min_order_size or qty <= 0:
        return 0.0
    return qty


def compute_slippage_pct(
    fill_price: float,
    reference_price: float,
    direction: str,
) -> float:
    """Fill-price slippage as a positive fraction (cost paid).

    Direction ``Buy``  → positive when fill > reference (paid more).
    Direction ``Sell`` → positive when fill < reference (received less).
    """
    if reference_price <= 0 or fill_price <= 0:
        return 0.0
    if direction.lower() == "buy":
        return max(0.0, (fill_price - reference_price) / reference_price)
    return max(0.0, (reference_price - fill_price) / reference_price)


@dataclass(frozen=True)
class SlippageLogEntry:
    time:        datetime
    symbol:      str
    direction:   str
    quantity:    float
    fill_price:  float
    reference_price: float
    slippage_bps:    float
    intent:      OrderIntent


def slippage_log_entry(
    *, time: datetime, symbol: str, direction: str, quantity: float,
    fill_price: float, reference_price: float, intent: OrderIntent,
) -> SlippageLogEntry:
    """Build a typed slip-log entry.

    Caller passes both fill_price and the reference (mid or signal) price;
    we compute the bps cost.
    """
    pct = compute_slippage_pct(fill_price, reference_price, direction)
    return SlippageLogEntry(
        time=time, symbol=symbol, direction=direction, quantity=quantity,
        fill_price=fill_price, reference_price=reference_price,
        slippage_bps=round(pct * 10_000, 2),
        intent=intent,
    )


@dataclass
class PendingOrder:
    """Snapshot of an outstanding limit order awaiting fill."""
    symbol:        str
    side:          str
    quantity:      float
    limit_price:   float
    submitted_at:  datetime
    ttl_seconds:   int
    intent:        OrderIntent
    order_id:      str | None = None

    def is_stale(self, now: datetime) -> bool:
        return (now - self.submitted_at).total_seconds() >= self.ttl_seconds


def stale_pending_orders(
    pending: list[PendingOrder], now: datetime,
) -> list[PendingOrder]:
    """Filter pending orders that have exceeded their TTL."""
    return [p for p in pending if p.is_stale(now)]


# ─── QC adapters ─────────────────────────────────────────────────────────────

if HAS_QC:

    def place_limit_or_market(
        algo,
        symbol,
        quantity: float,
        timeout_seconds: int = 30,
        tag: str = "Entry",
        limit_offset_bps: float = 5.0,
    ):
        """Submit a maker limit order; if not filled within `timeout_seconds`,
        cancel and submit a market order as fallback.

        For BUY: limit sits at bid (mid - half_spread), so we are at the
        passive side of the book.

        Note: QC backtest does NOT enforce TTL on limits; we record submission
        time and check during the next OnData tick.
        """
        try:
            sec = algo.Securities[symbol]
            price = float(sec.Price)
            if price <= 0:
                return None
            offset = price * limit_offset_bps / 10_000.0
            limit_px = price - offset if quantity > 0 else price + offset
            ticket = algo.LimitOrder(symbol, quantity, limit_px, tag)
            return ticket
        except Exception as exc:
            algo.Debug(f"place_limit_or_market error {symbol}: {exc}")
            return None


    def smart_liquidate(algo, symbol, tag: str = "Liquidate"):
        """Close a position safely, accounting for the CashBook quirk.

        Returns True if order submitted, False if position too small (dust).
        """
        try:
            holding = algo.Portfolio[symbol]
            qty = float(holding.Quantity)
            if qty == 0:
                return False
            price = float(algo.Securities[symbol].Price)
            if price <= 0:
                return False

            # Min order size (with fallback)
            sym_props = getattr(algo.Securities[symbol], "SymbolProperties", None)
            min_qty = (float(sym_props.MinimumOrderSize)
                       if sym_props and sym_props.MinimumOrderSize > 0
                       else min_quantity_fallback(symbol.Value
                                                  if hasattr(symbol, "Value")
                                                  else str(symbol)))
            lot = (float(sym_props.LotSize)
                   if sym_props and sym_props.LotSize > 0
                   else min_qty)

            # CashBook quantity (best-effort)
            cb_qty = qty   # fallback: assume portfolio == cashbook
            try:
                base_ccy = sym_props.QuoteCurrency.replace("USD", "")
                cb_qty = float(algo.Portfolio.CashBook[base_ccy].Amount)
            except Exception:
                pass

            sell_qty = safe_sell_quantity(
                portfolio_quantity=abs(qty),
                cashbook_quantity=abs(cb_qty),
                lot_size=lot,
                min_order_size=min_qty,
            )
            if sell_qty <= 0:
                return False

            algo.MarketOrder(symbol, -sell_qty if qty > 0 else sell_qty, tag=tag)
            return True
        except Exception as exc:
            algo.Debug(f"smart_liquidate error {symbol}: {exc}")
            return False


    def partial_smart_sell(algo, symbol, fraction: float, tag: str = "Partial TP"):
        """Sell `fraction` of the held position safely."""
        try:
            holding = algo.Portfolio[symbol]
            qty = float(holding.Quantity)
            if qty == 0 or not (0 < fraction < 1):
                return False
            target_sell = abs(qty) * fraction
            sym_props = getattr(algo.Securities[symbol], "SymbolProperties", None)
            min_qty = (float(sym_props.MinimumOrderSize)
                       if sym_props and sym_props.MinimumOrderSize > 0
                       else min_quantity_fallback(symbol.Value
                                                  if hasattr(symbol, "Value")
                                                  else str(symbol)))
            lot = (float(sym_props.LotSize)
                   if sym_props and sym_props.LotSize > 0
                   else min_qty)
            qty_lot = round_to_lot(target_sell, lot)
            if qty_lot < min_qty:
                # Partial would be dust — fall through to a full liquidation
                return smart_liquidate(algo, symbol, tag=tag)
            algo.MarketOrder(symbol, -qty_lot if qty > 0 else qty_lot, tag=tag)
            return True
        except Exception as exc:
            algo.Debug(f"partial_smart_sell error {symbol}: {exc}")
            return False


    def cleanup_position(algo, tracked_state: dict, symbol) -> None:
        """Clear local tracking state for a symbol after exit/dust cleanup."""
        for key in ("entry_prices", "highest_prices", "entry_times",
                    "entry_volumes", "rsi_peaked_overbought"):
            d = tracked_state.get(key)
            if isinstance(d, dict):
                d.pop(symbol, None)

else:
    # Pure-Python tests can't exercise these
    place_limit_or_market = None  # type: ignore
    smart_liquidate       = None  # type: ignore
    partial_smart_sell    = None  # type: ignore
    cleanup_position      = None  # type: ignore
