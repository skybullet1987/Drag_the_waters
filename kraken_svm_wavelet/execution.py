# region imports
from AlgorithmImports import *
# endregion


def set_holdings_limit(algo, symbol, target_weight, tag="SVMWavelet"):
    """
    Move position toward target_weight using a limit order at the current
    bid (buys) / ask (sells). Cash-account safe:
      - Never sells more than we currently own.
      - Never tries to spend more than free cash (after open orders).
      - Cancels stale open orders for the symbol before placing a new one.

    target_weight: signed fraction of TotalPortfolioValue, in [-1, 1].
    Returns the OrderTicket or None.
    """
    sec = algo.Securities[symbol]
    price = float(sec.Price or 0)
    if price <= 0:
        return None

    portfolio_value = float(algo.Portfolio.TotalPortfolioValue)
    target_value = portfolio_value * float(target_weight)
    target_qty = target_value / price

    current_qty = float(algo.Portfolio[symbol].Quantity)
    delta_qty = target_qty - current_qty

    # Cash account: can't sell more than we currently hold.
    if delta_qty < 0:
        delta_qty = max(delta_qty, -current_qty)

    # Cash account: can't spend more than the free cash, accounting for
    # cash already reserved by other open buy orders.
    if delta_qty > 0:
        free_cash = float(algo.Portfolio.Cash) * 0.98  # 2% buffer for fees/slippage
        for t in algo.Transactions.GetOpenOrders():
            try:
                if float(t.Quantity) > 0:
                    limit_px = None
                    try:
                        limit_px = float(t.Get(OrderField.LimitPrice) or 0)
                    except Exception:
                        limit_px = 0
                    px = limit_px if limit_px and limit_px > 0 else price
                    free_cash -= abs(float(t.Quantity)) * px
            except Exception:
                continue
        if free_cash <= 0:
            return None
        max_buy_qty = free_cash / price
        delta_qty = min(delta_qty, max_buy_qty)

    # Round to lot size; bail if dust (< $5 notional).
    lot_size = float(sec.SymbolProperties.LotSize or 1e-8)
    delta_qty = round(delta_qty / lot_size) * lot_size
    if abs(delta_qty * price) < 5.0:
        return None

    # Cancel any existing open orders on this symbol — otherwise stale
    # limits from yesterday accumulate and lock up cash.
    try:
        for t in algo.Transactions.GetOpenOrders(symbol):
            algo.Transactions.CancelOrder(t.Id)
    except Exception:
        pass

    # Limit price = bid for buys, ask for sells (post-only-style).
    bid = float(getattr(sec, "BidPrice", 0) or 0)
    ask = float(getattr(sec, "AskPrice", 0) or 0)
    if delta_qty > 0:
        limit_price = bid if bid > 0 else price
    else:
        limit_price = ask if ask > 0 else price

    try:
        return algo.LimitOrder(symbol, delta_qty, limit_price, tag=tag)
    except Exception as e:
        algo.Debug(f"Limit order failed for {symbol.Value}: {e}")
        return None


def cancel_stale_orders(algo, max_age_hours=24):
    """Cancel any limit orders older than `max_age_hours`. UTC-safe."""
    cutoff = algo.UtcTime - timedelta(hours=max_age_hours)
    for ticket in algo.Transactions.GetOpenOrders():
        try:
            order = algo.Transactions.GetOrderById(ticket.Id)
            if order is None:
                continue
            # order.Time is timezone-aware UTC; algo.UtcTime is too.
            if order.Time < cutoff:
                algo.Transactions.CancelOrder(ticket.Id)
        except Exception:
            pass
