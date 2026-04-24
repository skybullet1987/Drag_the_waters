# region imports
from AlgorithmImports import *
# endregion


def set_holdings_limit(algo, symbol, target_weight, tag="SVMWavelet"):
    """
    Move position toward target_weight using a limit order at the current
    bid (for buys) or ask (for sells). Falls back to MarketOrder if the
    quote is missing.

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

    # Round to lot size; bail if dust.
    lot_size = float(sec.SymbolProperties.LotSize or 1e-8)
    delta_qty = round(delta_qty / lot_size) * lot_size
    if abs(delta_qty * price) < 5.0:
        return None

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
        algo.Debug(f"Limit order failed for {symbol.Value}: {e}, falling back to market")
        try:
            return algo.MarketOrder(symbol, delta_qty, tag=tag)
        except Exception as e2:
            algo.Debug(f"Market fallback also failed for {symbol.Value}: {e2}")
            return None


def cancel_stale_orders(algo, max_age_hours=24):
    """Cancel any limit orders older than `max_age_hours`."""
    cutoff = algo.Time - timedelta(hours=max_age_hours)
    for ticket in algo.Transactions.GetOpenOrders():
        try:
            order = algo.Transactions.GetOrderById(ticket.Id)
            if order is not None and order.Time < cutoff:
                algo.Transactions.CancelOrder(ticket.Id)
        except Exception:
            pass
