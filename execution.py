# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
import random
from collections import deque
from datetime import timedelta
# endregion

random.seed(45)


ESTIMATED_ROUND_TRIP_FEE = 0.0065  # 0.65% round-trip

# STRUCTURAL blacklist: a-priori structural exclusions (not look-ahead biased).
SYMBOL_BLACKLIST_STRUCTURAL = {
    "BTCUSD",  # Reference symbol only — never trade (too expensive for small capital, always dust)
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
    # Forex pairs
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", "SGDUSD",
    "SEKUSD", "NOKUSD", "DKKUSD", "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", "INRUSD", "BRLUSD",
    "PLNUSD", "THBUSD",
}

SYMBOL_BLACKLIST_BACKTEST_DERIVED = {
    "XCNUSD",    # Catastrophic stop-loss losses (e.g. -$5,386 and -$1.9M single trades)
    "KTAUSD",    # Immediate stop losses on first trades
    "NANOUSD",   # Immediate stop losses on first trades
    "FWOGUSD",   # Repeated breakeven stops and poor fills
    "STRKUSD",   # Repeated breakeven stops and poor fills
}


INCLUDE_BACKTEST_DERIVED_BLACKLIST = False

# Combined blacklist.
SYMBOL_BLACKLIST = (
    SYMBOL_BLACKLIST_STRUCTURAL | SYMBOL_BLACKLIST_BACKTEST_DERIVED
    if INCLUDE_BACKTEST_DERIVED_BLACKLIST
    else SYMBOL_BLACKLIST_STRUCTURAL
)

# UTC hour thresholds for spread time-of-day multipliers (_estimate_backtest_spread).
_TOD_ASIAN_END = 8; _TOD_EU_END = 13; _TOD_US_END = 21  # session boundaries
# Backtest queue-priority: participation-rate rejection constants.
_QUEUE_PARTICIPATION_THRESHOLD = 0.02; _QUEUE_REJECTION_SLOPE = 10.0; _QUEUE_MAX_REJECTION_PROB = 0.90

# Known fiat currency codes used to filter forex pairs from the crypto universe
KNOWN_FIAT_CURRENCIES = frozenset({
    "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF", "CNY", "HKD", "SGD",
    "SEK", "NOK", "DKK", "KRW", "TRY", "ZAR", "MXN", "INR", "BRL", "PLN", "THB",
})


QUANTITY_HAIRCUT_FACTOR = 0.9999


QUANTITY_OVERSHOOT_TOLERANCE = 1.0001

KRAKEN_MIN_QTY_FALLBACK = {
    'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
    'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
    'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05,
    'ALGOUSD': 10.0, 'XLMUSD': 30.0, 'TRXUSD': 50.0, 'ENJUSD': 10.0,
    'BATUSD': 10.0, 'CRVUSD': 5.0, 'SNXUSD': 3.0, 'COMPUSD': 0.1,
    'AAVEUSD': 0.05, 'MKRUSD': 0.01, 'YFIUSD': 0.001, 'UNIUSD': 1.0,
    'SUSHIUSD': 5.0, '1INCHUSD': 5.0, 'GRTUSD': 10.0, 'FTMUSD': 10.0,
    'IMXUSD': 5.0, 'APEUSD': 2.0, 'GMTUSD': 10.0, 'OPUSD': 5.0,
    'LDOUSD': 5.0, 'ARBUSD': 5.0, 'LPTUSD': 5.0, 'KTAUSD': 10.0,
    'GUNUSD': 50.0, 'BANANAS31USD': 500.0, 'CHILLHOUSEUSD': 500.0,
    'PHAUSD': 50.0, 'MUSD': 50.0, 'ICNTUSD': 50.0,
    'SHIBUSD': 50000.0, 'XRPUSD': 2.0,
}

MIN_NOTIONAL_FALLBACK = {
    'EWTUSD': 2.0, 'SANDUSD': 8.0, 'CTSIUSD': 18.0, 'MKRUSD': 0.01,
    'AUDUSD': 10.0, 'LPTUSD': 0.3, 'OXTUSD': 40.0, 'ENJUSD': 15.0,
    'UNIUSD': 0.5, 'LSKUSD': 3.0, 'BCHUSD': 1.0,
}

# Fee buffer applied when computing sell quantity to prevent overselling due to
# Kraken deducting fees from the base asset after a buy (Cash Modeling discrepancy).
KRAKEN_SELL_FEE_BUFFER = 0.006  # 0.6% (0.4% base fee + 0.2% safety margin)


def get_min_quantity(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    try:
        if symbol in algo.Securities:
            sec = algo.Securities[symbol]
            if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                min_size = sec.SymbolProperties.MinimumOrderSize
                if min_size is not None and min_size > 0:
                    return float(min_size)
    except Exception as e:
        algo.Debug(f"Error getting min quantity for {ticker}: {e}")
        pass
    if ticker in KRAKEN_MIN_QTY_FALLBACK:
        return KRAKEN_MIN_QTY_FALLBACK[ticker]
    return estimate_min_qty(algo, symbol)


def estimate_min_qty(algo, symbol):
    try:
        price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    except Exception as e:
        algo.Debug(f"Error getting price for min qty estimate: {e}")
        price = 0
    if price <= 0: return 50.0
    if price < 0.001: return 1000.0
    elif price < 0.01: return 500.0
    elif price < 0.1: return 50.0
    elif price < 1.0: return 10.0
    elif price < 10.0: return 5.0
    elif price < 100.0: return 1.0
    elif price < 1000.0: return 0.1
    else: return 0.01


def get_min_notional_usd(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    fallback = MIN_NOTIONAL_FALLBACK.get(ticker, algo.min_notional)
    try:
        price = algo.Securities[symbol].Price
        min_qty = get_min_quantity(algo, symbol)
        implied = price * min_qty if price > 0 else fallback
        return max(fallback, implied, algo.min_notional)
    except Exception as e:
        algo.Debug(f"Error in get_min_notional_usd for {symbol.Value}: {e}")
        return max(fallback, algo.min_notional)


def round_quantity(algo, symbol, quantity):
    try:
        lot_size = algo.Securities[symbol].SymbolProperties.LotSize
        if lot_size is not None and lot_size > 0:
            rounded = float(math.floor(quantity / lot_size)) * lot_size
            # Clamp: never return more than the input (IEEE 754 safety)
            if rounded > quantity:
                rounded = rounded - lot_size
                if rounded < 0:
                    rounded = 0.0
            return rounded
        return quantity
    except Exception as e:
        algo.Debug(f"Error rounding quantity for {symbol.Value}: {e}")
        return quantity


def track_exit_order(algo, symbol, ticket, quantity):
    """Helper to track exit order in _submitted_orders for verification."""
    if hasattr(algo, '_submitted_orders') and ticket is not None:
        algo._submitted_orders[symbol] = {
            'order_id': ticket.OrderId,
            'time': algo.Time,
            'quantity': quantity,
            'intent': 'exit'
        }


def smart_liquidate(algo, symbol, tag="Liquidate"):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return False
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price * abs(holding_qty) < min_notional * 0.9:
        return False
    # Verify fee reserve before selling (Kraken cash account requirement)
    if algo.LiveMode and holding_qty > 0:
        estimated_fee = price * abs(holding_qty) * 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
        try:
            available_usd = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_usd = algo.Portfolio.Cash
        is_stop_loss = "Stop Loss" in tag or "Stop" in tag
        if available_usd < estimated_fee and not is_stop_loss:
            algo.Debug(f"⚠️ SKIP SELL {symbol.Value}: fee reserve too low "
                       f"(need ${estimated_fee:.4f}, have ${available_usd:.4f})")
            if symbol not in algo.entry_prices:
                algo.entry_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.highest_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.entry_times[symbol] = algo.Time
            return False
    algo.Transactions.CancelOpenOrders(symbol)
    # For exits: use MinimumOrderSize from the security (brokerage-enforced floor).
    # lot_size < MinimumOrderSize for some assets (e.g. ETHUSD: lot=0.001, min=0.002),
    # so using lot_size alone allows placing orders the brokerage will reject.
    try:
        sec = algo.Securities[symbol]
        min_order_size = sec.SymbolProperties.MinimumOrderSize
        if min_order_size is not None and min_order_size > 0:
            exit_min_qty = float(min_order_size)
        else:
            exit_min_qty = min_qty
    except Exception as e:
        algo.Debug(f"Warning: could not get MinimumOrderSize for {symbol.Value}: {e}")
        exit_min_qty = min_qty
    if abs(holding_qty) < exit_min_qty:
        return False
    # The portfolio already reflects the post-fee quantity (Kraken deducts fees from
    # the base asset at buy time), so no fee buffer is needed on the sell side.
    safe_qty = round_quantity(algo, symbol, abs(holding_qty))
    # Ensure we never attempt to sell more than the actual portfolio quantity.
    # round_quantity can round UP due to floating-point, causing "insufficient buying power"
    # errors on Kraken Cash Modeling (e.g. post-fee BTC holdings are slightly less than
    # the ordered quantity).  Apply a haircut before re-rounding to guarantee floor division
    # stays at or below actual_qty.
    actual_qty = abs(algo.Portfolio[symbol].Quantity)
    if safe_qty > actual_qty * QUANTITY_OVERSHOOT_TOLERANCE:  # tolerance for floating-point
        safe_qty = round_quantity(algo, symbol, actual_qty)
    # Final hard cap — never exceed actual holding regardless of floating-point rounding
    safe_qty = min(safe_qty, actual_qty)
    if safe_qty < exit_min_qty:
        # Position rounded down below MinimumOrderSize — treat as dust, cannot sell
        return False
    if safe_qty > 0:
        direction_mult = -1 if holding_qty > 0 else 1
        # Spread-aware exit logic
        is_stop_loss = "Stop Loss" in tag
        if not is_stop_loss:
            spread_pct = get_spread_pct(algo, symbol)
            if spread_pct is not None:
                if spread_pct > 0.03:  # 3% spread - log warning but still exit with market
                    algo.Debug(f"⚠️ WIDE SPREAD EXIT: {symbol.Value} spread={spread_pct:.2%}, using market order")
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
                elif algo.LiveMode and spread_pct > 0.015:  # 1.5% spread - use limit order with fallback (live mode only)
                    try:
                        sec = algo.Securities[symbol]
                        bid = sec.BidPrice
                        ask = sec.AskPrice
                        if bid > 0 and ask > 0:
                            mid = 0.5 * (bid + ask)
                            limit_order = algo.LimitOrder(symbol, safe_qty * direction_mult, mid, tag=tag)
                            # Track for fallback in VerifyOrderFills (90 second timeout handled there)
                            if hasattr(algo, '_submitted_orders'):
                                algo._submitted_orders[symbol] = {
                                    'order_id': limit_order.OrderId,
                                    'time': algo.Time,
                                    'quantity': safe_qty * direction_mult,  # Store signed quantity
                                    'is_limit_exit': True,
                                    'intent': 'exit'
                                }
                            algo.Debug(f"LIMIT EXIT: {symbol.Value} at mid ${mid:.4f} (spread={spread_pct:.2%})")
                            return True
                        else:
                            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                            track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                            return True
                    except Exception as e:
                        algo.Debug(f"Error placing limit exit for {symbol.Value}: {e}")
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                        track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                        return True
                else:
                    # Spread is acceptable – use limit order to capture maker fee
                    exit_price = algo.Securities[symbol].Price
                    try:
                        ticket = algo.LimitOrder(symbol, safe_qty * direction_mult, exit_price, tag=tag)
                    except Exception:
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
            else:
                # Spread unknown – use limit order at last price to capture maker fee
                exit_price = algo.Securities[symbol].Price
                try:
                    ticket = algo.LimitOrder(symbol, safe_qty * direction_mult, exit_price, tag=tag)
                except Exception:
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                return True
        else:
            # Stop loss - use limit order with 0.5% buffer to cap slippage on illiquid tokens
            STOP_LOSS_LIMIT_BUFFER = 0.005   # 0.5% price buffer below current price
            STOP_LOSS_LIMIT_TIMEOUT = 60     # seconds before market-order fallback
            exit_price = algo.Securities[symbol].Price
            if exit_price > 0:
                # Place limit at buffer% worse than current price to accept some slippage but cap it
                limit_price = exit_price * (1.0 - STOP_LOSS_LIMIT_BUFFER)
                try:
                    ticket = algo.LimitOrder(symbol, safe_qty * direction_mult, limit_price, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    # Track for quick market-order fallback if limit doesn't fill in time
                    if hasattr(algo, '_submitted_orders'):
                        algo._submitted_orders[symbol] = {
                            'order_id': ticket.OrderId,
                            'time': algo.Time,
                            'quantity': safe_qty * direction_mult,
                            'is_limit_exit': True,
                            'intent': 'exit',
                            'timeout_seconds': STOP_LOSS_LIMIT_TIMEOUT
                        }
                    return True
                except Exception:
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
            else:
                ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                return True
    else:
        algo.Debug(f"Warning: {symbol.Value} holding {holding_qty} rounds to 0")
        return False


def partial_smart_sell(algo, symbol, fraction, tag="Partial TP"):
    """
    Sell a fraction (0.0–1.0) of the current position.
    Both the sell portion and the remaining portion must meet the minimum
    order size; otherwise falls back to a full smart_liquidate.
    Returns True if an order was placed successfully, False otherwise.
    """
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price <= 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    sell_qty = round_quantity(algo, symbol, abs(holding_qty) * fraction)
    remaining_qty = round_quantity(algo, symbol, abs(holding_qty) * (1.0 - fraction))
    # Both halves must be tradeable; fall back to full exit if not
    if sell_qty < min_qty or remaining_qty < min_qty:
        return smart_liquidate(algo, symbol, tag)
    if sell_qty * price < min_notional * 0.9:
        return smart_liquidate(algo, symbol, tag)
    # Flag as partial so OnOrderEvent skips position cleanup
    if hasattr(algo, '_partial_sell_symbols'):
        algo._partial_sell_symbols.add(symbol)
    direction_mult = -1 if holding_qty > 0 else 1
    # Use limit order at current price to capture maker fee (0.25% vs 0.40% taker)
    try:
        ticket = algo.LimitOrder(symbol, sell_qty * direction_mult, price, tag=tag)
    except Exception:
        ticket = algo.MarketOrder(symbol, sell_qty * direction_mult, tag=tag)
    if ticket is not None:
        algo.Debug(f"PARTIAL SELL: {symbol.Value} | frac={fraction:.0%} qty={sell_qty:.6f} of {abs(holding_qty):.6f}")
    return ticket is not None



def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds


def cancel_stale_new_orders(algo):
    # Allow cancel gate when venue is online or unknown (fallback handled elsewhere)
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            sym_val = order.Symbol.Value
            if sym_val in algo._session_blacklist:
                continue
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                
                # Check if this order actually filled (fill event missed)
                if is_invested_not_dust(algo, order.Symbol):
                    # Position exists — order filled, we just missed the event
                    # Re-track instead of blacklisting
                    algo.Debug(f"STALE ORDER but position exists: {sym_val} — re-tracking")
                    holding = algo.Portfolio[order.Symbol]
                    algo.entry_prices[order.Symbol] = holding.AveragePrice
                    algo.highest_prices[order.Symbol] = holding.AveragePrice
                    algo.entry_times[order.Symbol] = algo.Time
                    algo.Transactions.CancelOrder(order.Id)  # Cancel the stale order
                    continue  # Don't blacklist
                
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
                
                # Only blacklist stale ENTRY orders, not EXIT orders
                # Exit orders that are stale just get cooldown to allow retry
                has_position_or_tracked = order.Symbol in algo.entry_prices or (
                    order.Symbol in algo.Portfolio and algo.Portfolio[order.Symbol].Quantity != 0
                )
                
                if has_position_or_tracked:
                    algo.Debug(f"STALE EXIT: {sym_val} - cooldown only, not blacklisted")
                else:
                    algo._symbol_entry_cooldowns[sym_val] = algo.Time + timedelta(minutes=15)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {sym_val} - entry cooldown 15min")
    except Exception as e:
        algo.Debug(f"Error in _cancel_stale_new_orders: {e}")


def is_invested_not_dust(algo, symbol):
    if symbol not in algo.Portfolio:
        return False
    h = algo.Portfolio[symbol]
    if not h.Invested or h.Quantity == 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else h.Price
    notional_ok = (price > 0) and (abs(h.Quantity) * price >= min_notional * 0.5)
    qty_ok = abs(h.Quantity) >= min_qty * 0.5
    return notional_ok or qty_ok


def get_actual_position_count(algo):
    return sum(1 for kvp in algo.Portfolio if is_invested_not_dust(algo, kvp.Key))


def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0


def has_non_stale_open_orders(algo, symbol):
    """Check if symbol has open orders that are NOT stale (younger than timeout)."""
    try:
        orders = algo.Transactions.GetOpenOrders(symbol)
        if len(orders) == 0:
            return False
        timeout_seconds = effective_stale_timeout(algo)
        for order in orders:
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age <= timeout_seconds:
                return True  # At least one order is not stale
        return False  # All orders are stale
    except Exception:
        return False


def get_spread_pct(algo, symbol):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                return (ask - bid) / mid
    except Exception as e:
        algo.Debug(f"Error getting spread for {symbol.Value}: {e}")
        pass
    return None


def _estimate_backtest_spread(algo, symbol):
    """
    Estimate bid-ask spread from dollar volume for backtest realism.
    Uses empirical relationship: spread ≈ k / sqrt(dollar_volume).
    Calibrated against typical Kraken altcoin spreads:
      - $1M daily vol → ~0.20% spread
      - $100K daily vol → ~0.63% spread
      - $10K daily vol → ~2.0% spread
    Only called in backtest when real quote data is unavailable.
    """
    crypto = algo.crypto_data.get(symbol)
    if crypto is None:
        return None

    # Use recent dollar volume (sum of last 20 bars × bar_interval_to_daily_scalar)
    dv_list = list(crypto.get('dollar_volume', []))
    if len(dv_list) < 5:
        return None

    # Average dollar volume per bar, extrapolated to daily
    avg_bar_dv = float(np.mean(dv_list[-20:]))
    # Bars per day: 288 for 5-min bars, 1440 for 1-min bars
    # The consolidator uses 5-min bars for indicator updates, but dollar_volume
    # is appended per 5-min bar in _update_symbol_data
    daily_dv = avg_bar_dv * 288  # 288 = 5-min bars per day (60min/5min × 24h)

    if daily_dv <= 0:
        return 0.05  # 5% default for zero-volume — will be rejected by spread cap

    # Empirical constant calibrated to Kraken:
    # k=2.0 gives spread=0.20% at $1M, 0.63% at $100K, 2.0% at $10K
    k = 2.0
    estimated_spread = k / (daily_dv ** 0.5)

    # Floor: even the most liquid alts have >= 0.05% spread on Kraken
    # Cap: don't estimate above 10% — if it's that wide, the filter will catch it
    estimated_spread = max(estimated_spread, 0.0005)
    estimated_spread = min(estimated_spread, 0.10)

    # Time-of-day multiplier: Kraken spreads widen outside US/EU hours.
    # 00-08 UTC Asian (×1.5), 08-13 EU (×1.0), 13-21 US/EU (×0.85), 21-24 wind-down (×1.2).
    h = algo.Time.hour
    if h < _TOD_ASIAN_END:
        tod_multiplier = 1.5
    elif h < _TOD_EU_END:
        tod_multiplier = 1.0
    elif h < _TOD_US_END:
        tod_multiplier = 0.85
    else:
        tod_multiplier = 1.2
    estimated_spread *= tod_multiplier

    # Re-apply cap after multiplier to avoid exceeding the hard ceiling
    estimated_spread = min(estimated_spread, 0.10)

    # Add the estimated spread to the spreads deque for median tracking
    crypto['spreads'].append(estimated_spread)

    return estimated_spread


def spread_ok(algo, symbol):
    sp = get_spread_pct(algo, symbol)
    if sp is None:
        if algo.LiveMode:
            # Allow unknown spreads in live — spread cap check will catch them once data arrives.
            now = algo.Time
            last_warn = algo._spread_warning_times.get(symbol.Value)
            if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                debug_limited(algo, f"SPREAD UNKNOWN: {symbol.Value} — allowing (no bid/ask yet)")
                algo._spread_warning_times[symbol.Value] = now
            return True
        else:
            # BACKTEST REALISM: estimate spread from dollar volume
            sp = _estimate_backtest_spread(algo, symbol)
            if sp is None:
                return True  # no data at all, allow (conservative fallback)
            # Fall through to normal spread checks below with estimated spread
    effective_spread_cap = algo.max_spread_pct
    if algo.volatility_regime == "high" or algo.market_regime == "sideways":
        effective_spread_cap = min(effective_spread_cap, 0.025)
    if sp > effective_spread_cap:
        return False
    crypto = algo.crypto_data.get(symbol)
    if crypto and len(crypto.get('spreads', [])) >= 4:
        median_sp = np.median(list(crypto['spreads']))
        if median_sp > 0 and sp > algo.spread_widen_mult * median_sp:
            return False
    return True


def intraday_volume_ok(algo, symbol, order_value):
    """
    Check that the recent intraday dollar volume is sufficient for the
    proposed order. Rejects entries where recent bars show very thin
    trading activity, even if the daily universe filter passed.

    Returns True if volume is adequate, False otherwise.
    """
    crypto = algo.crypto_data.get(symbol)
    if crypto is None:
        return True  # no data, allow (conservative)

    dv_list = list(crypto.get('dollar_volume', []))
    if len(dv_list) < 5:
        return True  # not enough history yet

    # Average dollar volume over last 10 bars (50 minutes at 5-min resolution)
    recent_avg_dv = float(np.mean(dv_list[-10:])) if len(dv_list) >= 10 else float(np.mean(dv_list))

    # Require at least $500 avg dollar volume per bar (≈ $144K daily pace)
    # This catches symbols in low-activity windows even if daily vol is high
    if recent_avg_dv < 500:
        return False

    # Participation rate check: order should be < 2% of recent bar volume
    if recent_avg_dv > 0 and order_value > 0:
        implied_participation = order_value / recent_avg_dv
        if implied_participation > algo.max_participation_rate:
            return False

    return True


def cleanup_position(algo, symbol, record_pnl=False, exit_price=None):
    """
    Clean up position tracking for a symbol.
    If record_pnl=True, records the PnL before cleanup using record_exit_pnl helper.
    """
    entry_price = algo.entry_prices.get(symbol, None)
    if record_pnl and entry_price is not None and entry_price > 0:
        if exit_price is None:
            try:
                exit_price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
            except Exception:
                exit_price = 0
        if exit_price > 0:
            record_exit_pnl(algo, symbol, entry_price, exit_price)
    # Then do existing cleanup
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    if symbol in algo.crypto_data:
        algo.crypto_data[symbol]['trail_stop'] = None
    if hasattr(algo, '_spike_entries'):
        algo._spike_entries.pop(symbol, None)
    if hasattr(algo, '_partial_tp_taken'):
        algo._partial_tp_taken.pop(symbol, None)
    if hasattr(algo, '_breakeven_stops'):
        algo._breakeven_stops.pop(symbol, None)


def sync_existing_positions(algo):
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        holding = algo.Portfolio[symbol]
        ticker = symbol.Value
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            try:
                algo.AddCrypto(ticker, Resolution.Minute, Market.Kraken)
            except Exception as e:
                algo.Debug(f"Error adding crypto {ticker}: {e}")
                continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.quick_take_profit:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.tight_stop_loss:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} positions")
    algo.Debug(f"Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        algo.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
        sold = smart_liquidate(algo, symbol, reason)
        if not sold:
            algo.Debug(f"⚠️ IMMEDIATE {reason} FAILED: {ticker} — position unsellable, cleaning up tracking")
            cleanup_position(algo, symbol)
        # Let OnOrderEvent handle cleanup and PnL tracking on fill


def debug_limited(algo, msg):
    if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
        algo.Debug(msg)
        return
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)


def slip_log(algo, symbol, direction, fill_price):
    """Enhanced slip_log with live outlier alert and symbol-level slippage tracking."""
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid <= 0 or ask <= 0:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        side = 1 if direction == OrderDirection.Buy else -1
        slip = side * (fill_price - mid) / mid
        abs_slip = abs(slip)
        algo._slip_abs.append(abs_slip)
        
        # Track slippage per symbol (store (time, abs_slippage) tuples for time-based decay)
        if hasattr(algo, '_symbol_slippage_history'):
            ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            if ticker not in algo._symbol_slippage_history:
                algo._symbol_slippage_history[ticker] = deque(maxlen=30)
            algo._symbol_slippage_history[ticker].append((algo.Time, abs_slip))
        
        # Live slippage alert for unusually high slippage
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")
        pass


def persist_state(algo):
    """Enhanced persist_state with trade_count and peak_value from main_opus."""
    if not algo.LiveMode:
        return
    try:
        spike_entries = [s.Value for s in getattr(algo, '_spike_entries', {}).keys() if hasattr(s, 'Value')]
        state = {
            "session_blacklist": list(algo._session_blacklist),
            "winning_trades": algo.winning_trades,
            "losing_trades": algo.losing_trades,
            "total_pnl": algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
            "peak_value": algo.peak_value if algo.peak_value is not None else 0,
            "spike_entries": spike_entries,
        }
        algo.ObjectStore.Save("live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    """Enhanced load_persisted_state with trade_count and peak_value from main_opus."""
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state"):
            raw = algo.ObjectStore.Read("live_state")
            data = json.loads(raw)
            algo._session_blacklist = set(data.get("session_blacklist", []))
            algo.winning_trades = data.get("winning_trades", 0)
            algo.losing_trades = data.get("losing_trades", 0)
            algo.total_pnl = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            # Restore spike entries: map string values back to Symbol objects
            spike_entry_values = set(data.get("spike_entries", []))
            if spike_entry_values and hasattr(algo, '_spike_entries'):
                for symbol in algo.Securities.Keys:
                    if hasattr(symbol, 'Value') and symbol.Value in spike_entry_values:
                        algo._spike_entries[symbol] = True
            algo.Debug(f"Loaded persisted state: blacklist {len(algo._session_blacklist)}, "
                       f"trades W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def cleanup_object_store(algo):
    """From main_opus._cleanup_object_store."""
    try:
        n = 0
        for i in algo.ObjectStore.GetEnumerator():
            k = i.Key if hasattr(i, 'Key') else str(i)
            if k != "live_state":
                try:
                    algo.ObjectStore.Delete(k)
                    n += 1
                except Exception as e:
                    algo.Debug(f"Error deleting key {k}: {e}")
                    pass
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")


def live_safety_checks(algo):
    """Extra safety checks for live trading from main_opus."""
    if not algo.LiveMode:
        return True
    
    # Check if we have minimum viable cash
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except Exception as e:
        algo.Debug(f"Error getting cash from CashBook, using Portfolio.Cash: {e}")
        cash = algo.Portfolio.Cash
    
    if cash < 2.0:
        debug_limited(algo, "LIVE SAFETY: Cash below $2, pausing new entries")
        return False
    
    # Rate limit: don't trade more than once per 90 seconds in live
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 90:
            return False
    
    return True


def kelly_fraction(algo):
    """
    Half-Kelly position sizing multiplier, normalized so 1.0 = neutral.

    Normalization: divide by 0.25, the expected half-Kelly for a typical strategy
    with full-Kelly ≈ 0.50. This means:
      half_kelly=0.25 → returns 1.0  (neutral sizing)
      half_kelly=0.375 → returns 1.5 (max boost, capped)
      half_kelly=0.125 → returns 0.5 (min floor)
    Result is clamped to [0.5, 1.5] to prevent extreme under/over-sizing.
    """
    if len(algo._rolling_wins) < 10:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win = np.mean(list(algo._rolling_win_sizes)) if len(algo._rolling_win_sizes) > 0 else 0.02
    avg_loss = np.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.02
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * 0.5
    return max(0.5, min(1.5, half_kelly / 0.25))


def get_slippage_penalty(algo, symbol):
    """
    Calculate position size multiplier based on historical slippage for a symbol.
    Returns a value between 0.3 and 1.0. Ignores slippage entries older than 48 hours.
    """
    if not hasattr(algo, '_symbol_slippage_history'):
        return 1.0
    
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    if ticker not in algo._symbol_slippage_history:
        return 1.0
    
    slippage_history = algo._symbol_slippage_history[ticker]
    if len(slippage_history) == 0:
        return 1.0
    
    # Filter out entries older than 48 hours
    cutoff = algo.Time - timedelta(hours=48)
    recent_slips = [slip for entry_time, slip in slippage_history if entry_time >= cutoff]
    if len(recent_slips) == 0:
        return 1.0
    
    avg_slippage = sum(recent_slips) / len(recent_slips)
    
    # Apply penalties based on average slippage
    if avg_slippage > 0.010:  # > 1.0%
        return 0.3
    elif avg_slippage > 0.005:  # > 0.5%
        return 0.6
    elif avg_slippage > 0.003:  # > 0.3%
        return 0.8
    else:
        return 1.0


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry"):
    """
    Place entry orders using limit orders to capture maker fees.
    In backtest: volatility-correlated non-fill simulation, then falls through
    to the same limit order logic as live.
    In live: use limit orders with timeout to capture maker fees.
    Returns the ticket from the order placement.
    """
    try:
        # BACKTEST REALISM: volatility-correlated non-fill simulation
        # Real non-fills are adversely selected — higher volatility = more non-fills
        # because price moves away faster. NOT random.
        if not algo.LiveMode:
            crypto = algo.crypto_data.get(symbol)
            if crypto and len(crypto.get('volatility', [])) > 0:
                recent_vol = float(crypto['volatility'][-1])
                # Base non-fill rate 8%, scales up to 30% in high-vol environments.
                # Real limit order fill rates on Kraken altcoins at bid+0.05% are 70-92%
                # within a 5-minute bar; the 8% base reflects a realistic rejection rate.
                non_fill_prob = min(0.08 + recent_vol * 3.0, 0.30)
            else:
                non_fill_prob = 0.15
            if random.random() < non_fill_prob:
                return None
        # Fall through to limit order logic for both live and backtest
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice

        if bid > 0 and ask > 0:
            # Place limit order just above best bid – still maker, improves fill odds
            limit_price = bid * 1.0005  # 0.05% above bid – still below mid, still maker

            # BACKTEST: simulate queue-priority cost when bid/ask data IS available.
            # Offset (0.12%) reflects real queue priority cost on Kraken altcoins (0.10-0.20%).
            if not algo.LiveMode:
                adverse_offset = 0.0012  # 0.12% queue-priority cost
                if quantity > 0:
                    limit_price *= (1 + adverse_offset)
                    limit_price = min(limit_price, ask)  # buy: never cross the ask
                else:
                    limit_price *= (1 - adverse_offset)
                    limit_price = max(limit_price, bid)  # sell: never go below the bid

                # Probabilistic rejection based on participation rate (>5% threshold).
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto.get('volume', [])) > 0:
                    bar_volume = float(crypto['volume'][-1])
                    bar_dollar_volume = bar_volume * float(limit_price) if limit_price > 0 else 0
                    order_notional = abs(quantity) * float(limit_price)
                    if bar_dollar_volume > 0:
                        participation_rate = order_notional / bar_dollar_volume
                        if participation_rate > _QUEUE_PARTICIPATION_THRESHOLD:
                            excess = participation_rate - _QUEUE_PARTICIPATION_THRESHOLD
                            rejection_prob = min(excess * _QUEUE_REJECTION_SLOPE, _QUEUE_MAX_REJECTION_PROB)
                            if random.random() < rejection_prob:
                                algo.Debug(
                                    f"BACKTEST QUEUE REJECT (bid/ask): {symbol.Value} "
                                    f"part={participation_rate:.1%} rej={rejection_prob:.1%}"
                                )
                                return None
            else:
                limit_price = min(limit_price, ask)  # live: never cross the spread
        else:
            # No bid/ask data: use last price as limit price.
            # In backtest this would fill immediately at the limit price,
            # giving unrealistic instant maker fills without queue priority.
            limit_price = sec.Price
            if limit_price <= 0:
                algo.Debug(f"Price unavailable for {symbol.Value}, using market order")
                return algo.MarketOrder(symbol, quantity, tag=tag)

            # BACKTEST: simulate queue-priority cost when no bid/ask data.
            # Apply adverse offset + participation-rate rejection.
            if not algo.LiveMode:
                # Adverse offset: algo is not first in line at the limit level.
                adverse_offset = 0.0010  # 0.10% queue-priority cost (slippage model handles the rest)
                if quantity > 0:
                    limit_price *= (1 + adverse_offset)
                else:
                    limit_price *= (1 - adverse_offset)

                # Probabilistic rejection based on participation rate (>5% threshold).
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto.get('volume', [])) > 0:
                    bar_volume = float(crypto['volume'][-1])
                    bar_dollar_volume = bar_volume * float(limit_price) if limit_price > 0 else 0
                    order_notional = abs(quantity) * float(limit_price)
                    if bar_dollar_volume > 0:
                        participation_rate = order_notional / bar_dollar_volume
                        if participation_rate > _QUEUE_PARTICIPATION_THRESHOLD:
                            excess = participation_rate - _QUEUE_PARTICIPATION_THRESHOLD
                            rejection_prob = min(excess * _QUEUE_REJECTION_SLOPE, _QUEUE_MAX_REJECTION_PROB)
                            if random.random() < rejection_prob:
                                algo.Debug(
                                    f"BACKTEST QUEUE REJECT: {symbol.Value} "
                                    f"part={participation_rate:.1%} rej={rejection_prob:.1%}"
                                )
                                return None

        # Place maker limit order
        limit_ticket = algo.LimitOrder(symbol, quantity, limit_price, tag=tag)

        # Track for fallback in VerifyOrderFills
        if hasattr(algo, '_submitted_orders'):
            algo._submitted_orders[symbol] = {
                'order_id': limit_ticket.OrderId,
                'time': algo.Time,
                'quantity': quantity,  # Store signed quantity
                'is_limit_entry': True,
                'timeout_seconds': timeout_seconds,
                'intent': 'entry'
            }

            algo.Debug(f"MAKER LIMIT: {symbol.Value} | qty={quantity} | bid=${bid:.4f} | limit=${limit_price:.4f} | timeout={timeout_seconds}s")
        return limit_ticket

    except Exception as e:
        algo.Debug(f"Error placing limit order for {symbol.Value}: {e}, falling back to market")
        return algo.MarketOrder(symbol, quantity, tag=tag)


def normalize_order_time(order_time):
    """Helper to normalize order time by removing timezone info if present."""
    return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time


def record_exit_pnl(algo, symbol, entry_price, exit_price, exit_tag="Unknown"):
    """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
    if entry_price <= 0 or exit_price <= 0:
        algo.Debug(f"⚠️ Cannot record PnL for {symbol.Value}: invalid prices (entry=${entry_price:.4f}, exit=${exit_price:.4f})")
        return None
    
    # Fee is a direct percentage-point reduction (matches convention in events.py).
    pnl = (exit_price - entry_price) / entry_price - ESTIMATED_ROUND_TRIP_FEE
    algo._rolling_wins.append(1 if pnl > 0 else 0)
    if pnl > 0:
        algo._rolling_win_sizes.append(pnl)
        algo.winning_trades += 1
        algo.consecutive_losses = 0
    else:
        algo._rolling_loss_sizes.append(abs(pnl))
        algo.losing_trades += 1
        algo.consecutive_losses += 1
    algo.total_pnl += pnl
    if not hasattr(algo, 'pnl_by_tag'):
        algo.pnl_by_tag = {}
    if exit_tag not in algo.pnl_by_tag:
        algo.pnl_by_tag[exit_tag] = []
    algo.pnl_by_tag[exit_tag].append(pnl)
    return pnl


def get_open_buy_orders_value(algo):
    """Calculate total value reserved by open buy orders."""
    total_reserved = 0
    for o in algo.Transactions.GetOpenOrders():
        if o.Direction == OrderDirection.Buy:
            if o.Price > 0:
                order_price = o.Price
            elif o.Symbol in algo.Securities:
                order_price = algo.Securities[o.Symbol].Price
                if order_price <= 0:
                    continue
            else:
                continue
            total_reserved += abs(o.Quantity) * order_price
    return total_reserved


def health_check(algo):
    """
    Enhanced health check with improved orphan detection and reverse resync.
    Implements Fix 3: check if ALL orders are stale before skipping orphan check.
    Includes reverse resync (phantom position detection) for live mode.
    """
    if algo.IsWarmingUp:
        return
    
    # Live resync to catch any holdings the event stream might have missed
    resync_holdings_full(algo)
    
    issues = []
    if algo.Portfolio.Cash < 5:
        issues.append(f"Low cash: ${algo.Portfolio.Cash:.2f}")
    
    for symbol in list(algo.entry_prices.keys()):
        open_orders = algo.Transactions.GetOpenOrders(symbol)
        if len(open_orders) > 0:
            # Check if all orders are stuck (older than timeout)
            all_stale = True
            for o in open_orders:
                order_time = normalize_order_time(o.Time)
                if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                    all_stale = False
                    break
            if not all_stale:
                continue  # Has non-stale orders, skip this symbol
            
            # All orders are stale - cancel them
            for o in open_orders:
                try:
                    algo.Transactions.CancelOrder(o.Id)
                    issues.append(f"Canceled stale order: {symbol.Value} (order {o.Id})")
                except Exception as e:
                    algo.Debug(f"Error canceling stale order for {symbol.Value}: {e}")
        
        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)
    
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
            issues.append(f"Untracked position: {kvp.Key.Value}")
    
    if len(algo._session_blacklist) > 50:
        issues.append(f"Large session blacklist: {len(algo._session_blacklist)}")
    
    open_orders = algo.Transactions.GetOpenOrders()
    if len(open_orders) > 0:
        issues.append(f"Open orders: {len(open_orders)}")
    
    if issues:
        algo.Debug("=== HEALTH CHECK ===")
        for issue in issues:
            algo.Debug(f"  ⚠️ {issue}")
    else:
        debug_limited(algo, "Health check: OK")


def resync_holdings_full(algo):
    """Live-only: backfills tracking for holdings missed via OnOrderEvent; detects phantom positions."""
    if algo.IsWarmingUp or not algo.LiveMode:
        return
    
    if not hasattr(algo, '_last_resync_log') or (algo.Time - algo._last_resync_log).total_seconds() > algo.resync_log_interval_seconds:
        algo.Debug(f"RESYNC CHECK: keys={len(list(algo.Portfolio.Keys))} tracked={len(algo.entry_prices)}")
        algo._last_resync_log = algo.Time
    
    # Forward resync: find holdings we're not tracking
    missing = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        if has_non_stale_open_orders(algo, symbol):
            continue
        missing.append(symbol)
    
    if missing:
        algo.Debug(f"RESYNC: {len(missing)} untracked holdings; backfilling.")
        for symbol in missing:
            try:
                if symbol not in algo.Securities:
                    algo.AddCrypto(symbol.Value, Resolution.Minute, Market.Kraken)
                holding = algo.Portfolio[symbol]
                entry = holding.AveragePrice
                algo.entry_prices[symbol] = entry
                algo.highest_prices[symbol] = entry
                algo.entry_times[symbol] = algo.Time
                cur = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                pnl_pct = (cur - entry) / entry if entry > 0 else 0
                algo.Debug(f"RESYNCED: {symbol.Value} entry=${entry:.4f} now=${cur:.4f} {pnl_pct:+.2%}")
            except Exception as e:
                algo.Debug(f"Resync error {symbol.Value}: {e}")
    
    # Reverse resync: detect phantom positions (tracked but broker has none)
    phantoms = []
    for symbol in list(algo.entry_prices.keys()):
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
        if holding is None or not holding.Invested or holding.Quantity == 0:
            open_orders = algo.Transactions.GetOpenOrders(symbol)
            if len(open_orders) > 0:
                all_stuck = all(
                    (algo.Time - normalize_order_time(o.Time)).total_seconds() > algo.live_stale_order_timeout_seconds
                    for o in open_orders
                )
                if not all_stuck:
                    continue
                for o in open_orders:
                    try:
                        algo.Transactions.CancelOrder(o.Id)
                    except Exception as e:
                        algo.Debug(f"Error canceling stuck order {symbol.Value}: {e}")
            phantoms.append(symbol)
    
    if phantoms:
        algo.Debug(f"⚠️ REVERSE RESYNC: {len(phantoms)} phantom positions")
        for symbol in phantoms:
            algo.Debug(f"⚠️ PHANTOM: {symbol.Value} tracked but qty=0, cleaning up")
            cleanup_position(algo, symbol, record_pnl=True)
            if hasattr(algo, 'exit_cooldown_hours') and hasattr(algo, '_exit_cooldowns'):
                algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        persist_state(algo)


def verify_order_fills(algo):
    """Verify submitted orders filled/timed-out; retry once before blacklisting."""
    if algo.IsWarmingUp:
        return
    
    current_time = algo.Time
    symbols_to_remove = []
    
    for symbol in list(algo._retry_pending.keys()):
        cancel_time = algo._retry_pending[symbol]
        if (current_time - cancel_time).total_seconds() >= algo.retry_pending_cooldown_seconds:
            if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                holding = algo.Portfolio[symbol]
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = current_time
                    algo.Debug(f"RETRY SKIPPED (position exists): {symbol.Value}")
                del algo._retry_pending[symbol]
            else:
                algo.Debug(f"RETRY NOW: {symbol.Value}")
                del algo._retry_pending[symbol]
    
    for symbol, order_info in list(algo._submitted_orders.items()):
        order_age_seconds = (current_time - order_info['time']).total_seconds()
        order_id = order_info['order_id']
        
        if order_info.get('is_limit_entry', False):
            timeout = order_info.get('timeout_seconds', 60)
        elif order_info.get('is_limit_exit', False):
            timeout = 90
        else:
            timeout = algo.order_timeout_seconds
        
        if order_age_seconds > algo.order_fill_check_threshold_seconds:
            intent = order_info.get('intent', 'entry')
            try:
                order = algo.Transactions.GetOrderById(order_id)
                if order is not None and order.Status == OrderStatus.Filled:
                    if intent == 'exit':
                        entry = algo.entry_prices.get(symbol, None)
                        if entry:
                            cur = algo.Securities[symbol].Price if symbol in algo.Securities else None
                            if cur is not None and cur > 0:
                                pnl = record_exit_pnl(algo, symbol, entry, cur)
                                algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} pnl={pnl:+.2%}" if pnl is not None else f"⚠️ MISSED EXIT FILL: {symbol.Value} invalid price")
                            else:
                                algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} no price")
                            cleanup_position(algo, symbol)
                        symbols_to_remove.append(symbol)
                        algo._order_retries.pop(order_id, None)
                        continue
                    else:
                        if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                            holding = algo.Portfolio[symbol]
                            entry_price = holding.AveragePrice
                            cur = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                            if symbol not in algo.entry_prices:
                                algo.entry_prices[symbol] = entry_price
                                algo.highest_prices[symbol] = max(cur, entry_price)
                                algo.entry_times[symbol] = order_info['time']
                                algo.daily_trade_count += 1
                                algo.Debug(f"FILL VERIFIED: {symbol.Value} entry=${entry_price:.4f}")
                            symbols_to_remove.append(symbol)
                            algo._order_retries.pop(order_id, None)
                            continue
            except Exception as e:
                algo.Debug(f"Error checking order {symbol.Value}: {e}")
            
            if intent == 'exit':
                holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
                if holding is None or not holding.Invested or holding.Quantity == 0:
                    entry = algo.entry_prices.get(symbol, None)
                    if entry:
                        cur = algo.Securities[symbol].Price if symbol in algo.Securities else None
                        if cur is not None and cur > 0:
                            pnl = record_exit_pnl(algo, symbol, entry, cur)
                            algo.Debug(f"⚠️ PHANTOM EXIT: {symbol.Value} pnl={pnl:+.2%}" if pnl is not None else f"⚠️ PHANTOM EXIT: {symbol.Value} invalid price")
                        else:
                            algo.Debug(f"⚠️ PHANTOM EXIT: {symbol.Value} no price")
                        cleanup_position(algo, symbol)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    continue
        
        if order_age_seconds > timeout:
            retry_count = algo._order_retries.get(order_id, 0)
            if retry_count == 0:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo.Debug(f"ORDER TIMEOUT attempt 1: {symbol.Value}")
                    algo._retry_pending[symbol] = current_time
                    algo._order_retries[order_id] = 1
                    symbols_to_remove.append(symbol)
                except Exception as e:
                    algo.Debug(f"Error canceling {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
            else:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo._session_blacklist.add(symbol.Value)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    algo.Debug(f"ORDER TIMEOUT attempt 2: {symbol.Value} blacklisted")
                except Exception as e:
                    algo.Debug(f"Error canceling {order_id}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
    
    for symbol in symbols_to_remove:
        algo._submitted_orders.pop(symbol, None)


def portfolio_sanity_check(algo):
    """
    Check for portfolio value mismatches between QC and tracked positions.
    Fixed: Use CashBook["USD"].Amount for actual USD cash, not Portfolio.Cash
    (which in Cash account mode includes crypto holdings value).
    Fix 6: Trigger resync_holdings_full on mismatch and log detailed breakdown.
    """
    if algo.IsWarmingUp:
        return
    
    total_qc = algo.Portfolio.TotalPortfolioValue
    
    # Use actual USD cash, NOT Portfolio.Cash (which double-counts in Cash accounts)
    try:
        usd_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        usd_cash = algo.Portfolio.Cash
    
    tracked_value = 0.0
    tracked_pos_count = 0
    
    for sym in list(algo.entry_prices.keys()):
        if sym in algo.Securities:
            price = algo.Securities[sym].Price
            if sym in algo.Portfolio:
                qty = algo.Portfolio[sym].Quantity
                tracked_value += abs(qty) * price
                tracked_pos_count += 1
    
    expected = usd_cash + tracked_value
    
    # Use both percentage and minimum absolute threshold to avoid spam; 1-hour cooldown.
    abs_diff = abs(total_qc - expected)
    if total_qc > 1.0:
        pct_diff = abs_diff / total_qc
        should_warn = pct_diff > algo.portfolio_mismatch_threshold and abs_diff > algo.portfolio_mismatch_min_dollars
        if should_warn:
            if algo._last_mismatch_warning is None or (algo.Time - algo._last_mismatch_warning).total_seconds() >= algo.portfolio_mismatch_cooldown_seconds:
                algo.Debug(f"⚠️ PORTFOLIO MISMATCH: QC=${total_qc:.2f} tracked=${expected:.2f} diff=${abs_diff:.2f} ({pct_diff:.2%})")
                algo.Debug(f"  cash=${usd_cash:.2f} tracked_pos={tracked_pos_count} val=${tracked_value:.2f}")
                resync_holdings_full(algo)
                algo._last_mismatch_warning = algo.Time


def review_performance(algo):
    """Review recent performance and adjust max_positions accordingly."""
    if algo.IsWarmingUp or len(algo.trade_log) < 10:
        return
    
    recent_trades = list(algo.trade_log)[-15:]
    if len(recent_trades) == 0:
        return
    
    recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
    recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
    old_max = algo.max_positions
    
    if recent_win_rate < 0.2 and recent_avg_pnl < -0.05:
        algo.max_positions = 1
        if old_max != 1:
            algo.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
    elif recent_win_rate > 0.35 and recent_avg_pnl > -0.01:
        algo.max_positions = algo.base_max_positions
        if old_max != algo.base_max_positions:
            algo.Debug(f"PERFORMANCE RECOVERY: max_pos={algo.base_max_positions}")


def daily_report(algo):
    """Generate daily report with portfolio status and position details."""
    if algo.IsWarmingUp:
        return

    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    avg = algo.total_pnl / total if total > 0 else 0
    algo.Debug(f"=== DAILY {algo.Time.date()} ===")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime} {algo.market_breadth:.0%}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    if algo._session_blacklist:
        algo.Debug(f"Blacklist: {len(algo._session_blacklist)}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            algo.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
    persist_state(algo)


def execute_buy(algo, symbol, quantity):
    """Place a market buy order for immediate fill."""
    algo.MarketOrder(symbol, quantity, tag="MG Market Buy")


def execute_sell(algo, symbol, quantity):
    """Place a market sell order for immediate fill."""
    algo.MarketOrder(symbol, -quantity, tag="MG Market Sell")
