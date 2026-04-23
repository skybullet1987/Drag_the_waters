# region imports
from AlgorithmImports import *
from execution import *
from collections import deque
# endregion


def on_order_event(algo, event):
    try:
        symbol = event.Symbol
        algo.Debug(f"ORDER: {symbol.Value} {event.Status} {event.Direction} qty={event.FillQuantity or event.Quantity} price={event.FillPrice} id={event.OrderId}")
        if event.Status == OrderStatus.Submitted:
            if symbol not in algo._pending_orders:
                algo._pending_orders[symbol] = 0
            intended_qty = abs(event.Quantity) if event.Quantity != 0 else abs(event.FillQuantity)
            algo._pending_orders[symbol] += intended_qty
            if symbol not in algo._submitted_orders:
                has_position = symbol in algo.Portfolio and algo.Portfolio[symbol].Invested
                if event.Direction == OrderDirection.Sell and has_position:
                    inferred_intent = 'exit'
                elif event.Direction == OrderDirection.Buy and not has_position:
                    inferred_intent = 'entry'
                else:
                    inferred_intent = 'entry' if event.Direction == OrderDirection.Buy else 'exit'
                algo._submitted_orders[symbol] = {
                    'order_id': event.OrderId,
                    'time': algo.Time,
                    'quantity': event.Quantity,
                    'intent': inferred_intent
                }
            else:
                algo._submitted_orders[symbol]['order_id'] = event.OrderId
        elif event.Status == OrderStatus.PartiallyFilled:
            if symbol in algo._pending_orders:
                algo._pending_orders[symbol] -= abs(event.FillQuantity)
                if algo._pending_orders[symbol] <= 0:
                    algo._pending_orders.pop(symbol, None)
            intended_qty = abs(event.Quantity)
            filled_qty = abs(event.FillQuantity)
            fill_pct = filled_qty / intended_qty if intended_qty > 0 else 0
            if fill_pct < 0.20:
                algo.Debug(f"Partial fill {fill_pct:.0%} on {symbol.Value} — letting it ride to stale-order timeout")
            if event.Direction == OrderDirection.Buy:
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = event.FillPrice
                    algo.highest_prices[symbol] = event.FillPrice
                    algo.entry_times[symbol] = algo.Time
            elif event.Direction == OrderDirection.Sell and fill_pct < 1.0:
                if not hasattr(algo, '_partial_sell_symbols'):
                    algo._partial_sell_symbols = set()
                algo._partial_sell_symbols.add(symbol)
                algo.Debug(f"PARTIAL EXIT: {symbol.Value} | {fill_pct:.1%} exited")
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Filled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Buy:
                algo.entry_prices[symbol] = event.FillPrice
                algo.highest_prices[symbol] = event.FillPrice
                algo.entry_times[symbol] = algo.Time
                algo.daily_trade_count += 1
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto['volume']) >= 1:
                    algo.entry_volumes[symbol] = crypto['volume'][-1]
                algo.rsi_peaked_overbought.pop(symbol, None)
            else:
                if symbol in algo._partial_sell_symbols:
                    algo._partial_sell_symbols.discard(symbol)
                else:
                    order = algo.Transactions.GetOrderById(event.OrderId)
                    exit_tag = order.Tag if order and order.Tag else "Unknown"
                    entry = algo.entry_prices.get(symbol, None)
                    if entry is None:
                        entry = event.FillPrice
                        algo.Debug(f"⚠️ WARNING: Missing entry price for {symbol.Value} sell, using fill price")
                    # Deduct estimated round-trip fee so Kelly/circuit-breakers
                    # operate on net-of-fee returns (same constant as execution.py).
                    pnl = (event.FillPrice - entry) / entry - ESTIMATED_ROUND_TRIP_FEE if entry > 0 else 0
                    algo._rolling_wins.append(1 if pnl > 0 else 0)
                    algo._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                    if pnl > 0:
                        algo._rolling_win_sizes.append(pnl)
                        algo.winning_trades += 1
                        algo.consecutive_losses = 0
                    else:
                        algo._rolling_loss_sizes.append(abs(pnl))
                        algo.losing_trades += 1
                        algo.consecutive_losses += 1
                    algo.total_pnl += pnl
                    # Track per-symbol performance for penalty logic
                    sym_val = symbol.Value
                    if not hasattr(algo, '_symbol_performance'):
                        algo._symbol_performance = {}
                    if sym_val not in algo._symbol_performance:
                        algo._symbol_performance[sym_val] = deque(maxlen=10)
                    algo._symbol_performance[sym_val].append(pnl)
                    if not hasattr(algo, 'pnl_by_tag'):
                        algo.pnl_by_tag = {}
                    if exit_tag not in algo.pnl_by_tag:
                        algo.pnl_by_tag[exit_tag] = []
                    algo.pnl_by_tag[exit_tag].append(pnl)
                    if len(algo.pnl_by_tag[exit_tag]) > 200:
                        algo.pnl_by_tag[exit_tag] = algo.pnl_by_tag[exit_tag][-200:]
                    algo.trade_log.append({
                        'time': algo.Time,
                        'symbol': symbol.Value,
                        'pnl_pct': pnl,
                        'exit_reason': exit_tag,
                    })
                    if len(algo._recent_trade_outcomes) >= 16:
                        recent_wr = sum(algo._recent_trade_outcomes) / len(algo._recent_trade_outcomes)
                        if recent_wr < 0.15:
                            algo._cash_mode_until = algo.Time + timedelta(minutes=45)
                            algo.Debug(f"⚠️ CASH MODE: WR={recent_wr:.0%} over {len(algo._recent_trade_outcomes)} trades. Pausing 45min.")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_attempts.pop(symbol, None)
                    algo._failed_exit_counts.pop(symbol, None)
            slip_log(algo, symbol, event.Direction, event.FillPrice)
        elif event.Status == OrderStatus.Canceled:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell and symbol not in algo.entry_prices:
                if is_invested_not_dust(algo, symbol):
                    holding = algo.Portfolio[symbol]
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = algo.Time
                    algo.Debug(f"RE-TRACKED after cancel: {symbol.Value}")
        elif event.Status == OrderStatus.Invalid:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            algo._order_retries.pop(event.OrderId, None)
            if event.Direction == OrderDirection.Sell:
                price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
                min_notional = get_min_notional_usd(algo, symbol)
                if price > 0 and symbol in algo.Portfolio and abs(algo.Portfolio[symbol].Quantity) * price < min_notional:
                    algo.Debug(f"DUST CLEANUP on invalid sell: {symbol.Value} — releasing tracking")
                    cleanup_position(algo, symbol)
                    algo._failed_exit_counts.pop(symbol, None)
                else:
                    fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
                    algo._failed_exit_counts[symbol] = fail_count
                    algo.Debug(f"Invalid sell #{fail_count}: {symbol.Value}")
                    if fail_count >= 3:
                        algo.Debug(f"FORCE CLEANUP: {symbol.Value} after {fail_count} failed exits — releasing tracking")
                        cleanup_position(algo, symbol)
                        algo._failed_exit_counts.pop(symbol, None)
                    elif symbol not in algo.entry_prices:
                        if is_invested_not_dust(algo, symbol):
                            holding = algo.Portfolio[symbol]
                            algo.entry_prices[symbol] = holding.AveragePrice
                            algo.highest_prices[symbol] = holding.AveragePrice
                            algo.entry_times[symbol] = algo.Time
                            algo.Debug(f"RE-TRACKED after invalid: {symbol.Value}")
            algo._session_blacklist.add(symbol.Value)
    except Exception as e:
        algo.Debug(f"OnOrderEvent error: {e}")
    if algo.LiveMode:
        persist_state(algo)
