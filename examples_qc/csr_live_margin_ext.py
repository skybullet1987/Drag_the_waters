# QuantConnect: upload with main.py for IB paper/live ONLY (LiveMode). Ignored in backtests.


def _any_invested(algo):
    for h in algo.Portfolio.Values:
        if h.Invested:
            return True
    return False


def run_samebar_trade(algo, target_ticker, weight, raw_signal):
    """
    Paper/live: on ticker change, liquidate first; buy next pipeline tick when flat.
    Returns True if this call fully handled trading (caller should return).
    """
    if float(getattr(algo, "margin_safety_pct", 1.0)) > 0.95:
        algo.margin_safety_pct = 0.95
    defer = getattr(algo, "_defer_buy", None)
    if defer is not None:
        t, w, _raw = defer
        if _any_invested(algo):
            return True
        sym = algo.symbols[t]
        w_exec = algo._set_holdings_buying_power_clamped(sym, w, True)
        algo._defer_buy = None
        algo._last_target_ticker = t
        algo._last_trade_time = algo.Time
        algo._last_executed_weight = w_exec
        algo.Debug(
            "%s ROTATE_BUY %s w=%.3f (deferred)"
            % (algo.Time.strftime("%Y-%m-%d"), t, w_exec)
        )
        return True

    last = getattr(algo, "_last_target_ticker", None)
    if last and last != target_ticker and _any_invested(algo):
        algo.Liquidate()
        algo._defer_buy = (target_ticker, float(weight), raw_signal)
        algo.Debug(
            "%s ROTATE_DEFER liquidated; pending buy %s w=%.3f"
            % (algo.Time.strftime("%Y-%m-%d"), target_ticker, float(weight))
        )
        return True

    sym = algo.symbols[target_ticker]
    w_exec = algo._set_holdings_buying_power_clamped(sym, weight, True)
    algo._last_target_ticker = target_ticker
    algo._last_trade_time = algo.Time
    algo._last_executed_weight = w_exec
    algo.Debug(
        "%s samebar target=%s w=%.3f raw=%s"
        % (algo.Time.strftime("%Y-%m-%d"), target_ticker, w_exec, raw_signal)
    )
    return True
