# QuantConnect: upload with main.py for IB paper/live ONLY (LiveMode). Ignored in backtests.


def wire_live_paper_schedule(algo):
    """One EOD decision per day — avoids OnData firing twice and canceling orders."""
    if not getattr(algo, "LiveMode", False) or algo.use_eod_next_bar_execution:
        return
    spy = algo.symbols.get("SPY")
    if spy is None:
        return
    algo.Schedule.On(
        algo.DateRules.EveryDay(spy),
        algo.TimeRules.AfterMarketClose(spy, 1),
        algo._live_flush_eod_trade,
    )
    algo.Debug("LIVE_PAPER: coalesce signals in OnData; trade once AfterMarketClose.")


def queue_live_signal(algo, target_ticker, weight, raw_signal):
    algo._live_day_signal = (target_ticker, float(weight), raw_signal)


def flush_live_eod_trade(algo):
    sig = getattr(algo, "_live_day_signal", None)
    if sig is None:
        return
    if getattr(algo, "_live_day_flushed_date", None) == algo.Time.date():
        return
    t, w, raw = sig
    algo.Debug(
        "%s LIVE_EOD_FLUSH %s w=%.3f raw=%s"
        % (algo.Time.strftime("%Y-%m-%d"), t, w, raw)
    )
    if algo._use_margin_safe_rotation():
        run_samebar_trade(algo, t, w, raw)
    else:
        sym = algo.symbols[t]
        w_exec = algo._set_holdings_buying_power_clamped(sym, w, True)
        algo._last_target_ticker = t
        algo._last_trade_time = algo.Time
        algo._last_executed_weight = w_exec
        algo.Debug(
            "%s samebar target=%s w=%.3f raw=%s"
            % (algo.Time.strftime("%Y-%m-%d"), t, w_exec, raw)
        )
    algo._live_day_flushed_date = algo.Time.date()


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
