# QuantConnect: upload with main.py when ACTIVE_BASELINE=maximize_hold

_HOLD_TICKERS = frozenset({"TQQQ", "SOXL", "QLD", "TECL", "SPXL"})


def apply_maximize_hold_profile(algo):
    """Maximize signals + same-bar; hold leveraged winners (trail stop, trend filter)."""
    algo.maximize_hold_active = True
    algo.hold_winners_enabled = True
    algo.hold_trail_pct = 0.08
    algo.hold_min_gain = 0.15
    algo._hold_entry_price = {}
    algo._hold_peak_pv = {}
    algo.Debug(
        "MAXIMIZE_HOLD: maximize + hold winners (trail=%.0f%%, min_gain=%.0f%%)."
        % (algo.hold_trail_pct * 100, algo.hold_min_gain * 100)
    )
    algo.maximize_backtest_equity = True
    algo._apply_maximize_backtest_equity_profile()


def on_position_opened(algo, ticker):
    if not getattr(algo, "hold_winners_enabled", False) or ticker not in algo.symbols:
        return
    if ticker not in _HOLD_TICKERS:
        return
    sym = algo.symbols[ticker]
    px = float(algo.Securities[sym].Price)
    if px > 0:
        algo._hold_entry_price[ticker] = px
    algo._hold_peak_pv[ticker] = float(algo.Portfolio.TotalPortfolioValue)


def _clear_hold_track(algo, ticker):
    algo._hold_entry_price.pop(ticker, None)
    algo._hold_peak_pv.pop(ticker, None)


def min_hold_blocks_switch_target(algo, proposed_ticker):
    if getattr(algo, "hold_winners_enabled", False) and hold_blocks_switch(
        algo, proposed_ticker
    ):
        return True
    if algo.min_hold_days <= 0 or algo._last_trade_time is None:
        return False
    if algo.use_drawdown_guard and algo._drawdown_guard_active:
        return False
    if proposed_ticker == algo._last_target_ticker:
        return False
    return (algo.Time - algo._last_trade_time).days < algo.min_hold_days


def after_trade_open(algo, ticker, prev_ticker):
    if ticker != prev_ticker and getattr(algo, "hold_winners_enabled", False):
        on_position_opened(algo, ticker)


def hold_blocks_switch(algo, proposed_ticker):
    """Return True to keep current position (block switch to proposed_ticker)."""
    if not getattr(algo, "hold_winners_enabled", False):
        return False
    if algo.use_drawdown_guard and algo._drawdown_guard_active:
        return False
    cur = algo._last_target_ticker
    if not cur or proposed_ticker == cur:
        return False
    if cur not in _HOLD_TICKERS or cur not in algo.symbols:
        return False

    sym = algo.symbols[cur]
    entry = algo._hold_entry_price.get(cur)
    if entry is None or entry <= 0:
        on_position_opened(algo, cur)
        entry = algo._hold_entry_price.get(cur)
    if not entry or entry <= 0:
        return False

    px = float(algo.Securities[sym].Price)
    if px <= 0:
        return False
    gain = (px / entry) - 1.0
    pv = float(algo.Portfolio.TotalPortfolioValue)
    peak = algo._hold_peak_pv.get(cur, pv)
    if pv > peak:
        algo._hold_peak_pv[cur] = pv
        peak = pv
    dd_peak = 1.0 - (pv / peak) if peak > 0 else 0.0
    trail = float(getattr(algo, "hold_trail_pct", 0.08))
    if dd_peak >= trail:
        _clear_hold_track(algo, cur)
        return False

    min_gain = float(getattr(algo, "hold_min_gain", 0.15))
    if gain < min_gain:
        return True

    if cur == "TQQQ" and "TQQQ_SMA20" in algo.indicators:
        sma = algo.indicators["TQQQ_SMA20"].Current.Value
        if sma > 0 and px > sma:
            return True
    elif cur == "SOXL" and "SOXL_SMA20" in algo.indicators:
        sma = algo.indicators["SOXL_SMA20"].Current.Value
        if sma > 0 and px > sma:
            return True

    _clear_hold_track(algo, cur)
    return False
