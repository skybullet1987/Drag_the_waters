# DEPRECATED: maximize_plus and aggressive_120x underperformed maximize/ml_overlay.
# QuantConnect: optional with main.py for ACTIVE_BASELINE aggressive_120x | maximize_plus.


def apply_aggressive_120x_baseline(algo):
    """Maximize signals + in-repo aggressive_120x bundle (hot vol, gross~1.35)."""
    algo.aggressive_preset_active = True
    algo.Debug(
        "AGGRESSIVE_120X (DEPRECATED): maximize + hot vol 0.75/0.85/0.55, gross~1.35, SOXL/UVXY RSI."
    )
    algo.maximize_backtest_equity = True
    algo._apply_maximize_backtest_equity_profile()
    algo.aggressive_120x_research = True
    algo._apply_aggressive_120x_research_bundle()


def apply_maximize_plus_baseline(algo):
    """Moderate step above maximize before full 120x (gross 1.15, hotter vol, rails still off)."""
    algo.maximize_plus_active = True
    algo.Debug(
        "MAXIMIZE_PLUS (DEPRECATED): maximize signals, gross 1.15, vol 0.65/0.78/0.52 (between max and 120x)."
    )
    algo.maximize_backtest_equity = True
    algo._apply_maximize_backtest_equity_profile()
    algo.max_gross_exposure = 1.15
    algo.max_position_weight = 1.15
    algo.target_ann_vol = 0.65
    algo.target_ann_vol_bull = 0.78
    algo.target_ann_vol_bear = 0.52
