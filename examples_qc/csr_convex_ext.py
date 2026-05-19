# DEPRECATED May 2026: ~$628k vs ml_overlay ~$4.4M. Use maximize or ml_overlay.


def apply_convex_profile(algo):
    """
    Maximize RSI / same-bar signals with crash stack and moderate vol targeting.
    Not institutional (no ladder, no bear-sleeve suppress).
    """
    algo.Debug(
        "CONVEX (DEPRECATED): failed vs ml_overlay — do not use for new backtests."
    )
    algo.convex_preset_active = True
    algo.maximize_backtest_equity = True
    algo._apply_maximize_backtest_equity_profile()

    algo.institutional_suppress_bear_leverage = False
    algo.institutional_soft_drawdown = False
    algo.use_probabilistic_regime = False
    algo.use_bull_leverage_ladder = False
    algo.scale_weight_by_regime_score = False

    algo.use_drawdown_guard = True
    algo.max_drawdown_pct = 0.32
    algo.drawdown_release_frac = 0.75
    algo.use_tiered_drawdown = True
    algo.tier1_drawdown = 0.10
    algo.tier1_mult = 0.88
    algo.tier2_drawdown = 0.18
    algo.tier2_mult = 0.65

    algo.gap_cooldown_days = 3
    algo.gap_cooldown_pct = -0.10

    algo.use_vix_delever = True
    algo.vix_delever_ratio = 1.18
    algo.vix_delever_mult = 0.58

    algo.use_vol_targeting = True
    algo.use_regime_vol_target = True
    algo.target_ann_vol = 0.28
    algo.target_ann_vol_bull = 0.32
    algo.target_ann_vol_bear = 0.22

    algo.use_rebalance_bands = True
    algo.min_weight_change_to_trade = max(algo.min_weight_change_to_trade, 0.02)
    if algo.max_daily_weight_change <= 0.0:
        algo.max_daily_weight_change = 0.25
    if algo.max_days_without_rebalance <= 0:
        algo.max_days_without_rebalance = 7

    algo.min_hold_days = max(algo.min_hold_days, 1)
    algo.max_gross_exposure = min(float(algo.max_gross_exposure), 1.0)
    algo.max_position_weight = min(float(algo.max_position_weight), 1.0)
