# QuantConnect: optional second file with main.py for ACTIVE_BASELINE=institutional only.


def apply_institutional_profile(algo):
    """Lower-DD preset: EOD, ~25% vol, ladder, no bear 3x, soft drawdown."""
    algo.Debug("INSTITUTIONAL v2: EOD, vol~25%, ladder, no bear 3x, soft DD.")
    algo.maximize_backtest_equity = False
    algo.production_safe_defaults = True
    algo._apply_production_safe_profile()
    algo.regime_mode = "spy_and_qqq"
    algo.use_probabilistic_regime = True
    algo.regime_score_min_bull = 0.50
    algo.regime_hysteresis_days = 1
    algo.use_bull_leverage_ladder = True
    algo.bull_ladder_tqqq_min = 0.60
    algo.bull_ladder_qld_min = 0.38
    algo.scale_weight_by_regime_score = False
    algo.use_rsp_breadth_proxy = True
    algo.use_vix_delever = True
    algo.vix_delever_ratio = 1.25
    algo.vix_delever_mult = 0.65
    algo.bull_tqqq_momentum_days = 5
    algo.min_hold_days = 2
    algo.disable_bull_uvxy = True
    algo.institutional_suppress_bear_leverage = True
    algo.institutional_soft_drawdown = True
    algo.target_ann_vol = 0.25
    algo.target_ann_vol_bull = 0.30
    algo.target_ann_vol_bear = 0.18
    algo.max_drawdown_pct = 0.32
    algo.drawdown_release_frac = 0.75
    algo.tier1_drawdown = 0.12
    algo.tier1_mult = 0.90
    algo.tier2_drawdown = 0.22
    algo.tier2_mult = 0.70
    algo.max_gross_exposure = 1.0
    algo.max_position_weight = 1.0
