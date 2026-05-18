# region imports
from AlgorithmImports import *

# endregion

# QuantConnect: add alongside main.py (no multiple inheritance — module functions only).


def parameter_was_set(algo, name):
    if not getattr(algo, "_use_qc_ui_parameters", True):
        return False
    raw = algo.GetParameter(name)
    return raw is not None and str(raw).strip() != ""


def apply_production_safe_profile(algo):
    algo.Debug("PRODUCTION_SAFE: EOD, rails, bands, drawdown guard.")
    algo.maximize_backtest_equity = False
    algo.use_eod_next_bar_execution = True
    algo.use_rebalance_bands = True
    algo.min_weight_change_to_trade = max(algo.min_weight_change_to_trade, 0.02)
    if algo.max_daily_weight_change <= 0.0:
        algo.max_daily_weight_change = 0.15
    if algo.max_days_without_rebalance <= 0:
        algo.max_days_without_rebalance = 5
    algo.vol_etp_confirm_days = max(algo.vol_etp_confirm_days, 1)
    if algo.max_consecutive_vol_etp_days <= 0:
        algo.max_consecutive_vol_etp_days = 5
    algo.gap_cooldown_days = max(algo.gap_cooldown_days, 3)
    algo.use_drawdown_guard = True
    algo.use_tiered_drawdown = True
    algo.use_vol_targeting = True
    algo.use_regime_vol_target = True


def apply_maximize_backtest_equity_profile(algo):
    """~60x in-sample baseline: same-bar, rails off, vol targets 0.58/0.68/0.48."""
    algo.Debug("MAXIMIZE_BACKTEST_EQUITY: same-bar, rails off, vol targets 0.58/0.68/0.48.")
    algo.use_eod_next_bar_execution = False
    algo.use_rebalance_bands = False
    algo.min_weight_change_to_trade = 0.0
    algo.max_daily_weight_change = 0.0
    algo.max_days_without_rebalance = 0
    algo.vol_etp_confirm_days = 0
    algo.max_consecutive_vol_etp_days = 0
    algo.max_consecutive_uvxy_days = 0
    algo.max_consecutive_svxy_days = 0
    algo.gap_cooldown_days = 0
    algo.gap_cooldown_pct = -0.12
    algo.use_drawdown_guard = False
    algo._drawdown_guard_active = False
    algo.use_tiered_drawdown = False
    algo.use_vol_targeting = True
    algo.use_regime_vol_target = True
    algo.target_ann_vol = 0.58
    algo.target_ann_vol_bull = 0.68
    algo.target_ann_vol_bear = 0.48
    algo.min_hold_days = 0
    algo.max_gross_exposure = 1.0
    algo.max_position_weight = 1.0
    algo.margin_safety_pct = 1.0
    # Skip tiny vol-scaler weight tweaks (0.0 caused ~2k orders / ~32x vs ~900 / ~60x).
    algo.min_rebalance_weight_delta = 0.03
    algo._bull_sleeve_mode = False
    algo._vol_target_off_in_bull = False
    algo.th_rsi_qqq_bull_uvxy = 90.0
    algo.th_rsi_spy_bull_uvxy = 89.0
    algo.th_rsi_uvxy_elevated = 82.0
    algo.th_rsi_uvxy_extreme = 93.0
    algo.th_rsi_soxl_bull = 34.0
    if algo.maximize_include_svxy:
        algo.use_svxy_calm = True
    if algo.maximize_disable_vol_target:
        algo.use_vol_targeting = False


def apply_maximize_60x_baseline(algo):
    """Single entry for hardcoded QC upload — no parameter panel."""
    algo.maximize_backtest_equity = True
    algo.maximize_include_svxy = False
    algo.maximize_disable_vol_target = False
    algo.lift_120x_research = False
    algo.production_safe_defaults = False
    apply_maximize_backtest_equity_profile(algo)


def apply_lift_120x_research_bundle(algo):
    algo.Debug(
        "LIFT_120X: bull_sleeve, vol_off_in_bull, bull_gross~1.15, min_hold=2, SOXL RSI 32."
    )
    algo.maximize_backtest_equity = True
    algo._bull_sleeve_mode = True
    algo._vol_target_off_in_bull = True
    algo.disable_bull_uvxy = True
    algo.use_svxy_calm = False
    algo.maximize_include_svxy = False
    algo._use_vix_gate = False
    algo._prefer_soxl_on_outperform = False
    algo.bull_uvxy_require_both = False
    algo._bull_gross_cap = max(
        1.0, min(2.0, algo._float_parameter("bull_gross_cap", 1.15))
    )
    algo.max_gross_exposure = max(
        1.0,
        min(2.0, algo._float_parameter("max_gross_exposure", algo._bull_gross_cap)),
    )
    algo.max_position_weight = max(
        0.01,
        min(
            algo.max_gross_exposure,
            algo._float_parameter("max_position_weight", algo.max_gross_exposure),
        ),
    )
    algo.min_hold_days = max(0, algo._int_parameter("min_hold_days", 2))
    algo.th_rsi_soxl_bull = 32.0
    if not parameter_was_set(algo, "margin_safety_pct"):
        algo.margin_safety_pct = 0.98
    algo.min_rebalance_weight_delta = max(
        0.0, min(0.25, algo._float_parameter("min_rebalance_weight_delta", 0.03))
    )


def reload_user_overrides_after_profile(algo):
    if not getattr(algo, "_use_qc_ui_parameters", True):
        return
    if parameter_was_set(algo, "max_gross_exposure"):
        algo.max_gross_exposure = max(
            1.0,
            min(2.0, algo._float_parameter("max_gross_exposure", 1.0)),
        )
    if parameter_was_set(algo, "max_position_weight"):
        algo.max_position_weight = max(
            0.01,
            min(
                algo.max_gross_exposure,
                algo._float_parameter("max_position_weight", 1.0),
            ),
        )
    if parameter_was_set(algo, "target_ann_vol"):
        algo.target_ann_vol = max(
            0.01, algo._float_parameter("target_ann_vol", algo.target_ann_vol)
        )
    if parameter_was_set(algo, "target_ann_vol_bull"):
        algo.target_ann_vol_bull = max(
            0.01,
            algo._float_parameter("target_ann_vol_bull", algo.target_ann_vol_bull),
        )
    if parameter_was_set(algo, "target_ann_vol_bear"):
        algo.target_ann_vol_bear = max(
            0.01,
            algo._float_parameter("target_ann_vol_bear", algo.target_ann_vol_bear),
        )
    if parameter_was_set(algo, "th_rsi_qqq_bull_uvxy"):
        algo.th_rsi_qqq_bull_uvxy = algo._float_parameter(
            "th_rsi_qqq_bull_uvxy", algo.th_rsi_qqq_bull_uvxy
        )
    if parameter_was_set(algo, "th_rsi_spy_bull_uvxy"):
        algo.th_rsi_spy_bull_uvxy = algo._float_parameter(
            "th_rsi_spy_bull_uvxy", algo.th_rsi_spy_bull_uvxy
        )
    if parameter_was_set(algo, "th_rsi_soxl_bull"):
        algo.th_rsi_soxl_bull = algo._float_parameter(
            "th_rsi_soxl_bull", algo.th_rsi_soxl_bull
        )
    if parameter_was_set(algo, "disable_bull_uvxy"):
        algo.disable_bull_uvxy = algo._bool_parameter("disable_bull_uvxy", False)
    if parameter_was_set(algo, "bull_uvxy_require_both"):
        algo.bull_uvxy_require_both = algo._bool_parameter(
            "bull_uvxy_require_both", False
        )
    if parameter_was_set(algo, "maximize_disable_vol_target"):
        if algo._bool_parameter("maximize_disable_vol_target", False):
            algo.use_vol_targeting = False
    if parameter_was_set(algo, "use_vol_targeting"):
        algo.use_vol_targeting = algo._bool_parameter(
            "use_vol_targeting", algo.use_vol_targeting
        )
    if parameter_was_set(algo, "vix_min_bull_uvxy"):
        algo.vix_min_bull_uvxy = max(
            10.0, algo._float_parameter("vix_min_bull_uvxy", algo.vix_min_bull_uvxy)
        )
    if parameter_was_set(algo, "soxl_outperform_days"):
        algo.soxl_outperform_days = max(
            2, algo._int_parameter("soxl_outperform_days", algo.soxl_outperform_days)
        )
    if parameter_was_set(algo, "vol_lookback"):
        algo.vol_lookback = max(
            5, algo._int_parameter("vol_lookback", algo.vol_lookback)
        )
    if parameter_was_set(algo, "vol_anchor_ticker"):
        raw = algo.GetParameter("vol_anchor_ticker")
        if raw is not None and str(raw).strip() != "":
            algo.vol_anchor_ticker = str(raw).strip().upper()
