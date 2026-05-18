# region imports
from AlgorithmImports import *

# endregion

# QuantConnect: add this file alongside main.py (algorithm class inherits CSRProfileMixin).


class CSRProfileMixin:
    def _parameter_was_set(self, name):
        raw = self.GetParameter(name)
        return raw is not None and str(raw).strip() != ""

    def _apply_production_safe_profile(self):
        self.Debug("PRODUCTION_SAFE: EOD, rails, bands, drawdown guard.")
        self.maximize_backtest_equity = False
        self.use_eod_next_bar_execution = True
        self.use_rebalance_bands = True
        self.min_weight_change_to_trade = max(self.min_weight_change_to_trade, 0.02)
        if self.max_daily_weight_change <= 0.0:
            self.max_daily_weight_change = 0.15
        if self.max_days_without_rebalance <= 0:
            self.max_days_without_rebalance = 5
        self.vol_etp_confirm_days = max(self.vol_etp_confirm_days, 1)
        if self.max_consecutive_vol_etp_days <= 0:
            self.max_consecutive_vol_etp_days = 5
        self.gap_cooldown_days = max(self.gap_cooldown_days, 3)
        self.use_drawdown_guard = True
        self.use_tiered_drawdown = True
        self.use_vol_targeting = True
        self.use_regime_vol_target = True

    def _apply_maximize_backtest_equity_profile(self):
        self.Debug("MAXIMIZE_BACKTEST_EQUITY: same-bar, rails off, vol targets 0.58/0.68/0.48.")
        self.use_eod_next_bar_execution = False
        self.use_rebalance_bands = False
        self.min_weight_change_to_trade = 0.0
        self.max_daily_weight_change = 0.0
        self.max_days_without_rebalance = 0
        self.vol_etp_confirm_days = 0
        self.max_consecutive_vol_etp_days = 0
        self.gap_cooldown_days = 0
        self.use_drawdown_guard = False
        self._drawdown_guard_active = False
        self.use_tiered_drawdown = False
        self.use_vol_targeting = True
        self.use_regime_vol_target = True
        self.target_ann_vol = 0.58
        self.target_ann_vol_bull = 0.68
        self.target_ann_vol_bear = 0.48
        self.min_hold_days = 0
        self.th_rsi_qqq_bull_uvxy = 90.0
        self.th_rsi_spy_bull_uvxy = 89.0
        self.th_rsi_uvxy_elevated = 82.0
        self.th_rsi_uvxy_extreme = 93.0
        self.th_rsi_soxl_bull = 34.0
        if self.maximize_include_svxy:
            self.use_svxy_calm = True
        if self.maximize_disable_vol_target:
            self.use_vol_targeting = False

    def _apply_lift_120x_research_bundle(self):
        self.Debug(
            "LIFT_120X: bull_sleeve, vol_off_in_bull, bull_gross~1.15, min_hold=2, SOXL RSI 32."
        )
        self.maximize_backtest_equity = True
        self._bull_sleeve_mode = True
        self._vol_target_off_in_bull = True
        self.disable_bull_uvxy = True
        self.use_svxy_calm = False
        self.maximize_include_svxy = False
        self._use_vix_gate = False
        self._prefer_soxl_on_outperform = False
        self.bull_uvxy_require_both = False
        self._bull_gross_cap = max(
            1.0, min(2.0, self._float_parameter("bull_gross_cap", 1.15))
        )
        self.max_gross_exposure = max(
            1.0,
            min(2.0, self._float_parameter("max_gross_exposure", self._bull_gross_cap)),
        )
        self.max_position_weight = max(
            0.01,
            min(
                self.max_gross_exposure,
                self._float_parameter("max_position_weight", self.max_gross_exposure),
            ),
        )
        self.min_hold_days = max(0, self._int_parameter("min_hold_days", 2))
        self.th_rsi_soxl_bull = 32.0
        if not self._parameter_was_set("margin_safety_pct"):
            self.margin_safety_pct = 0.99

    def _reload_user_overrides_after_profile(self):
        if self._parameter_was_set("max_gross_exposure"):
            self.max_gross_exposure = max(
                1.0,
                min(2.0, self._float_parameter("max_gross_exposure", 1.0)),
            )
        if self._parameter_was_set("max_position_weight"):
            self.max_position_weight = max(
                0.01,
                min(
                    self.max_gross_exposure,
                    self._float_parameter("max_position_weight", 1.0),
                ),
            )
        if self._parameter_was_set("target_ann_vol"):
            self.target_ann_vol = max(
                0.01, self._float_parameter("target_ann_vol", self.target_ann_vol)
            )
        if self._parameter_was_set("target_ann_vol_bull"):
            self.target_ann_vol_bull = max(
                0.01,
                self._float_parameter("target_ann_vol_bull", self.target_ann_vol_bull),
            )
        if self._parameter_was_set("target_ann_vol_bear"):
            self.target_ann_vol_bear = max(
                0.01,
                self._float_parameter("target_ann_vol_bear", self.target_ann_vol_bear),
            )
        if self._parameter_was_set("th_rsi_qqq_bull_uvxy"):
            self.th_rsi_qqq_bull_uvxy = self._float_parameter(
                "th_rsi_qqq_bull_uvxy", self.th_rsi_qqq_bull_uvxy
            )
        if self._parameter_was_set("th_rsi_spy_bull_uvxy"):
            self.th_rsi_spy_bull_uvxy = self._float_parameter(
                "th_rsi_spy_bull_uvxy", self.th_rsi_spy_bull_uvxy
            )
        if self._parameter_was_set("th_rsi_soxl_bull"):
            self.th_rsi_soxl_bull = self._float_parameter(
                "th_rsi_soxl_bull", self.th_rsi_soxl_bull
            )
        if self._parameter_was_set("disable_bull_uvxy"):
            self.disable_bull_uvxy = self._bool_parameter("disable_bull_uvxy", False)
        if self._parameter_was_set("bull_uvxy_require_both"):
            self.bull_uvxy_require_both = self._bool_parameter(
                "bull_uvxy_require_both", False
            )
        if self._parameter_was_set("maximize_disable_vol_target"):
            if self._bool_parameter("maximize_disable_vol_target", False):
                self.use_vol_targeting = False
        if self._parameter_was_set("use_vol_targeting"):
            self.use_vol_targeting = self._bool_parameter(
                "use_vol_targeting", self.use_vol_targeting
            )
        if self._parameter_was_set("vix_min_bull_uvxy"):
            self.vix_min_bull_uvxy = max(
                10.0, self._float_parameter("vix_min_bull_uvxy", self.vix_min_bull_uvxy)
            )
        if self._parameter_was_set("soxl_outperform_days"):
            self.soxl_outperform_days = max(
                2, self._int_parameter("soxl_outperform_days", self.soxl_outperform_days)
            )
        if self._parameter_was_set("vol_lookback"):
            self.vol_lookback = max(
                5, self._int_parameter("vol_lookback", self.vol_lookback)
            )
        if self._parameter_was_set("vol_anchor_ticker"):
            raw = self.GetParameter("vol_anchor_ticker")
            if raw is not None and str(raw).strip() != "":
                self.vol_anchor_ticker = str(raw).strip().upper()
