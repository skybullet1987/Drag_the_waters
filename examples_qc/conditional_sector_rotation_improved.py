# region imports
from AlgorithmImports import *
from datetime import datetime

# endregion

# Conditional sector rotation (QC/IB). Each .py file must stay under 63,000 bytes.
# maximize_hold → +csr_hold_ext.py. See QUANTCONNECT_DEPLOY.txt.
# ACTIVE_BASELINE: maximize | ml_overlay | ib_paper | ml_overlay_diversified

USE_QC_UI_PARAMETERS = False
ACTIVE_BASELINE = "ib_paper"  # ib_paper=IB; maximize | ml_overlay = research



class ConditionalSectorRotationImproved(QCAlgorithm):

    def Initialize(self):
        self._use_qc_ui_parameters = USE_QC_UI_PARAMETERS

        if not self._use_qc_ui_parameters:
            cash = 100000
            self.SetStartDate(2020, 1, 1)
            self.SetEndDate(2026, 5, 17)
            self.SetCash(cash)
            self.Debug(
                "HARDCODED_BASELINE: 2020-2026 $100k ACTIVE_BASELINE="
                f"{globals().get('ACTIVE_BASELINE', 'maximize')!r}"
            )
        else:
            sy = self._int_parameter("start_year", 2020)
            sm = self._int_parameter("start_month", 1)
            sd = self._int_parameter("start_day", 1)
            self.SetStartDate(sy, sm, sd)

            if not self._bool_parameter("run_to_present", False):
                ey = self._int_parameter("end_year", 2026)
                em = self._int_parameter("end_month", 5)
                ed = self._int_parameter("end_day", 17)
                self.SetEndDate(ey, em, ed)

            cash = max(1000, self._int_parameter("starting_cash", 100000))
            self.SetCash(cash)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin,
        )

        preset = ""
        if self._use_qc_ui_parameters:
            preset = str(self.GetParameter("research_preset") or "").strip().lower()
        self._research_preset = preset
        self._preset_force_production = preset in (
            "production",
            "prod",
            "live_safe",
        )
        self._preset_force_max_equity = preset in ("max_equity", "maximize", "is_max")
        self._preset_force_institutional = preset in (
            "institutional", "inst", "low_dd", "lowdd",
        )
        self._preset_force_ml_overlay = preset in (
            "ml_overlay", "ml_maximize", "ml", "track2",
        )
        self._preset_force_ml_div = preset in (
            "ml_overlay_diversified", "ml_div", "ml_diversified",
        )
        self._preset_force_ib_paper = preset in (
            "ib_paper", "ibkr_paper", "paper", "live_paper",
        )
        self._preset_force_convex = preset in (
            "convex", "crisis", "convexity", "low_dd_convex",
        )
        self._preset_force_maximize_plus = preset in (
            "maximize_plus", "plus", "max_plus",
        )
        self._preset_force_maximize_hold = preset in (
            "maximize_hold", "hold", "let_winners_run",
        )
        self._preset_force_aggressive_120x = preset in (
            "aggressive_120x",
            "120x",
            "max_120x",
        )
        if self._preset_force_production and self._preset_force_max_equity:
            self.Debug(
                "research_preset conflict: production/live_safe wins over max_equity"
            )
            self._preset_force_max_equity = False

        # ── Optional constant slippage (per share) — set before AddEquity ─
        self._equity_slippage_dollars = max(
            0.0, self._float_parameter("constant_slippage_per_share", 0.0)
        )

        # ── Execution mode ──────────────────────────────────────────────
        self.use_eod_next_bar_execution = self._bool_parameter(
            "use_eod_next_bar_execution", True
        )

        # ── Rebalance hygiene ───────────────────────────────────────────
        self.use_rebalance_bands = self._bool_parameter("use_rebalance_bands", True)
        self.min_weight_change_to_trade = max(
            0.0, min(1.0, self._float_parameter("min_weight_change_to_trade", 0.02))
        )
        self.max_daily_weight_change = max(
            0.0, min(1.0, self._float_parameter("max_daily_weight_change", 0.15))
        )
        self.max_days_without_rebalance = max(
            0, self._int_parameter("max_days_without_rebalance", 0)
        )

        # ── Vol-ETP confirmation (entries) ──────────────────────────────
        self.vol_etp_confirm_days = max(
            0, self._int_parameter("vol_etp_confirm_days", 0)
        )

        # ── Regime-based vol target ─────────────────────────────────────
        self.use_regime_vol_target = self._bool_parameter(
            "use_regime_vol_target", False
        )
        self.target_ann_vol_bull = max(
            0.01, self._float_parameter("target_ann_vol_bull", 0.28)
        )
        self.target_ann_vol_bear = max(
            0.01, self._float_parameter("target_ann_vol_bear", 0.18)
        )

        # ── Indicator periods ───────────────────────────────────────────
        self.rsi_period = self._int_parameter("rsi_period", 10)
        self.spy_sma_period = self._int_parameter("spy_sma_period", 200)
        self.qqq_sma_period = self._int_parameter("qqq_sma_period", 20)
        self.tqqq_sma_period = self._int_parameter("tqqq_sma_period", 20)
        self.soxl_sma_period = self._int_parameter("soxl_sma_period", 20)
        self.regime_qqq_sma_period = max(
            2, self._int_parameter("regime_qqq_sma_period", 50)
        )
        raw_regime = self.GetParameter("regime_mode")
        rm = "" if raw_regime is None else str(raw_regime).strip().lower()
        if rm in ("spy_and_qqq", "spy+qqq", "dual", "both"):
            self.regime_mode = "spy_and_qqq"
        elif rm in ("qqq", "qqq_only", "qqq_sma"):
            self.regime_mode = "qqq"
        else:
            self.regime_mode = "spy"
        self.min_hold_days = max(0, self._int_parameter("min_hold_days", 0))
        self.bull_tqqq_momentum_days = max(
            0, self._int_parameter("bull_tqqq_momentum_days", 0)
        )

        self.use_probabilistic_regime = self._bool_parameter("use_probabilistic_regime", False)
        self.regime_score_min_bull = max(0.0, min(1.0, self._float_parameter("regime_score_min_bull", 0.55)))
        self.regime_hysteresis_days = max(0, self._int_parameter("regime_hysteresis_days", 0))
        self.use_bull_leverage_ladder = self._bool_parameter("use_bull_leverage_ladder", False)
        self.bull_ladder_tqqq_min = max(0.0, min(1.0, self._float_parameter("bull_ladder_tqqq_min", 0.65)))
        self.bull_ladder_qld_min = max(0.0, min(1.0, self._float_parameter("bull_ladder_qld_min", 0.40)))
        if self.bull_ladder_qld_min > self.bull_ladder_tqqq_min:
            self.bull_ladder_qld_min = self.bull_ladder_tqqq_min
        self.scale_weight_by_regime_score = self._bool_parameter("scale_weight_by_regime_score", False)
        self.use_rsp_breadth_proxy = self._bool_parameter("use_rsp_breadth_proxy", False)
        self.rsp_breadth_sma_period = max(5, self._int_parameter("rsp_breadth_sma_period", 20))
        self.use_vix_delever = self._bool_parameter("use_vix_delever", False)
        self.vix_delever_ratio = max(1.0, self._float_parameter("vix_delever_ratio", 1.20))
        self.vix_delever_mult = max(0.05, min(1.0, self._float_parameter("vix_delever_mult", 0.55)))
        self.vix_sma_period = max(5, self._int_parameter("vix_sma_period", 20))

        self.use_ml_overlay = self._bool_parameter("use_ml_overlay", False)
        self.ml_train_bars = max(120, self._int_parameter("ml_train_bars", 500))
        self.ml_forward_days = max(1, self._int_parameter("ml_forward_days", 5))
        self.ml_retrain_days = max(21, self._int_parameter("ml_retrain_days", 63))
        self.ml_veto_prob = max(0.0, min(1.0, self._float_parameter("ml_veto_prob", 0.32)))
        self.ml_floor_mult = max(0.2, min(1.0, self._float_parameter("ml_floor_mult", 0.78)))
        self.ml_boost_cap = max(0.5, min(1.5, self._float_parameter("ml_boost_cap", 1.12)))
        self.ml_bear_offensive_prob = max(0.0, min(1.0, self._float_parameter("ml_bear_offensive_prob", 0.38)))
        self.ml_filter_bear_offensive = self._bool_parameter("ml_filter_bear_offensive", True)
        self.institutional_suppress_bear_leverage = self._bool_parameter(
            "institutional_suppress_bear_leverage", False
        )
        self.institutional_soft_drawdown = self._bool_parameter(
            "institutional_soft_drawdown", False
        )

        # ── RSI thresholds ────────────────────────────────────────────
        self.th_rsi_qqq_bull_uvxy = self._float_parameter("th_rsi_qqq_bull_uvxy", 81.0)
        self.th_rsi_spy_bull_uvxy = self._float_parameter("th_rsi_spy_bull_uvxy", 80.0)
        self.th_rsi_tqqq_bear_tecl = self._float_parameter("th_rsi_tqqq_bear_tecl", 30.0)
        self.th_rsi_spy_bear_spxl = self._float_parameter("th_rsi_spy_bear_spxl", 30.0)
        self.th_rsi_uvxy_elevated = self._float_parameter("th_rsi_uvxy_elevated", 74.0)
        self.th_rsi_uvxy_extreme = self._float_parameter("th_rsi_uvxy_extreme", 84.0)
        self.th_rsi_sqqq_tecs_qqq_above = self._float_parameter(
            "th_rsi_sqqq_tecs_qqq_above", 31.0
        )
        self.th_rsi_sqqq_tecs_tqqq_above = self._float_parameter(
            "th_rsi_sqqq_tecs_tqqq_above", 34.0
        )
        self.th_rsi_soxl_bull = self._float_parameter("th_rsi_soxl_bull", 40.0)
        self.th_rsi_uvxy_calm = self._float_parameter("th_rsi_uvxy_calm", 35.0)
        self.th_rsi_soxs_bear = self._float_parameter("th_rsi_soxs_bear", 55.0)

        # ── Feature flags ───────────────────────────────────────────────
        self.include_defensive_etfs = self._bool_parameter(
            "include_defensive_etfs", False
        )
        self.include_leveraged_defensives = self._bool_parameter(
            "include_leveraged_defensives", False
        )
        self.use_soxl_bull = self._bool_parameter("use_soxl_bull", True)
        self.use_svxy_calm = self._bool_parameter("use_svxy_calm", False)
        self.disable_bull_uvxy = self._bool_parameter("disable_bull_uvxy", False)
        self.bull_uvxy_require_both = self._bool_parameter(
            "bull_uvxy_require_both", False
        )
        self.aggressive_120x_research = self._bool_parameter(
            "aggressive_120x_research", False
        ) or self._preset_force_aggressive_120x
        self.maximize_disable_vol_target = self._bool_parameter(
            "maximize_disable_vol_target", False
        )
        self._soxl_skip_spy_rsi_filter = False

        # ── Volatility targeting ────────────────────────────────────────
        self.use_vol_targeting = self._bool_parameter("use_vol_targeting", True)
        self.target_ann_vol = max(0.01, self._float_parameter("target_ann_vol", 0.25))
        self.vol_lookback = max(5, self._int_parameter("vol_lookback", 20))
        self.max_gross_exposure = max(
            1.0, min(2.0, self._float_parameter("max_gross_exposure", 1.0))
        )
        self.max_position_weight = max(
            0.01,
            min(
                self.max_gross_exposure,
                self._float_parameter("max_position_weight", 1.0),
            ),
        )
        self.vol_etp_max_weight = max(
            0.01, min(1.0, self._float_parameter("vol_etp_max_weight", 1.0))
        )
        raw_vol_anchor = self.GetParameter("vol_anchor_ticker")
        self.vol_anchor_ticker = (
            "TQQQ"
            if raw_vol_anchor is None or str(raw_vol_anchor).strip() == ""
            else str(raw_vol_anchor).strip().upper()
        )

        # ── Vol ETP rails ───────────────────────────────────────────────
        self.max_consecutive_vol_etp_days = max(
            0, self._int_parameter("max_consecutive_vol_etp_days", 5)
        )
        self.max_consecutive_uvxy_days = max(
            0, self._int_parameter("max_consecutive_uvxy_days", 0)
        )
        self.max_consecutive_svxy_days = max(
            0, self._int_parameter("max_consecutive_svxy_days", 0)
        )
        self.gap_cooldown_pct = min(
            -0.01, self._float_parameter("gap_cooldown_pct", -0.12)
        )
        self.gap_cooldown_days = max(0, self._int_parameter("gap_cooldown_days", 3))

        # ── Drawdown: hard guard + tiered scaling ─────────────────────
        self.use_drawdown_guard = self._bool_parameter("use_drawdown_guard", True)
        self.max_drawdown_pct = max(
            0.05, min(0.95, self._float_parameter("max_drawdown_pct", 0.35))
        )
        self.drawdown_release_frac = max(
            0.05, min(1.0, self._float_parameter("drawdown_release_frac", 0.50))
        )
        self.use_tiered_drawdown = self._bool_parameter("use_tiered_drawdown", True)
        self.tier1_drawdown = max(0.0, self._float_parameter("tier1_drawdown", 0.15))
        self.tier1_mult = max(0.0, min(1.0, self._float_parameter("tier1_mult", 0.90)))
        self.tier2_drawdown = max(0.0, self._float_parameter("tier2_drawdown", 0.25))
        self.tier2_mult = max(0.0, min(1.0, self._float_parameter("tier2_mult", 0.70)))

        raw_risk_off = self.GetParameter("risk_off_ticker")
        self.risk_off_ticker = (
            "BSV"
            if raw_risk_off is None or str(raw_risk_off).strip() == ""
            else str(raw_risk_off).strip().upper()
        )

        if not self._use_qc_ui_parameters:
            self.trade_start = datetime(2020, 1, 1)
            self.trade_end = None
            self.production_safe_defaults = False
            self.maximize_backtest_equity = False
            self.aggressive_120x_research = False
            self.maximize_include_svxy = False
            _base = str(globals().get("ACTIVE_BASELINE", "maximize")).strip().lower()
            if _base in ("institutional", "inst", "low_dd", "lowdd"):
                self._apply_institutional_profile()
            elif _base in ("ml_overlay_aggressive", "ml_aggressive"):
                from csr_ml_overlay import apply_ml_aggressive_profile
                apply_ml_aggressive_profile(self)
            elif _base in ("ml_overlay", "ml_maximize", "ml", "track2"):
                self._apply_ml_maximize_profile()
            elif _base in ("ml_overlay_diversified", "ml_div", "ml_diversified"):
                from csr_defensive_sleeve_ext import apply_ml_defensive_diversified_profile
                apply_ml_defensive_diversified_profile(self)
            elif _base in ("ib_paper", "ibkr_paper", "paper", "live_paper"):
                from csr_ib_paper_ext import apply_ib_paper_profile
                apply_ib_paper_profile(self)
            elif _base in ("maximize_hold", "hold", "let_winners_run"):
                from csr_hold_ext import apply_maximize_hold_profile
                apply_maximize_hold_profile(self)
            else:
                self.maximize_backtest_equity = True
                self._apply_maximize_backtest_equity_profile()
            self.disable_bull_uvxy = True
        else:
            tsy = self._int_parameter("trade_start_year", sy)
            tsm = self._int_parameter("trade_start_month", sm)
            tsd = self._int_parameter("trade_start_day", sd)
            self.trade_start = datetime(tsy, tsm, tsd)

            tey = self._int_parameter("trade_end_year", 0)
            if tey > 0:
                tem = max(1, min(12, self._int_parameter("trade_end_month", 12)))
                ted = max(1, min(31, self._int_parameter("trade_end_day", 31)))
                self.trade_end = datetime(tey, tem, ted)
                if self.trade_end.date() < self.trade_start.date():
                    self.Debug("trade_end before trade_start — ignoring trade_end")
                    self.trade_end = None
            else:
                self.trade_end = None

            prod_user = self._bool_parameter("production_safe_defaults", False)
            self.production_safe_defaults = bool(
                prod_user or self._preset_force_production
            )
            self.headline_qc_default = self._bool_parameter("headline_qc_default", True)
            if self.headline_qc_default and not self.production_safe_defaults:
                max_user = True
                self.maximize_backtest_equity = True
            else:
                max_user = self._bool_parameter("maximize_backtest_equity", True)
                self.maximize_backtest_equity = bool(
                    (max_user or self._preset_force_max_equity)
                    and not self.production_safe_defaults
                )
            self.maximize_include_svxy = self._bool_parameter(
                "maximize_include_svxy", False
            )

            if self._preset_force_ib_paper:
                from csr_ib_paper_ext import apply_ib_paper_profile
                apply_ib_paper_profile(self)
            elif self.production_safe_defaults:
                self._apply_production_safe_profile()
            elif self._preset_force_institutional:
                self._apply_institutional_profile()
            elif self._preset_force_ml_div:
                from csr_defensive_sleeve_ext import apply_ml_defensive_diversified_profile
                apply_ml_defensive_diversified_profile(self)
            elif self._preset_force_ml_overlay:
                self._apply_ml_maximize_profile()
            elif self._preset_force_maximize_hold:
                from csr_hold_ext import apply_maximize_hold_profile
                apply_maximize_hold_profile(self)
            elif self._preset_force_convex:
                self._apply_convex_profile()
            elif self._preset_force_aggressive_120x:
                from csr_aggressive_ext import apply_aggressive_120x_baseline
                apply_aggressive_120x_baseline(self)
            elif self._preset_force_maximize_plus:
                from csr_aggressive_ext import apply_maximize_plus_baseline
                apply_maximize_plus_baseline(self)
            elif self.maximize_backtest_equity:
                self._apply_maximize_backtest_equity_profile()

            if (
                (self.aggressive_120x_research or self._preset_force_aggressive_120x)
                and not getattr(self, "aggressive_preset_active", False)
            ):
                self._apply_aggressive_120x_research_bundle()

            if (
                self.maximize_backtest_equity
                or self.production_safe_defaults
                or getattr(self, "ib_paper_active", False)
            ):
                self._reload_user_overrides_after_profile()

        if not self.use_eod_next_bar_execution:
            _bm = str(globals().get("ACTIVE_BASELINE", "maximize")).strip().lower()
            self.Debug(f"EXECUTION_MODE=same_bar ({_bm})")

        if preset in ("realistic", "realistic_backtest") and self._equity_slippage_dollars <= 0.0:
            self._equity_slippage_dollars = 0.001
            self.Debug(
                "research_preset=realistic: default constant_slippage_per_share=0.001"
            )

        if self._equity_slippage_dollars > 0.0:
            self.SetSecurityInitializer(self._equity_slippage_initializer)

        self._log_active_research_profile()
        if getattr(self, "use_ml_overlay", False) and not hasattr(self, "_mlh"):
            from csr_ml_overlay import wire_ml_overlay
            wire_ml_overlay(self)
        if getattr(self, "use_defensive_sleeve", False) and not hasattr(self, "_dsh"):
            from csr_defensive_sleeve_ext import wire_defensive_sleeve
            wire_defensive_sleeve(self)

        # ── Universe ────────────────────────────────────────────────────
        self.tickers = [
            "SPY", "QQQ", "TQQQ", "UVXY",
            "TECL", "SPXL", "SQQQ", "TECS", "BSV",
            "SOXL", "SOXS", "SVXY", "FAS",
        ]
        for _dt in getattr(self, "_defensive_sleeve_extra", ()):
            if _dt not in self.tickers:
                self.tickers.append(_dt)
        if self.include_defensive_etfs:
            self.tickers.extend(["TLT", "GLD"])
        if self.include_leveraged_defensives:
            for t in ["TMF", "ERX"]:
                if t not in self.tickers:
                    self.tickers.append(t)

        if self.vol_anchor_ticker not in self.tickers:
            raise ValueError(
                f"vol_anchor_ticker {self.vol_anchor_ticker!r} not in universe {self.tickers}"
            )
        if self.risk_off_ticker not in self.tickers:
            raise ValueError(
                f"risk_off_ticker {self.risk_off_ticker!r} not in universe {self.tickers}"
            )

        raw_bench = self.GetParameter("benchmark_ticker")
        _bs = "" if raw_bench is None else str(raw_bench).strip().upper()
        self.benchmark_ticker = _bs if _bs else "TQQQ"
        if self.benchmark_ticker not in self.tickers:
            self.tickers.append(self.benchmark_ticker)
        if self.use_bull_leverage_ladder and "QLD" not in self.tickers:
            self.tickers.append("QLD")
        if self.use_rsp_breadth_proxy and "RSP" not in self.tickers:
            self.tickers.append("RSP")

        self.symbols = {}
        self.indicators = {}
        self._vix_index_symbol = None
        for ticker in self.tickers:
            sym = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols[ticker] = sym
            self.indicators[self._rsi_key(ticker)] = self.RSI(
                sym, self.rsi_period, MovingAverageType.Wilders, Resolution.Daily
            )

        if self.use_vix_delever or getattr(self, "use_ml_overlay", False):
            try:
                vix_sym = self.AddIndex("VIX", Resolution.Daily).Symbol
                self._vix_index_symbol = vix_sym
                self.symbols["VIX"] = vix_sym
                self.indicators["VIX_SMA"] = self.SMA(vix_sym, self.vix_sma_period, Resolution.Daily)
            except Exception as ex:
                self.use_vix_delever = False
                self.Debug(f"VIX unavailable; use_vix_delever off: {ex}")

        self._sym_to_ticker = {self.symbols[k]: k for k in self.symbols.keys()}

        for key, ticker, period in [
            ("SPY_SMA200", "SPY", self.spy_sma_period),
            ("QQQ_SMA20", "QQQ", self.qqq_sma_period),
            ("TQQQ_SMA20", "TQQQ", self.tqqq_sma_period),
            ("SOXL_SMA20", "SOXL", self.soxl_sma_period),
        ]:
            self.indicators[key] = self.SMA(
                self.symbols[ticker], period, Resolution.Daily
            )

        if self.regime_mode in ("spy_and_qqq", "qqq"):
            self.indicators["QQQ_SMA_REGIME"] = self.SMA(
                self.symbols["QQQ"], self.regime_qqq_sma_period, Resolution.Daily
            )
        if self.use_rsp_breadth_proxy and "RSP" in self.symbols:
            self.indicators["RSP_SMA_BREADTH"] = self.SMA(
                self.symbols["RSP"], self.rsp_breadth_sma_period, Resolution.Daily
            )

        self.SetBenchmark(self.symbols[self.benchmark_ticker])
        warm = max(
            260,
            self.spy_sma_period + 60,
            self.qqq_sma_period + 60,
            self.tqqq_sma_period + 60,
            self.soxl_sma_period + 60,
            self.regime_qqq_sma_period + 60 if self.regime_mode in ("spy_and_qqq", "qqq") else 0,
            self.rsp_breadth_sma_period + 60 if self.use_rsp_breadth_proxy else 0,
            self.vix_sma_period + 60 if self.use_vix_delever else 0,
            self.rsi_period + 60,
            self.vol_lookback + 10,
        )
        self.SetWarmUp(warm, Resolution.Daily)

        spy = self.symbols["SPY"]
        if self.use_eod_next_bar_execution:
            self.Schedule.On(
                self.DateRules.EveryDay(spy),
                self.TimeRules.AfterMarketClose(spy, 10),
                self._after_market_close_plan,
            )
            self.Schedule.On(
                self.DateRules.EveryDay(spy),
                self.TimeRules.BeforeMarketOpen(spy, 5),
                self._before_market_open_execute,
            )

        # ── State ───────────────────────────────────────────────────────
        self._last_target_ticker = None
        self._last_trade_time = None
        self._peak_equity = float(cash)
        self._drawdown_guard_active = False

        self._pending_ticker = None
        self._pending_weight = 0.0

        self._consec_uvxy_days = 0
        self._consec_svxy_days = 0
        self._cooldown_remaining = 0

        self._vol_etp_confirm_name = None
        self._vol_etp_confirm_count = 0
        self._regime_bull_live = False
        self._regime_bull_streak = 0
        self._regime_bear_streak = 0
        self._last_regime_score = 0.0
        self.convex_preset_active = getattr(self, "convex_preset_active", False)
        self.aggressive_preset_active = getattr(self, "aggressive_preset_active", False)
        self.maximize_plus_active = getattr(self, "maximize_plus_active", False)
        self.maximize_hold_active = getattr(self, "maximize_hold_active", False)
        self.hold_winners_enabled = getattr(self, "hold_winners_enabled", False)
        self._hold_entry_price = getattr(self, "_hold_entry_price", {})
        self._hold_peak_pv = getattr(self, "_hold_peak_pv", {})
        self._ml_weights = None
        self._ml_last_prob = 0.5
        self._ml_last_train_day = None
        self._ml_bull_tickers = frozenset({"TQQQ", "SOXL", "QLD", "QQQ"})
        self._ml_bear_offensive = frozenset({"TECL", "SPXL", "TECS", "SOXS", "SQQQ", "UVXY"})

    def _equity_slippage_initializer(self, security):
        if security.Type != SecurityType.Equity:
            return
        if self._equity_slippage_dollars <= 0.0:
            return
        security.SetSlippageModel(ConstantSlippageModel(self._equity_slippage_dollars))

    def _log_active_research_profile(self):
        raw_max = self.GetParameter("maximize_backtest_equity")
        raw_prod = self.GetParameter("production_safe_defaults")
        raw_head = self.GetParameter("headline_qc_default")
        self.Debug(
            "QC_PARAMS_RAW "
            f"headline_qc_default={raw_head!r} maximize_backtest_equity={raw_max!r} "
            f"production_safe_defaults={raw_prod!r} research_preset={self._research_preset!r}"
        )
        if getattr(self, "maximize_hold_active", False):
            self.Debug(f"ACTIVE_PROFILE=maximize_hold (max_gross={self.max_gross_exposure:.2f}).")
        elif getattr(self, "ib_paper_active", False):
            self.Debug("ACTIVE_PROFILE=ib_paper (ML v2 + EOD production_safe).")
        elif getattr(self, "use_defensive_sleeve", False) and getattr(self, "use_ml_overlay", False):
            self.Debug("ACTIVE_PROFILE=ml_overlay_diversified (defensive sleeve).")
        elif getattr(self, "use_ml_overlay", False) and self.maximize_backtest_equity:
            self.Debug("ACTIVE_PROFILE=ml_overlay (%s)." % getattr(self, "ml_overlay_mode", "aggressive"))
        elif self.maximize_backtest_equity:
            self.Debug(f"ACTIVE_PROFILE=maximize (same-bar, max_gross={self.max_gross_exposure:.2f}).")
        elif getattr(self, "_preset_force_institutional", False) or (
            self.use_probabilistic_regime and self.use_bull_leverage_ladder
        ):
            self.Debug("ACTIVE_PROFILE=institutional.")
        elif self.production_safe_defaults:
            self.Debug(
                "ACTIVE_PROFILE=production_safe (EOD, rails, bands). "
                "research_preset or production_safe_defaults triggered this."
            )
        else:
            self.Debug(
                "ACTIVE_PROFILE=custom (maximize off, headline_qc_default false). "
                "Tune use_eod_next_bar_execution, rails, and slippage explicitly."
            )


    def _apply_ml_maximize_profile(self):
        from csr_ml_overlay import apply_ml_maximize_profile
        apply_ml_maximize_profile(self)

    def _ml_maybe_train(self, force=False):
        h = getattr(self, "_mlh", None)
        if h is not None:
            h.maybe_train(force)

    def _ml_apply_signal_filter(self, signal):
        h = getattr(self, "_mlh", None)
        if h is not None:
            return h.apply_signal_filter(signal)
        return signal

    def _ml_overlay_multiplier(self, ticker):
        h = getattr(self, "_mlh", None)
        if h is not None:
            return h.overlay_multiplier(ticker)
        return 1.0

    def _ds_apply(self, signal):
        h = getattr(self, "_dsh", None)
        return h.apply_signal(signal) if h else signal

    def _apply_institutional_profile(self):
        from csr_institutional_ext import apply_institutional_profile
        apply_institutional_profile(self)

    def _apply_convex_profile(self):
        from csr_convex_ext import apply_convex_profile
        apply_convex_profile(self)

    def _apply_production_safe_profile(self):
        from csr_profiles import apply_production_safe_profile
        apply_production_safe_profile(self)

    def _apply_maximize_backtest_equity_profile(self):
        from csr_profiles import apply_maximize_backtest_equity_profile
        apply_maximize_backtest_equity_profile(self)

    def _apply_aggressive_120x_research_bundle(self):
        from csr_profiles import apply_aggressive_120x_research_bundle
        apply_aggressive_120x_research_bundle(self)

    def _parameter_was_set(self, name):
        if not getattr(self, "_use_qc_ui_parameters", True):
            return False
        raw = self.GetParameter(name)
        return raw is not None and str(raw).strip() != ""

    def _reload_user_overrides_after_profile(self):
        from csr_profiles import reload_user_overrides_after_profile
        reload_user_overrides_after_profile(self)

    def _gross_cap(self):
        return max(1.0, min(2.0, float(getattr(self, "max_gross_exposure", 1.0))))


    # ── QC callbacks ─────────────────────────────────────────────────────

    def OnData(self, data):
        if not self.use_eod_next_bar_execution:
            self._run_intraday_pipeline()

    # ── Scheduled: EOD plan → BMO execute ───────────────────────────────

    def _after_market_close_plan(self):
        if self.IsWarmingUp or not self._indicators_ready():
            return

        if self._past_trade_end():
            self._pending_ticker = None
            self._pending_weight = 0.0
            return

        self._update_drawdown_guard()
        self._ml_maybe_train()
        self._refresh_regime_state()
        gap_fired = self._mark_eod_gap_cooldown()
        if gap_fired:
            self._cooldown_remaining = self.gap_cooldown_days

        raw_signal = self._compute_signal()
        if raw_signal is None:
            self._pending_ticker = None
            self._pending_weight = 0.0
            return

        raw_signal = self._apply_vol_etp_entry_confirmation(raw_signal)
        signal = self._apply_vol_etp_rails(raw_signal)
        signal = self._apply_gap_cooldown_filter(signal)
        signal = self._ml_apply_signal_filter(signal)
        signal = self._ds_apply(signal)

        if self.use_drawdown_guard and self._drawdown_guard_active:
            if getattr(self, "institutional_soft_drawdown", False) and self._effective_is_bull_regime():
                target_ticker = self._bull_leverage_ticker()
                base_weight = 0.55
            else:
                target_ticker = self.risk_off_ticker
                base_weight = 1.0
        else:
            target_ticker = signal
            base_weight = 1.0
            base_weight *= self._tiered_drawdown_multiplier()
            if self.use_vol_targeting:
                base_weight *= self._vol_target_multiplier()
            base_weight *= self._institutional_risk_multipliers()
        ml_m = self._ml_overlay_multiplier(target_ticker)
        if ml_m <= 0.0:
            target_ticker, base_weight = self.risk_off_ticker, 1.0
        else:
            base_weight *= ml_m
        target_ticker = self._ds_apply(target_ticker)

        base_weight = max(0.0, min(self._gross_cap(), float(base_weight)))

        if self.Time.date() < self.trade_start.date():
            self._pending_ticker = None
            self._pending_weight = 0.0
            return

        if self._min_hold_blocks_switch_target(target_ticker):
            self._pending_ticker = self._last_target_ticker
            self._pending_weight = self._last_pending_weight_or_default()
            return

        target_ticker, base_weight = self._apply_rebalance_friction(
            target_ticker, base_weight
        )
        target_ticker, base_weight = self._apply_execution_caps(
            target_ticker, base_weight
        )

        self._pending_ticker = target_ticker
        self._pending_weight = base_weight

        guard = " [DD_GUARD]" if (self.use_drawdown_guard and self._drawdown_guard_active) else ""
        self.Debug(
            f"EOD {self.Time:%Y-%m-%d} pending={self._pending_ticker} w={self._pending_weight:.3f}"
            f"{guard} raw={raw_signal} adj={signal}"
        )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def _before_market_open_execute(self):
        if self.IsWarmingUp or not self._indicators_ready():
            return

        if self._past_trade_end():
            if self.Portfolio.Invested:
                self.Liquidate()
            self._last_target_ticker = None
            self._pending_ticker = None
            self._pending_weight = 0.0
            return

        if self.Time.date() < self.trade_start.date():
            return

        if self._pending_ticker is None:
            if self.Portfolio.Invested:
                self.Liquidate()
            self._last_target_ticker = None
            return

        t = self._pending_ticker
        w = float(self._pending_weight)
        sym = self.symbols[t]

        if w <= 0.0:
            self.Liquidate()
            self._last_target_ticker = None
            self._last_trade_time = self.Time
            return

        prev_t = self._last_target_ticker
        self.SetHoldings(sym, w, True)
        if getattr(self, "hold_winners_enabled", False):
            from csr_hold_ext import after_trade_open
            after_trade_open(self, t, prev_t)
        self._last_target_ticker = t
        self._last_trade_time = self.Time
        self._last_executed_weight = w

    # ── Same-bar fallback (original style) ────────────────────────────────

    def _run_intraday_pipeline(self):
        if self.IsWarmingUp or not self._indicators_ready():
            return

        if self._past_trade_end():
            if self.Portfolio.Invested:
                self.Liquidate()
            self._last_target_ticker = None
            return

        if self.Time.date() < self.trade_start.date():
            return

        self._update_drawdown_guard()
        self._ml_maybe_train()
        self._refresh_regime_state()
        gap_fired = self._mark_eod_gap_cooldown()
        if gap_fired:
            self._cooldown_remaining = self.gap_cooldown_days

        raw_signal = self._compute_signal()
        if raw_signal is None:
            return

        raw_signal = self._apply_vol_etp_entry_confirmation(raw_signal)
        signal = self._apply_vol_etp_rails(raw_signal)
        signal = self._apply_gap_cooldown_filter(signal)
        signal = self._ml_apply_signal_filter(signal)
        signal = self._ds_apply(signal)

        if self.use_drawdown_guard and self._drawdown_guard_active:
            if getattr(self, "institutional_soft_drawdown", False) and self._effective_is_bull_regime():
                target_ticker = self._bull_leverage_ticker()
                w = 0.55
            else:
                target_ticker = self.risk_off_ticker
                w = 1.0
        else:
            target_ticker = signal
            w = 1.0
            w *= self._tiered_drawdown_multiplier()
            if self.use_vol_targeting:
                w *= self._vol_target_multiplier()
            w *= self._institutional_risk_multipliers()
        ml_m = self._ml_overlay_multiplier(target_ticker)
        if ml_m <= 0.0:
            target_ticker, w = self.risk_off_ticker, 1.0
        else:
            w *= ml_m
        target_ticker = self._ds_apply(target_ticker)
        w = max(0.0, min(self._gross_cap(), float(w)))

        if self._min_hold_blocks_switch_target(target_ticker):
            return

        target_ticker, w = self._apply_rebalance_friction(target_ticker, w)
        target_ticker, w = self._apply_execution_caps(target_ticker, w)

        if target_ticker == self._last_target_ticker and abs(
            w - getattr(self, "_last_executed_weight", 0.0)
        ) < 1e-9:
            return

        sym = self.symbols[target_ticker]
        prev_t = self._last_target_ticker
        self.SetHoldings(sym, w, True)
        if getattr(self, "hold_winners_enabled", False):
            from csr_hold_ext import after_trade_open
            after_trade_open(self, target_ticker, prev_t)
        self._last_target_ticker = target_ticker
        self._last_trade_time = self.Time
        self._last_executed_weight = w

        self.Debug(
            f"{self.Time:%Y-%m-%d} samebar target={target_ticker} w={w:.3f} raw={raw_signal}"
        )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ── Regime + vol-ETP confirmation ───────────────────────────────────

    def _past_trade_end(self):
        if self.trade_end is None:
            return False
        return self.Time.date() > self.trade_end.date()

    def _apply_execution_caps(self, ticker, weight):
        cap = self._gross_cap()
        w = max(0.0, min(cap, float(weight)))
        mw = max(0.01, min(cap, float(self.max_position_weight)))
        w = min(w, mw)
        if ticker in self._vol_etp_names():
            vw = max(0.01, min(1.0, float(self.vol_etp_max_weight)))
            w = min(w, vw)
        return ticker, w

    def _is_qqq_bull_trend(self):
        key = "QQQ_SMA_REGIME"
        if key not in self.indicators:
            return True
        ind = self.indicators[key]
        if not ind.IsReady:
            return False
        price_qqq = self.Securities[self.symbols["QQQ"]].Price
        sma = ind.Current.Value
        if price_qqq <= 0 or sma <= 0:
            return False
        return price_qqq > sma

    def _tqqq_momentum_positive(self, days):
        d = int(days)
        if d <= 0:
            return True
        hist = self.History(self.symbols["TQQQ"], d + 1, Resolution.Daily)
        if hist is None or getattr(hist, "empty", True):
            return True
        try:
            closes = hist["close"].dropna()
        except Exception:
            try:
                closes = hist.xs("TQQQ", level=0)["close"].dropna()
            except Exception:
                return True
        if closes is None or len(closes) < d + 1:
            return True
        c0 = float(closes.iloc[-(d + 1)])
        c1 = float(closes.iloc[-1])
        if c0 <= 0 or c1 <= 0:
            return True
        return (c1 / c0) - 1.0 > 0.0

    def _clamp01(self, x):
        return max(0.0, min(1.0, float(x)))

    def _trend_score(self, price, sma, scale=0.05):
        if price <= 0 or sma <= 0:
            return 0.0
        if price <= sma:
            return self._clamp01((price / sma) - 0.92) * 0.35
        return self._clamp01(0.55 + ((price / sma) - 1.0) / scale)

    def _regime_score(self):
        price_spy = self.Securities[self.symbols["SPY"]].Price
        sma_spy = self.indicators["SPY_SMA200"].Current.Value
        trend_s = self._trend_score(price_spy, sma_spy, 0.05)
        breadth_s = 0.5
        if self.use_rsp_breadth_proxy and "RSP" in self.symbols:
            p_rsp = self.Securities[self.symbols["RSP"]].Price
            key = "RSP_SMA_BREADTH"
            if p_rsp > 0 and price_spy > 0 and key in self.indicators and self.indicators[key].IsReady:
                ratio = p_rsp / price_spy
                sma_r = self.indicators[key].Current.Value
                if sma_r > 0:
                    breadth_s = self._clamp01(0.45 + (ratio / sma_r - 1.0) * 5.0)
        vol_s = 0.5
        hist = self.History(self.symbols["SPY"], self.vol_lookback + 1, Resolution.Daily)
        if hist is not None and not getattr(hist, "empty", True):
            try:
                closes = hist["close"].dropna()
                rets = closes.pct_change().dropna()
                if len(rets) >= max(5, self.vol_lookback - 1):
                    rv = float(rets.iloc[-self.vol_lookback :].std()) * (252.0 ** 0.5)
                    vol_s = self._clamp01(1.0 - (rv / 0.35))
            except Exception:
                pass
        qqq_s = 0.5
        if self.regime_mode in ("spy_and_qqq", "qqq"):
            p_qqq = self.Securities[self.symbols["QQQ"]].Price
            key = "QQQ_SMA_REGIME"
            if key in self.indicators and self.indicators[key].IsReady:
                qqq_s = self._trend_score(p_qqq, self.indicators[key].Current.Value, 0.04)
        elif price_spy > 0 and sma_spy > 0:
            qqq_s = trend_s
        self._last_regime_score = self._clamp01(
            0.35 * trend_s + 0.25 * qqq_s + 0.20 * breadth_s + 0.20 * vol_s
        )
        return self._last_regime_score

    def _raw_bull_regime(self):
        if self.use_probabilistic_regime:
            return self._regime_score() >= self.regime_score_min_bull
        return self._is_bull_regime()

    def _update_regime_hysteresis(self, raw_bull):
        n = int(self.regime_hysteresis_days)
        if n <= 0:
            self._regime_bull_live = bool(raw_bull)
            return
        if raw_bull:
            self._regime_bull_streak += 1
            self._regime_bear_streak = 0
            if self._regime_bull_streak >= n:
                self._regime_bull_live = True
        else:
            self._regime_bear_streak += 1
            self._regime_bull_streak = 0
            if self._regime_bear_streak >= n:
                self._regime_bull_live = False

    def _refresh_regime_state(self):
        if self.use_probabilistic_regime:
            self._regime_score()
        self._update_regime_hysteresis(self._raw_bull_regime())

    def _effective_is_bull_regime(self):
        if self.regime_hysteresis_days > 0:
            return self._regime_bull_live
        return self._raw_bull_regime()

    def _bull_leverage_ticker(self):
        if not self.use_bull_leverage_ladder:
            return "TQQQ"
        s = self._last_regime_score if self.use_probabilistic_regime else (
            1.0 if self._is_bull_regime() else 0.0
        )
        if s >= self.bull_ladder_tqqq_min:
            return "TQQQ"
        if s >= self.bull_ladder_qld_min:
            return "QLD"
        return "QQQ"

    def _vix_stress_multiplier(self):
        if not self.use_vix_delever or "VIX" not in self.symbols:
            return 1.0
        key = "VIX_SMA"
        if key not in self.indicators or not self.indicators[key].IsReady:
            return 1.0
        vix = self.Securities[self.symbols["VIX"]].Price
        sma = self.indicators[key].Current.Value
        if vix <= 0 or sma <= 0:
            return 1.0
        if vix > sma * self.vix_delever_ratio:
            return float(self.vix_delever_mult)
        return 1.0

    def _institutional_risk_multipliers(self):
        m = self._vix_stress_multiplier()
        if self.scale_weight_by_regime_score and self._regime_bull_live:
            m *= max(0.35, self._last_regime_score)
        return m

    def _bull_risk_on_momentum(self, ticker):
        if ticker not in ("TQQQ", "SOXL", "QLD", "QQQ"):
            return ticker
        if self.bull_tqqq_momentum_days <= 0:
            return ticker
        if self._tqqq_momentum_positive(self.bull_tqqq_momentum_days):
            return ticker
        return self.risk_off_ticker

    def _is_bull_regime(self):
        price_spy = self.Securities[self.symbols["SPY"]].Price
        sma_spy = self.indicators["SPY_SMA200"].Current.Value
        spy_ok = price_spy > 0 and sma_spy > 0
        spy_bull = spy_ok and (price_spy > sma_spy)
        qqq_bull = self._is_qqq_bull_trend()
        if self.regime_mode == "spy":
            if not spy_ok:
                return False
            return spy_bull
        if self.regime_mode == "spy_and_qqq":
            if not spy_ok:
                return False
            return spy_bull and qqq_bull
        if self.regime_mode == "qqq":
            return qqq_bull
        if not spy_ok:
            return False
        return spy_bull

    def _vol_etp_standby_ticker(self):
        if self._effective_is_bull_regime():
            return self._bull_leverage_ticker()
        return self.risk_off_ticker

    def _apply_vol_etp_entry_confirmation(self, signal):
        vset = self._vol_etp_names()
        if self.vol_etp_confirm_days <= 0:
            self._vol_etp_confirm_name = None
            self._vol_etp_confirm_count = 0
            return signal

        if signal in vset:
            if self._vol_etp_confirm_name == signal:
                self._vol_etp_confirm_count += 1
            else:
                self._vol_etp_confirm_name = signal
                self._vol_etp_confirm_count = 1
            if self._vol_etp_confirm_count >= self.vol_etp_confirm_days:
                return signal
            return self._vol_etp_standby_ticker()

        self._vol_etp_confirm_name = None
        self._vol_etp_confirm_count = 0
        return signal

    # ── Rebalance bands + daily weight cap ──────────────────────────────

    def _dominant_holding(self):
        pv = self.Portfolio.TotalPortfolioValue
        if pv <= 0:
            return None, 0.0
        best_t, best_f = None, 0.0
        for sym in self.symbols.values():
            h = self.Portfolio[sym]
            if not h.Invested:
                continue
            f = abs(float(h.HoldingsValue)) / pv
            if f > best_f:
                best_f = f
                best_t = self._sym_to_ticker.get(sym, None)
        return best_t, best_f

    def _stale_rebalance_needed(self):
        if self.max_days_without_rebalance <= 0:
            return False
        if self._last_trade_time is None:
            return False
        return (self.Time - self._last_trade_time).days >= self.max_days_without_rebalance

    def _apply_rebalance_friction(self, target_ticker, base_weight):
        """
        Ticker change: always trade to target (min-hold handled earlier).
        Same ticker: clamp one-day weight delta, then apply min-change band.
        """
        if self.use_drawdown_guard and self._drawdown_guard_active:
            return target_ticker, base_weight

        dom_t, dom_w = self._dominant_holding()
        if dom_t is None:
            return target_ticker, base_weight

        if target_ticker != dom_t:
            return target_ticker, base_weight

        w_tgt = float(base_weight)
        w_cur = float(dom_w)

        w_adj = w_tgt
        cap = float(self.max_daily_weight_change)
        if cap > 0.0:
            delta = w_adj - w_cur
            if delta > cap:
                w_adj = w_cur + cap
            elif delta < -cap:
                w_adj = w_cur - cap

        w_adj = max(0.0, min(self._gross_cap(), w_adj))

        if not self.use_rebalance_bands or self.min_weight_change_to_trade <= 0.0:
            return target_ticker, w_adj

        if abs(w_adj - w_cur) < self.min_weight_change_to_trade:
            if self._stale_rebalance_needed():
                return target_ticker, w_adj
            return dom_t, w_cur

        return target_ticker, w_adj

    # ── Parameter helpers ───────────────────────────────────────────────

    def _int_parameter(self, name, default):
        if not getattr(self, "_use_qc_ui_parameters", True):
            return default
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _float_parameter(self, name, default):
        if not getattr(self, "_use_qc_ui_parameters", True):
            return float(default)
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    def _bool_parameter(self, name, default):
        if not getattr(self, "_use_qc_ui_parameters", True):
            return default
        raw = self.GetParameter(name)
        if raw is None:
            return default
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return default

    def _rsi_key(self, ticker):
        return f"{ticker}_RSI_{self.rsi_period}_day"

    def _rsi(self, ticker):
        return self.indicators[self._rsi_key(ticker)].Current.Value

    def _indicators_ready(self):
        return all(ind.IsReady for ind in self.indicators.values())

    def _defensive_rsi_candidates(self):
        base = ["TECS", "SOXS", "BSV"]
        if self.include_defensive_etfs:
            base.extend(["TLT", "GLD"])
        if self.include_leveraged_defensives:
            if "TMF" in self.tickers:
                base.append("TMF")
            if "ERX" in self.tickers:
                base.append("ERX")
            if "FAS" in self.tickers:
                base.append("FAS")
        return base

    def _update_drawdown_guard(self):
        if not self.use_drawdown_guard:
            self._drawdown_guard_active = False
            return
        pv = self.Portfolio.TotalPortfolioValue
        if pv <= 0:
            return
        if pv > self._peak_equity:
            self._peak_equity = pv
        dd = 1.0 - (pv / self._peak_equity) if self._peak_equity > 0 else 0.0
        if dd >= self.max_drawdown_pct:
            self._drawdown_guard_active = True
        elif self._drawdown_guard_active:
            if dd <= self.max_drawdown_pct * self.drawdown_release_frac:
                self._drawdown_guard_active = False

    def _tiered_drawdown_multiplier(self):
        if not self.use_tiered_drawdown:
            return 1.0
        pv = self.Portfolio.TotalPortfolioValue
        if pv <= 0 or self._peak_equity <= 0:
            return 1.0
        dd = 1.0 - (pv / self._peak_equity)
        if dd >= self.tier2_drawdown:
            return self.tier2_mult
        if dd >= self.tier1_drawdown:
            return self.tier1_mult
        return 1.0

    def _effective_target_ann_vol(self):
        if self.use_regime_vol_target:
            return (
                self.target_ann_vol_bull
                if self._effective_is_bull_regime()
                else self.target_ann_vol_bear
            )
        return self.target_ann_vol

    def _vol_target_multiplier(self):
        t = self.vol_anchor_ticker
        hist = self.History(self.symbols[t], self.vol_lookback + 1, Resolution.Daily)
        if hist is None or hist.empty:
            return 1.0
        try:
            closes = hist["close"].dropna()
        except Exception:
            try:
                closes = hist.xs(t, level=0)["close"].dropna()
            except Exception:
                return 1.0
        if closes is None or len(closes) < self.vol_lookback:
            return 1.0
        rets = closes.pct_change().dropna()
        if len(rets) < max(5, self.vol_lookback - 1):
            return 1.0
        rv = float(rets.iloc[-self.vol_lookback :].std())
        if rv <= 1e-12:
            return 1.0
        ann = rv * (252.0 ** 0.5)
        tgt = self._effective_target_ann_vol()
        return min(self._gross_cap(), tgt / ann)

    def _vol_etp_names(self):
        return {"UVXY", "SVXY"}

    def _uvxy_rail_limit(self):
        if self.max_consecutive_uvxy_days > 0:
            return self.max_consecutive_uvxy_days
        return self.max_consecutive_vol_etp_days

    def _svxy_rail_limit(self):
        if self.max_consecutive_svxy_days > 0:
            return self.max_consecutive_svxy_days
        return self.max_consecutive_vol_etp_days

    def _apply_vol_etp_rails(self, signal):
        uv_lim = self._uvxy_rail_limit()
        sv_lim = self._svxy_rail_limit()
        if signal == "UVXY":
            if uv_lim > 0 and self._consec_uvxy_days >= uv_lim:
                self._consec_uvxy_days = 0
                self._consec_svxy_days = 0
                return "TQQQ"
            self._consec_uvxy_days += 1
            self._consec_svxy_days = 0
            return signal
        if signal == "SVXY":
            if sv_lim > 0 and self._consec_svxy_days >= sv_lim:
                self._consec_uvxy_days = 0
                self._consec_svxy_days = 0
                return "TQQQ"
            self._consec_svxy_days += 1
            self._consec_uvxy_days = 0
            return signal
        self._consec_uvxy_days = 0
        self._consec_svxy_days = 0
        return signal

    def _apply_gap_cooldown_filter(self, signal):
        if self._cooldown_remaining <= 0 or self.gap_cooldown_days <= 0:
            return signal
        allowed = {"TQQQ", "BSV", self.risk_off_ticker}
        if signal in allowed:
            return signal
        return self.risk_off_ticker if self.risk_off_ticker in allowed else "BSV"

    def _min_hold_blocks_switch_target(self, proposed_ticker):
        from csr_hold_ext import min_hold_blocks_switch_target
        return min_hold_blocks_switch_target(self, proposed_ticker)

    def _last_pending_weight_or_default(self):
        w = getattr(self, "_last_executed_weight", None)
        if w is None:
            return 1.0 if self._last_target_ticker else 0.0
        return float(w)

    def _dominant_equity_ticker_from_portfolio(self):
        t, _ = self._dominant_holding()
        return t

    def _mark_eod_gap_cooldown(self):
        if self.gap_cooldown_days <= 0:
            return False
        t = self._dominant_equity_ticker_from_portfolio()
        if t is None:
            return False
        hist = self.History(self.symbols[t], 2, Resolution.Daily)
        if hist is None or getattr(hist, "empty", True):
            return False
        try:
            closes = hist["close"].dropna()
        except Exception:
            try:
                closes = hist.xs(t, level=0)["close"].dropna()
            except Exception:
                return False
        if closes is None or len(closes) < 2:
            return False
        c0 = float(closes.iloc[-2])
        c1 = float(closes.iloc[-1])
        if c0 <= 0 or c1 <= 0:
            return False
        day_ret = (c1 / c0) - 1.0
        if day_ret <= self.gap_cooldown_pct:
            self.Debug(
                f"{self.Time:%Y-%m-%d} GAP_COOLDOWN triggered on {t} ret={day_ret:.3f}"
            )
            return True
        return False

    def _get_max_rsi_ticker(self, ticker_list):
        best, highest = None, -1.0
        for ticker in ticker_list:
            v = self._rsi(ticker)
            if v > highest:
                highest, best = v, ticker
        return best

    def _compute_signal(self):
        price_spy = self.Securities[self.symbols["SPY"]].Price
        price_qqq = self.Securities[self.symbols["QQQ"]].Price
        price_tqqq = self.Securities[self.symbols["TQQQ"]].Price
        price_soxl = self.Securities[self.symbols["SOXL"]].Price

        if any(p <= 0 for p in (price_spy, price_qqq, price_tqqq, price_soxl)):
            return None

        rsi_qqq = self._rsi("QQQ")
        rsi_spy = self._rsi("SPY")
        rsi_tqqq = self._rsi("TQQQ")
        rsi_sqqq = self._rsi("SQQQ")
        rsi_uvxy = self._rsi("UVXY")
        rsi_soxl = self._rsi("SOXL")
        rsi_soxs = self._rsi("SOXS")

        sma_qqq = self.indicators["QQQ_SMA20"].Current.Value
        sma_tqqq = self.indicators["TQQQ_SMA20"].Current.Value
        sma_soxl = self.indicators["SOXL_SMA20"].Current.Value

        if self._effective_is_bull_regime():
            if not self.disable_bull_uvxy:
                qqq_hot = rsi_qqq > self.th_rsi_qqq_bull_uvxy
                spy_hot = rsi_spy > self.th_rsi_spy_bull_uvxy
                if self.bull_uvxy_require_both:
                    bull_uvxy = qqq_hot and spy_hot
                else:
                    bull_uvxy = qqq_hot or spy_hot
                if bull_uvxy:
                    return "UVXY"
            if self.use_svxy_calm and rsi_uvxy < self.th_rsi_uvxy_calm:
                return "SVXY"
            soxl_ok = (
                self.use_soxl_bull
                and price_soxl > sma_soxl
                and rsi_soxl > self.th_rsi_soxl_bull
            )
            if soxl_ok and not getattr(self, "_soxl_skip_spy_rsi_filter", False):
                soxl_ok = rsi_soxl > rsi_spy
            if soxl_ok:
                return self._bull_risk_on_momentum("SOXL")
            return self._bull_risk_on_momentum(self._bull_leverage_ticker())

        if getattr(self, "institutional_suppress_bear_leverage", False):
            return self.risk_off_ticker

        if rsi_tqqq < self.th_rsi_tqqq_bear_tecl:
            return "TECL"
        if rsi_spy < self.th_rsi_spy_bear_spxl:
            return "SPXL"

        if rsi_uvxy > self.th_rsi_uvxy_elevated:
            if rsi_uvxy > self.th_rsi_uvxy_extreme:
                if price_qqq > sma_qqq:
                    if rsi_soxs > self.th_rsi_soxs_bear:
                        return "SOXS"
                    if rsi_sqqq < self.th_rsi_sqqq_tecs_qqq_above:
                        return "TECS"
                    return "TECL"
                return self._get_max_rsi_ticker(self._defensive_rsi_candidates())
            return "UVXY"

        if price_tqqq > sma_tqqq:
            if rsi_sqqq < self.th_rsi_sqqq_tecs_tqqq_above:
                return "TECS"
            return "TECL"

        return self._get_max_rsi_ticker(self._defensive_rsi_candidates())



    def OnEndOfAlgorithm(self):
        self.Debug("Algorithm finished.")
