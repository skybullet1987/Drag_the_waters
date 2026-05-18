# region imports
from AlgorithmImports import *
from datetime import datetime
import csr_profiles as csr

# endregion

# Conditional sector rotation (QuantConnect / IB). Deploy main.py + csr_profiles.py (<64k each).
# Default: headline maximize only (~60x). LIFT_120x: lift_120x_research=true or research_preset=bull_sleeve_120x.


class ConditionalSectorRotationImproved(QCAlgorithm):

    def Initialize(self):
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

        preset = str(self.GetParameter("research_preset") or "").strip().lower()
        self._research_preset = preset
        self._preset_force_production = preset in (
            "production",
            "prod",
            "live_safe",
        )
        self._preset_force_max_equity = preset in ("max_equity", "maximize", "is_max")
        self._preset_force_aggressive_120x = preset in (
            "aggressive_120x",
            "max_120x",
        )
        self._preset_force_target_120x = preset in (
            "target_120x",
            "target_120",
            "120x_target",
            "target_120_plus",
        )
        self._preset_force_bull_sleeve = preset in (
            "bull_sleeve_120x",
            "bull_sleeve",
            "lift_120x",
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
        self.target_120x_research = (
            self._bool_parameter("target_120x_research", False)
            or self._preset_force_target_120x
        )
        self.lift_120x_research = (
            self._bool_parameter("lift_120x_research", False)
            or self._preset_force_bull_sleeve
        )
        self.use_plain_maximize_only = self._bool_parameter(
            "use_plain_maximize_only", False
        )
        self.ignore_qc_parameter_overrides = self._bool_parameter(
            "ignore_qc_parameter_overrides", True
        )
        self.min_rebalance_weight_delta = max(
            0.0,
            min(0.25, self._float_parameter("min_rebalance_weight_delta", 0.0)),
        )
        self.maximize_disable_vol_target = self._bool_parameter(
            "maximize_disable_vol_target", False
        )
        self._soxl_skip_spy_rsi_filter = False
        self._bull_sleeve_mode = False
        self._vol_target_off_in_bull = False
        self._bull_gross_cap = 1.0
        self._use_vix_gate = False
        self._prefer_soxl_on_outperform = False
        self.vix_min_bull_uvxy = max(
            10.0, self._float_parameter("vix_min_bull_uvxy", 18.0)
        )
        self.soxl_outperform_days = max(
            2, self._int_parameter("soxl_outperform_days", 5)
        )
        self.soxl_outperform_rsi_bonus = max(
            0.0, self._float_parameter("soxl_outperform_rsi_bonus", 4.0)
        )

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
        self.margin_safety_pct = max(
            0.50, min(1.0, self._float_parameter("margin_safety_pct", 1.0))
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

        # ── Walk-forward style: begin trading after this date ───────────
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
        if self._preset_force_target_120x:
            self.production_safe_defaults = False
            self.maximize_backtest_equity = True
            self.lift_120x_research = False
            self.target_120x_research = True
        self.headline_qc_default = self._bool_parameter("headline_qc_default", True)
        if self.production_safe_defaults:
            self.target_120x_research = False
            self.lift_120x_research = False
        elif self.use_plain_maximize_only:
            self.target_120x_research = False
            self.lift_120x_research = False
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

        if self.production_safe_defaults:
            csr.apply_production_safe_profile(self)
        elif self.maximize_backtest_equity:
            csr.apply_maximize_backtest_equity_profile(self)

        if self.target_120x_research or self.aggressive_120x_research:
            self.Debug("DEPRECATED preset/bundle -> LIFT_120X")
            self.lift_120x_research = True
        if self.lift_120x_research:
            csr.apply_lift_120x_research_bundle(self)

        skip_reload = self.ignore_qc_parameter_overrides and self.lift_120x_research
        if (self.maximize_backtest_equity or self.production_safe_defaults) and not skip_reload:
            csr.reload_user_overrides_after_profile(self)

        if preset in ("realistic", "realistic_backtest") and self._equity_slippage_dollars <= 0.0:
            self._equity_slippage_dollars = 0.001
            self.Debug(
                "research_preset=realistic: default constant_slippage_per_share=0.001"
            )

        if self._equity_slippage_dollars > 0.0:
            self.SetSecurityInitializer(self._equity_slippage_initializer)

        self._log_active_research_profile()
        self._log_effective_config()

        # ── Universe ────────────────────────────────────────────────────
        self.tickers = [
            "SPY", "QQQ", "TQQQ", "UVXY",
            "TECL", "SPXL", "SQQQ", "TECS", "BSV",
            "SOXL", "SOXS", "SVXY", "FAS",
        ]
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
        if getattr(self, "_use_vix_gate", False) and "VIX" not in self.tickers:
            self.tickers.append("VIX")

        self.symbols = {}
        self.indicators = {}
        for ticker in self.tickers:
            if ticker == "VIX":
                try:
                    sym = self.AddIndex("VIX", Resolution.Daily).Symbol
                except Exception:
                    self.Debug("VIX index unavailable — bull UVXY VIX gate disabled")
                    self._use_vix_gate = False
                    continue
            else:
                sym = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols[ticker] = sym
            self.indicators[self._rsi_key(ticker)] = self.RSI(
                sym, self.rsi_period, MovingAverageType.Wilders, Resolution.Daily
            )

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

        self.SetBenchmark(self.symbols[self.benchmark_ticker])
        self.Debug(f"Benchmark={self.benchmark_ticker} (set benchmark_ticker parameter to override)")

        warm = max(
            260,
            self.spy_sma_period + 60,
            self.qqq_sma_period + 60,
            self.tqqq_sma_period + 60,
            self.soxl_sma_period + 60,
            self.regime_qqq_sma_period + 60 if self.regime_mode in ("spy_and_qqq", "qqq") else 0,
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

        self._init_attribution_state()

    def _init_attribution_state(self):
        self._attrib_signal_days = {}
        self._attrib_signal_total = 0
        self._attrib_weight_sum = 0.0
        self._attrib_executions = 0
        self._attrib_skipped_small = 0
        self._attrib_ticker_switch = 0

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
        if getattr(self, "lift_120x_research", False):
            bs = "on" if getattr(self, "_bull_sleeve_mode", False) else "off"
            vb = "off" if getattr(self, "_vol_target_off_in_bull", False) else "on"
            self.Debug(
                "ACTIVE_PROFILE=lift_120x (maximize + LIFT: bull_sleeve="
                f"{bs}, vol_in_bull={vb}, bull_gross={getattr(self, '_bull_gross_cap', 1):.2f}, "
                f"min_hold={self.min_hold_days})."
            )
        elif self.maximize_backtest_equity:
            self.Debug(
                "ACTIVE_PROFILE=maximize_backtest_equity (~60x baseline; same-bar, "
                f"rails off, max_gross={self.max_gross_exposure:.2f}). "
                "For LIFT/bull_sleeve: lift_120x_research=true. EOD: research_preset=production."
            )
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

    def _gross_cap(self):
        return max(1.0, min(2.0, float(getattr(self, "max_gross_exposure", 1.0))))

    def _effective_gross_cap(self):
        cap = self._gross_cap()
        if not getattr(self, "_bull_sleeve_mode", False):
            return cap
        if self._is_bull_regime():
            bg = max(1.0, min(2.0, float(getattr(self, "_bull_gross_cap", cap))))
            return min(cap, bg)
        return min(cap, 1.0)

    def _log_effective_config(self):
        self.Debug(
            "EFFECTIVE_CONFIG "
            f"plain_maximize={self.use_plain_maximize_only} lift_120x={self.lift_120x_research} "
            f"bull_sleeve={getattr(self, '_bull_sleeve_mode', False)} "
            f"vol_off_in_bull={getattr(self, '_vol_target_off_in_bull', False)} "
            f"max_gross={self.max_gross_exposure:.2f} bull_gross={getattr(self, '_bull_gross_cap', 1):.2f} "
            f"vol_anchor={self.vol_anchor_ticker} ignore_qc_overrides={self.ignore_qc_parameter_overrides} "
            f"min_rebal_delta={self.min_rebalance_weight_delta:.3f} margin_safety={self.margin_safety_pct:.3f}"
        )

    def _record_signal_attribution(self, ticker, weight):
        if self.IsWarmingUp or self.Time.date() < self.trade_start.date():
            return
        if ticker is None:
            return
        self._attrib_signal_days[ticker] = self._attrib_signal_days.get(ticker, 0) + 1
        self._attrib_signal_total += 1
        self._attrib_weight_sum += float(weight)

    def _log_backtest_attribution(self):
        if self._attrib_signal_total <= 0:
            self.Debug("ATTRIBUTION no signal days recorded")
            return
        parts = []
        for t in sorted(self._attrib_signal_days.keys()):
            n = self._attrib_signal_days[t]
            pct = 100.0 * n / self._attrib_signal_total
            parts.append(f"{t}={pct:.1f}%")
        avg_w = self._attrib_weight_sum / max(1, self._attrib_signal_total)
        self.Debug(
            "ATTRIBUTION signal_days="
            + ",".join(parts)
            + f" | avg_target_w={avg_w:.3f} executions={self._attrib_executions} "
            f"skipped_small_rebal={self._attrib_skipped_small} ticker_switches={self._attrib_ticker_switch}"
        )
        tqqq_soxl = self._attrib_signal_days.get("TQQQ", 0) + self._attrib_signal_days.get(
            "SOXL", 0
        )
        pct_bull_beta = 100.0 * tqqq_soxl / self._attrib_signal_total
        self.Debug(
            f"ATTRIBUTION TQQQ+SOXL share={pct_bull_beta:.1f}% (bull beta proxy; higher helps 120x hunt)"
        )

    def _portfolio_weight_in_symbol(self, sym):
        pv = float(self.Portfolio.TotalPortfolioValue)
        if pv <= 0:
            return 0.0
        h = self.Portfolio[sym]
        if not h.Invested:
            return 0.0
        return abs(float(h.HoldingsValue)) / pv

    def _safe_set_holdings(self, sym, weight, liquidate_existing=True):
        """
        Scale target weight to available margin before SetHoldings.
        Prevents 'Insufficient buying power' on leveraged ETFs (TQQQ/SOXL) when w≈1.
        Skips tiny same-symbol weight changes to avoid vol-scaler churn (~2k orders).
        """
        w = max(0.0, min(self._effective_gross_cap(), float(weight)))
        if w <= 1e-9:
            if liquidate_existing:
                self.Liquidate(sym)
            return 0.0

        cur_w = self._portfolio_weight_in_symbol(sym)
        band = float(getattr(self, "min_rebalance_weight_delta", 0.03))
        if band > 0 and cur_w > 1e-9 and abs(w - cur_w) < band:
            self._attrib_skipped_small += 1
            return float(cur_w)

        buf = max(0.50, min(1.0, float(getattr(self, "margin_safety_pct", 0.98))))
        w = w * buf

        pv = float(self.Portfolio.TotalPortfolioValue)
        if pv > 0:
            try:
                bp = float(self.Portfolio.GetBuyingPower(sym, OrderDirection.Buy))
                if bp > 0:
                    w = min(w, bp / pv)
            except Exception:
                pass

        w_exec = w
        qty = 0
        for _ in range(16):
            try:
                qty = int(self.CalculateOrderQuantity(sym, w_exec))
            except Exception:
                qty = 0
            if qty != 0:
                break
            w_exec *= 0.98
            if w_exec < 0.05:
                w_exec = 0.0
                break

        if w_exec <= 1e-9 or qty == 0:
            if liquidate_existing:
                self.Liquidate(sym)
            if weight > 0.1:
                self.Debug(
                    f"{self.Time:%Y-%m-%d} MARGIN_CLAMP {sym.Value} "
                    f"requested={float(weight):.3f} affordable=0"
                )
            return 0.0

        if w_exec < w * 0.995:
            self.Debug(
                f"{self.Time:%Y-%m-%d} MARGIN_CLAMP {sym.Value} "
                f"requested={float(weight):.3f} exec={w_exec:.3f}"
            )

        tkr = self._sym_to_ticker.get(sym, None)
        prev_t = getattr(self, "_last_target_ticker", None)
        if prev_t is not None and tkr is not None and prev_t != tkr:
            self._attrib_ticker_switch += 1
        self._attrib_executions += 1
        self.SetHoldings(sym, w_exec, liquidate_existing)
        return float(w_exec)

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

        if self.use_drawdown_guard and self._drawdown_guard_active:
            target_ticker = self.risk_off_ticker
            base_weight = 1.0
        else:
            target_ticker = signal
            base_weight = 1.0
            base_weight *= self._tiered_drawdown_multiplier()
            if self.use_vol_targeting:
                base_weight *= self._vol_target_multiplier()

        base_weight = max(0.0, min(self._effective_gross_cap(), float(base_weight)))

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

        self._record_signal_attribution(target_ticker, base_weight)
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

        w_exec = self._safe_set_holdings(sym, w, True)
        self._last_target_ticker = t
        self._last_trade_time = self.Time
        self._last_executed_weight = w_exec

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
        gap_fired = self._mark_eod_gap_cooldown()
        if gap_fired:
            self._cooldown_remaining = self.gap_cooldown_days

        raw_signal = self._compute_signal()
        if raw_signal is None:
            return

        raw_signal = self._apply_vol_etp_entry_confirmation(raw_signal)
        signal = self._apply_vol_etp_rails(raw_signal)
        signal = self._apply_gap_cooldown_filter(signal)

        if self.use_drawdown_guard and self._drawdown_guard_active:
            target_ticker = self.risk_off_ticker
            w = 1.0
        else:
            target_ticker = signal
            w = 1.0
            w *= self._tiered_drawdown_multiplier()
            if self.use_vol_targeting:
                w *= self._vol_target_multiplier()
        w = max(0.0, min(self._effective_gross_cap(), float(w)))

        if self._min_hold_blocks_switch_target(target_ticker):
            return

        target_ticker, w = self._apply_rebalance_friction(target_ticker, w)
        target_ticker, w = self._apply_execution_caps(target_ticker, w)

        self._record_signal_attribution(target_ticker, w)

        if target_ticker == self._last_target_ticker and abs(
            w - getattr(self, "_last_executed_weight", 0.0)
        ) < 1e-9:
            return

        sym = self.symbols[target_ticker]
        w_exec = self._safe_set_holdings(sym, w, True)
        self._last_target_ticker = target_ticker
        self._last_trade_time = self.Time
        self._last_executed_weight = w_exec

        self.Debug(
            f"{self.Time:%Y-%m-%d} samebar target={target_ticker} w={w_exec:.3f} raw={raw_signal}"
        )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ── Regime + vol-ETP confirmation ───────────────────────────────────

    def _past_trade_end(self):
        if self.trade_end is None:
            return False
        return self.Time.date() > self.trade_end.date()

    def _apply_execution_caps(self, ticker, weight):
        cap = self._effective_gross_cap()
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

    def _asset_return_over_days(self, ticker, days):
        d = int(days)
        if d <= 0 or ticker not in self.symbols:
            return None
        hist = self.History(self.symbols[ticker], d + 1, Resolution.Daily)
        if hist is None or getattr(hist, "empty", True):
            return None
        try:
            closes = hist["close"].dropna()
        except Exception:
            try:
                closes = hist.xs(ticker, level=0)["close"].dropna()
            except Exception:
                return None
        if closes is None or len(closes) < d + 1:
            return None
        c0 = float(closes.iloc[-(d + 1)])
        c1 = float(closes.iloc[-1])
        if c0 <= 0 or c1 <= 0:
            return None
        return (c1 / c0) - 1.0

    def _soxl_outperformed_tqqq(self, days=None):
        d = int(days if days is not None else self.soxl_outperform_days)
        r_soxl = self._asset_return_over_days("SOXL", d)
        r_tqqq = self._asset_return_over_days("TQQQ", d)
        if r_soxl is None or r_tqqq is None:
            return False
        return r_soxl > r_tqqq

    def _vix_ok_for_bull_uvxy(self):
        if not getattr(self, "_use_vix_gate", False):
            return True
        if "VIX" not in self.symbols:
            return True
        vix = self.Securities[self.symbols["VIX"]].Price
        if vix <= 0:
            return True
        return float(vix) >= float(self.vix_min_bull_uvxy)

    def _bull_risk_on_momentum(self, ticker):
        if ticker not in ("TQQQ", "SOXL"):
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
        return "TQQQ" if self._is_bull_regime() else self.risk_off_ticker

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

        w_adj = max(0.0, min(self._effective_gross_cap(), w_adj))

        if not self.use_rebalance_bands or self.min_weight_change_to_trade <= 0.0:
            return target_ticker, w_adj

        if abs(w_adj - w_cur) < self.min_weight_change_to_trade:
            if self._stale_rebalance_needed():
                return target_ticker, w_adj
            return dom_t, w_cur

        return target_ticker, w_adj

    # ── Parameter helpers ───────────────────────────────────────────────

    def _int_parameter(self, name, default):
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _float_parameter(self, name, default):
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    def _bool_parameter(self, name, default):
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
                if self._is_bull_regime()
                else self.target_ann_vol_bear
            )
        return self.target_ann_vol

    def _vol_target_multiplier(self):
        if getattr(self, "_vol_target_off_in_bull", False) and self._is_bull_regime():
            return self._effective_gross_cap()
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
        return min(self._effective_gross_cap(), tgt / ann)

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
        if self.min_hold_days <= 0 or self._last_trade_time is None:
            return False
        if self.use_drawdown_guard and self._drawdown_guard_active:
            return False
        if proposed_ticker == self._last_target_ticker:
            return False
        return (self.Time - self._last_trade_time).days < self.min_hold_days

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

    def _compute_bull_signal(
        self, rsi_qqq, rsi_spy, rsi_uvxy, rsi_soxl, price_soxl, sma_soxl
    ):
        if getattr(self, "_bull_sleeve_mode", False):
            soxl_ok = (
                self.use_soxl_bull
                and price_soxl > sma_soxl
                and rsi_soxl > self.th_rsi_soxl_bull
                and rsi_soxl > rsi_spy
            )
            if soxl_ok:
                return self._bull_risk_on_momentum("SOXL")
            return self._bull_risk_on_momentum("TQQQ")

        if not self.disable_bull_uvxy:
            qqq_hot = rsi_qqq > self.th_rsi_qqq_bull_uvxy
            spy_hot = rsi_spy > self.th_rsi_spy_bull_uvxy
            if self.bull_uvxy_require_both:
                bull_uvxy = qqq_hot and spy_hot
            else:
                bull_uvxy = qqq_hot or spy_hot
            if bull_uvxy and self._vix_ok_for_bull_uvxy():
                return "UVXY"
        if self.use_svxy_calm and rsi_uvxy < self.th_rsi_uvxy_calm:
            return "SVXY"
        soxl_rsi_th = float(self.th_rsi_soxl_bull)
        if getattr(self, "_prefer_soxl_on_outperform", False) and self._soxl_outperformed_tqqq():
            soxl_rsi_th = max(22.0, soxl_rsi_th - float(self.soxl_outperform_rsi_bonus))
        soxl_ok = (
            self.use_soxl_bull
            and price_soxl > sma_soxl
            and rsi_soxl > soxl_rsi_th
        )
        if soxl_ok and not getattr(self, "_soxl_skip_spy_rsi_filter", False):
            soxl_ok = rsi_soxl > rsi_spy
        if soxl_ok:
            return self._bull_risk_on_momentum("SOXL")
        return self._bull_risk_on_momentum("TQQQ")

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

        if self._is_bull_regime():
            return self._compute_bull_signal(
                rsi_qqq,
                rsi_spy,
                rsi_uvxy,
                rsi_soxl,
                price_soxl,
                sma_soxl,
            )

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
        self._log_backtest_attribution()
        self.Debug("Algorithm finished.")
