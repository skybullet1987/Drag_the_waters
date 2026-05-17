# region imports
from AlgorithmImports import *
from datetime import datetime

# endregion

# =============================================================================
# Conditional sector rotation — improved research template (QuantConnect / IB)
#
# Adds (vs a simple OnData same-bar version):
#   1) Optional end-of-day decision + next-session open execution (Schedule).
#   2) Volatility targeting: scales gross exposure toward a target ann. vol.
#   3) Vol-ETP rails: cap consecutive days in UVXY/SVXY; optional gap cooldown.
#   4) Tiered drawdown scaling: gradually reduces risk exposure before hard stop.
#   5) Optional "trade start" date to mimic walk-forward / frozen-parameter live.
#
# Educational / research only. Leveraged and inverse ETFs can gap and decay.
# Past performance does not guarantee future results.
# =============================================================================


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

        # ── Execution mode ──────────────────────────────────────────────
        self.use_eod_next_bar_execution = self._bool_parameter(
            "use_eod_next_bar_execution", True
        )

        # ── Indicator periods ───────────────────────────────────────────
        self.rsi_period = self._int_parameter("rsi_period", 10)
        self.spy_sma_period = self._int_parameter("spy_sma_period", 200)
        self.qqq_sma_period = self._int_parameter("qqq_sma_period", 20)
        self.tqqq_sma_period = self._int_parameter("tqqq_sma_period", 20)
        self.soxl_sma_period = self._int_parameter("soxl_sma_period", 20)
        self.min_hold_days = max(0, self._int_parameter("min_hold_days", 0))

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

        # ── Volatility targeting ────────────────────────────────────────
        self.use_vol_targeting = self._bool_parameter("use_vol_targeting", True)
        self.target_ann_vol = max(0.01, self._float_parameter("target_ann_vol", 0.25))
        self.vol_lookback = max(5, self._int_parameter("vol_lookback", 20))
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

        self.symbols = {}
        self.indicators = {}
        for ticker in self.tickers:
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

        self.SetBenchmark(self.symbols["SPY"])

        warm = max(
            260,
            self.spy_sma_period + 60,
            self.qqq_sma_period + 60,
            self.tqqq_sma_period + 60,
            self.soxl_sma_period + 60,
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

        self._consec_vol_etp_days = 0
        self._cooldown_remaining = 0

    # ── QC callbacks ─────────────────────────────────────────────────────

    def OnData(self, data):
        if not self.use_eod_next_bar_execution:
            self._run_intraday_pipeline()

    # ── Scheduled: EOD plan → BMO execute ───────────────────────────────

    def _after_market_close_plan(self):
        if self.IsWarmingUp or not self._indicators_ready():
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

        base_weight = max(0.0, min(1.0, float(base_weight)))

        if self.Time.date() < self.trade_start.date():
            self._pending_ticker = None
            self._pending_weight = 0.0
            return

        if self._min_hold_blocks_switch_target(target_ticker):
            self._pending_ticker = self._last_target_ticker
            self._pending_weight = self._last_pending_weight_or_default()
            return

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

        self.SetHoldings(sym, w, True)
        self._last_target_ticker = t
        self._last_trade_time = self.Time
        self._last_executed_weight = w

    # ── Same-bar fallback (original style) ────────────────────────────────

    def _run_intraday_pipeline(self):
        if self.IsWarmingUp or not self._indicators_ready():
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
        w = max(0.0, min(1.0, float(w)))

        if self._min_hold_blocks_switch_target(target_ticker):
            return

        if target_ticker == self._last_target_ticker and abs(
            w - getattr(self, "_last_executed_weight", 0.0)
        ) < 1e-6:
            return

        sym = self.symbols[target_ticker]
        self.SetHoldings(sym, w, True)
        self._last_target_ticker = target_ticker
        self._last_trade_time = self.Time
        self._last_executed_weight = w

        self.Debug(
            f"{self.Time:%Y-%m-%d} samebar target={target_ticker} w={w:.3f} raw={raw_signal}"
        )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

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
        return min(1.0, self.target_ann_vol / ann)

    def _vol_etp_names(self):
        return {"UVXY", "SVXY"}

    def _apply_vol_etp_rails(self, signal):
        vset = self._vol_etp_names()
        if signal in vset:
            lim = self.max_consecutive_vol_etp_days
            if lim > 0 and self._consec_vol_etp_days >= lim:
                self._consec_vol_etp_days = 0
                return "TQQQ"
            self._consec_vol_etp_days += 1
        else:
            self._consec_vol_etp_days = 0
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
        best_t, best_abs = None, 0.0
        for sym in self.symbols.values():
            h = self.Portfolio[sym]
            if not h.Invested:
                continue
            v = abs(float(h.HoldingsValue))
            if v > best_abs:
                best_abs = v
                best_t = self._sym_to_ticker.get(sym, None)
        return best_t

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

        sma_spy = self.indicators["SPY_SMA200"].Current.Value
        sma_qqq = self.indicators["QQQ_SMA20"].Current.Value
        sma_tqqq = self.indicators["TQQQ_SMA20"].Current.Value
        sma_soxl = self.indicators["SOXL_SMA20"].Current.Value

        if price_spy > sma_spy:
            if rsi_qqq > self.th_rsi_qqq_bull_uvxy or rsi_spy > self.th_rsi_spy_bull_uvxy:
                return "UVXY"
            if self.use_svxy_calm and rsi_uvxy < self.th_rsi_uvxy_calm:
                return "SVXY"
            if (
                self.use_soxl_bull
                and price_soxl > sma_soxl
                and rsi_soxl > self.th_rsi_soxl_bull
                and rsi_soxl > rsi_spy
            ):
                return "SOXL"
            return "TQQQ"

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
