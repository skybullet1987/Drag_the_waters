# Conditional sector rotation — QuantConnect / IB example (research template)
#
# =============================================================================
# HOW TO USE
# =============================================================================
# 1. New Python project on QuantConnect; paste this file as main.py (or import).
# 2. Live on Interactive Brokers: deploy from QC cloud, select IB brokerage.
# 3. Tune via Project → Parameters (see list at bottom of this docstring).
#
# =============================================================================
# READING BACKTESTS (e.g. high CAGR + large drawdown)
# =============================================================================
# - **PSR (Probabilistic Sharpe)** near ~0.5–0.8 is *not* proof of edge; treat
#   **~0.95+** as a stricter sanity bar if you use PSR at all.
# - **Huge end equity vs tiny start** can be **internally consistent** with a
#   high reported CAGR over many years in a bull-heavy sample; it is still not
#   a promise of live results (fees, fills, borrow, and path all differ).
# - **Large drawdowns** are common for single-ticker leveraged rotation; the
#   backtest assumes you **held** the plan through recovery unless you enable
#   the **drawdown guard** below.
# - **Alpha/Beta** with `SetBenchmark(SPY)` (enabled) are more interpretable than
#   uninitialized benchmark behavior — still sample-dependent.
# - **Strategy capacity** on QC flags liquidity; **Total fees** scale with
#   turnover and growing portfolio value in simulation.
# - **Many RSI/SMA cutoffs** = **overfitting risk**: walk-forward, sweeps, and
#   stress windows (2008, 2018, 2022) still matter.
#
# =============================================================================
# IMPLEMENTED IMPROVEMENTS
# =============================================================================
# - Parameterized thresholds, dates, starting cash, run_to_present.
# - SetBenchmark(SPY); trade only on target change; optional min_hold_days.
# - Optional TLT/GLD in defensive max-RSI basket (include_defensive_etfs).
# - **Drawdown guard**: optional switch to `risk_off_ticker` (default BSV) when
#   peak-to-trough drawdown exceeds `max_drawdown_pct`, with hysteresis release.
#   While the guard is active, **min_hold_days does not block** de-risk / guard
#   rotation (it still blocks speculative churn when the guard is off).
#
# =============================================================================
# DO YOU NEED MORE ETFs?
# =============================================================================
# **Not automatically.** Optional **TLT / GLD** only enter the defensive max-RSI
# tie-break when `include_defensive_etfs=true`. For true multi-asset weights,
# use a separate template.
#
# =============================================================================
# PARAMETERS (QC Project → Parameters; all optional)
# =============================================================================
# Dates (ignored partially if run_to_present for end):
#   start_year (2012), start_month (1), start_day (1),
#   end_year (2024), end_month (12), end_day (31)
# Cash: starting_cash (100000)
# Integers: rsi_period (10), spy_sma_period (200), qqq_sma_period (20),
#   tqqq_sma_period (20), min_hold_days (0 = off)
# Floats — RSI tree:
#   th_rsi_qqq_bull_uvxy (81), th_rsi_spy_bull_uvxy (80),
#   th_rsi_tqqq_bear_tecl (30), th_rsi_spy_bear_spxl (30),
#   th_rsi_uvxy_elevated (74), th_rsi_uvxy_extreme (84),
#   th_rsi_sqqq_tecs_qqq_above (31), th_rsi_sqqq_tecs_tqqq_above (34)
# Drawdown guard:
#   use_drawdown_guard ("false"), max_drawdown_pct (0.35) = 35% from peak,
#   drawdown_release_frac (0.50) release when DD <= max * this fraction,
#   risk_off_ticker ("BSV") must exist in the universe (core or defensive list)
# Bools / strings: run_to_present, include_defensive_etfs
#
# =============================================================================
# DISCLAIMER
# =============================================================================
# Educational / research only. Leveraged and inverse ETFs can lose most or all
# of their value intraday in extreme moves.

from AlgorithmImports import *


class ConditionalSectorRotation(QCAlgorithm):

    def Initialize(self):
        sy, sm, sd = (
            self._int_parameter("start_year", 2012),
            self._int_parameter("start_month", 1),
            self._int_parameter("start_day", 1),
        )
        self.SetStartDate(sy, sm, sd)

        run_open = self._bool_parameter("run_to_present", False)
        if not run_open:
            ey, em, ed = (
                self._int_parameter("end_year", 2024),
                self._int_parameter("end_month", 12),
                self._int_parameter("end_day", 31),
            )
            self.SetEndDate(ey, em, ed)

        cash = max(1000, self._int_parameter("starting_cash", 100000))
        self.SetCash(cash)

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

        self.rsi_period = self._int_parameter("rsi_period", 10)
        self.spy_sma_period = self._int_parameter("spy_sma_period", 200)
        self.qqq_sma_period = self._int_parameter("qqq_sma_period", 20)
        self.tqqq_sma_period = self._int_parameter("tqqq_sma_period", 20)
        self.min_hold_days = max(0, self._int_parameter("min_hold_days", 0))

        self.th_rsi_qqq_bull_uvxy = self._float_parameter("th_rsi_qqq_bull_uvxy", 81.0)
        self.th_rsi_spy_bull_uvxy = self._float_parameter("th_rsi_spy_bull_uvxy", 80.0)
        self.th_rsi_tqqq_bear_tecl = self._float_parameter("th_rsi_tqqq_bear_tecl", 30.0)
        self.th_rsi_spy_bear_spxl = self._float_parameter("th_rsi_spy_bear_spxl", 30.0)
        self.th_rsi_uvxy_elevated = self._float_parameter("th_rsi_uvxy_elevated", 74.0)
        self.th_rsi_uvxy_extreme = self._float_parameter("th_rsi_uvxy_extreme", 84.0)
        self.th_rsi_sqqq_tecs_qqq_above = self._float_parameter(
            "th_rsi_sqqq_tecs_qqq_above", 31.0)
        self.th_rsi_sqqq_tecs_tqqq_above = self._float_parameter(
            "th_rsi_sqqq_tecs_tqqq_above", 34.0)

        self.include_defensive_etfs = self._bool_parameter(
            "include_defensive_etfs", False)

        self.use_drawdown_guard = self._bool_parameter("use_drawdown_guard", False)
        self.max_drawdown_pct = max(
            0.05, min(0.95, self._float_parameter("max_drawdown_pct", 0.35)))
        self.drawdown_release_frac = max(
            0.05, min(1.0, self._float_parameter("drawdown_release_frac", 0.50)))

        raw_risk_off = self.GetParameter("risk_off_ticker")
        if raw_risk_off is None or str(raw_risk_off).strip() == "":
            self.risk_off_ticker = "BSV"
        else:
            self.risk_off_ticker = str(raw_risk_off).strip().upper()

        self.tickers = [
            "SPY", "QQQ", "TQQQ", "UVXY",
            "TECL", "SPXL", "SQQQ", "TECS", "BSV",
        ]
        if self.include_defensive_etfs:
            self.tickers.extend(["TLT", "GLD"])

        if self.risk_off_ticker not in self.tickers:
            raise ValueError(
                f"risk_off_ticker {self.risk_off_ticker!r} not in universe {self.tickers}"
            )

        self.symbols = {}
        self.indicators = {}

        for ticker in self.tickers:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols[ticker] = symbol
            self.indicators[self._rsi_key(ticker)] = self.RSI(
                symbol,
                self.rsi_period,
                MovingAverageType.Wilders,
                Resolution.Daily,
            )

        self.indicators["SPY_SMA200"] = self.SMA(
            self.symbols["SPY"], self.spy_sma_period, Resolution.Daily)
        self.indicators["QQQ_SMA20"] = self.SMA(
            self.symbols["QQQ"], self.qqq_sma_period, Resolution.Daily)
        self.indicators["TQQQ_SMA20"] = self.SMA(
            self.symbols["TQQQ"], self.tqqq_sma_period, Resolution.Daily)

        self.SetBenchmark(self.symbols["SPY"])

        warm = max(260, self.spy_sma_period + 60,
                   self.qqq_sma_period + 60, self.tqqq_sma_period + 60,
                   self.rsi_period + 60)
        self.SetWarmUp(warm, Resolution.Daily)

        self._last_target_ticker = None
        self._last_trade_time = None
        self._peak_equity = float(cash)
        self._drawdown_guard_active = False

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

    def _indicators_ready(self):
        for ind in self.indicators.values():
            if not ind.IsReady:
                return False
        return True

    def _defensive_rsi_candidates(self):
        base = ["TECS", "BSV"]
        if self.include_defensive_etfs:
            base.extend(["TLT", "GLD"])
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
            release_level = self.max_drawdown_pct * self.drawdown_release_frac
            if dd <= release_level:
                self._drawdown_guard_active = False

    def _min_hold_blocks_switch(self):
        if self.min_hold_days <= 0 or self._last_trade_time is None:
            return False
        if self._drawdown_guard_active:
            return False
        return (self.Time - self._last_trade_time).days < self.min_hold_days

    def OnData(self, data):
        if self.IsWarmingUp or not self._indicators_ready():
            return

        self._update_drawdown_guard()

        price_spy = self.Securities[self.symbols["SPY"]].Price
        price_qqq = self.Securities[self.symbols["QQQ"]].Price
        price_tqqq = self.Securities[self.symbols["TQQQ"]].Price

        if price_spy <= 0 or price_qqq <= 0 or price_tqqq <= 0:
            return

        rsi_qqq = self.indicators[self._rsi_key("QQQ")].Current.Value
        rsi_spy = self.indicators[self._rsi_key("SPY")].Current.Value
        rsi_tqqq = self.indicators[self._rsi_key("TQQQ")].Current.Value
        rsi_sqqq = self.indicators[self._rsi_key("SQQQ")].Current.Value
        rsi_uvxy = self.indicators[self._rsi_key("UVXY")].Current.Value

        sma_spy = self.indicators["SPY_SMA200"].Current.Value
        sma_qqq = self.indicators["QQQ_SMA20"].Current.Value
        sma_tqqq = self.indicators["TQQQ_SMA20"].Current.Value

        signal_target = None

        if price_spy > sma_spy:
            if rsi_qqq > self.th_rsi_qqq_bull_uvxy:
                signal_target = "UVXY"
            elif rsi_spy > self.th_rsi_spy_bull_uvxy:
                signal_target = "UVXY"
            else:
                signal_target = "TQQQ"
        else:
            if rsi_tqqq < self.th_rsi_tqqq_bear_tecl:
                signal_target = "TECL"
            elif rsi_spy < self.th_rsi_spy_bear_spxl:
                signal_target = "SPXL"
            elif rsi_uvxy > self.th_rsi_uvxy_elevated:
                if rsi_uvxy > self.th_rsi_uvxy_extreme:
                    if price_qqq > sma_qqq:
                        if rsi_sqqq < self.th_rsi_sqqq_tecs_qqq_above:
                            signal_target = "TECS"
                        else:
                            signal_target = "TECL"
                    else:
                        signal_target = self._get_max_rsi_ticker(
                            self._defensive_rsi_candidates())
                else:
                    signal_target = "UVXY"
            else:
                if price_tqqq > sma_tqqq:
                    if rsi_sqqq < self.th_rsi_sqqq_tecs_tqqq_above:
                        signal_target = "TECS"
                    else:
                        signal_target = "TECL"
                else:
                    signal_target = self._get_max_rsi_ticker(
                        self._defensive_rsi_candidates())

        if signal_target is None:
            return

        if self.use_drawdown_guard and self._drawdown_guard_active:
            target_ticker = self.risk_off_ticker
        else:
            target_ticker = signal_target

        if target_ticker == self._last_target_ticker:
            return

        if self._min_hold_blocks_switch():
            return

        self.SetHoldings(self.symbols[target_ticker], 1.0, True)
        self._last_target_ticker = target_ticker
        self._last_trade_time = self.Time

        guard = " [DD_GUARD]" if (
            self.use_drawdown_guard and self._drawdown_guard_active) else ""
        self.Debug(
            f"{self.Time:%Y-%m-%d} target={target_ticker}{guard} signal={signal_target}"
        )

    def _get_max_rsi_ticker(self, ticker_list):
        best = None
        highest = -1.0
        for ticker in ticker_list:
            rsi_val = self.indicators[self._rsi_key(ticker)].Current.Value
            if rsi_val > highest:
                highest = rsi_val
                best = ticker
        return best
