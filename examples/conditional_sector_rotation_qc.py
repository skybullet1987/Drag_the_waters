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
# - **Large drawdowns** (e.g. 50–60%+) are common for single-ticker leveraged
#   rotation; the backtest assumes you **fully held** the plan through recovery.
# - **Alpha/Beta near zero** often reflects **benchmark / reporting setup**, not
#   “market neutral.” Call `SetBenchmark` (done below) so QC can attribute risk.
# - **Strategy capacity** on QC flags where **size** hits **liquidity** (UVXY,
#   small inverse funds). Irrelevant at small AUM; critical if you scale.
# - **Many fixed RSI/SMA cutoffs** = **overfitting risk**. Use **walk-forward**
#   (train on an early window, validate on a later one) and **parameter sweeps**
#   (nudge thresholds ±5–10%; if results collapse, the rule set is fragile).
# - **Leveraged / inverse daily-reset ETFs** are **path-dependent**; long samples
#   still do not guarantee future paths. Stress **2008, 2018, 2022** separately.
#
# =============================================================================
# IDEAS TO IMPROVE (implemented vs research-only)
# =============================================================================
# Implemented in this file:
#   - Parameterized thresholds (sensitivity / OOS testing without code edits).
#   - **SetBenchmark(SPY)** for more meaningful risk stats vs raw zeros.
#   - **Trade only when the target sleeve changes** (fewer redundant orders).
#   - Optional **min_hold_days** to reduce flip-flopping and fee churn.
#   - Optional **TLT / GLD** in the defensive “max RSI” basket (see ETFs section).
#   - Optional **run_to_present** to omit `SetEndDate` for rolling research.
#
# Research you still do in QC (not auto-coded here):
#   - Walk-forward / rolling train–test; Monte Carlo on returns.
#   - Realism: your IB **commission tier**, **margin**, **partial fills**.
#   - Vol targeting overlay, max DD circuit breaker, multi-sleeve weights (!=100%
#     one name), or regime filter on realized vol.
#
# =============================================================================
# DO YOU NEED MORE ETFs?
# =============================================================================
# **Not automatically.** More symbols add **degrees of freedom** → easier to
# **overfit** unless each sleeve has a **clear economic role**.
# - This strategy is **100% one ETF at a time**; extra tickers only help if the
#   **logic** can choose them (e.g. bonds/gold as **risk-off** when vol is high).
# - Optional **TLT** (long Treasuries) and **GLD** (gold) are wired into the
#   **max-RSI defensive tie-break** branches when `include_defensive_etfs=true`.
#   They are **not** a full risk-parity redesign—just more **escape valves**.
# - If you want true diversification, consider a **separate** template: static
#   or vol-weighted **multi-asset** basket (e.g. equity + bonds + gold) with
#   **infrequent** rebalance, instead of piling symbols into this RSI tree.
#
# =============================================================================
# PARAMETERS (QC Project → Parameters; all optional)
# =============================================================================
# Integers:
#   rsi_period (default 10), spy_sma_period (200), qqq_sma_period (20),
#   tqqq_sma_period (20), min_hold_days (0 = off)
# Floats (RSI / SMA tree):
#   th_rsi_qqq_bull_uvxy (81), th_rsi_spy_bull_uvxy (80),
#   th_rsi_tqqq_bear_tecl (30), th_rsi_spy_bear_spxl (30),
#   th_rsi_uvxy_elevated (74), th_rsi_uvxy_extreme (84),
#   th_rsi_sqqq_tecs_qqq_above (31), th_rsi_sqqq_tecs_tqqq_above (34)
# Strings / bool-like:
#   run_to_present = "true"  → do not call SetEndDate (backtest to “now”).
#   include_defensive_etfs = "true" → add TLT, GLD to universe & max-RSI picks.
#
# =============================================================================
# DISCLAIMER
# =============================================================================
# Educational / research only. Past performance does not guarantee future
# results. Leveraged and inverse ETFs can lose most or all of their value
# intraday in extreme moves.

from AlgorithmImports import *


class ConditionalSectorRotation(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2012, 1, 1)
        if not self._bool_parameter("run_to_present", False):
            self.SetEndDate(2024, 12, 31)

        self.SetCash(100000)
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

        self.tickers = [
            "SPY", "QQQ", "TQQQ", "UVXY",
            "TECL", "SPXL", "SQQQ", "TECS", "BSV",
        ]
        if self.include_defensive_etfs:
            self.tickers.extend(["TLT", "GLD"])

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

    def _min_hold_blocks_switch(self):
        if self.min_hold_days <= 0 or self._last_trade_time is None:
            return False
        return (self.Time - self._last_trade_time).days < self.min_hold_days

    def OnData(self, data):
        if self.IsWarmingUp or not self._indicators_ready():
            return

        price_spy = self.Securities[self.symbols["SPY"]].Price
        price_qqq = self.Securities[self.symbols["QQQ"]].Price
        price_tqqq = self.Securities[self.symbols["TQQQ"]].Price

        rsi_qqq = self.indicators[self._rsi_key("QQQ")].Current.Value
        rsi_spy = self.indicators[self._rsi_key("SPY")].Current.Value
        rsi_tqqq = self.indicators[self._rsi_key("TQQQ")].Current.Value
        rsi_sqqq = self.indicators[self._rsi_key("SQQQ")].Current.Value
        rsi_uvxy = self.indicators[self._rsi_key("UVXY")].Current.Value

        sma_spy = self.indicators["SPY_SMA200"].Current.Value
        sma_qqq = self.indicators["QQQ_SMA20"].Current.Value
        sma_tqqq = self.indicators["TQQQ_SMA20"].Current.Value

        target_ticker = None

        if price_spy > sma_spy:
            if rsi_qqq > self.th_rsi_qqq_bull_uvxy:
                target_ticker = "UVXY"
            elif rsi_spy > self.th_rsi_spy_bull_uvxy:
                target_ticker = "UVXY"
            else:
                target_ticker = "TQQQ"
        else:
            if rsi_tqqq < self.th_rsi_tqqq_bear_tecl:
                target_ticker = "TECL"
            elif rsi_spy < self.th_rsi_spy_bear_spxl:
                target_ticker = "SPXL"
            elif rsi_uvxy > self.th_rsi_uvxy_elevated:
                if rsi_uvxy > self.th_rsi_uvxy_extreme:
                    if price_qqq > sma_qqq:
                        if rsi_sqqq < self.th_rsi_sqqq_tecs_qqq_above:
                            target_ticker = "TECS"
                        else:
                            target_ticker = "TECL"
                    else:
                        target_ticker = self._get_max_rsi_ticker(
                            self._defensive_rsi_candidates())
                else:
                    target_ticker = "UVXY"
            else:
                if price_tqqq > sma_tqqq:
                    if rsi_sqqq < self.th_rsi_sqqq_tecs_tqqq_above:
                        target_ticker = "TECS"
                    else:
                        target_ticker = "TECL"
                else:
                    target_ticker = self._get_max_rsi_ticker(
                        self._defensive_rsi_candidates())

        if target_ticker is None:
            return

        if target_ticker == self._last_target_ticker:
            return

        if self._min_hold_blocks_switch():
            return

        self.SetHoldings(self.symbols[target_ticker], 1.0, True)
        self._last_target_ticker = target_ticker
        self._last_trade_time = self.Time
        self.Debug(f"{self.Time:%Y-%m-%d} target={target_ticker}")

    def _get_max_rsi_ticker(self, ticker_list):
        best = None
        highest = -1.0
        for ticker in ticker_list:
            rsi_val = self.indicators[self._rsi_key(ticker)].Current.Value
            if rsi_val > highest:
                highest = rsi_val
                best = ticker
        return best
