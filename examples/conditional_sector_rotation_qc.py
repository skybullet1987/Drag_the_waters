# Conditional sector rotation — QuantConnect / IB drop-in example
#
# How to use
# -----------
# 1. Create a new Python project on QuantConnect (or open LEAN locally with QC data).
# 2. Replace main.py contents with this file, **or** paste the class below into your
#    algorithm module and set the project entry class name to match.
# 3. For live trading with Interactive Brokers, deploy from the QC cloud and select
#    Interactive Brokers as the brokerage (US equity symbols as written).
#
# Notes
# -----
# - Uses **US-listed** ETFs (SPY, QQQ, TQQQ, …). Symbols and margin rules on IB
#   depend on your account jurisdiction (e.g. Canada often trades these in USD).
# - Leveraged and inverse ETFs reset daily; long backtests are for research only.
# - Parameters (optional, set in QC project Parameters): rsi_period, spy_sma_period,
#   qqq_sma_period, tqqq_sma_period — all integers.

from AlgorithmImports import *


class ConditionalSectorRotation(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2012, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Closer fee/slippage assumptions when routing live to IB (backtest only).
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

        self.rsi_period = self._int_parameter("rsi_period", 10)
        self.spy_sma_period = self._int_parameter("spy_sma_period", 200)
        self.qqq_sma_period = self._int_parameter("qqq_sma_period", 20)
        self.tqqq_sma_period = self._int_parameter("tqqq_sma_period", 20)

        self.tickers = [
            "SPY", "QQQ", "TQQQ", "UVXY",
            "TECL", "SPXL", "SQQQ", "TECS", "BSV",
        ]

        self.symbols = {}
        self.indicators = {}

        for ticker in self.tickers:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.symbols[ticker] = symbol
            key = self._rsi_key(ticker)
            self.indicators[key] = self.RSI(
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

        # Enough daily bars for 200+ period SMA/RSI; increase if you raise spy_sma_period.
        self.SetWarmUp(260, Resolution.Daily)

    def _int_parameter(self, name, default):
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _rsi_key(self, ticker):
        return f"{ticker}_RSI_{self.rsi_period}_day"

    def _indicators_ready(self):
        for ind in self.indicators.values():
            if not ind.IsReady:
                return False
        return True

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
            if rsi_qqq > 81:
                target_ticker = "UVXY"
            elif rsi_spy > 80:
                target_ticker = "UVXY"
            else:
                target_ticker = "TQQQ"
        else:
            if rsi_tqqq < 30:
                target_ticker = "TECL"
            elif rsi_spy < 30:
                target_ticker = "SPXL"
            elif rsi_uvxy > 74:
                if rsi_uvxy > 84:
                    if price_qqq > sma_qqq:
                        if rsi_sqqq < 31:
                            target_ticker = "TECS"
                        else:
                            target_ticker = "TECL"
                    else:
                        target_ticker = self._get_max_rsi_ticker(["TECS", "BSV"])
                else:
                    target_ticker = "UVXY"
            else:
                if price_tqqq > sma_tqqq:
                    if rsi_sqqq < 34:
                        target_ticker = "TECS"
                    else:
                        target_ticker = "TECL"
                else:
                    target_ticker = self._get_max_rsi_ticker(["TECS", "BSV"])

        if target_ticker is None:
            return

        # Third argument: liquidate other holdings first (100% in one symbol).
        self.SetHoldings(self.symbols[target_ticker], 1.0, True)

    def _get_max_rsi_ticker(self, ticker_list):
        best = None
        highest = -1.0
        for ticker in ticker_list:
            rsi_val = self.indicators[self._rsi_key(ticker)].Current.Value
            if rsi_val > highest:
                highest = rsi_val
                best = ticker
        return best
