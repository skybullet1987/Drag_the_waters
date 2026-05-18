# region imports
from AlgorithmImports import *
from datetime import datetime

# endregion

# =============================================================================
# 100% TQQQ buy-and-hold baseline (QuantConnect research)
#
# Use the same project parameters as ConditionalSectorRotationImproved for dates
# and cash (start_year, start_month, start_day, end_year, end_month, end_day,
# starting_cash, run_to_present) so you can compare charts vs the rotation algo.
#
# Optional: trade_end_year/month/day — liquidate after that calendar date (OOS window).
# Optional: constant_slippage_per_share (e.g. 0.001) — applied before AddEquity via
# SetSecurityInitializer for closer live-style friction vs the main template.
#
# Benchmark defaults to QQQ (parameter benchmark_ticker: SPY, IWM, DIA, …).
# Do NOT set benchmark to TQQQ while holding 100% TQQQ — Alpha/PSR/IR vs yourself are meaningless.
#
# In QuantConnect: point the project main file at this class temporarily, or
# duplicate the project and swap the algorithm type.
# =============================================================================


class TqqqBuyHoldBenchmark(QCAlgorithm):

    def Initialize(self):
        sy = self._int_param("start_year", 2020)
        sm = self._int_param("start_month", 1)
        sd = self._int_param("start_day", 1)
        self.SetStartDate(sy, sm, sd)

        if not self._bool_param("run_to_present", False):
            ey = self._int_param("end_year", 2026)
            em = self._int_param("end_month", 5)
            ed = self._int_param("end_day", 17)
            self.SetEndDate(ey, em, ed)

        cash = max(1000, self._int_param("starting_cash", 100000))
        self.SetCash(cash)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin,
        )

        self._trade_end = self._parse_trade_end(sy, sm, sd)

        slip = max(0.0, self._float_param("constant_slippage_per_share", 0.0))
        self._slippage_dollars = slip
        if slip > 0.0:
            self.SetSecurityInitializer(self._equity_slippage_initializer)

        self._tqqq = self.AddEquity("TQQQ", Resolution.Daily).Symbol

        raw_bench = self.GetParameter("benchmark_ticker")
        s = "" if raw_bench is None else str(raw_bench).strip().upper()
        bench = s if s else "QQQ"
        if bench == "TQQQ":
            self.Debug(
                "benchmark_ticker=TQQQ with 100% TQQQ portfolio makes Alpha/PSR vs "
                "benchmark meaningless; using QQQ instead."
            )
            bench = "QQQ"
        self._bench = self.AddEquity(bench, Resolution.Daily).Symbol
        self.SetBenchmark(self._bench)
        self.Debug(f"TqqqBuyHoldBenchmark: hold TQQQ, benchmark={bench}")

        self.SetWarmUp(5, Resolution.Daily)

    def _equity_slippage_initializer(self, security):
        if security.Type != SecurityType.Equity:
            return
        if self._slippage_dollars <= 0.0:
            return
        security.SetSlippageModel(ConstantSlippageModel(self._slippage_dollars))

    def _parse_trade_end(self, sy, sm, sd):
        tey = self._int_param("trade_end_year", 0)
        if tey <= 0:
            return None
        tem = max(1, min(12, self._int_param("trade_end_month", 12)))
        ted = max(1, min(31, self._int_param("trade_end_day", 31)))
        end_dt = datetime(tey, tem, ted)
        start_dt = datetime(sy, sm, sd)
        if end_dt.date() < start_dt.date():
            self.Debug("trade_end before start — ignoring trade_end")
            return None
        return end_dt

    def _past_trade_end(self):
        if self._trade_end is None:
            return False
        return self.Time.date() > self._trade_end.date()

    def OnData(self, data):
        if self.IsWarmingUp:
            return
        if self._past_trade_end():
            if self.Portfolio.Invested:
                self.Liquidate()
            return
        if not self.Portfolio.Invested:
            self.SetHoldings(self._tqqq, 1.0, True)

    def _float_param(self, name, default):
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    def _int_param(self, name, default):
        raw = self.GetParameter(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _bool_param(self, name, default):
        raw = self.GetParameter(name)
        if raw is None:
            return default
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return default
