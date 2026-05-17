# region imports
from AlgorithmImports import *

# endregion

# =============================================================================
# 100% TQQQ buy-and-hold baseline (QuantConnect research)
#
# Use the same project parameters as ConditionalSectorRotationImproved for dates
# and cash (start_year, start_month, start_day, end_year, end_month, end_day,
# starting_cash, run_to_present) so you can compare charts vs the rotation algo.
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

    def OnData(self, data):
        if self.IsWarmingUp:
            return
        if not self.Portfolio.Invested:
            self.SetHoldings(self._tqqq, 1.0, True)

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
