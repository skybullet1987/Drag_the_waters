"""Unit tests for defensive sleeve scoring (no QuantConnect runtime)."""

import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/examples_qc")

from csr_defensive_sleeve_ext import CSRDefensiveSleeveHelper


class _Ind(object):
    def __init__(self, val, ready=True):
        self.Current = SimpleNamespace(Value=val)
        self.IsReady = ready


class _MockAlgo(object):
    def __init__(self):
        self.use_defensive_sleeve = True
        self.risk_off_ticker = "BSV"
        self.symbols = {t: t for t in ("SPY", "DBMF", "TLT", "GLD", "DBC", "XLV", "BSV")}
        self.indicators = {
            "SPY_SMA200": _Ind(400.0),
            "SPY_RSI_10_day": _Ind(45.0),
            "DBMF_RSI_10_day": _Ind(55.0),
            "TLT_RSI_10_day": _Ind(48.0),
            "GLD_RSI_10_day": _Ind(58.0),
            "DBC_RSI_10_day": _Ind(50.0),
            "XLV_RSI_10_day": _Ind(54.0),
            "BSV_RSI_10_day": _Ind(42.0),
            "VIX_SMA": _Ind(22.0),
        }
        self.Securities = {
            self.symbols["SPY"]: SimpleNamespace(Price=380.0),
            self.symbols.get("VIX", "VIX"): SimpleNamespace(Price=28.0),
        }
        self.symbols["VIX"] = "VIX"
        self._regime_bull_live = False
        self._last_regime_score = 0.44
        self.use_probabilistic_regime = True
        self.defensive_flat_score_min = 0.36
        self.defensive_flat_score_max = 0.52
        self.defensive_sleeve_tickers = ("DBMF", "TLT", "GLD", "DBC", "XLV", "BSV")
        self.regime_hysteresis_days = 2
        self._defensive_sleeve_redirect = frozenset({"BSV", "TECS", "TLT", "GLD"})
        self._defensive_sleeve_offensive = frozenset({"TECL", "SPXL"})
        self.defensive_flat_suppress_offensive = True

    def _effective_is_bull_regime(self):
        return self._regime_bull_live

    def _rsi_key(self, ticker):
        return "%s_RSI_10_day" % ticker

    def _rsi(self, ticker):
        return self.indicators[self._rsi_key(ticker)].Current.Value

    def History(self, sym, n, res):
        return None


def test_redirect_bsv_picks_ticker():
    algo = _MockAlgo()
    h = CSRDefensiveSleeveHelper(algo)
    out = h.apply_signal("BSV")
    assert out in algo.defensive_sleeve_tickers


def test_flat_suppresses_tecl():
    algo = _MockAlgo()
    h = CSRDefensiveSleeveHelper(algo)
    out = h.apply_signal("TECL")
    assert out != "TECL"
    assert out in algo.defensive_sleeve_tickers


def test_bull_passes_through():
    algo = _MockAlgo()
    algo._regime_bull_live = True
    h = CSRDefensiveSleeveHelper(algo)
    assert h.apply_signal("BSV") == "BSV"
