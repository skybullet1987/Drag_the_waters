import sys
from types import SimpleNamespace

sys.path.insert(0, "/workspace/examples_qc")


class _H(object):
    def __init__(self, invested):
        self.Invested = invested


class _Algo(object):
    def __init__(self):
        self.Portfolio = SimpleNamespace(
            Values=[_H(True)],
            TotalPortfolioValue=100000.0,
        )
        self.symbols = {"SOXL": "SOXL", "TQQQ": "TQQQ"}
        self._last_target_ticker = "TQQQ"
        self._defer_buy = None
        self.Time = SimpleNamespace(strftime=lambda f: "2026-05-26")
        self._last_executed_weight = 1.0
        self._last_trade_time = None
        self._clamp_calls = []

    def Liquidate(self):
        self.Portfolio.Values = []

    def _set_holdings_buying_power_clamped(self, sym, w, liq):
        self._clamp_calls.append((sym, w))
        return w * 0.95

    def Debug(self, msg):
        pass


def test_rotate_defers_when_invested():
    import csr_live_margin_ext as ext
    a = _Algo()
    assert ext.run_samebar_trade(a, "SOXL", 1.0, "SOXL") is True
    assert a._defer_buy == ("SOXL", 1.0, "SOXL")
    assert not a.Portfolio.Values[0].Invested if a.Portfolio.Values else True
    assert len(a._clamp_calls) == 0


def test_margin_safe_rotation_live_only():
    class _A(object):
        LiveMode = False

    def _rot(a):
        forced = getattr(a, "force_margin_safe_trades", None)
        if forced is not None:
            return bool(forced)
        return bool(getattr(a, "LiveMode", False))

    assert _rot(_A()) is False
    _A.LiveMode = True
    assert _rot(_A()) is True


def test_queue_live_signal_overwrites():
    import datetime
    import csr_live_margin_ext as ext

    class _A(object):
        Time = SimpleNamespace(date=datetime.date(2026, 5, 28))

    a = _A()
    ext.queue_live_signal(a, "SOXL", 1.0, "SOXL")
    ext.queue_live_signal(a, "TQQQ", 0.95, "TQQQ")
    assert a._live_day_signal[0] == "TQQQ"


def test_deferred_buy_when_flat():
    import csr_live_margin_ext as ext
    a = _Algo()
    a._defer_buy = ("SOXL", 1.0, "SOXL")
    a.Portfolio.Values = []
    assert ext.run_samebar_trade(a, "SOXL", 1.0, "SOXL") is True
    assert a._defer_buy is None
    assert a._last_target_ticker == "SOXL"
    assert len(a._clamp_calls) == 1
