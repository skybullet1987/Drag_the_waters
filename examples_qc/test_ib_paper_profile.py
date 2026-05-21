"""Profile flags for IB paper preset (no QuantConnect)."""

import sys
import types


class _Algo(object):
    max_drawdown_pct = 0.35
    vol_etp_confirm_days = 0
    min_hold_days = 0
    maximize_backtest_equity = True
    use_eod_next_bar_execution = False

    def Debug(self, msg):
        pass


def test_ib_paper_profile_flags(monkeypatch):
    fake_ml = types.ModuleType("csr_ml_overlay")
    fake_ml.wire_ml_overlay = lambda a: None

    def _prod(a):
        a.use_eod_next_bar_execution = True
        a.maximize_backtest_equity = False

    fake_prof = types.ModuleType("csr_profiles")
    fake_prof.apply_production_safe_profile = _prod
    monkeypatch.setitem(sys.modules, "csr_ml_overlay", fake_ml)
    monkeypatch.setitem(sys.modules, "csr_profiles", fake_prof)
    sys.modules.pop("csr_ib_paper_ext", None)
    import csr_ib_paper_ext as ext

    a = _Algo()
    ext.apply_ib_paper_profile(a)
    assert a.ib_paper_active is True
    assert a.use_ml_overlay is True
    assert a.maximize_backtest_equity is False
    assert a.use_eod_next_bar_execution is True
    assert a.ml_overlay_mode == "defensive"
    assert a.ml_veto_prob == 0.32
