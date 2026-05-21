# QuantConnect: upload with main.py + csr_ml_overlay.py when ACTIVE_BASELINE=ml_overlay_diversified
try:
    from AlgorithmImports import Resolution
except ImportError:
    class Resolution(object):
        Daily = 0
# Bear / flat sleeve: DBMF, TLT, GLD, DBC, XLV (+ BSV fallback). ML overlay v2 defensive base.


class CSRDefensiveSleeveHelper(object):
    def __init__(self, algo):
        self.a = algo

    def apply_signal(self, signal):
        a = self.a
        if not getattr(a, "use_defensive_sleeve", False) or signal is None:
            return signal
        if a._effective_is_bull_regime():
            return signal
        redirect = getattr(
            a,
            "_defensive_sleeve_redirect",
            frozenset({"BSV", "TECS", "TLT", "GLD"}),
        )
        if signal == a.risk_off_ticker or signal in redirect:
            return self._pick()
        off = getattr(a, "_defensive_sleeve_offensive", frozenset({"TECL", "SPXL"}))
        if getattr(a, "defensive_flat_suppress_offensive", True) and self._is_flat():
            if signal in off:
                return self._pick()
        return signal

    def _is_flat(self):
        a = self.a
        if getattr(a, "use_probabilistic_regime", False):
            s = float(getattr(a, "_last_regime_score", 0.0))
            lo = float(getattr(a, "defensive_flat_score_min", 0.36))
            hi = float(getattr(a, "defensive_flat_score_max", 0.52))
            return lo <= s <= hi
        ps = a.Securities[a.symbols["SPY"]].Price
        sma = a.indicators["SPY_SMA200"].Current.Value
        if ps <= 0 or sma <= 0:
            return False
        band = float(getattr(a, "defensive_flat_spy_band", 0.025))
        return abs(ps / sma - 1.0) <= band

    def _pick(self):
        a = self.a
        tickers = getattr(
            a, "defensive_sleeve_tickers", ("DBMF", "TLT", "GLD", "DBC", "XLV", "BSV")
        )
        cands = [t for t in tickers if t in a.symbols]
        if not cands:
            return a.risk_off_ticker
        rsi_spy = a._rsi("SPY")
        spy_below = self._spy_below_sma200()
        vix_hot = self._vix_stress()
        scores = {}
        for t in cands:
            key = a._rsi_key(t)
            ind = a.indicators.get(key)
            if ind is None or not ind.IsReady:
                continue
            r = a._rsi(t)
            sc = 0.0
            if t == "DBMF":
                if vix_hot:
                    sc += 2.0
                if spy_below:
                    sc += 1.5
                sc += 0.25 * (r / 100.0)
            elif t == "TLT":
                if spy_below and self._mom(t, 20) > 0:
                    sc += 2.2
                if vix_hot:
                    sc += 1.0
            elif t == "GLD":
                if r > rsi_spy + 3:
                    sc += 2.0
                if spy_below:
                    sc += 1.0
                else:
                    sc += 0.4
            elif t == "DBC":
                m63 = self._mom(t, 63)
                if m63 > 0:
                    sc += 2.5
                if spy_below:
                    sc += 0.5
            elif t == "XLV":
                if r > 52 and rsi_spy < 50:
                    sc += 2.0
                if spy_below:
                    sc += 0.5
            elif t == "BSV":
                sc += 0.65
            scores[t] = sc
        if not scores:
            return a.risk_off_ticker
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def _spy_below_sma200(self):
        a = self.a
        key = "SPY_SMA200"
        if key not in a.indicators or not a.indicators[key].IsReady:
            return False
        ps = a.Securities[a.symbols["SPY"]].Price
        return ps > 0 and ps < a.indicators[key].Current.Value

    def _vix_stress(self):
        a = self.a
        if "VIX" not in a.symbols:
            return False
        vk = "VIX_SMA"
        if vk not in a.indicators or not a.indicators[vk].IsReady:
            return False
        vix = a.Securities[a.symbols["VIX"]].Price
        ratio = float(getattr(a, "defensive_vix_ratio", 1.1))
        return vix > 0 and vix > a.indicators[vk].Current.Value * ratio

    def _mom(self, ticker, days):
        a = self.a
        sym = a.symbols.get(ticker)
        if sym is None:
            return 0.0
        try:
            hist = a.History(sym, days + 2, Resolution.Daily)
        except Exception:
            return 0.0
        if hist is None or getattr(hist, "empty", True):
            return 0.0
        try:
            closes = hist["close"].dropna()
        except Exception:
            return 0.0
        if len(closes) < days + 1:
            return 0.0
        c0, c1 = float(closes.iloc[-(days + 1)]), float(closes.iloc[-1])
        if c0 <= 0:
            return 0.0
        return c1 / c0 - 1.0


def wire_defensive_sleeve(algo):
    algo._dsh = CSRDefensiveSleeveHelper(algo)


def apply_ml_defensive_diversified_profile(algo):
    from csr_ml_overlay import wire_ml_overlay

    wire_ml_overlay(algo)
    algo._mlh.apply_ml_defensive_profile()
    algo.Debug("ML_OVERLAY_DIV: v2 ML + bear/flat DBMF/TLT/GLD/DBC/XLV sleeve.")
    algo.use_defensive_sleeve = True
    algo.defensive_sleeve_tickers = ("DBMF", "TLT", "GLD", "DBC", "XLV", "BSV")
    algo._defensive_sleeve_extra = ("DBMF", "TLT", "GLD", "DBC", "XLV")
    algo.include_defensive_etfs = True
    algo.use_probabilistic_regime = True
    algo.regime_score_min_bull = 0.52
    algo.regime_hysteresis_days = 2
    algo.defensive_flat_score_min = 0.36
    algo.defensive_flat_score_max = 0.52
    algo.defensive_flat_spy_band = 0.025
    algo.defensive_flat_suppress_offensive = True
    algo.defensive_vix_ratio = 1.1
    algo.ml_bear_offensive_prob = 0.40
    algo._defensive_sleeve_redirect = frozenset({"BSV", "TECS", "TLT", "GLD"})
    algo._defensive_sleeve_offensive = frozenset({"TECL", "SPXL"})
