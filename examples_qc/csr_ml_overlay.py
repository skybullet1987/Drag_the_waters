# region imports
from AlgorithmImports import *

# endregion

# QuantConnect: upload with main.py. No multiple inheritance — use CSRMLOverlayHelper + wire_ml_overlay().


class CSRMLOverlayHelper(object):
    """ML overlay via composition (QCAlgorithm cannot use Python mixins)."""

    def __init__(self, algo):
        self.a = algo

    def apply_ml_maximize_profile(self):
        a = self.a
        a.Debug("ML_OVERLAY: maximize + logistic filter (numpy, periodic retrain).")
        a.maximize_backtest_equity = True
        a._apply_maximize_backtest_equity_profile()
        a.disable_bull_uvxy = True
        a.use_ml_overlay = True
        a.ml_train_bars = 500
        a.ml_forward_days = 5
        a.ml_retrain_days = 63
        # v2: lighter touch — v1 veto@0.42/floor@0.45 capped ~$2.4M vs ~$6M maximize
        a.ml_veto_prob = 0.32
        a.ml_floor_mult = 0.78
        a.ml_boost_cap = 1.12
        a.ml_bear_offensive_prob = 0.38
        a.ml_filter_bear_offensive = True

    def maybe_train(self, force=False):
        a = self.a
        if not getattr(a, "use_ml_overlay", False) or a.IsWarmingUp:
            return
        d = a.Time.date()
        if not force and a._ml_last_train_day is not None:
            if (d - a._ml_last_train_day).days < int(a.ml_retrain_days):
                return
        w = self.fit_from_history()
        if w is not None:
            a._ml_weights = w
            a._ml_last_train_day = d
            a.Debug("ML_TRAIN prob=%.3f" % a._ml_last_prob)

    def hist_closes(self, ticker, n):
        a = self.a
        sym = a.symbols.get(ticker)
        if sym is None:
            return None
        hist = a.History(sym, n + 2, Resolution.Daily)
        if hist is None or getattr(hist, "empty", True):
            return None
        try:
            s = hist["close"].dropna()
        except Exception:
            try:
                s = hist.xs(ticker, level=0)["close"].dropna()
            except Exception:
                return None
        if s is None or len(s) < max(30, n // 3):
            return None
        return [float(x) for x in s.values[-n:]]

    @staticmethod
    def nret(c, d):
        if c is None or len(c) < d + 1:
            return 0.0
        o, p = c[-(d + 1)], c[-1]
        return (p / o - 1.0) if o > 0 and p > 0 else 0.0

    def build_xy(self, spy, qqq, tqqq, soxl, vix, i, fwd):
        if i < 220 or i + fwd >= len(tqqq):
            return None
        sp, st = spy[i], tqqq[i]
        if sp <= 0 or st <= 0:
            return None
        sma = sum(spy[i - 199 : i + 1]) / 200.0
        spy_tr = (sp / sma - 1.0) if sma > 0 else 0.0
        rs = [(spy[j] / spy[j - 1] - 1.0) for j in range(i - 19, i + 1) if spy[j - 1] > 0]
        rv = (sum(r * r for r in rs) / max(1, len(rs))) ** 0.5 * (252.0 ** 0.5) if rs else 0.2
        vz = 0.0
        if vix and len(vix) > i and i >= 20:
            vs = sum(vix[i - 19 : i + 1]) / 20.0
            if vs > 0:
                vz = vix[i] / vs - 1.0
        y = 1.0 if (tqqq[i + fwd] / st - 1.0) > 0 else 0.0
        x = [
            1.0,
            spy_tr,
            self.nret(qqq[: i + 1], 20),
            self.nret(tqqq[: i + 1], 20),
            self.nret(tqqq[: i + 1], 20) - self.nret(soxl[: i + 1], 20),
            rv,
            vz,
        ]
        return x, y

    def fit_from_history(self):
        import numpy as np

        a = self.a
        n = int(a.ml_train_bars) + 260
        spy = self.hist_closes("SPY", n)
        qqq = self.hist_closes("QQQ", n)
        tqqq = self.hist_closes("TQQQ", n)
        soxl = self.hist_closes("SOXL", n)
        vix = self.hist_closes("VIX", n) if "VIX" in a.symbols else None
        if not spy or not tqqq or len(tqqq) < 280:
            return None
        fwd = int(a.ml_forward_days)
        rows, ys = [], []
        for i in range(len(tqqq)):
            xy = self.build_xy(spy, qqq, tqqq, soxl, vix, i, fwd)
            if xy:
                rows.append(xy[0])
                ys.append(xy[1])
        if len(rows) < 80:
            return None
        X = np.asarray(rows, dtype=float)
        y = np.asarray(ys, dtype=float)
        w = np.zeros(X.shape[1])
        lr = 0.12 / max(1, len(y))
        for _ in range(100):
            z = np.clip(X.dot(w), -18.0, 18.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w -= lr * (X.T.dot(p - y) / len(y))
        a._ml_last_prob = float(1.0 / (1.0 + np.exp(-float(X[-1].dot(w)))))
        return [float(v) for v in w]

    def bull_probability(self):
        import numpy as np

        a = self.a
        if not getattr(a, "use_ml_overlay", False) or not a._ml_weights:
            return getattr(a, "_ml_last_prob", 0.5)
        spy = self.hist_closes("SPY", 230)
        qqq = self.hist_closes("QQQ", 30)
        tqqq = self.hist_closes("TQQQ", 30)
        soxl = self.hist_closes("SOXL", 30)
        vix = self.hist_closes("VIX", 30) if "VIX" in a.symbols else None
        if not spy or not tqqq:
            return a._ml_last_prob
        xy = self.build_xy(spy, qqq, tqqq, soxl, vix, len(tqqq) - 1, int(a.ml_forward_days))
        if xy is None:
            return a._ml_last_prob
        z = float(np.clip(np.dot(a._ml_weights, xy[0]), -18.0, 18.0))
        a._ml_last_prob = float(1.0 / (1.0 + np.exp(-z)))
        return a._ml_last_prob

    def overlay_multiplier(self, ticker):
        a = self.a
        if not getattr(a, "use_ml_overlay", False):
            return 1.0
        p = self.bull_probability()
        if ticker not in a._ml_bull_tickers:
            return 1.0
        if p < float(a.ml_veto_prob):
            return 0.0
        span = max(1e-6, 1.0 - float(a.ml_veto_prob))
        s = (p - float(a.ml_veto_prob)) / span
        lo, hi = float(a.ml_floor_mult), float(a.ml_boost_cap)
        return min(hi, max(lo, lo + (1.0 - lo) * s))

    def apply_signal_filter(self, signal):
        a = self.a
        if not getattr(a, "use_ml_overlay", False) or signal is None:
            return signal
        p = self.bull_probability()
        if signal in a._ml_bull_tickers and p < float(a.ml_veto_prob):
            return a.risk_off_ticker
        if getattr(a, "ml_filter_bear_offensive", True) and signal in a._ml_bear_offensive:
            bear_thr = float(getattr(a, "ml_bear_offensive_prob", 0.38))
            if p < bear_thr:
                return a.risk_off_ticker
        return signal


def wire_ml_overlay(algo):
    """Attach ML helper; call from Initialize when use_ml_overlay is True."""
    algo._mlh = CSRMLOverlayHelper(algo)


def apply_ml_maximize_profile(algo):
    """Entry for ACTIVE_BASELINE=ml_overlay (no mixin on QCAlgorithm)."""
    wire_ml_overlay(algo)
    algo._mlh.apply_ml_maximize_profile()
