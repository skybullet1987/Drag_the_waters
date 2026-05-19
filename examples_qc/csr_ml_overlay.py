# region imports
from AlgorithmImports import *
# endregion

class CSRMLOverlayMixin(object):
    def _apply_ml_maximize_profile(self):
        self.Debug("ML_OVERLAY: maximize + logistic filter (numpy, periodic retrain).")
        self.maximize_backtest_equity = True
        self._apply_maximize_backtest_equity_profile()
        self.disable_bull_uvxy = True
        self.use_ml_overlay = True
        self.ml_train_bars = 500
        self.ml_forward_days = 5
        self.ml_retrain_days = 63
        self.ml_veto_prob = 0.42
        self.ml_floor_mult = 0.45
        self.ml_boost_cap = 1.08
        self.ml_filter_bear_offensive = True

    def _ml_maybe_train(self, force=False):
        if not getattr(self, "use_ml_overlay", False) or self.IsWarmingUp:
            return
        d = self.Time.date()
        if not force and self._ml_last_train_day is not None:
            if (d - self._ml_last_train_day).days < int(self.ml_retrain_days):
                return
        w = self._ml_fit_from_history()
        if w is not None:
            self._ml_weights = w
            self._ml_last_train_day = d
            self.Debug("ML_TRAIN prob=%.3f" % self._ml_last_prob)

    def _ml_hist_closes(self, ticker, n):
        sym = self.symbols.get(ticker)
        if sym is None:
            return None
        hist = self.History(sym, n + 2, Resolution.Daily)
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
    def _ml_nret(c, d):
        if c is None or len(c) < d + 1:
            return 0.0
        a, b = c[-(d + 1)], c[-1]
        return (b / a - 1.0) if a > 0 and b > 0 else 0.0

    def _ml_build_xy(self, spy, qqq, tqqq, soxl, vix, i, fwd):
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
        x = [1.0, spy_tr, self._ml_nret(qqq[: i + 1], 20), self._ml_nret(tqqq[: i + 1], 20),
             self._ml_nret(tqqq[: i + 1], 20) - self._ml_nret(soxl[: i + 1], 20), rv, vz]
        return x, y

    def _ml_fit_from_history(self):
        import numpy as np
        n = int(self.ml_train_bars) + 260
        spy = self._ml_hist_closes("SPY", n)
        qqq = self._ml_hist_closes("QQQ", n)
        tqqq = self._ml_hist_closes("TQQQ", n)
        soxl = self._ml_hist_closes("SOXL", n)
        vix = self._ml_hist_closes("VIX", n) if "VIX" in self.symbols else None
        if not spy or not tqqq or len(tqqq) < 280:
            return None
        fwd = int(self.ml_forward_days)
        rows, ys = [], []
        for i in range(len(tqqq)):
            xy = self._ml_build_xy(spy, qqq, tqqq, soxl, vix, i, fwd)
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
        self._ml_last_prob = float(1.0 / (1.0 + np.exp(-float(X[-1].dot(w)))))
        return [float(v) for v in w]

    def _ml_bull_probability(self):
        if not getattr(self, "use_ml_overlay", False) or not self._ml_weights:
            return getattr(self, "_ml_last_prob", 0.5)
        import numpy as np
        spy = self._ml_hist_closes("SPY", 230)
        qqq = self._ml_hist_closes("QQQ", 30)
        tqqq = self._ml_hist_closes("TQQQ", 30)
        soxl = self._ml_hist_closes("SOXL", 30)
        vix = self._ml_hist_closes("VIX", 30) if "VIX" in self.symbols else None
        if not spy or not tqqq:
            return self._ml_last_prob
        xy = self._ml_build_xy(spy, qqq, tqqq, soxl, vix, len(tqqq) - 1, int(self.ml_forward_days))
        if xy is None:
            return self._ml_last_prob
        z = float(np.clip(np.dot(self._ml_weights, xy[0]), -18.0, 18.0))
        self._ml_last_prob = float(1.0 / (1.0 + np.exp(-z)))
        return self._ml_last_prob

    def _ml_overlay_multiplier(self, ticker):
        if not getattr(self, "use_ml_overlay", False):
            return 1.0
        p = self._ml_bull_probability()
        if ticker not in self._ml_bull_tickers:
            return 1.0
        if p < float(self.ml_veto_prob):
            return 0.0
        span = max(1e-6, 1.0 - float(self.ml_veto_prob))
        s = (p - float(self.ml_veto_prob)) / span
        lo, hi = float(self.ml_floor_mult), float(self.ml_boost_cap)
        return min(hi, max(lo, lo + (1.0 - lo) * s))

    def _ml_apply_signal_filter(self, signal):
        if not getattr(self, "use_ml_overlay", False) or signal is None:
            return signal
        p = self._ml_bull_probability()
        if signal in self._ml_bull_tickers and p < float(self.ml_veto_prob):
            return self.risk_off_ticker
        if getattr(self, "ml_filter_bear_offensive", True) and signal in self._ml_bear_offensive:
            if p < 0.50:
                return self.risk_off_ticker
        return signal
