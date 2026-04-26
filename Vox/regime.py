# ── Vox Regime Filter ─────────────────────────────────────────────────────────
#
# Uses a 4-hour BTC SMA(20) + linear slope gate to block alt-coin longs during
# bearish Bitcoin macro regimes.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from AlgorithmImports import *


class RegimeFilter:
    """
    Bitcoin-based macro regime filter.

    Logic
    -----
    The filter collects 4-hour BTC OHLCV bars via a QC consolidator.  On each
    decision call it evaluates two conditions:

    1. **SMA gate** — the latest 4h close must be *above* the 20-bar SMA.
    2. **Slope gate** — a linear least-squares slope over the last 5 bars must
       be positive (i.e. BTC is in an upward trend on the 4h frame).

    If either condition fails the filter returns ``False`` (block alt longs).
    BTC itself is always permitted regardless of regime.

    Insufficient history (< 20 bars) is treated as "risk on" so that the
    strategy can start trading during warm-up rather than sitting out entirely.

    Usage
    -----
    In ``initialize``::

        self._regime = RegimeFilter()
        self._regime.update_btc(self, self._btc_sym)

    In ``on_data``::

        if not self._regime.is_risk_on(sym):
            continue
    """

    _SMA_PERIOD   = 20
    _SLOPE_PERIOD = 5

    def __init__(self):
        self._closes = deque(maxlen=self._SMA_PERIOD + self._SLOPE_PERIOD)

    # ── Consolidator wiring ───────────────────────────────────────────────────

    def update_btc(self, algorithm, btc_sym):
        """
        Register a 4-hour consolidator on *btc_sym* that feeds this filter.

        Call once from ``algorithm.initialize()``.

        Parameters
        ----------
        algorithm : QCAlgorithm
        btc_sym   : Symbol — the BTC/USD symbol object returned by add_crypto.
        """
        algorithm.consolidate(
            btc_sym,
            timedelta(hours=4),
            self._on_4h_bar,
        )

    def _on_4h_bar(self, bar):
        """Receive a closed 4-hour bar and append the close price."""
        self._closes.append(float(bar.close))

    # ── Public API ────────────────────────────────────────────────────────────

    def is_risk_on(self, btc_sym, sym=None):
        """
        Return True if the macro regime permits a long entry.

        Parameters
        ----------
        btc_sym : Symbol — the BTC symbol used to bypass the filter for BTC.
        sym     : Symbol or None — the symbol being evaluated.  If it equals
                  *btc_sym*, the method always returns True.

        Returns
        -------
        bool
        """
        # BTC itself is exempt from the regime gate
        if sym is not None and sym == btc_sym:
            return True

        closes = list(self._closes)

        # Not enough history → allow trading
        if len(closes) < self._SMA_PERIOD:
            return True

        # ── SMA(20) gate ──────────────────────────────────────────────────────
        sma20 = float(np.mean(closes[-self._SMA_PERIOD:]))
        latest = closes[-1]
        if latest < sma20:
            return False

        # ── Linear slope gate (last 5 bars) ───────────────────────────────────
        if len(closes) >= self._SLOPE_PERIOD:
            window = np.asarray(closes[-self._SLOPE_PERIOD:], dtype=float)
            x      = np.arange(self._SLOPE_PERIOD, dtype=float)
            slope  = float(np.polyfit(x, window, 1)[0])
            if slope < 0.0:
                return False

        return True
