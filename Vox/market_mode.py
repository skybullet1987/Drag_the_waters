# ── market_mode.py: MarketModeDetector ───────────────────────────────────────
#
# Moved from core.py to keep core.py under the QuantConnect 63KB file limit.
# Re-exported from core.py for backward compatibility.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from datetime import timedelta


"""
Market mode / regime detection for Vox ruthless mode.

Detects market regime from BTC price data and supplies a
mode string used by the ruthless confirmation gate.
"""


MARKET_MODES = ("risk_on_trend", "pump", "chop", "selloff", "high_vol_reversal")


class MarketModeDetector:
    """Lightweight rules-based market mode detector.

    Uses BTC 4-hour closes to classify the current regime into one of:
      risk_on_trend   — BTC trending up, moderate volatility
      pump            — BTC rising fast, high volume
      chop            — BTC oscillating, low directional momentum
      selloff         — BTC trending down
      high_vol_reversal — extreme volatility with conflicting signals

    Call update_btc() from initialize() to register a consolidator,
    or feed closes directly via detect() for testing.
    """

    _WINDOW = 24   # 4h bars ~ 4 days

    def __init__(self):
        self._closes  = deque(maxlen=self._WINDOW + 4)
        self._volumes = deque(maxlen=self._WINDOW + 4)
        self._mode    = "chop"   # default until enough data

    # ── Public interface ──────────────────────────────────────────────────────

    def update_btc(self, algorithm, btc_sym):
        """Register a 4-hour BTC consolidator feeding this detector."""
        algorithm.consolidate(
            btc_sym,
            timedelta(hours=4),
            self._on_4h_bar,
        )

    @property
    def mode(self):
        """Current detected market mode string."""
        return self._mode

    def detect(self, closes, volumes=None):
        """Classify regime from *closes* (and optionally *volumes*).
        Updates self._mode and returns the mode string.

        Parameters
        ----------
        closes  : array-like of float, at least 5 bars
        volumes : array-like of float or None
        """
        c = list(closes)
        v = list(volumes) if volumes else []
        if len(c) < 5:
            return "chop"

        ret_4  = (c[-1] - c[-5])  / c[-5]  if c[-5]  != 0 else 0.0
        ret_1  = (c[-1] - c[-2])  / c[-2]  if c[-2]  != 0 else 0.0

        if len(c) >= 13:
            ret_12 = (c[-1] - c[-13]) / c[-13] if c[-13] != 0 else 0.0
        else:
            ret_12 = ret_4

        # SMA slope
        if len(c) >= 10:
            sma_now  = float(np.mean(c[-5:]))
            sma_prev = float(np.mean(c[-10:-5]))
            sma_slope = (sma_now - sma_prev) / sma_prev if sma_prev != 0 else 0.0
        else:
            sma_slope = ret_4

        # Volatility (normalised std of last 8 rets)
        if len(c) >= 9:
            rets = np.diff(c[-9:]) / np.where(np.array(c[-9:-1]) != 0,
                                               np.array(c[-9:-1]), 1.0)
            vol = float(np.std(rets))
        else:
            vol = abs(ret_1)

        # Volume ratio
        vol_ratio = 1.0
        if len(v) >= 6:
            avg_v = float(np.mean(v[-6:-1]))
            if avg_v > 0:
                vol_ratio = min(float(v[-1]) / avg_v, 10.0)

        # Range efficiency (trend purity)
        if len(c) >= 9:
            net_move  = abs(c[-1] - c[-9])
            sum_moves = float(np.sum(np.abs(np.diff(c[-9:]))))
            range_eff = net_move / sum_moves if sum_moves > 0 else 0.0
        else:
            range_eff = 0.5

        # ── Classification rules ──────────────────────────────────────────────
        if ret_4 < -0.04 and sma_slope < -0.01:
            mode = "selloff"
        elif vol > 0.025 and range_eff < 0.30:
            mode = "high_vol_reversal"
        elif ret_4 > 0.05 and vol_ratio > 2.0:
            mode = "pump"
        elif ret_4 > 0.01 and sma_slope > 0.003 and range_eff > 0.40:
            mode = "risk_on_trend"
        else:
            mode = "chop"

        self._mode = mode
        return mode

    # ── Consolidator callback ─────────────────────────────────────────────────

    def _on_4h_bar(self, bar):
        self._closes.append(float(bar.close))
        self._volumes.append(float(bar.volume))
        self.detect(self._closes, self._volumes)
