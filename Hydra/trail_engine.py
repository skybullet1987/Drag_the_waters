# trail_engine.py — HYDRA dual-mode trailing stop engine
#
# TWO MODES:
#   SCALP: tight trail (1.5%) for weak signals — capture +2-3%
#   RUNNER: wide trail (5%) for strong signals — ride +10-20%
#
# The key insight: we were capturing +2% of +15% moves.
# Runner mode lets big pumps run while scalp mode handles small ones.

import numpy as np


class TrailEngine:
    """Dual-mode trailing stop: SCALP (tight) vs RUNNER (wide)."""

    # Scalp mode: quick in-and-out, small profits
    SCALP_SL       = 0.012   # -1.2% stop
    SCALP_BE       = 0.015   # breakeven at +1.5% (was 0.8% — too early, caused BE losses)
    SCALP_TRAIL_ARM = 0.018  # arm trail at +1.8%
    SCALP_TRAIL    = 0.012   # 1.2% trail

    # Runner mode: ride the pump, capture big moves
    RUNNER_SL      = 0.018   # -1.8% stop (tighter — limit runner losses)
    RUNNER_BE      = 0.02    # breakeven at +2%
    RUNNER_TRAIL_ARM = 0.035 # arm trail at +3.5% (was 4%)
    RUNNER_TRAIL   = 0.045   # 4.5% trail (was 5% — slightly tighter)
    RUNNER_BOOST_AT = 0.08   # at +8%: tighten trail to 3% to lock profits
    RUNNER_BOOST_TRAIL = 0.03

    EMERGENCY_SL   = 0.04    # -4% hard stop both modes

    def __init__(self, mode="scalp", **kwargs):
        self._mode = mode
        self._entry_px = 0.0
        self._high_water = 0.0
        self._be_active = False
        self._trail_active = False
        self._apply_mode(mode)

    def _apply_mode(self, mode):
        if mode == "runner":
            self._sl = self.RUNNER_SL
            self._be_at = self.RUNNER_BE
            self._trail_arm = self.RUNNER_TRAIL_ARM
            self._trail_pct = self.RUNNER_TRAIL
        else:
            self._sl = self.SCALP_SL
            self._be_at = self.SCALP_BE
            self._trail_arm = self.SCALP_TRAIL_ARM
            self._trail_pct = self.SCALP_TRAIL

    def reset(self, entry_px, mode="scalp"):
        self._entry_px = float(entry_px)
        self._high_water = float(entry_px)
        self._be_active = False
        self._trail_active = False
        self._mode = mode
        self._apply_mode(mode)

    def check(self, price, elapsed_hours=0, timeout_hours=36):
        if self._entry_px <= 0:
            return None

        price = float(price)
        ret = (price - self._entry_px) / self._entry_px

        if price > self._high_water:
            self._high_water = price

        max_ret = (self._high_water - self._entry_px) / self._entry_px

        # Emergency SL
        if ret <= -self.EMERGENCY_SL:
            return "EXIT_EMERGENCY_SL"

        # Regular SL
        if not self._be_active and ret <= -self._sl:
            return "EXIT_SL"

        # Breakeven: move SL to entry (no exit, just tighten stop)
        if not self._be_active and max_ret >= self._be_at:
            self._be_active = True
            self._sl = 0.002  # tighten SL to near-entry instead of exiting

        # Activate trail
        if not self._trail_active and max_ret >= self._trail_arm:
            self._trail_active = True

        # Runner boost: tighten trail after big gain to lock profits
        if self._mode == "runner" and self._trail_active and max_ret >= self.RUNNER_BOOST_AT:
            self._trail_pct = self.RUNNER_BOOST_TRAIL

        # Trailing stop
        if self._trail_active:
            trail_stop = self._high_water * (1 - self._trail_pct)
            if price <= trail_stop:
                return "EXIT_TRAIL"

        # Timeout
        if elapsed_hours >= timeout_hours:
            if abs(ret) > 0.003:
                return "EXIT_TIMEOUT"
            if elapsed_hours >= timeout_hours + 6:
                return "EXIT_TIMEOUT"

        return None

    @property
    def mode(self):
        return self._mode

    @property
    def is_trailing(self):
        return self._trail_active

    @property
    def high_water(self):
        return self._high_water
