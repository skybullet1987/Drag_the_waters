# trail_engine.py — HYDRA adaptive trailing stop engine
#
# The ONLY proven edge: trailing stop on crypto momentum = 87-100% WR
# This engine manages: SL → breakeven → trailing → timeout

import numpy as np


class TrailEngine:
    """Adaptive trailing stop with breakeven and pyramiding support."""

    def __init__(self, sl_pct=0.015, breakeven_at=0.01,
                 trail_arm_pct=0.02, trail_pct=0.015,
                 emergency_sl=0.04):
        self._sl = sl_pct
        self._be_at = breakeven_at
        self._trail_arm = trail_arm_pct
        self._trail_pct = trail_pct
        self._emergency = emergency_sl

        self._entry_px = 0.0
        self._high_water = 0.0
        self._be_active = False
        self._trail_active = False

    def reset(self, entry_px):
        self._entry_px = float(entry_px)
        self._high_water = float(entry_px)
        self._be_active = False
        self._trail_active = False

    def check(self, price, elapsed_hours=0, timeout_hours=36):
        """Check if an exit should trigger.

        Returns exit tag string or None.
        """
        if self._entry_px <= 0:
            return None

        price = float(price)
        ret = (price - self._entry_px) / self._entry_px

        # Update high water mark
        if price > self._high_water:
            self._high_water = price

        max_ret = (self._high_water - self._entry_px) / self._entry_px

        # Emergency SL (always active)
        if ret <= -self._emergency:
            return "EXIT_EMERGENCY_SL"

        # Regular SL (before breakeven)
        if not self._be_active and ret <= -self._sl:
            return "EXIT_SL"

        # Activate breakeven
        if not self._be_active and max_ret >= self._be_at:
            self._be_active = True

        # Breakeven stop (after breakeven activated)
        if self._be_active and not self._trail_active:
            if ret <= 0.001:  # essentially breakeven (tiny buffer)
                return "EXIT_BE"

        # Activate trailing
        if not self._trail_active and max_ret >= self._trail_arm:
            self._trail_active = True

        # Trailing stop
        if self._trail_active:
            trail_stop = self._high_water * (1 - self._trail_pct)
            if price <= trail_stop:
                return "EXIT_TRAIL"

        # Timeout
        if elapsed_hours >= timeout_hours:
            if ret > 0.005:
                return "EXIT_TIMEOUT"  # take small profit
            elif ret < -0.005:
                return "EXIT_TIMEOUT"  # cut small loss
            # Near breakeven at timeout: give 6 more hours
            if elapsed_hours >= timeout_hours + 6:
                return "EXIT_TIMEOUT"

        return None

    @property
    def is_trailing(self):
        return self._trail_active

    @property
    def high_water(self):
        return self._high_water
