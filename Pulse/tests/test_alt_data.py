"""Tests for Pulse.alt_data — F&G signal layer."""

from __future__ import annotations

import pytest

from Pulse.alt_data import (
    FG_REGIMES,
    fg_regime, fg_size_multiplier, fg_max_positions_multiplier,
    fg_block_new_entries, fg_bias_toward_bounce,
    FGSignal,
)


# ───────────────────────────────────────────────────────────────────────────────
# fg_regime — boundary classification
# ───────────────────────────────────────────────────────────────────────────────

def test_fg_regimes_constants():
    assert FG_REGIMES == (
        "extreme_fear", "fear", "neutral", "greed", "extreme_greed",
    )


def test_fg_regime_extreme_fear():
    assert fg_regime(0)  == "extreme_fear"
    assert fg_regime(15) == "extreme_fear"
    assert fg_regime(25) == "extreme_fear"  # boundary inclusive


def test_fg_regime_fear():
    assert fg_regime(26) == "fear"
    assert fg_regime(40) == "fear"
    assert fg_regime(49) == "fear"


def test_fg_regime_neutral():
    assert fg_regime(50) == "neutral"


def test_fg_regime_greed():
    assert fg_regime(51) == "greed"
    assert fg_regime(60) == "greed"
    assert fg_regime(74) == "greed"


def test_fg_regime_extreme_greed():
    assert fg_regime(75) == "extreme_greed"   # boundary inclusive
    assert fg_regime(90) == "extreme_greed"
    assert fg_regime(100) == "extreme_greed"


def test_fg_regime_none_returns_neutral():
    assert fg_regime(None) == "neutral"


# ───────────────────────────────────────────────────────────────────────────────
# fg_size_multiplier
# ───────────────────────────────────────────────────────────────────────────────

def test_size_mult_extreme_fear_boost():
    """Extreme fear → 1.2× (capitulation buy)."""
    assert fg_size_multiplier(20) == 1.2


def test_size_mult_extreme_greed_halve():
    """Extreme greed → 0.5× (everyone is long; danger)."""
    assert fg_size_multiplier(85) == 0.5


def test_size_mult_neutral_unchanged():
    assert fg_size_multiplier(50) == 1.0


def test_size_mult_monotonic_decreasing():
    """As fear → greed, size multiplier should monotonically decrease."""
    s = [fg_size_multiplier(v) for v in [10, 30, 50, 60, 80, 95]]
    for i in range(1, len(s)):
        assert s[i] <= s[i - 1] + 1e-9


# ───────────────────────────────────────────────────────────────────────────────
# fg_max_positions_multiplier
# ───────────────────────────────────────────────────────────────────────────────

def test_max_positions_mult_extreme_greed_halves():
    assert fg_max_positions_multiplier(80) == 0.5


def test_max_positions_mult_greed_quarter_off():
    assert fg_max_positions_multiplier(60) == 0.75


def test_max_positions_mult_neutral_full():
    assert fg_max_positions_multiplier(50) == 1.0


def test_max_positions_mult_fear_no_expansion():
    """Even in fear we don't expand position count, just resize."""
    assert fg_max_positions_multiplier(20) == 1.0


# ───────────────────────────────────────────────────────────────────────────────
# fg_block_new_entries
# ───────────────────────────────────────────────────────────────────────────────

def test_block_new_entries_off_by_default_at_neutral():
    assert fg_block_new_entries(50) is False


def test_block_new_entries_blocks_at_panic_greed():
    assert fg_block_new_entries(95) is True


def test_block_new_entries_threshold_default_90():
    assert fg_block_new_entries(89) is False
    assert fg_block_new_entries(90) is True


def test_block_new_entries_custom_threshold():
    assert fg_block_new_entries(60, block_above=55) is True
    assert fg_block_new_entries(60, block_above=70) is False


def test_block_none_safe():
    assert fg_block_new_entries(None) is False


# ───────────────────────────────────────────────────────────────────────────────
# fg_bias_toward_bounce
# ───────────────────────────────────────────────────────────────────────────────

def test_bias_bounce_in_fear():
    assert fg_bias_toward_bounce(20) is True


def test_bias_bounce_not_in_neutral():
    assert fg_bias_toward_bounce(50) is False


def test_bias_bounce_default_threshold_30():
    assert fg_bias_toward_bounce(30) is True
    assert fg_bias_toward_bounce(31) is False


def test_bias_bounce_none_safe():
    assert fg_bias_toward_bounce(None) is False


# ───────────────────────────────────────────────────────────────────────────────
# FGSignal one-shot
# ───────────────────────────────────────────────────────────────────────────────

def test_fg_signal_from_extreme_fear():
    s = FGSignal.from_value(15)
    assert s.value == 15
    assert s.regime == "extreme_fear"
    assert s.size_multiplier == 1.2
    assert s.max_positions_multiplier == 1.0
    assert not s.block_new_entries
    assert s.bias_toward_bounce


def test_fg_signal_from_extreme_greed():
    s = FGSignal.from_value(85)
    assert s.regime == "extreme_greed"
    assert s.size_multiplier == 0.5
    assert s.max_positions_multiplier == 0.5
    assert not s.block_new_entries     # default block_above=90
    assert not s.bias_toward_bounce


def test_fg_signal_panic_greed_blocks():
    s = FGSignal.from_value(92)
    assert s.regime == "extreme_greed"
    assert s.block_new_entries


def test_fg_signal_neutral_default_unchanged():
    s = FGSignal.from_value(50)
    assert s.regime == "neutral"
    assert s.size_multiplier == 1.0
    assert s.max_positions_multiplier == 1.0


def test_fg_signal_none_value_safe_neutral():
    s = FGSignal.from_value(None)
    assert s.regime == "neutral"
    assert s.size_multiplier == 1.0


# ───────────────────────────────────────────────────────────────────────────────
# Integration: typical strategy use
# ───────────────────────────────────────────────────────────────────────────────

def test_strategy_use_pattern():
    """Realistic example: scalp engine sees F&G=80 and adjusts.

    base_max_positions=6, base_alloc=$500.
    Expected: max_positions = round(6 * 0.5) = 3
              alloc = 500 * 0.5 = 250
    """
    fg_value = 80   # extreme greed
    s = FGSignal.from_value(fg_value)
    base_max = 6
    base_alloc = 500.0
    new_max  = round(base_max * s.max_positions_multiplier)
    new_alloc = base_alloc * s.size_multiplier
    assert new_max == 3
    assert new_alloc == 250.0
    assert not s.block_new_entries


# ───────────────────────────────────────────────────────────────────────────────
# Tier C.4: Funding rate proxy
# ───────────────────────────────────────────────────────────────────────────────

from Pulse.alt_data import (
    FUNDING_REGIMES,
    funding_regime, funding_size_modifier,
    funding_block_new_entries, funding_bias_toward_bounce,
    FundingSignal,
)


def test_funding_regimes_constant():
    assert FUNDING_REGIMES == (
        "deep_short_squeeze", "shorts_paying", "balanced",
        "longs_paying", "deep_long_crowd",
    )


# ── Regime classification at boundary values ──────────────────────────────

def test_funding_regime_deep_short_squeeze():
    assert funding_regime(-0.0010) == "deep_short_squeeze"
    assert funding_regime(-0.0005) == "deep_short_squeeze"  # boundary


def test_funding_regime_shorts_paying():
    assert funding_regime(-0.0003) == "shorts_paying"
    assert funding_regime(-0.0001) == "shorts_paying"   # boundary


def test_funding_regime_balanced():
    assert funding_regime(0.0) == "balanced"
    assert funding_regime(-0.00005) == "balanced"
    assert funding_regime(0.00005) == "balanced"


def test_funding_regime_longs_paying():
    assert funding_regime(0.0001) == "longs_paying"   # boundary
    assert funding_regime(0.0003) == "longs_paying"


def test_funding_regime_deep_long_crowd():
    assert funding_regime(0.0005) == "deep_long_crowd"   # boundary
    assert funding_regime(0.0010) == "deep_long_crowd"


def test_funding_regime_none_returns_balanced():
    assert funding_regime(None) == "balanced"


# ── Size modifier (Pulse is long-only on Kraken cash) ────────────────────

def test_funding_size_mod_deep_squeeze_boosts():
    """Deep short squeeze = juicy long opportunity → 1.20×."""
    assert funding_size_modifier(-0.001) == 1.20


def test_funding_size_mod_deep_long_crowd_halves():
    """Crowded longs = danger → halve size."""
    assert funding_size_modifier(0.001) == 0.50


def test_funding_size_mod_balanced_is_one():
    assert funding_size_modifier(0.0) == 1.0


def test_funding_size_mod_monotonic_decreasing():
    """As funding rises (longs more crowded), size mult decreases."""
    rates = [-0.001, -0.0003, 0.0, 0.0003, 0.001]
    mods  = [funding_size_modifier(r) for r in rates]
    for i in range(1, len(mods)):
        assert mods[i] <= mods[i - 1] + 1e-9


# ── Hard block on extreme positive funding ────────────────────────────────

def test_funding_block_at_panic_positive():
    """Default block_above=+0.10% per 8h."""
    assert funding_block_new_entries(0.0015)  # 0.15% — block
    assert not funding_block_new_entries(0.0005)  # not yet


def test_funding_block_default_threshold():
    """Exactly at 0.10% → blocked (boundary inclusive)."""
    assert funding_block_new_entries(0.001)


def test_funding_block_none_safe():
    assert not funding_block_new_entries(None)


def test_funding_block_custom_threshold():
    assert funding_block_new_entries(0.0006, block_above=0.0005)
    assert not funding_block_new_entries(0.0006, block_above=0.001)


# ── Bias toward bounce when shorts crowded ───────────────────────────────

def test_funding_bias_bounce_when_shorts_squeezed():
    assert funding_bias_toward_bounce(-0.0010)


def test_funding_bias_no_bounce_at_balanced():
    assert not funding_bias_toward_bounce(0.0)


def test_funding_bias_bounce_default_threshold():
    """Default threshold -0.05% per 8h."""
    assert funding_bias_toward_bounce(-0.0005)
    assert not funding_bias_toward_bounce(-0.0004)


def test_funding_bias_none_safe():
    assert not funding_bias_toward_bounce(None)


# ── FundingSignal one-shot snapshot ───────────────────────────────────────

def test_funding_signal_from_deep_squeeze():
    s = FundingSignal.from_rate(-0.001)
    assert s.regime == "deep_short_squeeze"
    assert s.size_modifier == 1.20
    assert not s.block_new_entries
    assert s.bias_toward_bounce


def test_funding_signal_from_deep_long_crowd():
    s = FundingSignal.from_rate(0.0006)
    assert s.regime == "deep_long_crowd"
    assert s.size_modifier == 0.5
    assert not s.block_new_entries  # 0.06% not yet at panic threshold


def test_funding_signal_from_panic_long():
    s = FundingSignal.from_rate(0.0015)
    assert s.regime == "deep_long_crowd"
    assert s.block_new_entries


def test_funding_signal_balanced():
    s = FundingSignal.from_rate(0.0)
    assert s.regime == "balanced"
    assert s.size_modifier == 1.0
    assert not s.block_new_entries
    assert not s.bias_toward_bounce


def test_funding_signal_none_safe():
    s = FundingSignal.from_rate(None)
    assert s.regime == "balanced"
    assert s.size_modifier == 1.0


# ── Realistic strategy use ───────────────────────────────────────────────

def test_funding_strategy_pattern_combines_with_other_modifiers():
    """Show how funding combines multiplicatively with other size modifiers."""
    funding = FundingSignal.from_rate(-0.001)   # deep squeeze → 1.20×
    base_size = 500.0
    size_after_funding = base_size * funding.size_modifier
    assert size_after_funding == pytest.approx(600.0)


def test_funding_blocks_when_extreme_long_crowd():
    """At +0.15% funding, no new long entries even with strong scalp signal."""
    funding = FundingSignal.from_rate(0.0015)
    assert funding.block_new_entries
