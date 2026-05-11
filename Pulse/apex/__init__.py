"""Pulse.apex — edge-first crypto strategy.

Architecture:
  L1 DATA      → on-chain, ETF flows, stablecoin mints, news, macro
  L2 SIGNALS   → 12 normalized scores (pure-Python, unit-testable)
  L3 ML        → XGBoost + LightGBM + LogReg ensemble (joblib)
  L4 EXECUTION → 4h-scheduled engine with Kelly sizing & ATR trail

Lazy imports — no QC dependency at module load. Submodules import
QC lazily where needed.
"""

# Pure-Python primitives (no QC deps)
from Pulse.apex.config import (
    APEX_VERSION,
    APEX_ENTRY_THRESHOLD,
    APEX_EXIT_THRESHOLD,
    APEX_MAX_POSITIONS,
    APEX_KELLY_FRACTION,
    APEX_REBALANCE_HOURS,
    APEX_HOLD_DAYS_MAX,
    APEX_FEATURE_DIM,
)
from Pulse.apex.registry import (
    SignalScore,
    SignalRegistry,
    register_signal,
    list_registered_signals,
)
from Pulse.apex.feature_vector import (
    FeatureVector,
    build_feature_vector,
    feature_names,
    DEFAULT_FEATURE_NAMES,
)

__all__ = [
    "APEX_VERSION",
    "APEX_ENTRY_THRESHOLD",
    "APEX_EXIT_THRESHOLD",
    "APEX_MAX_POSITIONS",
    "APEX_KELLY_FRACTION",
    "APEX_REBALANCE_HOURS",
    "APEX_HOLD_DAYS_MAX",
    "APEX_FEATURE_DIM",
    "SignalScore",
    "SignalRegistry",
    "register_signal",
    "list_registered_signals",
    "FeatureVector",
    "build_feature_vector",
    "feature_names",
    "DEFAULT_FEATURE_NAMES",
]
