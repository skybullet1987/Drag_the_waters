"""Pulse.apex.integration — one-call bootstrap that wires Apex into Pulse.

Usage from PulseAlgorithm.Initialize():

    from Pulse.apex.integration import bootstrap_apex, ApexBundle
    self._apex: ApexBundle = bootstrap_apex(self)

Then in a 4-hour scheduled callback:

    self._apex.engine.on_4h_tick(
        now=self.Time,
        universe=[s.Value for s in self._curated_universe],
        market_context_provider=self._apex.context_provider,
        equity=float(self.Portfolio.TotalPortfolioValue),
        place_order_fn=self._apex.place_order,
        ...
    )

And in OnData (every minute):

    self._apex.engine.on_minute_tick(
        now=self.Time,
        current_prices=current_prices,
        latest_probs=self._apex.latest_probs,
        place_exit_fn=self._apex.place_exit,
    )

The bundle holds all the in-process stores + signal registry + engine
+ data feeders. Each store handles its own update_from_slice / record
calls — main.py just feeds them at the right cadence.

Designed to be imported lazily so unit tests of Pulse.apex don't depend
on Pulse.main.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from Pulse.apex.config import (
    APEX_ENTRY_THRESHOLD, APEX_EXIT_THRESHOLD, APEX_MAX_POSITIONS,
    APEX_KELLY_FRACTION, APEX_MODEL_FILENAME, APEX_MODEL_REQUIRED,
    APEX_REBALANCE_HOURS, APEX_UNIVERSE_TIERS,
)
from Pulse.apex.registry import SignalRegistry
from Pulse.apex.feature_vector import build_feature_vector, DEFAULT_SIGNAL_ORDER
from Pulse.apex.ml.predict import ApexInference
from Pulse.apex.apex_engine import ApexEngine

# Native QC signals
from Pulse.apex.signals.btc_onchain import (
    BitcoinMetadataHistoryStore, register_btc_onchain,
)
from Pulse.apex.signals.btc_dominance import (
    CoinGeckoDominanceStore, register_btc_dominance,
)
from Pulse.apex.signals.funding_native import (
    BinanceFundingRateStore, register_funding_native,
)
from Pulse.apex.signals.cross_asset import (
    CrossAssetSeriesStore, register_cross_asset,
)

# Custom data signals
from Pulse.apex.data.etf_flows import (
    ETFFlowStore, register_etf_flow,
)
from Pulse.apex.data.stablecoin_supply import (
    StablecoinSupplyStore, register_stablecoin,
)
from Pulse.apex.data.token_unlocks import (
    UnlockCalendarStore, register_token_unlock,
)
from Pulse.apex.data.news_sentiment import (
    NewsSentimentStore, register_news_sentiment,
)
from Pulse.apex.data.onchain_valuation import (
    OnchainValuationStore, register_mvrv,
)


# ─── Bundle dataclass ────────────────────────────────────────────────────────


@dataclass
class ApexBundle:
    """Everything bootstrap_apex constructs. Stored on the PulseAlgorithm."""
    registry:             SignalRegistry
    engine:               ApexEngine
    inference:            ApexInference

    # Per-signal data stores — main.py feeds these as data arrives
    btc_onchain_store:    BitcoinMetadataHistoryStore
    dominance_store:      CoinGeckoDominanceStore
    funding_store:        BinanceFundingRateStore
    cross_asset_store:    CrossAssetSeriesStore
    etf_flow_store:       ETFFlowStore
    stablecoin_store:     StablecoinSupplyStore
    unlock_store:         UnlockCalendarStore
    news_store:           NewsSentimentStore
    onchain_val_store:    OnchainValuationStore

    # Latest scored probabilities (refreshed at every 4h tick) — used
    # by the per-minute exit pass for prob-flip logic.
    latest_probs:         dict = field(default_factory=dict)

    # Wire-in callables provided by the host (PulseAlgorithm)
    place_order:          Optional[Callable[[str, float, str, float], None]] = None
    place_exit:           Optional[Callable[[str, float, str], None]] = None
    context_provider:     Optional[Callable[[str], dict]] = None
    now_provider:         Optional[Callable[[dict], Any]] = None

    def signal_count(self) -> int:
        return len(self.registry)

    def stats(self) -> dict:
        return {
            "registered_signals": self.registry.names(),
            "engine":             self.engine.stats(),
            "btc_onchain_days":   len(self.btc_onchain_store.history()
                                       .get("hash_rate", [])),
            "etf_flows_days":     len(self.etf_flow_store.get()),
            "funding_pairs":      len(self.funding_store._by_perp),
            "news_symbols":       len(self.news_store.by_symbol),
            "unlock_events":      len(self.unlock_store.events),
        }


# ─── Bootstrap ──────────────────────────────────────────────────────────────


def bootstrap_apex(
    *,
    model_path: Optional[str] = APEX_MODEL_FILENAME,
    fallback_only: bool = not APEX_MODEL_REQUIRED,
    entry_threshold: float = APEX_ENTRY_THRESHOLD,
    exit_threshold:  float = APEX_EXIT_THRESHOLD,
    max_positions:   int   = APEX_MAX_POSITIONS,
    derate:          float = APEX_KELLY_FRACTION,
    now_callable: Optional[Callable[[dict], Any]] = None,
) -> ApexBundle:
    """Build all stores, register signals, create the engine, return bundle.

    Args
    ----
    model_path     : path to joblib (relative to QC project root). Set to None
                     to force fallback (used by tests).
    fallback_only  : if True, never even attempt to load the joblib.
    now_callable   : context → datetime resolver for the unlock signal.
                     Defaults to context.get("now") or datetime.utcnow().
    """
    # ── Stores ────────────────────────────────────────────────────────────
    btc_onchain_store = BitcoinMetadataHistoryStore()
    dominance_store   = CoinGeckoDominanceStore()
    funding_store     = BinanceFundingRateStore()
    cross_asset_store = CrossAssetSeriesStore()
    etf_flow_store    = ETFFlowStore()
    stablecoin_store  = StablecoinSupplyStore()
    unlock_store      = UnlockCalendarStore()
    news_store        = NewsSentimentStore()
    onchain_val_store = OnchainValuationStore()

    # ── Signal registry (per-bundle, NOT process-default) ─────────────────
    registry = SignalRegistry()

    register_btc_onchain(registry, btc_onchain_store.history)
    register_btc_dominance(registry, dominance_store.dominance_series)
    register_funding_native(registry, funding_store.get)
    register_cross_asset(registry, cross_asset_store.series)
    register_etf_flow(registry, etf_flow_store.get)
    register_stablecoin(registry, stablecoin_store.get)

    # token_unlock needs a now-provider in addition to events
    if now_callable is None:
        from datetime import datetime as _dt
        def _default_now(ctx: dict):
            n = ctx.get("now") if isinstance(ctx, dict) else None
            return n if n is not None else _dt.utcnow()
        now_callable = _default_now

    register_token_unlock(registry, unlock_store.get, now_callable)
    register_news_sentiment(registry, news_store.get)
    register_mvrv(registry, onchain_val_store.mvrv_series)

    # ── ML inference (joblib or fallback) ─────────────────────────────────
    inference = ApexInference(model_path=model_path,
                               fallback_only=fallback_only)

    # ── Engine ────────────────────────────────────────────────────────────
    engine = ApexEngine(
        inference=inference, registry=registry,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        max_positions=max_positions, derate=derate,
    )

    return ApexBundle(
        registry=registry, engine=engine, inference=inference,
        btc_onchain_store=btc_onchain_store,
        dominance_store=dominance_store,
        funding_store=funding_store,
        cross_asset_store=cross_asset_store,
        etf_flow_store=etf_flow_store,
        stablecoin_store=stablecoin_store,
        unlock_store=unlock_store,
        news_store=news_store,
        onchain_val_store=onchain_val_store,
        now_provider=now_callable,
    )


def attach_callbacks(
    bundle: ApexBundle, *,
    place_order: Callable[[str, float, str, float], None],
    place_exit:  Callable[[str, float, str], None],
    context_provider: Optional[Callable[[str], dict]] = None,
) -> ApexBundle:
    """Attach host callbacks (PulseAlgorithm provides these)."""
    bundle.place_order = place_order
    bundle.place_exit  = place_exit
    bundle.context_provider = context_provider or (lambda s: {})
    return bundle
