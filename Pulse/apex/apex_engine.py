"""Pulse.apex.apex_engine — top-level orchestrator for the Apex strategy.

Public API
----------
ApexEngine
    .on_4h_tick(now, universe, market_context_provider, equity, place_order_fn)
        Score every symbol, decide entries, place market orders.
    .on_minute_tick(now, current_prices, latest_probs, place_exit_fn)
        Run exit logic across open positions; place liquidations.
    .open_positions       dict[symbol → ApexPosition]
    .last_decision_log    list of recent decisions for diagnostics

Designed to be called from PulseAlgorithm (Phase 6 wiring) without
importing QC at module load. Place-order callables are injected so
the engine is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional

from Pulse.apex.config import (
    APEX_ENTRY_THRESHOLD,
    APEX_EXIT_THRESHOLD,
    APEX_MAX_POSITIONS,
    APEX_HOLD_DAYS_MAX,
    APEX_PER_TRADE_HARD_KILL_PCT,
    APEX_ATR_TRAIL_MULT,
    APEX_KELLY_FRACTION,
    APEX_MIN_POSITION_USD,
    APEX_REBALANCE_HOURS,
    APEX_DAILY_DD_FREEZE_PCT,
)
from Pulse.apex.feature_vector import (
    build_feature_vector, build_feature_matrix, DEFAULT_SIGNAL_ORDER,
)
from Pulse.apex.registry import SignalRegistry, get_default_registry
from Pulse.apex.ml.predict import ApexInference
from Pulse.apex.exits import (
    ApexPosition, ExitDecision, evaluate_exits,
)
from Pulse.apex.sizing import compute_position_usd, SizeDecision


# ─── Decision log ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ApexDecision:
    timestamp:     datetime
    symbol:        str
    kind:          str           # "ENTRY" | "EXIT" | "SKIP"
    prob:          float
    size_usd:      float
    quantity:      float
    reason:        str
    feature_vec:   list[float] = field(default_factory=list)


# ─── Engine ──────────────────────────────────────────────────────────────────


class ApexEngine:
    """Apex strategy orchestrator. Stateless w.r.t. market data — all
    in/out via injected providers + callbacks.

    Args
    ----
    inference         : ApexInference instance (loaded model or fallback)
    registry          : SignalRegistry of registered signals
                        (defaults to the process-level default registry)
    entry_threshold   : prob ≥ this → enter (override APEX_ENTRY_THRESHOLD)
    exit_threshold    : prob < this → exit by prob-flip
    max_positions     : concurrent open positions cap
    """

    def __init__(self,
                 inference: ApexInference,
                 registry: Optional[SignalRegistry] = None,
                 *,
                 entry_threshold: float = APEX_ENTRY_THRESHOLD,
                 exit_threshold:  float = APEX_EXIT_THRESHOLD,
                 max_positions:   int   = APEX_MAX_POSITIONS,
                 max_new_per_tick: int  = 3,
                 hold_days_max:    float = APEX_HOLD_DAYS_MAX,
                 hard_kill_pct:    float = APEX_PER_TRADE_HARD_KILL_PCT,
                 atr_trail_mult:   float = APEX_ATR_TRAIL_MULT,
                 derate:           float = APEX_KELLY_FRACTION,
                 min_position_usd: float = APEX_MIN_POSITION_USD,
                 daily_dd_freeze:  float = APEX_DAILY_DD_FREEZE_PCT,
                 decision_log_size: int = 200,
                 ) -> None:
        self.inference = inference
        self.registry  = registry or get_default_registry()
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.max_positions   = max_positions
        self.max_new_per_tick = max_new_per_tick
        self.hold_days_max   = hold_days_max
        self.hard_kill_pct   = hard_kill_pct
        self.atr_trail_mult  = atr_trail_mult
        self.derate          = derate
        self.min_position_usd = min_position_usd
        self.daily_dd_freeze = daily_dd_freeze
        self._decision_log_size = decision_log_size

        self.open_positions: dict[str, ApexPosition] = {}
        self.last_decision_log: list[ApexDecision] = []
        self._daily_high_equity: Optional[float] = None
        self._daily_high_date: Optional[str] = None

    # ── Decision-log helper ────────────────────────────────────────────────

    def _log(self, dec: ApexDecision) -> None:
        self.last_decision_log.append(dec)
        if len(self.last_decision_log) > self._decision_log_size:
            del self.last_decision_log[0]

    # ── Daily DD freeze ────────────────────────────────────────────────────

    def update_daily_equity_high(self, now: datetime, equity: float) -> None:
        date_str = now.strftime("%Y-%m-%d")
        if self._daily_high_date != date_str:
            self._daily_high_date = date_str
            self._daily_high_equity = equity
        elif equity > (self._daily_high_equity or 0):
            self._daily_high_equity = equity

    def daily_dd_pct(self, equity: float) -> float:
        if self._daily_high_equity is None or self._daily_high_equity <= 0:
            return 0.0
        return (equity - self._daily_high_equity) / self._daily_high_equity

    def is_dd_frozen(self, equity: float) -> bool:
        return self.daily_dd_pct(equity) <= -abs(self.daily_dd_freeze)

    # ── 4-hour entry tick ──────────────────────────────────────────────────

    def on_4h_tick(
        self,
        now: datetime,
        universe: Iterable[str],
        market_context_provider: Callable[[str], dict],
        equity: float,
        place_order_fn: Callable[[str, float, str, float], None],
        *,
        tier_max_pos_usd_provider: Callable[[str], float] = lambda s: 1500.0,
        regime_mult_provider: Callable[[str], float] = lambda s: 1.0,
        current_price_provider: Callable[[str], float] = lambda s: 0.0,
        current_atr_provider: Callable[[str], float] = lambda s: 0.0,
    ) -> list[ApexDecision]:
        """Score the universe, decide which symbols to enter, place orders.

        place_order_fn signature: (symbol, qty, tag, fill_price_estimate) → None
        Returns the list of decisions produced this tick.
        """
        out: list[ApexDecision] = []
        self.update_daily_equity_high(now, equity)
        if self.is_dd_frozen(equity):
            dec = ApexDecision(now, "GLOBAL", "SKIP", 0.0, 0.0, 0.0,
                               reason="daily_dd_freeze")
            self._log(dec)
            return [dec]

        # Build feature matrix for all candidates not already open
        symbols = [s for s in universe
                   if s not in self.open_positions]
        if len(self.open_positions) >= self.max_positions:
            return []   # capacity full

        ctx_per_sym = {s: market_context_provider(s) for s in symbols}
        _, matrix, fvs = build_feature_matrix(
            symbols, ctx_per_sym, registry=self.registry,
            signal_order=DEFAULT_SIGNAL_ORDER,
        )
        if not symbols:
            return []
        probs = self.inference.predict_many(matrix)

        # Rank by prob desc; pick top max_new_per_tick that pass entry gate
        ranked = sorted(
            [(s, p, vec) for s, p, vec in zip(symbols, probs, matrix)],
            key=lambda t: t[1], reverse=True,
        )
        slots_open = self.max_positions - len(self.open_positions)
        new_entries = 0
        for sym, p, vec in ranked:
            if new_entries >= self.max_new_per_tick or new_entries >= slots_open:
                break
            if p < self.entry_threshold:
                self._log(ApexDecision(now, sym, "SKIP", p, 0.0, 0.0,
                                        f"prob<{self.entry_threshold}", vec))
                continue
            tier_cap = tier_max_pos_usd_provider(sym)
            regime  = regime_mult_provider(sym)
            decision = compute_position_usd(
                p, available_equity=equity, tier_max_pos_usd=tier_cap,
                regime_size_mult=regime, derate=self.derate,
                entry_threshold=self.entry_threshold,
                min_position_usd=self.min_position_usd,
            )
            if decision.skip_reason:
                self._log(ApexDecision(now, sym, "SKIP", p, 0.0, 0.0,
                                        decision.skip_reason, vec))
                continue
            cur_price = current_price_provider(sym)
            if cur_price <= 0:
                self._log(ApexDecision(now, sym, "SKIP", p, decision.position_usd,
                                        0.0, "no_price", vec))
                continue
            qty = decision.position_usd / cur_price
            place_order_fn(sym, qty, "APEX_ENTRY", cur_price)
            self.open_positions[sym] = ApexPosition(
                symbol=sym, entry_time=now, entry_price=cur_price,
                quantity=qty, entry_atr=current_atr_provider(sym) or cur_price * 0.02,
                high_water=cur_price, last_seen_prob=p,
            )
            entry_dec = ApexDecision(now, sym, "ENTRY", p, decision.position_usd,
                                      qty, "ENTRY", vec)
            self._log(entry_dec)
            out.append(entry_dec)
            new_entries += 1

        return out

    # ── Minute tick (exits) ────────────────────────────────────────────────

    def on_minute_tick(
        self,
        now: datetime,
        current_prices: dict[str, float],
        latest_probs: dict[str, float],
        place_exit_fn: Callable[[str, float, str], None],
    ) -> list[ApexDecision]:
        """Evaluate exits for every open position. Liquidations placed via
        place_exit_fn(symbol, qty, reason)."""
        out: list[ApexDecision] = []
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            cur = current_prices.get(sym, 0.0)
            if cur <= 0:
                continue
            d = evaluate_exits(
                pos,
                current_price=cur,
                now=now,
                latest_prob=latest_probs.get(sym),
                hard_kill_pct=self.hard_kill_pct,
                max_days=self.hold_days_max,
                trail_mult=self.atr_trail_mult,
                exit_threshold=self.exit_threshold,
            )
            if d.should_exit:
                place_exit_fn(sym, pos.quantity, d.reason or "EXIT")
                exit_dec = ApexDecision(
                    now, sym, "EXIT", pos.last_seen_prob, 0.0, pos.quantity,
                    d.reason or "EXIT",
                )
                self._log(exit_dec)
                out.append(exit_dec)
                self.open_positions.pop(sym, None)
        return out

    # ── Diagnostics ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.open_positions.clear()
        self.last_decision_log.clear()
        self._daily_high_equity = None
        self._daily_high_date = None

    def stats(self) -> dict:
        n = len(self.last_decision_log)
        kinds: dict[str, int] = {}
        for d in self.last_decision_log:
            kinds[d.kind] = kinds.get(d.kind, 0) + 1
        return {
            "open_positions":   len(self.open_positions),
            "decisions_logged": n,
            "by_kind":          kinds,
            "in_fallback_mode": self.inference.in_fallback_mode,
        }
