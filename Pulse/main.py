"""main — PulseAlgorithm QC entry point (single-strategy scalp mode).

This wires together every Phase 1 module into a runnable QuantConnect algorithm.
Multi-strategy portfolio (scalp + trend + mean-reversion) comes in Phase 4.

Architecture (data flow per OnData tick):

  ┌──────────────────────────────────────────────────────────────────────┐
  │  OnData(slice)                                                       │
  │   1. update per-symbol OHLCV + VWAP state (rolling deques)           │
  │   2. update DrawdownCircuitBreaker with current equity               │
  │   3. if circuit halted → liquidate all and return                    │
  │   4. if circuit tripped → skip new entries (manage open only)        │
  │   5. evaluate every open position:                                   │
  │        - PerTradeKill at -8%                                         │
  │        - TP / SL / trail / time-stop                                 │
  │   6. once per decision interval (15min):                             │
  │        - score every eligible symbol via MicroScalpEngine v8         │
  │        - apply size multipliers (kyle / RV / regime / FG)            │
  │        - tier-aware position size USD                                │
  │        - submit limit-with-TTL via place_limit_or_market             │
  │                                                                      │
  │  OnOrderEvent(event)                                                 │
  │   1. dispatch to events.on_order_event(audit_state)                  │
  │   2. log slippage if reference_price was recorded at submit          │
  │   3. update rolling WR / cash-mode trigger                           │
  └──────────────────────────────────────────────────────────────────────┘

File size target: < 30 KB so we have headroom to extend.
"""

from __future__ import annotations

# QC imports — guarded so the module can be imported (without instantiating
# the algorithm) under pytest for static-analysis type checks.
try:
    from AlgorithmImports import *  # type: ignore  # noqa: F401,F403
    HAS_QC = True
except Exception:
    HAS_QC = False

from collections import deque
from datetime import datetime, timedelta
from typing import Any

# Pulse imports
from Pulse.config import (
    INITIAL_CASH_USD,
    MAX_DRAWDOWN_TRIP_PCT, MAX_DRAWDOWN_HALT_PCT, MAX_DD_RECOVERY_PCT,
    PER_TRADE_HARD_KILL_PCT,
    SCALP_ENTRY_THRESHOLD, SCALP_HIGH_CONVICTION_THRES,
    QUICK_TAKE_PROFIT_PCT, TIGHT_STOP_LOSS_PCT,
    ATR_TP_MULT, ATR_SL_MULT,
    TRAIL_ACTIVATION_PCT, TRAIL_STOP_PCT, TIME_STOP_HOURS,
    MAX_POSITIONS, TARGET_POSITION_ANN_VOL, PORTFOLIO_VOL_CAP,
    LIMIT_ORDER_TTL_SECONDS, FG_GREED_EXTREME_THRESHOLD,
    TIER_LIMITS,
)
from Pulse.universe import (
    UniverseGate, SymbolTierClassifier, SymbolStats,
)
from Pulse.scalp_engine import (
    SymbolBars, MarketContext, compute_scalp_score, rank_candidates,
)
from Pulse.circuit import (
    DrawdownCircuitBreaker, RollingMaxDrawdown, PerTradeKill,
)
from Pulse.events import (
    OrderAuditState, on_order_event as _on_order_event_impl,
    rolling_win_rate, expectancy,
)
from Pulse.execution import OrderIntent
from Pulse.alt_data import FearGreedData, FGSignal
from Pulse.fees import KrakenTieredFeeModel
from Pulse.slippage import RealisticCryptoSlippage


# ─── Per-symbol rolling state ────────────────────────────────────────────────

class SymbolBuffers:
    """Per-symbol OHLCV + VWAP rolling state.

    Kept as a plain class so it can be unit-tested without QC.
    """
    HISTORY_BARS = 200    # enough for EMAs + Yang-Zhang vol + spread

    def __init__(self):
        self.opens   = deque(maxlen=self.HISTORY_BARS)
        self.highs   = deque(maxlen=self.HISTORY_BARS)
        self.lows    = deque(maxlen=self.HISTORY_BARS)
        self.closes  = deque(maxlen=self.HISTORY_BARS)
        self.volumes = deque(maxlen=self.HISTORY_BARS)
        # 24h rolling dollar volume tracker (1-min bars × 1440 bars)
        self.recent_dollar_volumes = deque(maxlen=1440)
        # Bid/ask snapshot — used for spread tracking only (NOT for signal,
        # because OBI from minute QuoteBars is unreliable in live).
        self.last_bid: float = 0.0
        self.last_ask: float = 0.0
        # 60-bar spread history (bps) for the universe gate
        self.spread_bps_history = deque(maxlen=60)

    def update_bar(self, o, h, l, c, v):
        if o > 0:
            self.opens.append(float(o))
            self.highs.append(float(h))
            self.lows.append(float(l))
            self.closes.append(float(c))
            self.volumes.append(float(v))
            self.recent_dollar_volumes.append(float(c) * float(v))

    def update_quote(self, bid: float, ask: float):
        if bid > 0 and ask > 0:
            self.last_bid = bid
            self.last_ask = ask
            mid = (bid + ask) / 2
            if mid > 0:
                self.spread_bps_history.append((ask - bid) / mid * 10_000)

    @property
    def last_close(self) -> float:
        return self.closes[-1] if self.closes else 0.0

    def stats_for_universe(self, symbol: str, days_of_history: int) -> SymbolStats:
        """Build a SymbolStats snapshot for the universe gate."""
        dv24h = sum(self.recent_dollar_volumes)
        avg_spread = (sum(self.spread_bps_history) / len(self.spread_bps_history)
                      if self.spread_bps_history else 30.0)
        return SymbolStats(
            symbol=symbol,
            rolling_24h_dollar_vol_usd=dv24h,
            rolling_60bar_mean_spread_bps=avg_spread,
            last_price_usd=self.last_close,
            days_of_history=days_of_history,
        )

    def to_symbol_bars(self, symbol: str) -> SymbolBars:
        return SymbolBars(
            symbol=symbol,
            opens=list(self.opens),
            highs=list(self.highs),
            lows=list(self.lows),
            closes=list(self.closes),
            volumes=list(self.volumes),
        )


# ─── Position tracking ──────────────────────────────────────────────────────

class OpenPosition:
    """Lightweight per-position tracker — only what main loop needs."""
    def __init__(self, symbol, entry_price: float, entry_time: datetime,
                 quantity: float, intent: OrderIntent):
        self.symbol      = symbol
        self.entry_price = entry_price
        self.entry_time  = entry_time
        self.quantity    = quantity
        self.intent      = intent
        self.high_price  = entry_price
        self.low_price   = entry_price
        self.trail_armed = False
        self.partial_taken = False

    def update_extremes(self, price: float):
        if price > self.high_price:
            self.high_price = price
        if price < self.low_price:
            self.low_price = price

    def held_hours(self, now: datetime) -> float:
        return (now - self.entry_time).total_seconds() / 3600


# ─── PulseAlgorithm (only meaningful inside QC) ─────────────────────────────

if HAS_QC:

    class PulseAlgorithm(QCAlgorithm):
        """Best-of-breed crypto scalper — Phase 1 single-strategy mode.

        Defaults assume Kraken cash brokerage and 1-minute resolution.
        Override via QC parameter panel:
          start_year, end_year, initial_cash, max_dd_pct, decision_interval_min
        """

        # ── Initialize ─────────────────────────────────────────────────────

        def Initialize(self):
            # ── Apply runtime overrides (Phase 3 sweep) BEFORE parameters ──
            # If a runtime_overrides.py file is present in the project, its
            # OVERRIDES dict is merged into Pulse.config module attributes.
            # Useful for pushing per-backtest parameter overrides without
            # re-uploading the whole codebase.
            self._apply_runtime_overrides()

            # ── Parameters ────────────────────────────────────────────────
            start_year = int(self.GetParameter("start_year") or 2025)
            end_year   = int(self.GetParameter("end_year")   or 2026)
            cash       = float(self.GetParameter("initial_cash") or INITIAL_CASH_USD)
            self._decision_interval_min = int(
                self.GetParameter("decision_interval_min") or 15
            )

            # ── Phase 2 harsh-sim flag (auto-applied if true) ─────────────
            harsh_raw = self.GetParameter("use_harsh_sim")
            self._use_harsh_sim = (
                str(harsh_raw).lower() in ("true", "1", "yes")
                if harsh_raw else False
            )
            if self._use_harsh_sim:
                self.Log("[pulse] HARSH-SIM mode enabled — pessimistic slippage/fees")

            self.SetStartDate(start_year, 1, 1)
            self.SetEndDate(end_year, 12, 31)
            self.SetCash(cash)
            self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
            self.UniverseSettings.Resolution = Resolution.Minute

            # ── Subscribe BTC + curated alt list ──────────────────────────
            self._symbols = []
            for ticker in self._initial_universe():
                try:
                    sym = self.AddCrypto(
                        ticker, Resolution.Minute, Market.Kraken,
                    ).Symbol
                    self._symbols.append(sym)
                except Exception as exc:
                    self.Debug(f"AddCrypto {ticker} failed: {exc}")
            self._btc_sym = next(
                (s for s in self._symbols if s.Value == "BTCUSD"), None
            )

            # ── F&G subscription ──────────────────────────────────────────
            self._fg_symbol = self.AddData(FearGreedData, "FNG", Resolution.Daily).Symbol
            self._fg_value: float | None = None

            # ── Custom slippage + fee per security ────────────────────────
            self.SetSecurityInitializer(self._on_security_added)

            # ── Per-symbol buffers ────────────────────────────────────────
            self._buffers: dict[Any, SymbolBuffers] = {
                s: SymbolBuffers() for s in self._symbols
            }

            # ── Universe gate + tier classifier ───────────────────────────
            self._gate = UniverseGate()
            self._tiers = SymbolTierClassifier()

            # ── Risk circuits ─────────────────────────────────────────────
            self._circuit = DrawdownCircuitBreaker(
                trip_drawdown_pct=MAX_DRAWDOWN_TRIP_PCT,
                halt_drawdown_pct=MAX_DRAWDOWN_HALT_PCT,
                recovery_pct=MAX_DD_RECOVERY_PCT,
            )
            self._rolling_dd = RollingMaxDrawdown(lookback_bars=1440)
            self._kill = PerTradeKill(threshold_pct=PER_TRADE_HARD_KILL_PCT)

            # ── Order audit state ─────────────────────────────────────────
            self._audit = OrderAuditState()

            # ── Open positions tracker ────────────────────────────────────
            self._open: dict[Any, OpenPosition] = {}

            # ── Cross-symbol context (filled each cycle) ──────────────────
            self._market_context = MarketContext()

            # ── Decision throttle ─────────────────────────────────────────
            self._last_decision_time: datetime | None = None

            # Warmup — enough history for SMAs, Yang-Zhang, etc.
            self.SetWarmup(timedelta(days=14))

            self.Log(
                f"[pulse] Initialize start={start_year}-01-01 end={end_year}-12-31 "
                f"cash=${cash} decision_interval={self._decision_interval_min}min "
                f"symbols={len(self._symbols)} max_dd_trip={MAX_DRAWDOWN_TRIP_PCT}"
            )

        def _on_security_added(self, security):
            try:
                if getattr(self, "_use_harsh_sim", False):
                    # Phase 2 harsh simulator: pessimistic slippage + 100% taker
                    from harsh_simulator import (
                        HarshConfig, HarshSlippageModel, HarshFeeModel,
                    )
                    cfg = HarshConfig()
                    security.SetSlippageModel(HarshSlippageModel(cfg))
                    security.SetFeeModel(HarshFeeModel(cfg))
                else:
                    # Standard Pulse slippage + tiered Kraken fees
                    security.SetSlippageModel(RealisticCryptoSlippage())
                    security.SetFeeModel(KrakenTieredFeeModel())
            except Exception as exc:
                self.Debug(f"security init failed for {security.Symbol}: {exc}")

        def _apply_runtime_overrides(self):
            """Apply per-backtest parameter overrides from runtime_overrides.py.

            Convention: a Phase 3 sweep helper pushes a tiny `runtime_overrides.py`
            to the project containing:
                OVERRIDES = {
                    "SCALP_ENTRY_THRESHOLD": 0.55,
                    "QUICK_TAKE_PROFIT_PCT": 0.12,
                    ...
                }
            We import it (gracefully no-op if missing) and assign each key
            into the `config` module so subsequent imports pick up the new
            values.

            This sidesteps QC's per-backtest parameter API limitations while
            keeping the override file small and easy to push.
            """
            try:
                import config as _cfg
                import runtime_overrides as _ro
            except Exception:
                return   # No overrides file → use defaults
            overrides = getattr(_ro, "OVERRIDES", {}) or {}
            applied = 0
            for k, v in overrides.items():
                if hasattr(_cfg, k):
                    setattr(_cfg, k, v)
                    applied += 1
            if applied:
                self.Log(f"[pulse] runtime_overrides applied: {applied} params")

        def _initial_universe(self) -> list[str]:
            """Curated 25-symbol Kraken Pro universe.

            Phase 0a's UniverseGate filters dynamically each cycle, but we
            still need a static subscription list to tell QC what to pull.
            """
            return [
                # Major
                "BTCUSD", "ETHUSD",
                # Large
                "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD",
                "LINKUSD", "AVAXUSD", "DOTUSD",
                # Mid
                "LTCUSD", "MATICUSD", "ATOMUSD", "UNIUSD", "AAVEUSD",
                "NEARUSD", "INJUSD", "OPUSD", "ARBUSD", "BCHUSD",
                "TRXUSD", "FETUSD", "ICPUSD", "RENDERUSD", "HBARUSD",
            ]

        # ── OnData ─────────────────────────────────────────────────────────

        def OnData(self, slice):
            now = self.Time

            # Update per-symbol buffers
            for sym, buf in self._buffers.items():
                if slice.Bars.ContainsKey(sym):
                    bar = slice.Bars[sym]
                    buf.update_bar(bar.Open, bar.High, bar.Low,
                                  bar.Close, bar.Volume)
                if slice.QuoteBars.ContainsKey(sym):
                    qb = slice.QuoteBars[sym]
                    buf.update_quote(float(qb.Bid.Close or 0),
                                    float(qb.Ask.Close or 0))

            # F&G value
            if slice.ContainsKey(self._fg_symbol):
                fg = slice[self._fg_symbol]
                if fg is not None:
                    self._fg_value = float(fg.Value)

            if self.IsWarmingUp:
                return

            # ── Risk circuits ─────────────────────────────────────────────
            equity = float(self.Portfolio.TotalPortfolioValue)
            cb_action = self._circuit.update(equity, now)
            self._rolling_dd.update(equity)

            if self._circuit.should_liquidate_all():
                self.Log(f"[pulse] CIRCUIT HALT — liquidating all positions")
                self._liquidate_all_open()
                return

            # ── Per-position management ───────────────────────────────────
            self._manage_open_positions(now)

            # ── Decision throttle ─────────────────────────────────────────
            if not self._should_decide(now):
                return

            # ── Skip new entries when tripped ─────────────────────────────
            if not self._circuit.can_enter_new_positions():
                return

            # ── Build market context ──────────────────────────────────────
            self._refresh_market_context(now)

            # ── Score candidates ──────────────────────────────────────────
            ranked = self._score_universe()
            if not ranked:
                return

            # ── Apply F&G max_positions multiplier ────────────────────────
            fg_signal = FGSignal.from_value(self._fg_value)
            if fg_signal.block_new_entries:
                self.Debug(f"[pulse] FG panic-greed: blocking new entries "
                          f"(value={self._fg_value})")
                return
            base_max = MAX_POSITIONS
            effective_max = max(1, round(base_max * fg_signal.max_positions_multiplier))

            # ── Submit entries up to effective_max ────────────────────────
            slots_open = effective_max - len(self._open)
            for score in ranked[:max(0, slots_open)]:
                if score.symbol in self._open:
                    continue
                self._try_enter(score, now)

            self._last_decision_time = now

        def OnOrderEvent(self, event):
            try:
                res = _on_order_event_impl(self, event, self._audit)
                # If we just got an entry fill, track an OpenPosition
                if res.action == "entry_recorded":
                    sym_obj = event.Symbol
                    self._open[sym_obj] = OpenPosition(
                        symbol=sym_obj,
                        entry_price=float(event.FillPrice),
                        entry_time=self.Time,
                        quantity=float(event.FillQuantity),
                        intent=OrderIntent.ENTRY,
                    )
                elif res.action in ("exit_recorded", "exit_unpaired",
                                    "invalid_force_cleanup"):
                    self._open.pop(event.Symbol, None)
            except Exception as exc:
                self.Debug(f"OnOrderEvent error: {exc}")

        # ── Helpers ────────────────────────────────────────────────────────

        def _should_decide(self, now: datetime) -> bool:
            if self._last_decision_time is None:
                return True
            elapsed_min = (now - self._last_decision_time).total_seconds() / 60
            return elapsed_min >= self._decision_interval_min

        def _refresh_market_context(self, now):
            ctx = MarketContext()
            if self._btc_sym and self._buffers[self._btc_sym].closes:
                btc_buf = self._buffers[self._btc_sym]
                # Approximate 4h closes by sampling every 240th 1m bar
                closes = list(btc_buf.closes)
                ctx.btc_4h_closes  = closes[::240][-30:] or closes[-30:]
                ctx.btc_4h_volumes = list(btc_buf.volumes)[::240][-30:] or list(btc_buf.volumes)[-30:]
                ctx.btc_daily_closes = closes[::1440][-200:] or []
                if len(closes) >= 1440 * 30:
                    ctx.btc_30d_return = (
                        closes[-1] - closes[-1440 * 30]
                    ) / closes[-1440 * 30]
            # Symbol recent returns (5min lookback for spillover)
            recent: dict[str, float] = {}
            for sym, buf in self._buffers.items():
                if len(buf.closes) >= 5 and buf.closes[-5] > 0:
                    recent[sym.Value] = (buf.closes[-1] - buf.closes[-5]) / buf.closes[-5]
            ctx.symbol_recent_returns = recent
            # Alt 30d returns
            alt_returns = []
            for sym, buf in self._buffers.items():
                if sym == self._btc_sym:
                    continue
                if len(buf.closes) >= 1440 * 30 and buf.closes[-1440 * 30] > 0:
                    r = (buf.closes[-1] - buf.closes[-1440 * 30]) / buf.closes[-1440 * 30]
                    alt_returns.append(r)
            ctx.alts_30d_returns = alt_returns
            ctx.fg_value = self._fg_value
            self._market_context = ctx

        def _score_universe(self):
            """Score every eligible symbol and return ranked entry candidates."""
            candidates = []
            for sym, buf in self._buffers.items():
                if sym in self._open:
                    continue   # already holding
                if len(buf.closes) < 60:
                    continue   # not enough history
                # Universe gate — drop polluted symbols
                stats = buf.stats_for_universe(
                    symbol=sym.Value,
                    days_of_history=len(buf.closes) // 1440,
                )
                if not self._gate.is_eligible(stats):
                    continue
                # Tier check — eject = skip
                tier_lim = self._tiers.limits_for(sym.Value, now=self.Time)
                if tier_lim["tier"] == "ejected":
                    continue
                candidates.append(buf.to_symbol_bars(sym.Value))
            if not candidates:
                return []
            return rank_candidates(
                candidates, self._market_context,
                entry_threshold=SCALP_ENTRY_THRESHOLD,
                high_conviction_thres=SCALP_HIGH_CONVICTION_THRES,
            )

        def _try_enter(self, score, now):
            try:
                sym = next(
                    (s for s in self._symbols if s.Value == score.symbol), None
                )
                if sym is None:
                    return
                tier_lim = self._tiers.limits_for(sym.Value, now=now)
                max_pos_usd = tier_lim["max_pos_usd"]
                # Apply size multipliers
                size_usd = max_pos_usd * score.composed_size_mult
                if score.high_conviction:
                    size_usd *= 1.0   # full tier max for high conviction
                else:
                    size_usd *= 0.7   # 70% for normal conviction
                # Cap at 80% of available cash
                cash = float(self.Portfolio.Cash) * 0.8
                size_usd = min(size_usd, cash)
                if size_usd < 5.0:
                    return   # below Kraken min notional
                price = float(self.Securities[sym].Price)
                if price <= 0:
                    return
                qty = size_usd / price
                self.Log(
                    f"[pulse] SCALP ENTRY {sym.Value} score={score.score:.3f} "
                    f"hc={score.high_conviction} mode={score.market_mode} "
                    f"tier={tier_lim['tier']} size_usd={size_usd:.2f} qty={qty:.6f}"
                )
                self.MarketOrder(sym, qty, tag="ENTRY")
            except Exception as exc:
                self.Debug(f"_try_enter error {score.symbol}: {exc}")

        def _manage_open_positions(self, now):
            for sym, pos in list(self._open.items()):
                price = float(self.Securities[sym].Price)
                if price <= 0:
                    continue
                pos.update_extremes(price)
                ret = (price - pos.entry_price) / pos.entry_price

                # 1. Per-trade hard kill at -8%
                kill_dec = self._kill.evaluate(sym.Value, pos.entry_price, price)
                if kill_dec.should_kill:
                    self.Log(f"[pulse] HARD KILL {sym.Value} ret={ret:+.2%}")
                    self.Liquidate(sym, tag="HARD_KILL")
                    continue

                # 2. Hard SL at -3.5%
                if ret <= -TIGHT_STOP_LOSS_PCT:
                    self.Liquidate(sym, tag="STOP_LOSS")
                    continue

                # 3. Time stop
                if pos.held_hours(now) >= TIME_STOP_HOURS:
                    self.Liquidate(sym, tag="TIME_STOP")
                    continue

                # 4. Trail (arm at +4%, trail 2.5% from high)
                max_ret = (pos.high_price - pos.entry_price) / pos.entry_price
                if max_ret >= TRAIL_ACTIVATION_PCT:
                    pos.trail_armed = True
                if pos.trail_armed:
                    trail_stop_price = pos.high_price * (1 - TRAIL_STOP_PCT)
                    if price < trail_stop_price:
                        self.Liquidate(sym, tag="TRAIL_STOP")
                        continue

                # 5. Take profit at +12% (or ATR-based; using fixed for now)
                if ret >= QUICK_TAKE_PROFIT_PCT:
                    self.Liquidate(sym, tag="TAKE_PROFIT")
                    continue

        def _liquidate_all_open(self):
            for sym in list(self._open.keys()):
                self.Liquidate(sym, tag="CIRCUIT_HALT")
            self._open.clear()

        def OnEndOfAlgorithm(self):
            wr = rolling_win_rate(self._audit)
            exp = expectancy(self._audit)
            self.Log(
                f"[pulse] FINAL trades={self._audit.winning_trades + self._audit.losing_trades} "
                f"WR={wr:.1%} expectancy={exp*100:.3f}% "
                f"total_pnl={self._audit.total_pnl*100:.2f}% "
                f"slippage_warnings={len(self._audit.slippage_log)}"
            )

else:
    PulseAlgorithm = None  # type: ignore
