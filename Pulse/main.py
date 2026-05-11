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

# ─── Pure-Python override helpers (testable without QC) ────────────────────

# Reserved keys the qc_sweep_runner may push to override per-window
# backtest dates and other QC-parameter-style settings.
SPECIAL_OVERRIDE_KEYS = (
    "PULSE_OVERRIDE_START_YEAR",
    "PULSE_OVERRIDE_START_MONTH",
    "PULSE_OVERRIDE_START_DAY",
    "PULSE_OVERRIDE_END_YEAR",
    "PULSE_OVERRIDE_END_MONTH",
    "PULSE_OVERRIDE_END_DAY",
    "use_harsh_sim",
    "start_year", "end_year", "initial_cash",
    "decision_interval_min",
)


def split_runtime_overrides(overrides_dict, config_module):
    """Split an overrides dict into (special_dict, config_assignments).

    Special keys are stashed for later lookup; config keys are immediately
    written into the supplied config_module. Returns the special dict.
    """
    special = {}
    for k, v in (overrides_dict or {}).items():
        if k in SPECIAL_OVERRIDE_KEYS:
            special[k] = v
        elif hasattr(config_module, k):
            setattr(config_module, k, v)
    return special


def resolve_param(name, runtime_overrides, qc_get_param_fn, default=None):
    """Read a parameter, preferring runtime_overrides over QC parameters.

    Args:
        name: parameter name
        runtime_overrides: dict (may be None / empty)
        qc_get_param_fn: callable(name) → str or None (e.g. self.GetParameter)
        default: returned when both sources are missing/empty

    Returns the resolved value (raw — caller casts to int/float/bool as needed).
    """
    ro = runtime_overrides or {}
    if name in ro:
        return ro[name]
    qc_val = qc_get_param_fn(name) if qc_get_param_fn else None
    return qc_val if qc_val not in (None, "") else default


# Pulse imports
# Read overridable values via the config MODULE at use-time so
# runtime_overrides take effect. Specifically: TIME_STOP_HOURS,
# MAX_POSITIONS, SCALP_ENTRY_THRESHOLD, SCALP_HIGH_CONVICTION_THRES.
# The bare names below are bound at IMPORT time and therefore reflect
# only the original config defaults — do NOT use them where overrides
# might apply at runtime.
import Pulse.config as _cfg
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
from Pulse.online_learning import (
    OnlineThresholdLearner,
    ENTRY_THRESHOLD_MIN as _LRN_ENTRY_MIN,
    ENTRY_THRESHOLD_MAX as _LRN_ENTRY_MAX,
    HC_THRESHOLD_MIN    as _LRN_HC_MIN,
    HC_THRESHOLD_MAX    as _LRN_HC_MAX,
)
from Pulse.optimal_execution import build_slice_plan, DEFAULT_LARGE_ORDER_THRESHOLD_BPS
from Pulse.execution import (
    min_quantity_fallback, KRAKEN_MIN_QTY_FALLBACK,
    safe_sell_quantity, round_to_lot,
)


# ─── Per-symbol rolling state ────────────────────────────────────────────────

class SymbolBuffers:
    """Per-symbol OHLCV + VWAP rolling state.

    Kept as a plain class so it can be unit-tested without QC.

    HISTORY_BARS sized for: 4h-sampled BTC context (240 bars/sample × 30
    samples = 7200 bars), plus rolling 60-bar feature windows.
    """
    HISTORY_BARS = 8_000   # ~5.5 days of 1-min bars; supports 4h sampling

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

            # ── Parameters (runtime_overrides win, then QC params, then defaults) ─
            start_year = int(self._param("start_year", 2025))
            end_year   = int(self._param("end_year",   2026))
            cash       = float(self._param("initial_cash", INITIAL_CASH_USD))
            self._decision_interval_min = int(self._param("decision_interval_min", 15))

            # Per-window date overrides from Phase 3 sweep (month + day, not just year)
            start_month = int(self._param("PULSE_OVERRIDE_START_MONTH", 1))
            start_day   = int(self._param("PULSE_OVERRIDE_START_DAY", 1))
            end_month   = int(self._param("PULSE_OVERRIDE_END_MONTH", 12))
            end_day     = int(self._param("PULSE_OVERRIDE_END_DAY", 31))
            # If the more granular override dates are present, use them
            ovr_start_y = self._param("PULSE_OVERRIDE_START_YEAR")
            ovr_end_y   = self._param("PULSE_OVERRIDE_END_YEAR")
            if ovr_start_y is not None:
                start_year = int(ovr_start_y)
            if ovr_end_y is not None:
                end_year = int(ovr_end_y)

            # ── Phase 2 harsh-sim flag (auto-applied if true) ─────────────
            harsh_raw = self._param("use_harsh_sim")
            self._use_harsh_sim = (
                str(harsh_raw).lower() in ("true", "1", "yes")
                if harsh_raw else False
            )
            if self._use_harsh_sim:
                self.Log("[pulse] HARSH-SIM mode enabled — pessimistic slippage/fees")

            self.SetStartDate(start_year, start_month, start_day)
            self.SetEndDate(end_year, end_month, end_day)
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

            # ── Binance funding rate subscriptions (for Apex) ─────────────
            # Native QC dataset — replays correctly in backtest. Subscribed
            # only for the perp tickers that map to our Kraken universe.
            self._binance_funding_symbols: dict[str, Any] = {}
            try:
                from QuantConnect.DataSource import BinanceFundingRate  # type: ignore
                from Pulse.apex.signals.funding_native import KRAKEN_TO_BINANCE_PERP
                for kraken_sym in [s.Value for s in self._symbols]:
                    perp = KRAKEN_TO_BINANCE_PERP.get(kraken_sym.upper())
                    if perp is None:
                        continue
                    try:
                        sub = self.AddData(
                            BinanceFundingRate, perp, Resolution.Hour,
                        )
                        self._binance_funding_symbols[perp] = sub.Symbol
                    except Exception as exc:
                        self.Debug(f"Binance funding sub failed for {perp}: {exc}")
                if self._binance_funding_symbols:
                    self.Log(f"[pulse] Binance funding subscribed: "
                             f"{len(self._binance_funding_symbols)} pairs")
            except Exception as exc:
                self.Debug(f"BinanceFundingRate import failed: {exc}")

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

            # ── Adaptive threshold learner (online learning) ──────────────
            # Auto-relax learner bounds if config thresholds are set lower
            # than the learner's defaults (e.g. diagnostic backtest with
            # SCALP_ENTRY_THRESHOLD=0.35). Otherwise the learner's __init__
            # validation rejects the initial value.
            # Read via _cfg so runtime_overrides take effect (the bare names
            # bound at import time would still show the unmodified defaults).
            _cfg_entry = float(_cfg.SCALP_ENTRY_THRESHOLD)
            _cfg_hc    = float(_cfg.SCALP_HIGH_CONVICTION_THRES)
            entry_min = min(_LRN_ENTRY_MIN, _cfg_entry)
            entry_max = max(_LRN_ENTRY_MAX, _cfg_entry)
            hc_min    = min(_LRN_HC_MIN,    _cfg_hc)
            hc_max    = max(_LRN_HC_MAX,    _cfg_hc)
            self._learner = OnlineThresholdLearner(
                initial_entry_threshold=_cfg_entry,
                initial_high_conviction_thres=_cfg_hc,
                entry_min=entry_min, entry_max=entry_max,
                hc_min=hc_min, hc_max=hc_max,
            )
            self._last_learner_tune_log: datetime | None = None

            # ── Open positions tracker ────────────────────────────────────
            self._open: dict[Any, OpenPosition] = {}

            # Per-symbol cooldown after invalid order or recent fill — blocks
            # rapid re-entry attempts on the same symbol (was a major source
            # of "Insufficient buying power" spam in the Jan-2025 backtest).
            self._symbol_cooldown_until: dict[str, datetime] = {}
            self._reentry_cooldown_minutes = 30

            # Per-symbol tracker for which symbols have a FRESH bar this tick.
            # Exits should ONLY fire on symbols that just received new data,
            # otherwise QC fills market orders at price=0 (data gap).
            self._symbols_with_fresh_bar_this_tick: set = set()

            # Per-symbol failed-exit backoff: after N failed exits, pause
            # exit attempts for M minutes so we stop spamming the order book.
            self._exit_failure_count: dict[str, int] = {}
            self._exit_pause_until: dict[str, datetime] = {}

            # ── Cross-symbol context (filled each cycle) ──────────────────
            self._market_context = MarketContext()

            # ── Apex engine (gated by runtime override `apex_enabled`) ────
            # When enabled, runs every 4h alongside the existing scalp
            # engine. ML model is loaded from `apex_model_v1.joblib` if
            # present; otherwise Apex runs in fallback (rule-based) mode.
            self._apex = None
            self._apex_last_tick: datetime | None = None
            apex_raw = self._param("apex_enabled", False)
            self._apex_enabled = (
                str(apex_raw).lower() in ("true", "1", "yes")
                if apex_raw else False
            )
            if self._apex_enabled:
                try:
                    from Pulse.apex.integration import (
                        bootstrap_apex, attach_callbacks,
                    )
                    self._apex = bootstrap_apex(
                        model_path="apex_model_v1.joblib",
                        fallback_only=False,
                    )
                    attach_callbacks(
                        self._apex,
                        place_order=self._apex_place_order,
                        place_exit=self._apex_place_exit,
                        context_provider=lambda s: {"now": self.Time},
                    )
                    # Bulk-load any bundled CSVs (ETF flows, unlocks).
                    # These are STATIC data refreshed offline — they don't
                    # need a live subscription. The Reader-based PythonData
                    # adapters are still available for live mode.
                    self._apex_load_static_csvs()
                    self.Log(
                        f"[pulse] APEX enabled — "
                        f"signals={self._apex.signal_count()} "
                        f"etf_days={len(self._apex.etf_flow_store.get())} "
                        f"unlocks={len(self._apex.unlock_store.events)} "
                        f"fallback={self._apex.inference.in_fallback_mode}"
                    )
                except Exception as exc:
                    self.Debug(f"[pulse] APEX bootstrap failed: {exc}")
                    self._apex = None
                    self._apex_enabled = False

            # ── Decision throttle ─────────────────────────────────────────
            self._last_decision_time: datetime | None = None

            # Warmup — only need enough for the rolling-window features
            # (Yang-Zhang vol uses 20 bars, EMA20 uses 20 bars, vol z-score
            # uses 60 bars). Keep this tight so the strategy starts trading
            # within a few hours of backtest start, not 14 days later.
            self.SetWarmup(timedelta(hours=6))

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
            """Load + apply runtime_overrides.py if present. See module-level
            split_runtime_overrides() for the underlying logic.

            Bare ``except:`` catches BaseException, since QC's Python.NET
            hosted runtime sometimes raises non-Exception subclasses on
            module-not-found errors. Defensive — overrides are always optional.
            """
            self._runtime_overrides: dict = {}
            try:
                import config as _cfg
            except BaseException:
                return
            try:
                import runtime_overrides as _ro
                overrides = getattr(_ro, "OVERRIDES", {}) or {}
            except BaseException:
                overrides = {}
            self._runtime_overrides = split_runtime_overrides(overrides, _cfg)
            if self._runtime_overrides or overrides:
                self.Log(
                    f"[pulse] runtime_overrides loaded: "
                    f"{len(self._runtime_overrides)} special + "
                    f"{len(overrides) - len(self._runtime_overrides)} config"
                )

        def _param(self, name: str, default=None):
            return resolve_param(
                name, getattr(self, "_runtime_overrides", None),
                self.GetParameter, default,
            )

        def _initial_universe(self) -> list[str]:
            """Curated Kraken Pro universe known to exist in QC's crypto data.

            Phase 0a's UniverseGate filters dynamically each cycle, but we
            still need a static subscription list to tell QC what to pull.

            Excluded:
              - DOGEUSD: Kraken QC data doesn't have this symbol (verified
                from live error: 'Crypto DOGEUSD symbol could not be found')
              - MATICUSD: rebranded to POLUSD; symbol may not be in QC data
              - RENDERUSD: newer; QC data coverage uncertain
            Add removed ones back ONLY after verifying via:
              algo.AddCrypto(ticker, Resolution.Minute, Market.Kraken)
            doesn't raise.
            """
            return [
                # Major
                "BTCUSD", "ETHUSD",
                # Large
                "SOLUSD", "XRPUSD", "ADAUSD",
                "LINKUSD", "AVAXUSD", "DOTUSD",
                # Mid
                "LTCUSD", "ATOMUSD", "UNIUSD", "AAVEUSD",
                "NEARUSD", "INJUSD", "OPUSD", "ARBUSD", "BCHUSD",
                "TRXUSD", "FETUSD", "ICPUSD", "HBARUSD",
            ]

        # ── OnData ─────────────────────────────────────────────────────────

        def OnData(self, slice):
            now = self.Time

            # Reset per-tick fresh-bar tracker (used by exit logic)
            self._symbols_with_fresh_bar_this_tick = set()
            # Update per-symbol buffers
            for sym, buf in self._buffers.items():
                if slice.Bars.ContainsKey(sym):
                    bar = slice.Bars[sym]
                    buf.update_bar(bar.Open, bar.High, bar.Low,
                                  bar.Close, bar.Volume)
                    self._symbols_with_fresh_bar_this_tick.add(sym)
                if slice.QuoteBars.ContainsKey(sym):
                    qb = slice.QuoteBars[sym]
                    # QC sometimes delivers QuoteBars with Bid or Ask = None
                    # (one-sided quotes when only one side has a recent fill).
                    bid_px = float(qb.Bid.Close) if (qb.Bid is not None
                                                     and qb.Bid.Close) else 0.0
                    ask_px = float(qb.Ask.Close) if (qb.Ask is not None
                                                     and qb.Ask.Close) else 0.0
                    if bid_px > 0 or ask_px > 0:
                        buf.update_quote(bid_px, ask_px)

            # F&G value
            if slice.ContainsKey(self._fg_symbol):
                fg = slice[self._fg_symbol]
                if fg is not None:
                    self._fg_value = float(fg.Value)

            # Apex: feed Binance funding rate ticks into the store
            if self._apex is not None and self._binance_funding_symbols:
                for perp, fund_sym in self._binance_funding_symbols.items():
                    try:
                        if slice.ContainsKey(fund_sym):
                            tick = slice[fund_sym]
                            if tick is not None and tick.Value is not None:
                                self._apex.funding_store.record(
                                    perp, float(tick.Value),
                                )
                    except Exception:
                        pass

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

            # ── Apex 4h scoring tick (gated) ──────────────────────────────
            if self._apex is not None and self._apex_should_run_4h_tick(now):
                self._apex_run_4h_tick(now, slice)

            # ── Apex per-minute exit pass (gated) ─────────────────────────
            if self._apex is not None and self._apex.engine.open_positions:
                cur_prices: dict[str, float] = {}
                for sym_str in list(self._apex.engine.open_positions.keys()):
                    try:
                        sym = self.Symbol(sym_str)
                        cur_prices[sym_str] = float(self.Securities[sym].Price)
                    except Exception:
                        cur_prices[sym_str] = 0.0
                try:
                    self._apex.engine.on_minute_tick(
                        now=now,
                        current_prices=cur_prices,
                        latest_probs={},   # only flip on next 4h re-score
                        place_exit_fn=self._apex_place_exit,
                    )
                except Exception as exc:
                    self.Debug(f"[apex] minute exit pass failed: {exc}")

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
            base_max = _cfg.MAX_POSITIONS
            effective_max = max(1, round(base_max * fg_signal.max_positions_multiplier))

            # ── Capital-aware position cap ─────────────────────────────────
            # On Kraken, min order USD is ~$5 + the per-symbol min-qty
            # constraint (e.g. BTC needs ≥0.0001 = ~$10 at $100k BTC).
            # If equity is too small to fit `effective_max` positions
            # of meaningful size, REDUCE the cap. Otherwise we open
            # positions too small to ever sell → INVALID exit spiral.
            equity = float(self.Portfolio.TotalPortfolioValue)
            min_pos_usd_floor = 12.0   # safe-above-Kraken-min for major
            capital_max = max(1, int(equity / min_pos_usd_floor))
            effective_max = min(effective_max, capital_max)

            # ── Submit entries up to effective_max ────────────────────────
            # Match by symbol VALUE (string), not by Symbol object.
            held_values = {s.Value for s in self._open.keys()}
            slots_open = effective_max - len(self._open)
            for score in ranked[:max(0, slots_open)]:
                if score.symbol in held_values:
                    continue
                # ALSO check actual portfolio quantity — the Sardine bug
                # was opening duplicates after force-cleanup wiped local
                # state but the position was still actually held.
                sym_obj = next(
                    (s for s in self._symbols if s.Value == score.symbol), None
                )
                if sym_obj is not None and \
                   abs(float(self.Portfolio[sym_obj].Quantity)) > 0:
                    continue
                # Per-symbol cooldown (post-invalid or recent-fill)
                cd = self._symbol_cooldown_until.get(score.symbol)
                if cd is not None and now < cd:
                    continue
                # Per-symbol exit-attempt pause: if we're paused on exits,
                # don't open a new position either (we couldn't sell it).
                ep = self._exit_pause_until.get(score.symbol)
                if ep is not None and now < ep:
                    continue
                self._try_enter(score, now)

            self._last_decision_time = now

        def OnOrderEvent(self, event):
            try:
                res = _on_order_event_impl(self, event, self._audit)
                sym_value = (event.Symbol.Value if hasattr(event.Symbol, "Value")
                              else str(event.Symbol))
                # Cooldown on invalid ENTRIES only (not exits).
                # Failed exits should NOT block future entries on the same
                # symbol — that prevents the strategy from ever re-engaging
                # with a coin after a transient data gap.
                # Direction 0 = Buy = entry; 1 = Sell = exit.
                status_str = str(event.Status).split(".")[-1]
                is_entry = False
                try:
                    from AlgorithmImports import OrderDirection
                    is_entry = (event.Direction == OrderDirection.Buy)
                except Exception:
                    is_entry = (str(event.Direction).endswith("Buy"))
                if status_str == "Invalid" and is_entry:
                    self._symbol_cooldown_until[sym_value] = (
                        self.Time + timedelta(minutes=self._reentry_cooldown_minutes)
                    )
                # Exit-side failure backoff: 3 strikes → 30-min pause on
                # exit attempts for THIS symbol. Stops the per-minute
                # spam pattern observed in the orders log.
                if status_str == "Invalid" and not is_entry:
                    cnt = self._exit_failure_count.get(sym_value, 0) + 1
                    self._exit_failure_count[sym_value] = cnt
                    if cnt >= 3:
                        self._exit_pause_until[sym_value] = (
                            self.Time + timedelta(minutes=30)
                        )
                        self._exit_failure_count[sym_value] = 0   # reset counter
                # Successful exit fill clears the failure counter
                if status_str == "Filled" and not is_entry:
                    self._exit_failure_count.pop(sym_value, None)
                    self._exit_pause_until.pop(sym_value, None)
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
                    # Record trade outcome for the online learner
                    if res.pnl_pct is not None:
                        # Use the latest entry score as a heuristic (we don't
                        # have per-trade score plumbing yet); pass 0.55 as a
                        # neutral default.
                        self._learner.record(
                            timestamp=self.Time, pnl_pct=res.pnl_pct,
                            score=0.55, high_conviction=False,
                        )
                        # Try a tune cycle (throttled to once per 12h internally)
                        action = self._learner.tune(self.Time)
                        if action.action in ("tightened", "loosened"):
                            self.Log(
                                f"[learner] {action.action} thresholds: "
                                f"entry {action.old_entry:.3f}→{action.new_entry:.3f} "
                                f"hc {action.old_hc:.3f}→{action.new_hc:.3f} "
                                f"({action.reason})"
                            )
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
            """Build cross-symbol context. Each block degrades gracefully
            so the strategy keeps working when the rolling buffer is
            shorter than the ideal lookback (e.g. early in a backtest)."""
            ctx = MarketContext()
            if self._btc_sym and self._buffers[self._btc_sym].closes:
                btc_buf = self._buffers[self._btc_sym]
                closes = list(btc_buf.closes)
                vols   = list(btc_buf.volumes)
                # Approximate 4h closes by sampling every 240th 1m bar
                # (max 30 samples). When we have fewer than 240 bars,
                # fall back to a coarser sampling so MarketModeDetector
                # has SOMETHING to work with.
                step_4h = max(1, len(closes) // 30)
                ctx.btc_4h_closes  = closes[::step_4h][-30:]
                ctx.btc_4h_volumes = vols[::step_4h][-30:]
                # Daily sampling: every 1440 bars OR coarser fallback
                step_d = max(1, len(closes) // 200)
                ctx.btc_daily_closes = closes[::step_d][-200:]
                # 30d return: prefer 30 days of bars; else use whatever we have
                lookback_30d = min(1440 * 30, len(closes) - 1)
                if lookback_30d >= 60 and closes[-lookback_30d] > 0:
                    ctx.btc_30d_return = (
                        closes[-1] - closes[-lookback_30d]
                    ) / closes[-lookback_30d]
            # Symbol recent returns (5min lookback for spillover)
            recent: dict[str, float] = {}
            for sym, buf in self._buffers.items():
                if len(buf.closes) >= 5 and buf.closes[-5] > 0:
                    recent[sym.Value] = (buf.closes[-1] - buf.closes[-5]) / buf.closes[-5]
            ctx.symbol_recent_returns = recent
            # Alt 30d returns — fall back to whatever lookback fits
            alt_returns = []
            for sym, buf in self._buffers.items():
                if sym == self._btc_sym or len(buf.closes) < 60:
                    continue
                lb = min(1440 * 30, len(buf.closes) - 1)
                if lb >= 60 and buf.closes[-lb] > 0:
                    r = (buf.closes[-1] - buf.closes[-lb]) / buf.closes[-lb]
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
                    continue   # not enough history for feature computation
                # Universe gate — primarily catches FARTCOIN/PEAQ-style
                # new-listing pollution. Our curated universe is all
                # established multi-year symbols on Kraken, so we pass
                # a safe high days_of_history value instead of computing
                # it from buffer length (which is intentionally short
                # for memory reasons).
                stats = buf.stats_for_universe(
                    symbol=sym.Value,
                    days_of_history=9999,
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
            # Live thresholds from the OnlineThresholdLearner (auto-adapt)
            return rank_candidates(
                candidates, self._market_context,
                entry_threshold=self._learner.entry_threshold,
                high_conviction_thres=self._learner.high_conviction_thres,
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

                # ── Equity-aware fair-share cap ────────────────────────────
                # The tier max (e.g. $5K major / $1.5K large) is a CEILING.
                # The actual size must also fit within the strategy's
                # current capital. We allocate by fair-share:
                #   per_position_share = total_equity / max_positions
                # so MAX_POSITIONS concurrent trades can all fit.
                total_equity = float(self.Portfolio.TotalPortfolioValue)
                fair_share = total_equity / max(_cfg.MAX_POSITIONS, 1)
                size_usd = min(max_pos_usd, fair_share)

                # Apply score multipliers
                size_usd *= score.composed_size_mult
                if score.high_conviction:
                    size_usd *= 1.0
                else:
                    size_usd *= 0.7

                # ── Hard cap by AVAILABLE USD CASH (not total NAV) ─────────
                # Lesson learned (live backtest 2025-01-02→01-06):
                # In QC crypto CASH accounts, Portfolio.MarginRemaining and
                # Portfolio.Cash both return TOTAL NAV (cash + crypto value).
                # The only API that returns true unspent USD is
                # Portfolio.CashBook["USD"].Amount. Use it FIRST.
                #
                # We also subtract the USD value of any open BUY orders that
                # are pending fill — QC doesn't auto-reserve cash for them
                # in cash mode, so multiple in-flight orders can stack up.
                try:
                    available = float(self.Portfolio.CashBook["USD"].Amount)
                except Exception:
                    try:
                        available = float(self.Portfolio.MarginRemaining)
                    except Exception:
                        available = float(self.Portfolio.Cash)
                # Subtract pending buy-order value (best-effort)
                try:
                    pending_usd = sum(
                        float(t.Quantity) * float(self.Securities[t.Symbol].Price)
                        for t in self.Transactions.GetOpenOrders()
                        if hasattr(t, "Direction") and t.Quantity > 0
                    )
                    available = max(0.0, available - pending_usd)
                except Exception:
                    pass
                # 95% of available — leave 5% headroom for fees/slippage
                size_usd = min(size_usd, available * 0.95)

                if size_usd < 5.0:
                    self.Debug(
                        f"[pulse] skip {sym.Value}: size_usd={size_usd:.2f} "
                        f"< 5.0 (avail={available:.2f}, equity={total_equity:.2f})"
                    )
                    return
                price = float(self.Securities[sym].Price)
                if price <= 0:
                    return
                qty = size_usd / price

                # ── CRITICAL: validate vs Kraken min order quantity ────────
                # If qty < Kraken min-qty, the buy might fill but the SELL
                # will be rejected as INVALID — leading to stuck positions
                # and the cascade we saw in 'Emotional Light Brown Sardine'
                # backtest. Reject the entry now, log the reason.
                min_qty = min_quantity_fallback(sym.Value)
                if qty < min_qty:
                    # Try to scale up to the min if cash allows; if not,
                    # skip and put the symbol in a 30-min cooldown so we
                    # don't repeatedly evaluate it.
                    needed_usd = min_qty * price * 1.05   # 5% buffer for fees
                    if needed_usd <= available * 0.95:
                        # Scale up to exactly min_qty
                        qty = min_qty * 1.001    # tiny safety margin above
                        size_usd = qty * price
                    else:
                        self._symbol_cooldown_until[sym.Value] = (
                            now + timedelta(minutes=30)
                        )
                        self.Debug(
                            f"[pulse] skip {sym.Value}: qty={qty:.8f} "
                            f"below Kraken min {min_qty:.8f} "
                            f"(would need ${needed_usd:.2f}, have ${available:.2f})"
                        )
                        return

                # Optional: slice large entries via Almgren-Chriss
                # (only for high-conviction trades that would slip ≥25bp)
                buf = self._buffers.get(sym)
                avg_bar_vol = (sum(buf.volumes) / len(buf.volumes)
                               if buf and buf.volumes else 0)
                slice_plan = build_slice_plan(
                    total_quantity=qty,
                    bar_volume_estimate=avg_bar_vol,
                    n_slices=5 if score.high_conviction else 1,
                    large_order_threshold_bps=DEFAULT_LARGE_ORDER_THRESHOLD_BPS,
                    min_qty_per_slice=min_qty,   # never split below exchange min
                )
                self.Log(
                    f"[pulse] SCALP ENTRY {sym.Value} score={score.score:.3f} "
                    f"hc={score.high_conviction} mode={score.market_mode} "
                    f"tier={tier_lim['tier']} size_usd={size_usd:.2f} qty={qty:.6f} "
                    f"slices={slice_plan.n_slices} "
                    f"slip_est={slice_plan.expected_total_slip_bps:.1f}bp "
                    f"saved={slice_plan.saved_bps:.1f}bp"
                )
                if slice_plan.skipped_reason or slice_plan.n_slices == 1:
                    self.MarketOrder(sym, qty, tag="ENTRY")
                else:
                    # Submit each child order back-to-back; in production
                    # these would be spaced across bars by a queue manager.
                    for step in slice_plan.steps:
                        if step.quantity > 0:
                            self.MarketOrder(sym, step.quantity,
                                             tag=f"ENTRY_S{step.idx}")
            except Exception as exc:
                self.Debug(f"_try_enter error {score.symbol}: {exc}")

        def _manage_open_positions(self, now):
            # ── Sync local state with actual portfolio ────────────────────
            for sym in list(self._open.keys()):
                actual_qty = float(self.Portfolio[sym].Quantity)
                if actual_qty == 0:
                    self._open.pop(sym, None)

            for sym, pos in list(self._open.items()):
                # Only attempt exits when a FRESH bar arrived this tick.
                # If we don't have new data, QC will fill the market order
                # at price=0 (rejected as Invalid). Defer to next tick.
                if sym not in self._symbols_with_fresh_bar_this_tick:
                    continue

                # Per-symbol exit-attempt pause (after N consecutive failures)
                pause = self._exit_pause_until.get(sym.Value)
                if pause is not None and now < pause:
                    continue

                price = float(self.Securities[sym].Price)
                if price <= 0:
                    # Defensive: even with a fresh bar, double-check price
                    continue
                pos.update_extremes(price)
                ret = (price - pos.entry_price) / pos.entry_price

                # ── Pick exit reason (cascade) ────────────────────────────
                exit_tag = None
                # 1. Per-trade hard kill at -8%
                kill_dec = self._kill.evaluate(sym.Value, pos.entry_price, price)
                if kill_dec.should_kill:
                    self.Log(f"[pulse] HARD KILL {sym.Value} ret={ret:+.2%}")
                    exit_tag = "HARD_KILL"
                # 2. Hard SL at -3.5%
                elif ret <= -TIGHT_STOP_LOSS_PCT:
                    exit_tag = "STOP_LOSS"
                # 3. Time stop
                elif pos.held_hours(now) >= _cfg.TIME_STOP_HOURS:
                    exit_tag = "TIME_STOP"
                else:
                    # 4. Trail (arm at +4%, trail 2.5% from high)
                    max_ret = (pos.high_price - pos.entry_price) / pos.entry_price
                    if max_ret >= TRAIL_ACTIVATION_PCT:
                        pos.trail_armed = True
                    if pos.trail_armed:
                        trail_stop_price = pos.high_price * (1 - TRAIL_STOP_PCT)
                        if price < trail_stop_price:
                            exit_tag = "TRAIL_STOP"
                    # 5. Take profit at +12%
                    if exit_tag is None and ret >= QUICK_TAKE_PROFIT_PCT:
                        exit_tag = "TAKE_PROFIT"

                if exit_tag is None:
                    continue

                # ── ROOT-CAUSE FIX: safe-sell quantity from CashBook ───────
                # The Vox CashBook bug, finally wired into the exit path.
                # In QC crypto cash mode, Portfolio[sym].Quantity can drift
                # SLIGHTLY ABOVE the true CashBook[base].Amount (fees are
                # paid from the base currency on each leg). self.Liquidate
                # tries to sell Portfolio.Quantity → CashBook says "you
                # don't have that much" → INVALID. We compute a safe sell
                # qty using the lower of (Portfolio, CashBook) minus a
                # 1-lot buffer, then submit a market order for that.
                self._submit_safe_exit(sym, exit_tag)

        def _submit_safe_exit(self, sym, tag: str):
            """Sell `sym` using safe_sell_quantity (CashBook-aware)."""
            try:
                portfolio_qty = float(self.Portfolio[sym].Quantity)
                if portfolio_qty == 0:
                    self._open.pop(sym, None)
                    return
                # Find the base currency: Kraken pairs are SYMUSD → SYM
                sym_value = sym.Value
                base_ccy = sym_value[:-3] if sym_value.endswith("USD") else None
                cashbook_qty = portfolio_qty
                if base_ccy:
                    try:
                        cashbook_qty = float(
                            self.Portfolio.CashBook[base_ccy].Amount
                        )
                    except Exception:
                        cashbook_qty = portfolio_qty
                # Lot size + min order from QC SymbolProperties (fallback to table)
                sec = self.Securities[sym]
                sp  = getattr(sec, "SymbolProperties", None)
                lot_size = (float(sp.LotSize)
                            if sp and sp.LotSize and float(sp.LotSize) > 0
                            else min_quantity_fallback(sym_value))
                min_ord  = (float(sp.MinimumOrderSize)
                            if sp and sp.MinimumOrderSize and float(sp.MinimumOrderSize) > 0
                            else min_quantity_fallback(sym_value))
                safe_qty = safe_sell_quantity(
                    portfolio_quantity=abs(portfolio_qty),
                    cashbook_quantity=abs(cashbook_qty),
                    lot_size=lot_size,
                    min_order_size=min_ord,
                    exit_qty_buffer_lots=1,
                )
                if safe_qty <= 0:
                    # Position is dust — clear local state + give up
                    self.Log(
                        f"[pulse] {tag} {sym_value}: dust qty "
                        f"(port={portfolio_qty:.10f} cashbook={cashbook_qty:.10f} "
                        f"min={min_ord:.10f}); clearing local state"
                    )
                    self._open.pop(sym, None)
                    return
                # Market sell the safe quantity
                self.MarketOrder(sym, -safe_qty, tag=tag)
            except Exception as exc:
                self.Debug(f"_submit_safe_exit error {sym}: {exc}")

        def _liquidate_all_open(self):
            for sym in list(self._open.keys()):
                self._submit_safe_exit(sym, tag="CIRCUIT_HALT")
            self._open.clear()
            # Also liquidate any Apex-managed positions
            if self._apex is not None:
                for sym_str in list(self._apex.engine.open_positions.keys()):
                    try:
                        sym = self.Symbol(sym_str)
                        self._submit_safe_exit(sym, tag="APEX_CIRCUIT_HALT")
                    except Exception:
                        pass
                self._apex.engine.reset()

        # ── Apex CSV loaders ───────────────────────────────────────────────
        def _apex_load_static_csvs(self) -> None:
            """Read any bundled static CSVs and seed the Apex stores.

            Tries QC's ObjectStore first (where files are mounted at
            runtime), then a few common path candidates. Silently skips
            missing files — Apex still runs (just with fewer signals).
            """
            for fname, loader in (
                ("apex_etf_flows.csv",
                 self._apex.etf_flow_store.load_csv),
                ("apex_token_unlocks.csv",
                 self._apex.unlock_store.load_csv),
                ("apex_stablecoin_supply.csv",
                 self._apex.stablecoin_store.load_csv),
                ("apex_news_sentiment.csv",
                 self._apex.news_store.load_csv),
            ):
                content = self._apex_read_static_file(fname)
                if not content:
                    continue
                try:
                    loader(content)
                    self.Log(f"[apex] loaded {fname} ({len(content)} bytes)")
                except Exception as exc:
                    self.Debug(f"[apex] {fname} parse failed: {exc}")

        def _apex_read_static_file(self, name: str) -> str:
            """Try several locations to find a bundled CSV."""
            # 1) QC ObjectStore (most reliable in cloud)
            try:
                if hasattr(self, "ObjectStore") and self.ObjectStore is not None:
                    if self.ObjectStore.ContainsKey(name):
                        return self.ObjectStore.Read(name) or ""
            except Exception:
                pass
            # 2) Common project-relative paths
            import os
            for path in (
                name,
                os.path.join("project", name),
                os.path.join("data", name),
            ):
                try:
                    if os.path.isfile(path):
                        with open(path, "r", encoding="utf-8") as f:
                            return f.read()
                except Exception:
                    continue
            return ""

        # ── Apex callbacks ─────────────────────────────────────────────────
        def _apex_place_order(self, sym_str: str, qty: float,
                               tag: str, price: float) -> None:
            """Apex entry callback — places a market buy on Kraken."""
            try:
                sym = self.Symbol(sym_str)
                if qty <= 0:
                    return
                self.MarketOrder(sym, qty, tag=tag)
                self.Log(f"[apex] ENTRY {sym_str} qty={qty:.6f} "
                         f"~${qty*price:.0f} tag={tag}")
            except Exception as exc:
                self.Debug(f"[apex] order failed for {sym_str}: {exc}")

        def _apex_place_exit(self, sym_str: str, qty: float,
                              reason: str) -> None:
            """Apex exit callback — uses safe_sell_quantity-based path."""
            try:
                sym = self.Symbol(sym_str)
                self._submit_safe_exit(sym, tag=f"APEX_{reason}")
                self.Log(f"[apex] EXIT  {sym_str} qty={qty:.6f} reason={reason}")
            except Exception as exc:
                self.Debug(f"[apex] exit failed for {sym_str}: {exc}")

        def _apex_should_run_4h_tick(self, now: datetime) -> bool:
            """True if it's been ≥ APEX_REBALANCE_HOURS since the last tick."""
            from Pulse.apex.config import APEX_REBALANCE_HOURS
            if self._apex_last_tick is None:
                return True
            elapsed = (now - self._apex_last_tick).total_seconds() / 3600
            return elapsed >= APEX_REBALANCE_HOURS

        def _apex_run_4h_tick(self, now: datetime, slice_) -> None:
            """Score the universe with Apex; place entries via callbacks."""
            if self._apex is None:
                return
            # Best-effort: feed BTCUSD price into the on-chain valuation store
            # (we don't have BitcoinMetadata or CoinGecko subscribed yet,
            # but the price feed is always available).
            try:
                btc_sym = self.Symbol("BTCUSD")
                btc_price = float(self.Securities[btc_sym].Price)
                if btc_price > 0:
                    # Placeholder volume = 1 (we lack on-chain volume here)
                    self._apex.onchain_val_store.record(btc_price, 1.0)
            except Exception:
                pass

            # Build tier-cap helper from existing tier classifier
            def _tier_cap(sym_str: str) -> float:
                try:
                    sym = self.Symbol(sym_str)
                    lim = self._tiers.limits_for(sym.Value, now=now)
                    return float(lim["max_pos_usd"])
                except Exception:
                    return 500.0

            def _price_for(sym_str: str) -> float:
                try:
                    sym = self.Symbol(sym_str)
                    return float(self.Securities[sym].Price)
                except Exception:
                    return 0.0

            equity = float(self.Portfolio.TotalPortfolioValue)
            universe = [s.Value for s in self._symbols]

            try:
                self._apex.engine.on_4h_tick(
                    now=now, universe=universe,
                    market_context_provider=self._apex.context_provider,
                    equity=equity,
                    place_order_fn=self._apex_place_order,
                    tier_max_pos_usd_provider=_tier_cap,
                    regime_mult_provider=lambda s: 1.0,
                    current_price_provider=_price_for,
                    current_atr_provider=lambda s: max(_price_for(s) * 0.02, 0.0),
                )
            except Exception as exc:
                self.Debug(f"[apex] 4h tick failed: {exc}")
            self._apex_last_tick = now

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
