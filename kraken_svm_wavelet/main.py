# region imports
from AlgorithmImports import *
import numpy as np

from svmwavelet import SVMWavelet
from universe import select_top_kraken_usd
from execution import set_holdings_limit, cancel_stale_orders
# endregion


class KrakenSvmWaveletAlgorithm(QCAlgorithm):
    """
    Multi-asset SVM + Wavelet daily forecaster for Kraken USD pairs.

    Architecture is a direct generalization of the QC HandsOnAITradingBook
    'FX SVM Wavelet Forecasting' sample to a top-N crypto universe with:
      - Limit-order execution (maker-fee preference).
      - Cached SVMWavelet refit every `refit_every_bars` daily bars
        (default 7 = weekly; set to 1 for full textbook fidelity).
      - Equal-weight cap per asset to bound concentration.
    """

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2026, 4, 1)
        self.set_cash(5000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # ---- Parameters ----
        self._period = int(self.get_parameter("period") or 152)
        self._weight_threshold = float(self.get_parameter("weight_threshold") or 0.005)
        self._max_universe_size = int(self.get_parameter("max_universe_size") or 50)
        self._refit_every_bars = int(self.get_parameter("refit_every_bars") or 7)
        self._max_per_asset_weight = float(self.get_parameter("max_per_asset_weight") or 0.10)

        # ---- Universe ----
        self.universe_settings.resolution = Resolution.MINUTE
        self.add_universe(CryptoUniverse.Kraken(
            lambda coarse: select_top_kraken_usd(coarse, max_count=self._max_universe_size)
        ))

        # ---- Per-symbol state ----
        self._wavelet = SVMWavelet()
        self._symbol_state = {}  # symbol -> dict(window, last_forecast, bars_since_refit)

        # ---- Schedule ----
        # Daily forecast + rebalance; cancel stale limits every 6 hours.
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(0, 5),
            self._daily_rebalance
        )
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=6)),
            lambda: cancel_stale_orders(self, max_age_hours=24)
        )

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            symbol = security.symbol
            if symbol in self._symbol_state:
                continue
            window = RollingWindow[float](self._period)
            # Warm up with daily history.
            try:
                history = self.history[TradeBar](symbol, self._period, Resolution.DAILY)
                for bar in history:
                    window.add(float(bar.close))
            except Exception as e:
                self.debug(f"History warmup failed for {symbol.Value}: {e}")
            self._symbol_state[symbol] = {
                "window": window,
                "last_forecast": None,
                "bars_since_refit": self._refit_every_bars,  # force refit on first call
            }
            # Daily consolidator to keep the window fresh.
            self.consolidate(
                symbol, Resolution.DAILY, TickType.TRADE,
                self._make_consolidation_handler(symbol)
            )
        for security in changes.removed_securities:
            symbol = security.symbol
            if symbol in self._symbol_state:
                del self._symbol_state[symbol]
            if self.portfolio[symbol].invested:
                self.liquidate(symbol)

    def _make_consolidation_handler(self, symbol):
        def handler(bar):
            state = self._symbol_state.get(symbol)
            if state is None:
                return
            state["window"].add(float(bar.close))
            state["bars_since_refit"] += 1
        return handler

    def _daily_rebalance(self):
        if self.is_warming_up:
            return

        # 1) Forecast each ready asset.
        target_weights = {}
        for symbol, state in self._symbol_state.items():
            window = state["window"]
            if not window.is_ready:
                continue
            need_refit = state["bars_since_refit"] >= self._refit_every_bars

            if need_refit or state["last_forecast"] is None:
                try:
                    prices = np.array(list(window))[::-1]
                    forecasted_value = float(self._wavelet.forecast(prices))
                    state["last_forecast"] = forecasted_value
                    state["bars_since_refit"] = 0
                except Exception as e:
                    self.debug(f"Forecast error {symbol.Value}: {e}")
                    continue

            current_price = float(self.securities[symbol].price or 0)
            if current_price <= 0 or state["last_forecast"] is None:
                continue
            raw_weight = (state["last_forecast"] / current_price) - 1.0

            # Threshold gate.
            if abs(raw_weight) < self._weight_threshold:
                target_weights[symbol] = 0.0
                continue

            # Cap per-asset exposure.
            capped = max(min(raw_weight, self._max_per_asset_weight),
                         -self._max_per_asset_weight)
            target_weights[symbol] = capped

        # 2) Normalize so sum(|weights|) <= 1 (cash account constraint).
        gross = sum(abs(w) for w in target_weights.values())
        if gross > 1.0:
            scale = 1.0 / gross
            target_weights = {s: w * scale for s, w in target_weights.items()}

        # 3) On a Kraken cash account, no shorts: clamp negatives to 0
        #    (treat negative forecast as "exit / stay flat").
        target_weights = {s: max(w, 0.0) for s, w in target_weights.items()}

        # 4) Submit limit orders.
        for symbol, weight in target_weights.items():
            set_holdings_limit(self, symbol, weight, tag="SVMWavelet")
        # Liquidate anything not in the target set that we still hold.
        for kvp in self.portfolio:
            symbol = kvp.key
            if symbol not in target_weights and kvp.value.invested:
                set_holdings_limit(self, symbol, 0.0, tag="SVMWavelet exit")
