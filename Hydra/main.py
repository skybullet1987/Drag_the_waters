# ══════════════════════════════════════════════════════════════════════════════
# HYDRA v8 — Relative Strength Momentum Rotation
#
# 100% into the hottest coin. Rotate when a hotter one appears.
# Regime filter prevents trading in bear markets.
#
# This is the simplest strategy that could 100x:
#   - Rank coins by 24h momentum every 4 hours
#   - Go 100% into the leader (maximum concentration = maximum compounding)
#   - Only rotate when new leader is 2%+ stronger
#   - Golden cross regime: cash during bear
#   - Trailing stop: 5% from high to protect gains
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np
from datetime import timedelta

COINS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD",
    "XDGUSD", "AVAXUSD", "LINKUSD", "DOTUSD", "LTCUSD",
    "NEARUSD", "SUIUSD", "TRXUSD", "BCHUSD", "TONUSD",
    "TAOUSD", "FETUSD", "HBARUSD", "RENDERUSD", "ICPUSD",
    "ALGOUSD", "STXUSD", "AAVEUSD", "UNIUSD", "PEPEUSD",
    "ONDOUSD", "KASUSD", "PENDLEUSD", "CRVUSD", "INJUSD",
]


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 12, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.02

        self._symbols = []
        for ticker in COINS:
            try:
                s = self.add_crypto(ticker, Resolution.HOUR, Market.KRAKEN).symbol
                self._symbols.append(s)
            except Exception:
                pass

        self._btc = None
        for s in self._symbols:
            if "BTC" in s.value:
                self._btc = s
                break

        self._current_leader = None
        self._entry_px = 0.0
        self._high_water = 0.0

        # Rebalance every 4 hours
        for hour in range(0, 24, 4):
            self.schedule.on(
                self.date_rules.every_day(),
                self.time_rules.at(hour, 0),
                self._rebalance,
            )

        # Trailing stop check every hour
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=1)),
            self._check_trail,
        )

        self.set_warmup(timedelta(days=60))

    # ── REGIME ───────────────────────────────────────────────────────────

    def _is_bull(self):
        """Golden cross + BTC above SMA50. Simple, proven."""
        if not self._btc:
            return False
        h = self.history(self._btc, 210, Resolution.DAILY)
        if h.empty or len(h) < 200:
            return True  # not enough data, assume bull (optimistic start)
        c = h["close"].values
        sma50 = np.mean(c[-50:])
        sma200 = np.mean(c[-200:])
        price = c[-1]
        return price > sma50 and sma50 > sma200

    # ── RANKING ──────────────────────────────────────────────────────────

    def _rank_by_momentum(self):
        """Rank all coins by 24h rate of change. Also check 50-SMA filter."""
        rankings = []
        for sym in self._symbols:
            h = self.history(sym, 52, Resolution.HOUR)  # ~2 days
            if h.empty or len(h) < 24:
                continue
            c = h["close"].values
            roc_24h = (c[-1] - c[-24]) / c[-24]

            # Must be above 50h SMA (not a dead cat bounce)
            h_long = self.history(sym, 55, Resolution.HOUR)
            if not h_long.empty and len(h_long) >= 50:
                sma50h = np.mean(h_long["close"].values[-50:])
                if c[-1] < sma50h:
                    continue  # below SMA = skip

            # Must have positive momentum
            if roc_24h <= 0:
                continue

            rankings.append((sym, roc_24h))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    # ── REBALANCE ────────────────────────────────────────────────────────

    def _rebalance(self):
        if self.is_warming_up:
            return

        # BEAR → liquidate, go cash
        if not self._is_bull():
            if self._current_leader:
                self.liquidate()
                self.log(f"[hydra] BEAR → CASH")
                self._current_leader = None
                self._entry_px = 0.0
                self._high_water = 0.0
            return

        # BULL → find the hottest coin
        rankings = self._rank_by_momentum()
        if not rankings:
            return

        best_sym, best_roc = rankings[0]

        # Already holding the best? Check if something is 2%+ better
        if self._current_leader and self._current_leader == best_sym:
            return  # already in the leader, hold

        if self._current_leader:
            # Current holding's momentum
            current_roc = 0
            for sym, roc in rankings:
                if sym == self._current_leader:
                    current_roc = roc
                    break

            # Only rotate if new leader is significantly stronger
            if best_roc < current_roc + 0.02:
                return  # not 2%+ better, don't rotate

        # ROTATE: sell current, buy new leader
        self.liquidate()
        self.set_holdings(best_sym, 0.98)  # 98% (leave 2% for fees)
        self._current_leader = best_sym
        self._entry_px = float(self.securities[best_sym].price)
        self._high_water = self._entry_px

        self.log(f"[hydra] ROTATE → {best_sym.value} ROC24h={best_roc:+.1%}"
                 f" px={self._entry_px:.4f} pv=${float(self.portfolio.total_portfolio_value):.2f}")

    # ── TRAILING STOP ────────────────────────────────────────────────────

    def _check_trail(self):
        if self.is_warming_up or not self._current_leader:
            return

        sym = self._current_leader
        price = float(self.securities[sym].price)
        if price <= 0 or self._entry_px <= 0:
            return

        # Update high water
        if price > self._high_water:
            self._high_water = price

        ret = (price - self._entry_px) / self._entry_px
        max_ret = (self._high_water - self._entry_px) / self._entry_px

        # Hard stop: -5% from entry
        if ret < -0.05:
            self.liquidate()
            self.log(f"[hydra] STOP LOSS {sym.value} ret={ret:+.1%}")
            self._current_leader = None
            return

        # Trailing: once we're +3% up, trail at 5% from high
        if max_ret >= 0.03:
            trail_stop = self._high_water * 0.95
            if price < trail_stop:
                self.liquidate()
                self.log(f"[hydra] TRAIL EXIT {sym.value} ret={ret:+.1%} max={max_ret:+.1%}")
                self._current_leader = None
                return

        # BTC fast crash check
        if self._btc:
            btc_h = self.history(self._btc, 25, Resolution.HOUR)
            if not btc_h.empty and len(btc_h) >= 24:
                btc_ret = (btc_h["close"].values[-1] - btc_h["close"].values[0]) / btc_h["close"].values[0]
                if btc_ret < -0.03:
                    self.liquidate()
                    self.log(f"[hydra] BTC CRASH {btc_ret:+.1%} → CASH")
                    self._current_leader = None

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED:
            if order_event.fill_quantity > 0:  # buy
                self._entry_px = float(order_event.fill_price)
                self._high_water = self._entry_px
