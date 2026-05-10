# ══════════════════════════════════════════════════════════════════════════════
# HYDRA v5 — "Whale Rider"
# Position trading for 100x at 1x spot. NO leverage needed.
#
# INSIGHT: stop scalping +2%. Catch +50-300% alt moves with wide trail.
# Nov-Dec 2024: SOL +73%, XRP +316%, DOGE +350%. ONE trade = +60-250%.
#
# STRATEGY:
#   1. DAILY bars (not 5-min) — hold for DAYS/WEEKS, not hours
#   2. BTC above 50-day SMA = BULL → trade. Below = CASH.
#   3. Buy top 1-2 momentum alts (strongest 7-day return)
#   4. Wide 10% trailing stop — ride the ENTIRE multi-week pump
#   5. When trail fires → rotate into next hottest alt
#   6. Compound: $100 → $160 → $480 → $1,680 → ...
#
# This is Turtle Trading for crypto. Original Turtles did 100x+.
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np
from collections import deque

COINS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD",
    "XDGUSD", "AVAXUSD", "LINKUSD", "DOTUSD", "LTCUSD",
    "NEARUSD", "SUIUSD", "RENDERUSD", "PEPEUSD", "ONDOUSD",
]

# ── Parameters ───────────────────────────────────────────────────────────────
TRAIL_PCT            = 0.10    # 10% trail — ride massive moves
SL_PCT               = 0.07    # 7% stop loss — give room for multi-day swings
ALLOC                = 0.90    # 90% per position — aggressive compounding
MOMENTUM_LOOKBACK    = 7       # 7-day momentum for ranking
BTC_SMA_PERIOD       = 50      # 50-day SMA for regime
REBALANCE_DAYS       = 3       # check for rotation every 3 days
MIN_MOMENTUM         = 0.02    # minimum 2% weekly return to qualify


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 10, 1)
        self.set_end_date(2025, 3, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.01

        # Daily resolution — no timeout issues, holds for days/weeks
        self._symbols = []
        for ticker in COINS:
            try:
                sym = self.add_crypto(ticker, Resolution.DAILY, Market.KRAKEN).symbol
                self._symbols.append(sym)
            except Exception:
                pass
        self.log(f"[hydra] {len(self._symbols)} coins loaded (DAILY resolution)")

        # BTC reference
        self._btc_sym = None
        for sym in self._symbols:
            if "BTC" in sym.value:
                self._btc_sym = sym
                break

        # Position state
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._high_water = 0.0

        # Schedule: check every day
        for sym in self._symbols:
            if sym != self._btc_sym:
                self.schedule.on(
                    self.date_rules.every_day(sym),
                    self.time_rules.after_market_open(sym, 1),
                    self._daily_check,
                )
                break  # just need one schedule

        self.set_warmup(60)  # 60 days warmup for SMA50

    def _is_bull_market(self):
        """BTC above 50-day SMA = bull market. Trade. Below = cash."""
        if not self._btc_sym:
            return False
        hist = self.history(self._btc_sym, 55, Resolution.DAILY)
        if hist.empty or len(hist) < 50:
            return False
        closes = hist["close"].values
        sma50 = np.mean(closes[-50:])
        current = closes[-1]
        # Bull: BTC above SMA50 AND SMA trending up
        sma50_prev = np.mean(closes[-55:-5]) if len(closes) >= 55 else sma50
        trending_up = sma50 > sma50_prev
        return current > sma50 and trending_up

    def _rank_alts_by_momentum(self):
        """Rank all alts by 7-day return. Strongest momentum = best trade."""
        rankings = []
        for sym in self._symbols:
            if sym == self._btc_sym:
                continue
            hist = self.history(sym, MOMENTUM_LOOKBACK + 2, Resolution.DAILY)
            if hist.empty or len(hist) < MOMENTUM_LOOKBACK:
                continue
            closes = hist["close"].values
            volumes = hist["volume"].values
            if closes[-1] <= 0 or closes[0] <= 0:
                continue

            ret_7d = (closes[-1] - closes[0]) / closes[0]
            avg_vol = np.mean(volumes)

            if ret_7d < MIN_MOMENTUM:
                continue
            if avg_vol <= 0:
                continue

            # Check trend: EMA8 > EMA21 on daily
            hist_long = self.history(sym, 25, Resolution.DAILY)
            if not hist_long.empty and len(hist_long) >= 21:
                cl = hist_long["close"].values
                ema8 = np.mean(cl[-8:])
                ema21 = np.mean(cl[-21:])
                if ema8 < ema21:
                    continue  # downtrending, skip

            rankings.append((sym, ret_7d, avg_vol))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _daily_check(self):
        """Main daily logic."""
        if self.is_warming_up:
            return

        # ── Check exit on current position ───────────────────────────────
        if self._pos_sym:
            price = float(self.securities[self._pos_sym].price)
            if price <= 0:
                return

            if price > self._high_water:
                self._high_water = price

            ret = (price - self._entry_px) / self._entry_px if self._entry_px > 0 else 0
            max_ret = (self._high_water - self._entry_px) / self._entry_px if self._entry_px > 0 else 0

            # Stop loss
            if ret <= -SL_PCT:
                self._exit("EXIT_SL")
                return

            # Trailing stop
            if max_ret >= 0.03:  # arm trail after +3%
                trail_stop = self._high_water * (1 - TRAIL_PCT)
                if price <= trail_stop:
                    self._exit("EXIT_TRAIL")
                    return

            # Regime change: BTC turned bearish → exit everything
            if not self._is_bull_market():
                if ret > 0:
                    self._exit("EXIT_REGIME_BULL")
                elif ret < -0.03:
                    self._exit("EXIT_REGIME_CUT")
                return

            # Check if we should rotate to a stronger coin
            days_held = (self.time - self._entry_time).days if self._entry_time else 0
            if days_held >= REBALANCE_DAYS:
                rankings = self._rank_alts_by_momentum()
                if rankings and rankings[0][0] != self._pos_sym:
                    top_sym, top_ret, _ = rankings[0]
                    current_ret = ret
                    # Only rotate if new coin has 2x the momentum
                    if top_ret > current_ret * 2 and top_ret > 0.05:
                        self._exit("EXIT_ROTATE")
                        return

            return  # holding position, no action needed

        # ── No position: look for entry ──────────────────────────────────

        # RULE 1: Bull market only
        if not self._is_bull_market():
            return

        # RULE 2: Find strongest momentum alt
        rankings = self._rank_alts_by_momentum()
        if not rankings:
            return

        sym, ret_7d, vol = rankings[0]
        price = float(self.securities[sym].price)
        if price <= 0:
            return

        # RULE 3: Size based on momentum strength
        if ret_7d > 0.15:
            alloc = 0.95   # exceptional: near-full
        elif ret_7d > 0.08:
            alloc = 0.85   # strong
        elif ret_7d > 0.04:
            alloc = 0.70   # moderate
        else:
            alloc = 0.50   # base

        pv = float(self.portfolio.total_portfolio_value)
        qty = (pv * alloc * 0.99) / price
        lot = self.securities[sym].symbol_properties.lot_size
        min_order = self.securities[sym].symbol_properties.minimum_order_size
        qty = max(0, (qty // lot) * lot)
        if qty < min_order:
            return

        self.log(
            f"[hydra] ENTRY {sym.value} ret7d={ret_7d:.1%} alloc={alloc:.0%}"
            f" px={price:.4f} qty={qty:.6f} pv=${pv:.2f}"
        )

        self._pos_sym = sym
        self._entry_px = price
        self._entry_time = self.time
        self._high_water = price
        self.market_order(sym, qty, tag="ENTRY")

    def _exit(self, tag):
        """Exit current position."""
        sym = self._pos_sym
        if not sym:
            return

        qty = float(self.portfolio[sym].quantity)
        if qty <= 0:
            self._clear()
            return

        # Safe sell: check CashBook
        try:
            quote = self.securities[sym].symbol_properties.quote_currency
            base = sym.value.replace(quote, "")
            if base in self.portfolio.cash_book:
                qty = min(qty, float(self.portfolio.cash_book[base].amount))
        except Exception:
            pass

        lot = self.securities[sym].symbol_properties.lot_size
        qty = max(0, (qty // lot) * lot - lot)
        if qty <= 0:
            self._clear()
            return

        ret = (float(self.securities[sym].price) - self._entry_px) / self._entry_px if self._entry_px > 0 else 0
        held_days = (self.time - self._entry_time).days if self._entry_time else 0
        self.log(
            f"[hydra] {tag} {sym.value} ret={ret:+.1%} held={held_days}d"
            f" hw={self._high_water:.4f} pv=${float(self.portfolio.total_portfolio_value):.2f}"
        )

        self.market_order(sym, -qty, tag=tag)
        self._clear()

    def _clear(self):
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._high_water = 0.0

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.INVALID:
            self._clear()
        elif order_event.status == OrderStatus.FILLED:
            tag = ""
            try:
                tag = self.transactions.get_order_by_id(order_event.order_id).tag
            except Exception:
                pass
            if tag == "ENTRY":
                self._entry_px = float(order_event.fill_price)
                self._high_water = self._entry_px
