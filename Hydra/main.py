# ══════════════════════════════════════════════════════════════════════════════
# HYDRA v7 — "Diamond Hands"
#
# STOP trading. START holding.
#
# 2024 reality:
#   SOL: $100 → $260 (+160%)
#   XRP: $0.50 → $2.50 (+400%)
#   DOGE: $0.08 → $0.45 (+462%)
#   Our algo: +15% at best. GARBAGE.
#
# Why: we made 100-300 trades when we should have made 10.
# Every trade = fees + chance of SL loss + missing the pump.
#
# NEW APPROACH: weekly momentum rotation. That's it.
#   Monday: rank alts by 30-day return
#   Buy top 2 with 45% each
#   HOLD for minimum 1 week
#   Only sell if BTC enters bear OR alt breaks below 50-day SMA
#   Rebalance weekly — rotate to stronger alt if 2x better exists
#
# Target: capture 50-200% alt moves with 2-5 trades per quarter
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np

ALTS = [
    # Mega cap
    "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "XDGUSD",
    "AVAXUSD", "LINKUSD", "DOTUSD", "LTCUSD",
    # Large cap
    "NEARUSD", "SUIUSD", "TRXUSD", "BCHUSD", "TONUSD",
    "TAOUSD", "FETUSD", "HBARUSD", "RENDERUSD", "ICPUSD",
    # Mid cap movers
    "ALGOUSD", "STXUSD", "AAVEUSD", "UNIUSD", "PEPEUSD",
    "ONDOUSD", "KASUSD", "PENDLEUSD", "CRVUSD", "INJUSD",
    # High beta
    "JUPUSD", "TIAUSD", "EIGENUSD", "LDOUSD", "ARBUSD",
    "BONKUSD", "SHIBUSD", "APTUSD", "FLRUSD", "ATOMUSD",
    "OPUSD", "SEIUSD", "GRTUSD", "WIFUSD", "POLUSD",
    "GALAUSD", "ETCUSD", "STRKUSD", "WLDUSD", "QNTUSD",
]


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 12, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.01

        # BTC for regime
        self._btc = self.add_crypto("BTCUSD", Resolution.DAILY, Market.KRAKEN).symbol

        # Alts
        self._alts = []
        for ticker in ALTS:
            try:
                sym = self.add_crypto(ticker, Resolution.DAILY, Market.KRAKEN).symbol
                self._alts.append(sym)
            except Exception:
                pass

        # Holdings
        self._held = {}  # sym -> entry_price

        # Weekly rebalance on Monday at noon UTC
        self.schedule.on(
            self.date_rules.every(DayOfWeek.MONDAY),
            self.time_rules.at(12, 0),
            self._weekly_rebalance,
        )

        # Daily regime check at 8am UTC
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(8, 0),
            self._daily_regime_check,
        )

        self.set_warmup(60)

    def _is_bull(self):
        """BTC above 50-day SMA AND SMA trending up = bull."""
        hist = self.history(self._btc, 55, Resolution.DAILY)
        if hist.empty or len(hist) < 50:
            return False
        c = hist["close"].values
        sma50 = np.mean(c[-50:])
        sma50_old = np.mean(c[-55:-5]) if len(c) >= 55 else sma50
        return c[-1] > sma50 and sma50 >= sma50_old * 0.995

    def _rank_alts(self, period=30):
        """Rank alts by N-day return. Returns [(sym, ret, above_sma50)]."""
        ranked = []
        for sym in self._alts:
            hist = self.history(sym, max(period + 5, 55), Resolution.DAILY)
            if hist.empty or len(hist) < period:
                continue
            c = hist["close"].values
            ret = (c[-1] - c[-period]) / c[-period]
            sma50 = np.mean(c[-min(50, len(c)):])
            above_sma = c[-1] > sma50
            ranked.append((sym, ret, above_sma))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _weekly_rebalance(self):
        """Core logic: rank alts, hold top 2, rotate if needed."""
        if self.is_warming_up:
            return

        bull = self._is_bull()

        if not bull:
            # BEAR: liquidate everything
            if self._held:
                self.log(f"[hydra] BEAR REGIME — liquidating all")
                self.liquidate()
                self._held.clear()
            return

        # BULL: find top 2 momentum alts
        ranked = self._rank_alts(30)
        # Must be positive momentum AND above SMA50
        qualified = [(s, r) for s, r, above in ranked if r > 0.01 and above]

        if not qualified:
            return

        top2 = qualified[:2]
        top_syms = {s for s, r in top2}

        # Sell anything not in top 2
        for sym in list(self._held.keys()):
            if sym not in top_syms:
                # Only rotate if something significantly better exists
                current_rank = next((i for i, (s, r, _) in enumerate(ranked) if s == sym), 99)
                if current_rank > 4:  # dropped out of top 5
                    px = float(self.securities[sym].price)
                    entry = self._held[sym]
                    ret = (px - entry) / entry if entry > 0 else 0
                    self.log(f"[hydra] ROTATE OUT {sym.value} ret={ret:+.1%} rank={current_rank}")
                    self._sell(sym)

        # Buy top 2 if not already holding
        pv = float(self.portfolio.total_portfolio_value)
        for sym, ret in top2:
            if sym in self._held:
                continue  # already holding
            if len(self._held) >= 2:
                continue  # full

            alloc = 0.45
            cash = pv * alloc * 0.99
            price = float(self.securities[sym].price)
            if price <= 0:
                continue

            qty = cash / price
            lot = self.securities[sym].symbol_properties.lot_size
            min_order = self.securities[sym].symbol_properties.minimum_order_size
            qty = max(0, (qty // lot) * lot)
            if qty < min_order:
                continue

            self.log(f"[hydra] BUY {sym.value} ret30d={ret:+.1%} alloc={alloc:.0%}"
                     f" px={price:.4f} pv=${pv:.2f}")
            self.market_order(sym, qty, tag="ENTRY")
            self._held[sym] = price

    def _daily_regime_check(self):
        """Emergency exit: BTC drops below SMA50 or crashes."""
        if self.is_warming_up or not self._held:
            return

        # Fast crash detection: BTC -5% in 3 days
        hist = self.history(self._btc, 5, Resolution.DAILY)
        if not hist.empty and len(hist) >= 3:
            c = hist["close"].values
            ret_3d = (c[-1] - c[0]) / c[0]
            if ret_3d < -0.05:
                self.log(f"[hydra] BTC CRASH {ret_3d:+.1%} — emergency liquidate")
                self.liquidate()
                self._held.clear()
                return

        # Regime change: BTC below SMA50
        if not self._is_bull():
            # Check individual positions — exit losers, keep winners
            for sym in list(self._held.keys()):
                px = float(self.securities[sym].price)
                entry = self._held[sym]
                ret = (px - entry) / entry if entry > 0 else 0
                if ret < 0:
                    self.log(f"[hydra] REGIME EXIT {sym.value} ret={ret:+.1%}")
                    self._sell(sym)

    def _sell(self, sym):
        """Sell entire position."""
        qty = float(self.portfolio[sym].quantity)
        if qty <= 0:
            self._held.pop(sym, None)
            return
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
            self._held.pop(sym, None)
            return

        px = float(self.securities[sym].price)
        entry = self._held.get(sym, px)
        ret = (px - entry) / entry if entry > 0 else 0
        self.log(f"[hydra] SELL {sym.value} ret={ret:+.1%} pv=${float(self.portfolio.total_portfolio_value):.2f}")
        self.market_order(sym, -qty, tag="EXIT")
        self._held.pop(sym, None)

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED:
            sym = order_event.symbol
            tag = ""
            try:
                tag = self.transactions.get_order_by_id(order_event.order_id).tag
            except Exception:
                pass
            if tag == "ENTRY":
                self._held[sym] = float(order_event.fill_price)
        elif order_event.status == OrderStatus.INVALID:
            sym = order_event.symbol
            self._held.pop(sym, None)
