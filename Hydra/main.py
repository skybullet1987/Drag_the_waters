# ══════════════════════════════════════════════════════════════════════════════
# HYDRA v4 — "Turtle Crypto"
# The simplest possible strategy that could produce 100x.
#
# RULES:
#   1. BTC at new 20-day high → BUY the hottest alt
#   2. BTC NOT at 20-day high → DO NOTHING (cash)
#   3. Trail at 3% from high → captures pumps
#   4. SL at -1.5% → tiny losses when wrong
#   5. That's it. No ML, no ensemble, no indicators.
#
# WHY: every complex iteration lost money. The ONLY edge proven across
# 10+ backtests is the trailing stop. The simplest entry that ONLY
# fires during bull markets + trailing stop = the highest probability
# path to profit.
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import timedelta

COINS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD",
    "XDGUSD", "AVAXUSD", "LINKUSD", "DOTUSD", "LTCUSD",
    "NEARUSD", "SUIUSD", "TRXUSD", "BCHUSD", "TONUSD",
    "TAOUSD", "FETUSD", "HBARUSD", "RENDERUSD", "ICPUSD",
    "ALGOUSD", "STXUSD", "AAVEUSD", "UNIUSD", "PEPEUSD",
    "ONDOUSD", "KASUSD", "PENDLEUSD", "CRVUSD", "INJUSD",
    "JUPUSD", "TIAUSD", "EIGENUSD", "LDOUSD", "ARBUSD",
    "BONKUSD", "SHIBUSD", "APTUSD", "FLRUSD", "ATOMUSD",
    "OPUSD", "SEIUSD", "GRTUSD", "WIFUSD", "POLUSD",
    "GALAUSD", "ETCUSD", "STRKUSD", "WLDUSD", "QNTUSD",
]

# ── Parameters ───────────────────────────────────────────────────────────────
BTC_HIGH_LOOKBACK    = 20 * 288  # 20 days at 5-min bars = 5760 bars
ALT_MOMENTUM_BARS    = 48       # 4h at 5-min = which alt is hottest
SCAN_INTERVAL        = 5        # check every 5 min — more opportunities
SL_PCT               = 0.015    # -1.5% stop loss
TRAIL_ARM_PCT        = 0.015    # arm trail at +1.5%
TRAIL_PCT            = 0.025    # trail 2.5% from high water mark
EMERGENCY_SL         = 0.04     # -4% hard stop
ALLOC                = 0.85     # 85% per trade
COOLDOWN_MIN         = 15       # 15 min after any exit
MAX_DAILY_SL         = 4
WARMUP_DAYS          = 25


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 10, 1)
        self.set_end_date(2025, 3, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.02

        self._symbols = []
        for ticker in COINS:
            try:
                sym = self.add_crypto(ticker, Resolution.MINUTE, Market.KRAKEN).symbol
                self._symbols.append(sym)
            except Exception:
                pass
        self.log(f"[hydra] {len(self._symbols)} coins loaded")

        self._state = {}
        for sym in self._symbols:
            self._state[sym] = {"closes": deque(maxlen=6000), "volumes": deque(maxlen=6000)}

        self._btc_sym = None
        for sym in self._symbols:
            if "BTC" in sym.value:
                self._btc_sym = sym
                break

        for sym in self._symbols:
            c = TradeBarConsolidator(timedelta(minutes=5))
            c.data_consolidated += self._on_bar
            self.subscription_manager.add_consolidator(sym, c)

        # Position state
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._high_water = 0.0
        self._trail_active = False
        self._pending_sym = None
        self._pending_oid = None
        self._exiting = False
        self._exit_time = None
        self._exit_retry = 0
        self._daily_sl = 0

        # Tracking
        self._portfolio_high = 100.0

        self.set_warmup(timedelta(days=WARMUP_DAYS))
        self.schedule.on(self.date_rules.every_day(), self.time_rules.midnight, self._daily_reset)

    def _on_bar(self, sender, bar):
        sym = bar.symbol
        self._state[sym]["closes"].append(float(bar.close))
        self._state[sym]["volumes"].append(float(bar.volume))

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Drawdown circuit breaker
        pv = float(self.portfolio.total_portfolio_value)
        if pv > self._portfolio_high:
            self._portfolio_high = pv
        if self._portfolio_high > 0 and (self._portfolio_high - pv) / self._portfolio_high > 0.25:
            return

        # Check exit
        if self._pos_sym and not self._exiting:
            if self._pos_sym in data.bars:
                self._check_exit(float(data.bars[self._pos_sym].close))

        # Scan for entries
        if self.time.minute % SCAN_INTERVAL != 0:
            return
        if self._pos_sym or self._pending_sym or self._exiting:
            return
        if self._daily_sl >= MAX_DAILY_SL:
            return
        if self._exit_time and (self.time - self._exit_time).total_seconds() / 60 < COOLDOWN_MIN:
            return

        self._scan()

    def _is_btc_bullish(self):
        """THE core rule: is BTC at or near its 20-day high?"""
        if not self._btc_sym:
            return False
        bc = self._state[self._btc_sym]["closes"]
        if len(bc) < min(BTC_HIGH_LOOKBACK, 2000):
            return False
        c = list(bc)
        lookback = min(BTC_HIGH_LOOKBACK, len(c))
        high_20d = max(c[-lookback:])
        current = c[-1]
        # BTC within 2% of 20-day high = bullish
        return current >= high_20d * 0.98

    def _find_best_asymmetric_trade(self):
        """Find the alt where +1.5% up is more likely than -0.5% down.

        Key insight: don't predict direction. Predict ASYMMETRY.
        The best trade is where the upside/downside ratio is highest.
        """
        candidates = []

        for sym in self._symbols:
            if sym == self._btc_sym:
                continue
            closes = self._state[sym]["closes"]
            volumes = self._state[sym]["volumes"]
            if len(closes) < 100:
                continue

            c = np.asarray(list(closes), dtype=float)
            v = np.asarray(list(volumes), dtype=float)
            price = c[-1]
            if price <= 0:
                continue

            # ── Momentum (trend strength) ────────────────────────────────
            ret_1h = (c[-1] - c[-13]) / c[-13] if len(c) >= 13 else 0
            ret_4h = (c[-1] - c[-49]) / c[-49] if len(c) >= 49 else 0

            # Must have positive momentum
            if ret_4h <= 0.003:
                continue

            # ── Volume acceleration (institutional interest) ─────────────
            vol_avg = np.mean(v[-20:])
            vol_ratio = v[-1] / max(vol_avg, 1e-9)

            # ── Volatility compression → expansion (breakout quality) ────
            atr_recent = np.mean(np.abs(np.diff(c[-10:])))
            atr_older = np.mean(np.abs(np.diff(c[-30:-10]))) if len(c) >= 30 else atr_recent
            vol_expansion = atr_recent / max(atr_older, 1e-9)

            # ── RSI filter ───────────────────────────────────────────────
            diffs = np.diff(c[-15:])
            gains = np.mean(np.where(diffs > 0, diffs, 0))
            loss_avg = np.mean(np.where(diffs < 0, -diffs, 0))
            rsi = 100 - 100 / (1 + gains / max(loss_avg, 1e-9))
            if rsi > 78:
                continue

            # ── Trend alignment (EMA 8 > 21) ────────────────────────────
            if len(c) >= 22:
                ema8 = c[-8:].mean()
                ema21 = c[-21:].mean()
                trend_aligned = ema8 > ema21
            else:
                trend_aligned = ret_4h > 0

            if not trend_aligned:
                continue

            # ── ASYMMETRY SCORE: probability of +1.5% before -0.5% ──────
            # Higher momentum + volume + expansion + trend = more asymmetric
            score = (
                0.30 * min(ret_4h / 0.04, 1.0) +          # 4h momentum
                0.20 * min(max(ret_1h, 0) / 0.02, 1.0) +  # 1h momentum
                0.20 * min(max(vol_ratio - 1, 0) / 2, 1.0) + # volume
                0.15 * min(max(vol_expansion - 1, 0) / 1, 1.0) + # expansion
                0.15 * (1.0 if 40 < rsi < 70 else 0.5)     # sweet spot RSI
            )

            candidates.append((sym, score, ret_4h))

        if not candidates:
            return None, 0

        # Return the best asymmetric candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][2]

    def _scan(self):
        """The entire strategy in 3 lines of logic."""
        # RULE 1: Is BTC bullish?
        if not self._is_btc_bullish():
            return  # NO → do nothing. Cash is a position.

        # RULE 2: Find the best asymmetric trade
        sym, ret = self._find_best_asymmetric_trade()
        if sym is None:
            return

        # RULE 3: Buy it
        price = float(self.securities[sym].price)
        if price <= 0:
            return

        pv = float(self.portfolio.total_portfolio_value)
        # DYNAMIC SIZING: this is where 100x happens
        # Strong momentum → big position, weak → small
        if ret > 0.05:
            alloc = 0.90   # exceptional: 90%
        elif ret > 0.03:
            alloc = 0.70   # strong: 70%
        elif ret > 0.015:
            alloc = 0.50   # moderate: 50%
        else:
            alloc = 0.30   # weak: 30%
        qty = (pv * alloc * 0.99) / price
        lot = self.securities[sym].symbol_properties.lot_size
        min_order = self.securities[sym].symbol_properties.minimum_order_size
        qty = max(0, (qty // lot) * lot)
        if qty < min_order:
            return

        self.log(f"[hydra] ENTRY {sym.value} ret4h={ret:.2%} px={price:.4f} qty={qty:.6f}")

        self._pending_sym = sym
        self._high_water = price
        self._trail_active = False
        order = self.market_order(sym, qty, tag="ENTRY")
        self._pending_oid = order.order_id

    def _check_exit(self, price):
        """Trail / SL / timeout."""
        if not self._entry_px or self._entry_px <= 0:
            return

        ret = (price - self._entry_px) / self._entry_px
        elapsed_h = (self.time - self._entry_time).total_seconds() / 3600

        # Update high water
        if price > self._high_water:
            self._high_water = price
        max_ret = (self._high_water - self._entry_px) / self._entry_px

        # Min hold: 10 min (only emergency SL)
        if elapsed_h < 10/60:
            if ret <= -EMERGENCY_SL:
                self._do_exit("EXIT_EMERGENCY_SL")
            return

        # SL
        if ret <= -SL_PCT:
            self._do_exit("EXIT_SL")
            return

        # Arm trail
        if not self._trail_active and max_ret >= TRAIL_ARM_PCT:
            self._trail_active = True

        # Trail stop
        if self._trail_active:
            trail_stop = self._high_water * (1 - TRAIL_PCT)
            if price <= trail_stop:
                self._do_exit("EXIT_TRAIL")
                return

        # Timeout: 48h
        if elapsed_h >= 48:
            self._do_exit("EXIT_TIMEOUT")

    def _do_exit(self, tag):
        if self._exiting:
            return
        sym = self._pos_sym
        if not sym:
            return

        # Safe sell qty
        qty = float(self.portfolio[sym].quantity)
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
        self.log(f"[hydra] {tag} {sym.value} ret={ret:+.2%} hw={self._high_water:.4f}")
        self._exiting = True
        self.market_order(sym, -qty, tag=tag)

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.INVALID:
            self._exit_retry += 1
            if self._exit_retry >= 3:
                if self._pos_sym:
                    self._daily_sl += 1
                self._exit_time = self.time
                self._clear()
            else:
                self._exiting = False
            return

        if order_event.status != OrderStatus.FILLED:
            return

        sym = order_event.symbol
        tag = ""
        try:
            tag = self.transactions.get_order_by_id(order_event.order_id).tag
        except Exception:
            pass

        if tag == "ENTRY":
            self._pos_sym = sym
            self._entry_px = float(order_event.fill_price)
            self._entry_time = self.time
            self._high_water = self._entry_px
            self._trail_active = False
            self._pending_sym = None
            self._exit_retry = 0
        elif tag.startswith("EXIT"):
            if "SL" in tag:
                self._daily_sl += 1
            self._exit_time = self.time
            self._clear()

    def _clear(self):
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._high_water = 0.0
        self._trail_active = False
        self._pending_sym = None
        self._exiting = False
        self._exit_retry = 0

    def _daily_reset(self):
        self._daily_sl = 0
