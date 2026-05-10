# ══════════════════════════════════════════════════════════════════════════════
# HYDRA v6 — "The Survivor"
# Built to work LIVE, not just backtest.
#
# Machine Gun did 75x in backtest then DIED live. Lesson learned.
# This algo is designed for LIVE survival:
#   - Hourly bars (not 1-min) — immune to execution/microstructure issues
#   - Wide stops — survives slippage, spread, gaps
#   - Trend following — doesn't need order book accuracy
#   - Regime-aware — sits in cash during bear markets
#   - Multi-strategy — dip buy + momentum + mean reversion rotation
#   - Position trading — holds hours to days, not seconds
#
# Target: 100x at 1x spot over 6-12 months in a bull market
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import timedelta

COINS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD",
    "XDGUSD", "AVAXUSD", "LINKUSD", "DOTUSD", "LTCUSD",
    "NEARUSD", "SUIUSD", "RENDERUSD", "PEPEUSD", "ONDOUSD",
]


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2026, 12, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.01

        self._symbols = []
        for ticker in COINS:
            try:
                sym = self.add_crypto(ticker, Resolution.HOUR, Market.KRAKEN).symbol
                self._symbols.append(sym)
            except Exception:
                pass
        self.log(f"[hydra] {len(self._symbols)} coins on HOURLY bars")

        self._btc = None
        for s in self._symbols:
            if "BTC" in s.value:
                self._btc = s
                break

        # State
        self._closes = {s: deque(maxlen=500) for s in self._symbols}
        self._volumes = {s: deque(maxlen=500) for s in self._symbols}
        self._highs = {s: deque(maxlen=500) for s in self._symbols}

        # Positions: up to 3 concurrent
        self._positions = {}  # sym -> {entry_px, entry_time, high_water, trail_active}
        self.MAX_POSITIONS = 3

        # Risk
        self._daily_sl = 0
        self._last_exit = {}  # sym -> time
        self._portfolio_high = 100.0

        self.set_warmup(timedelta(days=30))
        self.schedule.on(self.date_rules.every_day(),
                         self.time_rules.midnight, self._daily_reset)

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Update state from hourly bars
        for sym in self._symbols:
            if sym in data.bars:
                bar = data.bars[sym]
                self._closes[sym].append(float(bar.close))
                self._volumes[sym].append(float(bar.volume))
                self._highs[sym].append(float(bar.high))

        # Drawdown circuit breaker
        pv = float(self.portfolio.total_portfolio_value)
        if pv > self._portfolio_high:
            self._portfolio_high = pv
        if self._portfolio_high > 0 and (self._portfolio_high - pv) / self._portfolio_high > 0.20:
            return

        # FAST REGIME KILL SWITCH: exit ALL when BTC turns weak
        regime = self._get_regime()
        if regime == "bear" and self._positions:
            for sym in list(self._positions.keys()):
                pos = self._positions[sym]
                ret = (float(self.securities[sym].price) - pos["entry_px"]) / pos["entry_px"] if pos["entry_px"] > 0 else 0
                if ret < 0.01:  # exit anything not solidly profitable
                    self._do_exit(sym, "EXIT_REGIME_KILL")
            return  # don't scan for entries in bear

        # BTC momentum deterioration: exit if BTC drops 2%+ in 24h
        if self._btc and len(self._closes[self._btc]) >= 25:
            btc_c = list(self._closes[self._btc])
            btc_ret24h = (btc_c[-1] - btc_c[-25]) / btc_c[-25]
            if btc_ret24h < -0.03 and self._positions:
                for sym in list(self._positions.keys()):
                    self._do_exit(sym, "EXIT_BTC_WEAK")
                return

        # Check exits on all positions
        for sym in list(self._positions.keys()):
            if sym in data.bars:
                self._check_exit(sym, float(data.bars[sym].close))

        if self.time.hour % 4 != 0 or self.time.minute != 0:
            return
        if self._daily_sl >= 4:
            return

        self._scan_entries()

    # ══════════════════════════════════════════════════════════════════════
    # REGIME DETECTION
    # ══════════════════════════════════════════════════════════════════════

    def _get_regime(self):
        """Detect market regime from BTC.

        Returns: "bull", "strong_bull", "bear", "chop"
        """
        if not self._btc or len(self._closes[self._btc]) < 200:
            return "chop"

        c = np.array(list(self._closes[self._btc]))
        price = c[-1]
        sma50 = np.mean(c[-50:])
        sma200 = np.mean(c[-200:]) if len(c) >= 200 else np.mean(c)
        ret_24h = (c[-1] - c[-25]) / c[-25] if len(c) >= 25 else 0
        ret_7d = (c[-1] - c[-168]) / c[-168] if len(c) >= 168 else 0

        if price > sma50 and sma50 > sma200 and ret_7d > 0.05:
            return "strong_bull"
        elif price > sma50 and sma50 > sma200:
            return "bull"
        elif price < sma50 and price < sma200:
            return "bear"
        else:
            return "chop"

    # ══════════════════════════════════════════════════════════════════════
    # SIGNAL DETECTION (3 strategies, regime-dependent)
    # ══════════════════════════════════════════════════════════════════════

    def _scan_entries(self):
        regime = self._get_regime()

        # BEAR: no new entries
        if regime == "bear":
            return

        open_count = len(self._positions)
        if open_count >= self.MAX_POSITIONS:
            return

        candidates = []

        for sym in self._symbols:
            if sym == self._btc:
                continue
            if sym in self._positions:
                continue
            if sym in self._last_exit:
                if (self.time - self._last_exit[sym]).total_seconds() < 7200:
                    continue

            c = self._closes[sym]
            v = self._volumes[sym]
            if len(c) < 100:
                continue

            ca = np.array(list(c))
            va = np.array(list(v))
            price = ca[-1]
            if price <= 0:
                continue

            result = self._score_coin(ca, va, regime)
            if result and result[0] > 0:
                candidates.append((sym, result[0], result[1], result[2]))

        if not candidates:
            return

        # Sort by score, take best available
        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = self.MAX_POSITIONS - open_count

        for sym, score, reason, alloc in candidates[:slots]:
            self._enter(sym, score, reason, alloc)

    def _score_coin(self, c, v, regime):
        """Score a coin across 3 strategies. Returns (score, reason, alloc) or None."""
        price = c[-1]

        # Common indicators
        sma20 = np.mean(c[-20:])
        sma50 = np.mean(c[-50:]) if len(c) >= 50 else sma20
        ret_4h = (c[-1] - c[-5]) / c[-5] if len(c) >= 5 else 0
        ret_24h = (c[-1] - c[-25]) / c[-25] if len(c) >= 25 else 0
        ret_7d = (c[-1] - c[-168]) / c[-168] if len(c) >= 168 else 0
        vol_avg = np.mean(v[-20:]) if len(v) >= 20 else 1
        vol_ratio = v[-1] / max(vol_avg, 1e-9)

        # RSI
        diffs = np.diff(c[-15:])
        gains = np.mean(np.where(diffs > 0, diffs, 0))
        loss_avg = np.mean(np.where(diffs < 0, -diffs, 0))
        rsi = 100 - 100 / (1 + gains / max(loss_avg, 1e-9))

        # ADX proxy (trend strength via directional movement)
        if len(c) >= 15:
            ups = np.diff(np.array(list(self._highs.get(list(self._highs.keys())[0], c)))[-15:])
            downs = -np.diff(c[-15:])
            dm_plus = np.mean(np.where(ups > downs, ups, 0))
            dm_minus = np.mean(np.where(downs > ups, downs, 0))
            dx = abs(dm_plus - dm_minus) / max(dm_plus + dm_minus, 1e-9) * 100
        else:
            dx = 0

        # EMA alignment
        ema8 = np.mean(c[-8:])
        ema21 = np.mean(c[-21:]) if len(c) >= 21 else ema8
        trend_aligned = ema8 > ema21

        # VWAP
        if len(c) >= 20 and len(v) >= 20:
            vwap = np.sum(c[-20:] * v[-20:]) / max(np.sum(v[-20:]), 1e-9)
            above_vwap = price > vwap
        else:
            above_vwap = True

        best_score = 0
        best_reason = ""
        best_alloc = 0

        # ── STRATEGY 1: DIP BUY (bull + strong_bull) ────────────────────
        if regime in ("bull", "strong_bull") and price > sma50:
            dip_from_high = (max(c[-20:]) - price) / max(c[-20:])
            if 0.02 <= dip_from_high <= 0.10 and rsi < 45 and trend_aligned:
                score = (
                    0.30 * min(dip_from_high / 0.06, 1) +
                    0.25 * max(0, (45 - rsi) / 25) +
                    0.20 * (1 if above_vwap else 0.3) +
                    0.15 * min(max(ret_7d, 0) / 0.10, 1) +
                    0.10 * (1 if regime == "strong_bull" else 0.5)
                )
                if score > best_score:
                    best_score = score
                    best_reason = "dip_buy"
                    best_alloc = 0.55 if regime == "strong_bull" else 0.40

        # ── STRATEGY 2: MOMENTUM CONTINUATION (strong_bull) ─────────────
        if regime == "strong_bull":
            if ret_24h > 0.03 and vol_ratio > 1.5 and trend_aligned and rsi < 75:
                score = (
                    0.30 * min(ret_24h / 0.08, 1) +
                    0.25 * min((vol_ratio - 1) / 2, 1) +
                    0.20 * min(dx / 30, 1) +
                    0.15 * (1 if rsi > 50 and rsi < 70 else 0.5) +
                    0.10 * min(max(ret_7d, 0) / 0.15, 1)
                )
                if score > best_score:
                    best_score = score
                    best_reason = "momentum"
                    best_alloc = 0.50

        # ── STRATEGY 3: MEAN REVERSION (bull + chop) ────────────────────
        if regime in ("bull", "chop") and price > sma50:
            bb_mean = np.mean(c[-20:])
            bb_std = np.std(c[-20:])
            bb_lower = bb_mean - 2 * bb_std
            near_bb_lower = price < bb_lower * 1.01
            if near_bb_lower and rsi < 35 and vol_ratio < 2:
                score = (
                    0.35 * max(0, (35 - rsi) / 20) +
                    0.30 * min((bb_mean - price) / max(bb_std, 1e-9) / 2, 1) +
                    0.20 * (1 if trend_aligned else 0.3) +
                    0.15 * (1 if above_vwap else 0.5)
                )
                if score > best_score:
                    best_score = score
                    best_reason = "mean_revert"
                    best_alloc = 0.35

        if best_score < 0.55:
            return None

        return (best_score, best_reason, best_alloc)

    # ══════════════════════════════════════════════════════════════════════
    # ENTRY
    # ══════════════════════════════════════════════════════════════════════

    def _enter(self, sym, score, reason, alloc):
        price = float(self.securities[sym].price)
        if price <= 0:
            return

        pv = float(self.portfolio.total_portfolio_value)
        qty = (pv * alloc * 0.99) / price

        lot = self.securities[sym].symbol_properties.lot_size
        min_order = self.securities[sym].symbol_properties.minimum_order_size
        qty = max(0, (qty // lot) * lot)
        if qty < min_order:
            return

        self._positions[sym] = {
            "entry_px": price,
            "entry_time": self.time,
            "high_water": price,
            "trail_active": False,
            "reason": reason,
            "alloc": alloc,
        }

        self.log(f"[hydra] ENTRY {sym.value} reason={reason} score={score:.2f}"
                 f" alloc={alloc:.0%} px={price:.4f} regime={self._get_regime()}"
                 f" positions={len(self._positions)}/{self.MAX_POSITIONS}")

        self.market_order(sym, qty, tag=f"ENTRY_{reason}")

    # ══════════════════════════════════════════════════════════════════════
    # ADAPTIVE EXIT SYSTEM
    # 4 exit types: SL, trail, momentum fail, time decay
    # ══════════════════════════════════════════════════════════════════════

    def _check_exit(self, sym, price):
        pos = self._positions.get(sym)
        if not pos:
            return

        entry_px = pos["entry_px"]
        if entry_px <= 0:
            return

        ret = (price - entry_px) / entry_px
        elapsed_h = (self.time - pos["entry_time"]).total_seconds() / 3600

        # Update high water
        if price > pos["high_water"]:
            pos["high_water"] = price
        max_ret = (pos["high_water"] - entry_px) / entry_px

        # ── EXIT 1: Stop Loss (adaptive by reason) ──────────────────────
        sl = 0.04 if pos["reason"] == "momentum" else 0.035
        if ret <= -sl:
            self._do_exit(sym, "EXIT_SL")
            return

        # ── EXIT 2: Trailing Stop ────────────────────────────────────────
        trail_arm = 0.03 if pos["reason"] == "dip_buy" else 0.05
        trail_pct = 0.03  # 3% trail — wide enough for hourly bars

        if not pos["trail_active"] and max_ret >= trail_arm:
            pos["trail_active"] = True

        if pos["trail_active"]:
            trail_stop = pos["high_water"] * (1 - trail_pct)
            if price <= trail_stop:
                self._do_exit(sym, "EXIT_TRAIL")
                return

        # ── EXIT 3: Momentum Failure ─────────────────────────────────────
        if elapsed_h >= 6 and ret < -0.01:
            c = self._closes.get(sym)
            if c and len(c) >= 5:
                ret_4h = (list(c)[-1] - list(c)[-5]) / list(c)[-5]
                if ret_4h < -0.02:
                    self._do_exit(sym, "EXIT_MOM_FAIL")
                    return

        # ── EXIT 4: Time Decay ───────────────────────────────────────────
        if elapsed_h >= 48:
            self._do_exit(sym, "EXIT_TIMEOUT")
            return
        if elapsed_h >= 24 and ret < 0.005:
            self._do_exit(sym, "EXIT_TIMEOUT")
            return

        # ── EXIT 5: Regime Change ────────────────────────────────────────
        if self._get_regime() == "bear" and ret < 0:
            self._do_exit(sym, "EXIT_REGIME")

    def _do_exit(self, sym, tag):
        qty = float(self.portfolio[sym].quantity)
        if qty <= 0:
            self._positions.pop(sym, None)
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
            self._positions.pop(sym, None)
            return

        pos = self._positions.get(sym, {})
        ret = (float(self.securities[sym].price) - pos.get("entry_px", 0)) / pos.get("entry_px", 1) if pos.get("entry_px", 0) > 0 else 0

        self.log(f"[hydra] {tag} {sym.value} ret={ret:+.1%}"
                 f" reason={pos.get('reason','?')} held={((self.time-pos.get('entry_time',self.time)).total_seconds()/3600):.1f}h"
                 f" pv=${float(self.portfolio.total_portfolio_value):.2f}")

        self.market_order(sym, -qty, tag=tag)
        self._positions.pop(sym, None)
        self._last_exit[sym] = self.time
        if "SL" in tag:
            self._daily_sl += 1

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.INVALID:
            sym = order_event.symbol
            self._positions.pop(sym, None)
        elif order_event.status == OrderStatus.FILLED:
            tag = ""
            try:
                tag = self.transactions.get_order_by_id(order_event.order_id).tag
            except Exception:
                pass
            if tag.startswith("ENTRY"):
                sym = order_event.symbol
                if sym in self._positions:
                    self._positions[sym]["entry_px"] = float(order_event.fill_price)
                    self._positions[sym]["high_water"] = float(order_event.fill_price)

    def _daily_reset(self):
        self._daily_sl = 0
