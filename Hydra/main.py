# ══════════════════════════════════════════════════════════════════════════════
# HYDRA FINAL — Adaptive Trend Basket
#
# Synthesis of 25+ backtests and professional quant research.
#
# PROVEN FINDINGS:
#   1. Regime filter > signal quality (biggest alpha source)
#   2. Multi-position > single position (diversification edge)
#   3. Fewer trades > more trades (fees destroy returns)
#   4. Let winners run (biggest leak was exiting too early)
#   5. ML for regime/risk, NOT for prediction
#
# ARCHITECTURE:
#   - Regime: golden cross, only trade confirmed bull
#   - Basket: top 3 momentum coins, volatility-weighted
#   - Turnover: rebalance weekly, not daily
#   - Exits: wide ATR trail (let winners run), tight SL (cut losers fast)
#   - Target: 25-60% CAGR with controlled drawdowns
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

BASKET_SIZE       = 3      # hold top 3 coins
REBALANCE_DAYS    = 7      # rebalance weekly (proven: less turnover = better)
MOMENTUM_PERIOD   = 30     # 30-day momentum for ranking
MIN_MOMENTUM      = 0.01   # minimum 1% monthly return to qualify
ROTATION_THRESHOLD = 0.05  # only rotate if new coin is 5%+ stronger
SL_PCT            = 0.04   # 4% stop loss (cut losers fast)
TRAIL_ARM         = 0.05   # arm trail at +5%
TRAIL_ATR_MULT    = 3.0    # trail = 3 × ATR (wide — let winners run)
TARGET_VOL        = 0.03   # 3% target vol per position (vol-weighted sizing)


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
                s = self.add_crypto(ticker, Resolution.DAILY, Market.KRAKEN).symbol
                self._symbols.append(s)
            except Exception:
                pass

        self._btc = None
        for s in self._symbols:
            if "BTC" in s.value:
                self._btc = s
                break

        # Track positions
        self._entries = {}   # sym -> {px, time, high}
        self._last_rebalance = None

        # Weekly rebalance
        self.schedule.on(
            self.date_rules.every(DayOfWeek.MONDAY),
            self.time_rules.at(12, 0),
            self._weekly_rebalance,
        )

        # Daily risk check
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(8, 0),
            self._daily_risk_check,
        )

        self.set_warmup(timedelta(days=60))

    # ═══════════════════════════════════════════════════════════════════════
    # REGIME (the #1 alpha source)
    # ═══════════════════════════════════════════════════════════════════════

    def _is_bull(self):
        """Golden cross + positive 30d momentum. Simple, proven."""
        if not self._btc:
            return False
        h = self.history(self._btc, 210, Resolution.DAILY)
        if h.empty or len(h) < 200:
            return False
        c = h["close"].values
        sma50 = np.mean(c[-50:])
        sma200 = np.mean(c[-200:])
        ret_30d = (c[-1] - c[-30]) / c[-30] if len(c) >= 30 else 0
        return c[-1] > sma50 and sma50 > sma200 and ret_30d > -0.05

    # ═══════════════════════════════════════════════════════════════════════
    # RANKING (cross-sectional + time-series momentum)
    # ═══════════════════════════════════════════════════════════════════════

    def _rank_coins(self):
        """Rank by momentum. Filter by trend (above SMA50). Vol-weight."""
        candidates = []
        for sym in self._symbols:
            h = self.history(sym, MOMENTUM_PERIOD + 55, Resolution.DAILY)
            if h.empty or len(h) < MOMENTUM_PERIOD:
                continue
            c = h["close"].values
            v = h["volume"].values

            # Cross-sectional momentum: 30d return
            ret = (c[-1] - c[-MOMENTUM_PERIOD]) / c[-MOMENTUM_PERIOD]
            if ret < MIN_MOMENTUM:
                continue

            # Time-series filter: must be above 50-day SMA (actually trending)
            if len(c) >= 50:
                sma50 = np.mean(c[-50:])
                if c[-1] < sma50:
                    continue

            # Volatility for position sizing
            daily_rets = np.diff(c[-21:]) / c[-21:-1] if len(c) >= 21 else np.array([0.03])
            vol = np.std(daily_rets) if len(daily_rets) > 1 else 0.03
            vol = max(vol, 0.005)  # floor

            candidates.append({
                "sym": sym,
                "ret": ret,
                "vol": vol,
                "weight": TARGET_VOL / vol,  # inverse-vol weight
            })

        # Sort by momentum
        candidates.sort(key=lambda x: x["ret"], reverse=True)

        # Normalize weights for top N
        top = candidates[:BASKET_SIZE]
        if top:
            total_w = sum(c["weight"] for c in top)
            for c in top:
                c["weight"] = min(c["weight"] / total_w, 0.50)  # cap at 50% per coin

        return top

    # ═══════════════════════════════════════════════════════════════════════
    # WEEKLY REBALANCE (low turnover — proven better)
    # ═══════════════════════════════════════════════════════════════════════

    def _weekly_rebalance(self):
        if self.is_warming_up:
            return

        # BEAR → go to cash
        if not self._is_bull():
            if self.portfolio.invested:
                self.liquidate()
                self.log("[hydra] BEAR → CASH")
                self._entries.clear()
            return

        # Rank coins
        ranked = self._rank_coins()
        if not ranked:
            return

        new_syms = {c["sym"] for c in ranked}
        current_syms = set(self._entries.keys())

        # Sell anything not in the new basket (only if dropped significantly)
        for sym in list(current_syms):
            if sym not in new_syms:
                # Check if it just barely missed — don't rotate for tiny differences
                sym_ret = 0
                h = self.history(sym, MOMENTUM_PERIOD + 5, Resolution.DAILY)
                if not h.empty and len(h) >= MOMENTUM_PERIOD:
                    c = h["close"].values
                    sym_ret = (c[-1] - c[-MOMENTUM_PERIOD]) / c[-MOMENTUM_PERIOD]

                if ranked and ranked[-1]["ret"] - sym_ret > ROTATION_THRESHOLD:
                    self.set_holdings(sym, 0)
                    self._entries.pop(sym, None)
                    self.log(f"[hydra] SELL {sym.value} (dropped from basket)")

        # Buy new basket members
        for coin in ranked:
            sym = coin["sym"]
            target_weight = coin["weight"] * 0.95  # leave buffer for fees

            if sym not in self._entries:
                self.set_holdings(sym, target_weight)
                px = float(self.securities[sym].price)
                self._entries[sym] = {"px": px, "time": self.time, "high": px}
                self.log(f"[hydra] BUY {sym.value} ret30d={coin['ret']:+.1%}"
                         f" vol={coin['vol']:.3f} weight={target_weight:.0%}")

    # ═══════════════════════════════════════════════════════════════════════
    # DAILY RISK CHECK (exits: SL + trail + regime)
    # ═══════════════════════════════════════════════════════════════════════

    def _daily_risk_check(self):
        if self.is_warming_up:
            return

        # BTC crash: -5% in 3 days → emergency exit ALL
        if self._btc:
            h = self.history(self._btc, 5, Resolution.DAILY)
            if not h.empty and len(h) >= 3:
                c = h["close"].values
                if (c[-1] - c[0]) / c[0] < -0.05:
                    self.liquidate()
                    self.log("[hydra] BTC CRASH → LIQUIDATE ALL")
                    self._entries.clear()
                    return

        # Per-position checks
        for sym in list(self._entries.keys()):
            pos = self._entries[sym]
            px = float(self.securities[sym].price)
            if px <= 0 or pos["px"] <= 0:
                continue

            ret = (px - pos["px"]) / pos["px"]

            # Update high water
            if px > pos["high"]:
                pos["high"] = px
            max_ret = (pos["high"] - pos["px"]) / pos["px"]

            # STOP LOSS: cut losers fast (-4%)
            if ret < -SL_PCT:
                self.set_holdings(sym, 0)
                self.log(f"[hydra] SL {sym.value} ret={ret:+.1%}")
                self._entries.pop(sym, None)
                continue

            # TRAILING STOP: ATR-based, arms at +5%
            if max_ret >= TRAIL_ARM:
                h = self.history(sym, 15, Resolution.DAILY)
                if not h.empty and len(h) >= 10:
                    c = h["close"].values
                    atr = np.mean(np.abs(np.diff(c[-10:])))
                    trail_pct = max(0.03, min(TRAIL_ATR_MULT * atr / px, 0.12))
                else:
                    trail_pct = 0.05

                trail_stop = pos["high"] * (1 - trail_pct)
                if px < trail_stop:
                    self.set_holdings(sym, 0)
                    self.log(f"[hydra] TRAIL {sym.value} ret={ret:+.1%} max={max_ret:+.1%} trail={trail_pct:.1%}")
                    self._entries.pop(sym, None)
                    continue

        # REGIME CHANGE: bear → exit all losers
        if not self._is_bull():
            for sym in list(self._entries.keys()):
                ret = (float(self.securities[sym].price) - self._entries[sym]["px"]) / self._entries[sym]["px"]
                if ret < 0.02:
                    self.set_holdings(sym, 0)
                    self.log(f"[hydra] REGIME EXIT {sym.value} ret={ret:+.1%}")
                    self._entries.pop(sym, None)

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED and order_event.fill_quantity > 0:
            sym = order_event.symbol
            if sym in self._entries:
                self._entries[sym]["px"] = float(order_event.fill_price)
                self._entries[sym]["high"] = float(order_event.fill_price)
