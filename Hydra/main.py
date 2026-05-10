# ══════════════════════════════════════════════════════════════════════════════
# HYDRA — Multi-Signal Pump Hunter
# $100 → $10,000 through asymmetric momentum capture
#
# Architecture: REACT, don't predict.
#   4 signal detectors scan 50 coins every 5 min
#   Tiny SL (1.5%) + trailing stop (1.5% adaptive) = asymmetric risk
#   Pyramiding into winners compounds aggressively
#   Even at 35% WR: profitable due to 3:1+ win/loss ratio
# ══════════════════════════════════════════════════════════════════════════════

from AlgorithmImports import *
import numpy as np
from collections import deque
from datetime import timedelta
from signals import detect_all_signals
from trail_engine import TrailEngine

# ── Configuration ────────────────────────────────────────────────────────────
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

# Entry
SCAN_INTERVAL_MIN     = 5     # scan every 5 min
PUMP_RET_1H_MIN       = 0.02  # +2% in 1h = pump starting
PUMP_VOL_MULT         = 2.0   # 2x volume = real pump
BREAKOUT_LOOKBACK     = 20    # 20-bar high for breakout
BREAKOUT_VOL_MULT     = 1.5   # 1.5x volume for breakout
SQUEEZE_LOOKBACK      = 20    # Bollinger period
SQUEEZE_THRESHOLD     = 0.3   # bandwidth percentile for squeeze
VOL_ANOMALY_MULT      = 3.0   # 3x volume = institutional

# Execution — ASYMMETRIC RISK
SL_PCT                = 0.015  # -1.5% stop loss (TINY)
BREAKEVEN_AT          = 0.01   # move stop to entry at +1%
TRAIL_ARM_PCT         = 0.02   # arm trailing at +2%
TRAIL_PCT             = 0.015  # trail 1.5% from high (adaptive via ATR)
EMERGENCY_SL          = 0.04   # -4% hard stop
MIN_HOLD_MIN          = 15     # hold at least 15 min
TIMEOUT_HOURS         = 36     # max hold 36h

# Sizing
INITIAL_ALLOC         = 0.50   # 50% initial entry
PYRAMID_1_AT          = 0.015  # add at +1.5%
PYRAMID_1_SIZE        = 0.25   # add 25%
PYRAMID_2_AT          = 0.03   # add at +3%
PYRAMID_2_SIZE        = 0.20   # add 20%

# Risk
MAX_DAILY_SL          = 5      # pause after 5 SL per day
COOLDOWN_MIN          = 10     # 10 min after any exit
SL_COOLDOWN_MIN       = 30     # 30 min per-coin after SL
MAX_DD_PCT            = 0.25   # 25% drawdown circuit breaker

# BTC filter
BTC_CRASH_THRESHOLD   = -0.02  # skip entries if BTC ret_4h < -2%

WARMUP_DAYS           = 30


class HydraAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 10, 1)
        self.set_end_date(2025, 3, 31)
        self.set_cash(100)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.free_portfolio_value_percentage = 0.02

        # Add all coins
        self._symbols = []
        for ticker in COINS:
            try:
                sym = self.add_crypto(ticker, Resolution.MINUTE, Market.KRAKEN).symbol
                self._symbols.append(sym)
            except Exception:
                pass
        self.log(f"[hydra] {len(self._symbols)} coins loaded")

        # State: per-symbol OHLCV deques
        self._state = {}
        for sym in self._symbols:
            self._state[sym] = {
                "closes": deque(maxlen=500),
                "highs": deque(maxlen=500),
                "lows": deque(maxlen=500),
                "volumes": deque(maxlen=500),
            }

        # BTC reference
        self._btc_sym = None
        for sym in self._symbols:
            if "BTC" in sym.value:
                self._btc_sym = sym
                break

        # Consolidators: 5-min bars
        for sym in self._symbols:
            consolidator = TradeBarConsolidator(timedelta(minutes=5))
            consolidator.data_consolidated += self._on_5m
            self.subscription_manager.add_consolidator(sym, consolidator)

        # Position state
        self._trail = TrailEngine(
            sl_pct=SL_PCT, breakeven_at=BREAKEVEN_AT,
            trail_arm_pct=TRAIL_ARM_PCT, trail_pct=TRAIL_PCT,
            emergency_sl=EMERGENCY_SL,
        )
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._pending_sym = None
        self._pending_oid = None
        self._exiting = False
        self._pyramid_level = 0  # 0=initial, 1=first add, 2=second add
        self._exit_time = None

        # Risk tracking
        self._daily_sl_count = 0
        self._last_sl_time = {}   # sym -> datetime
        self._high_water = 100.0
        self._dd_halt = False

        # Warmup
        self.set_warmup(timedelta(days=WARMUP_DAYS))

        # Chronos forecast cache
        self._forecast_cache = {}
        self._forecast_last = None

        # Schedule daily reset
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.midnight,
            self._daily_reset,
        )

    def _on_5m(self, sender, bar):
        sym = bar.symbol
        st = self._state[sym]
        st["closes"].append(float(bar.close))
        st["highs"].append(float(bar.high))
        st["lows"].append(float(bar.low))
        st["volumes"].append(float(bar.volume))

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Drawdown circuit breaker
        pv = float(self.portfolio.total_portfolio_value)
        if pv > self._high_water:
            self._high_water = pv
        dd = (self._high_water - pv) / self._high_water if self._high_water > 0 else 0
        if dd > MAX_DD_PCT:
            if not self._dd_halt:
                self.log(f"[hydra] DRAWDOWN HALT: {dd:.1%} > {MAX_DD_PCT:.0%}")
                self._dd_halt = True
            return
        self._dd_halt = False

        # Check exit on current position
        if self._pos_sym and not self._exiting:
            if self._pos_sym in data.bars:
                self._check_exit(data.bars[self._pos_sym])

        # Check pyramiding
        if self._pos_sym and not self._exiting and self._pyramid_level < 2:
            self._check_pyramid()

        # Scan for entries every SCAN_INTERVAL_MIN
        if self.time.minute % SCAN_INTERVAL_MIN != 0:
            return
        if self._pos_sym or self._pending_sym or self._exiting:
            return
        if self._daily_sl_count >= MAX_DAILY_SL:
            return

        # Cooldown check
        if self._exit_time:
            elapsed = (self.time - self._exit_time).total_seconds() / 60
            if elapsed < COOLDOWN_MIN:
                return

        # Update Chronos forecasts periodically
        self._update_forecasts()

        self._scan_and_enter()

    def _scan_and_enter(self):
        """Scan all coins for momentum signals and enter the strongest."""
        # BTC crash filter
        btc_ret4 = 0.0
        if self._btc_sym:
            bc = self._state[self._btc_sym]["closes"]
            if len(bc) >= 49:  # 4h at 5-min = 48 bars
                btc_ret4 = (bc[-1] - bc[-49]) / bc[-49]
        if btc_ret4 < BTC_CRASH_THRESHOLD:
            return

        signals = []
        for sym in self._symbols:
            st = self._state[sym]
            if len(st["closes"]) < 50:
                continue

            # SL cooldown per coin
            if sym in self._last_sl_time:
                elapsed = (self.time - self._last_sl_time[sym]).total_seconds() / 60
                if elapsed < SL_COOLDOWN_MIN:
                    continue

            closes = list(st["closes"])
            highs = list(st["highs"])
            lows = list(st["lows"])
            volumes = list(st["volumes"])

            # Get Chronos forecast
            chronos_ret = self._forecast_cache.get(sym, 0.0)

            result = detect_all_signals(
                closes=closes, highs=highs, lows=lows, volumes=volumes,
                chronos_forecast=chronos_ret,
                pump_ret_min=PUMP_RET_1H_MIN, pump_vol_mult=PUMP_VOL_MULT,
                breakout_lookback=BREAKOUT_LOOKBACK, breakout_vol_mult=BREAKOUT_VOL_MULT,
                squeeze_lookback=SQUEEZE_LOOKBACK, squeeze_threshold=SQUEEZE_THRESHOLD,
                vol_anomaly_mult=VOL_ANOMALY_MULT,
            )
            if result["enter"]:
                signals.append((sym, result["strength"], result["reason"], result))

        if not signals:
            return

        # Pick strongest signal
        signals.sort(key=lambda x: x[1], reverse=True)
        sym, strength, reason, details = signals[0]
        price = float(self.securities[sym].price)
        if price <= 0:
            return

        # Size: initial allocation
        pv = float(self.portfolio.total_portfolio_value)
        alloc = INITIAL_ALLOC
        cash = pv * alloc * 0.99
        qty = cash / price
        if qty <= 0:
            return

        # Validate lot size
        lot = self.securities[sym].symbol_properties.lot_size
        min_order = self.securities[sym].symbol_properties.minimum_order_size
        qty = max(0, (qty // lot) * lot)
        if qty < min_order:
            return

        self.log(
            f"[hydra] ENTRY {sym.value} reason={reason} strength={strength:.3f}"
            f" px={price:.4f} qty={qty:.6f} alloc={alloc:.0%}"
            f" chronos={self._forecast_cache.get(sym, 0):.4f} btc4h={btc_ret4:.3f}"
        )

        # Enter
        self._pending_sym = sym
        self._trail.reset(price)
        self._pyramid_level = 0
        order = self.market_order(sym, qty, tag="ENTRY")
        self._pending_oid = order.order_id

    def _check_exit(self, bar):
        """Check trail/SL/timeout exits."""
        price = float(bar.close)
        elapsed_min = (self.time - self._entry_time).total_seconds() / 60

        # Min hold
        if elapsed_min < MIN_HOLD_MIN:
            # Only emergency SL during min hold
            ret = (price - self._entry_px) / self._entry_px
            if ret <= -EMERGENCY_SL:
                self._exit("EXIT_EMERGENCY_SL", price)
            return

        action = self._trail.check(price, elapsed_min / 60.0, TIMEOUT_HOURS)
        if action:
            self._exit(action, price)

    def _exit(self, tag, price):
        """Submit exit order."""
        if self._exiting:
            return
        qty = float(self.portfolio[self._pos_sym].quantity)
        if qty <= 0:
            self._clear_state()
            return

        # Floor to lot size
        lot = self.securities[self._pos_sym].symbol_properties.lot_size
        qty = (qty // lot) * lot
        if qty <= 0:
            self._clear_state()
            return

        ret = (price - self._entry_px) / self._entry_px if self._entry_px > 0 else 0
        self.log(
            f"[hydra] {tag} {self._pos_sym.value} ret={ret:+.2%}"
            f" px={price:.4f} entry={self._entry_px:.4f}"
            f" pyramid={self._pyramid_level}"
        )

        self._exiting = True
        self.market_order(self._pos_sym, -qty, tag=tag)

    def _check_pyramid(self):
        """Add to winning position at predefined levels."""
        if not self._pos_sym or self._entry_px <= 0:
            return
        price = float(self.securities[self._pos_sym].price)
        ret = (price - self._entry_px) / self._entry_px

        target = PYRAMID_1_AT if self._pyramid_level == 0 else PYRAMID_2_AT
        size = PYRAMID_1_SIZE if self._pyramid_level == 0 else PYRAMID_2_SIZE

        if ret >= target:
            pv = float(self.portfolio.total_portfolio_value)
            cash = pv * size * 0.99
            qty = cash / price
            lot = self.securities[self._pos_sym].symbol_properties.lot_size
            min_order = self.securities[self._pos_sym].symbol_properties.minimum_order_size
            qty = max(0, (qty // lot) * lot)
            if qty >= min_order:
                self._pyramid_level += 1
                self.market_order(self._pos_sym, qty, tag=f"PYRAMID_{self._pyramid_level}")
                self.log(
                    f"[hydra] PYRAMID_{self._pyramid_level} {self._pos_sym.value}"
                    f" ret={ret:+.2%} add_qty={qty:.6f}"
                )

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            if order_event.status == OrderStatus.INVALID:
                self._exiting = False
                self._pending_sym = None
            return

        sym = order_event.symbol
        tag = self.transactions.get_order_by_id(order_event.order_id).tag

        if tag == "ENTRY":
            self._pos_sym = sym
            self._entry_px = float(order_event.fill_price)
            self._entry_time = self.time
            self._pending_sym = None
            self._trail.reset(self._entry_px)

        elif tag.startswith("PYRAMID"):
            pass  # position already tracked

        elif tag.startswith("EXIT"):
            is_sl = "SL" in tag
            ret = (float(order_event.fill_price) - self._entry_px) / self._entry_px if self._entry_px > 0 else 0

            if is_sl:
                self._daily_sl_count += 1
                self._last_sl_time[sym] = self.time

            self._exit_time = self.time
            self._clear_state()

    def _clear_state(self):
        self._pos_sym = None
        self._entry_px = 0.0
        self._entry_time = None
        self._pending_sym = None
        self._exiting = False
        self._pyramid_level = 0
        self._trail.reset(0)

    def _daily_reset(self):
        self._daily_sl_count = 0

    def _update_forecasts(self):
        """Update Chronos forecasts for all symbols."""
        if self._forecast_last and (self.time - self._forecast_last).total_seconds() < 3600:
            return
        try:
            from forecast_features import chronos_forecast_return, wavelet_forecast_return
            for sym in self._symbols:
                closes = list(self._state[sym]["closes"])
                if len(closes) >= 64:
                    c_ret = chronos_forecast_return(closes)
                    w_ret = wavelet_forecast_return(closes)
                    self._forecast_cache[sym] = c_ret + w_ret  # combined forecast
            self._forecast_last = self.time
        except Exception:
            pass
