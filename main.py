# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

# ─────────────────────────────────────────────────────────────────────────────
# Kraken Top-Coin Rotation — 15-minute bars, single-position
#
# Universe    : 10 fixed Kraken USD pairs
# Cadence     : score all coins every 15 minutes
# Selection   : enter the top coin when its score >= SCORE_MIN and it
#               leads the runner-up by at least SCORE_GAP
# Sizing      : ALLOCATION fraction of portfolio, one position at a time
# Exits       : take-profit (+1.2 %), stop-loss (−0.7 %), or 2-hour timeout
# Risk guards : post-exit cooldown; daily stop-loss cap
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE = [
    "BTCUSD", "ETHUSD", "AVAXUSD", "XRPUSD", "ADAUSD",
    "LTCUSD", "LINKUSD", "DOTUSD", "TRXUSD", "SOLUSD",
]

# Default strategy constants — all overridable via the QC parameter panel
TAKE_PROFIT   = 0.012   # +1.2 %  close long when price rises this much
STOP_LOSS     = 0.007   # −0.7 %  close long when price falls this much
TIMEOUT_HOURS = 2.0     # close if neither TP nor SL triggered within 2 h
SCORE_MIN     = 0.60    # minimum composite score required to open a position
SCORE_GAP     = 0.04    # required score lead of #1 coin over #2 coin
ALLOCATION    = 0.80    # fraction of portfolio per trade
MAX_DAILY_SL  = 2       # halt new entries for the day after this many SL hits
COOLDOWN_MINS = 15      # minutes to wait after any exit before re-entering


class KrakenTopCoinAlgorithm(QCAlgorithm):
    """
    Single-file QuantConnect strategy: top-coin rotation on Kraken.

    Every 15 minutes a composite score (momentum + RSI + volume spike) is
    computed for each of the 10 fixed Kraken USD coins.  If the top coin
    scores at or above SCORE_MIN and its lead over second place meets
    SCORE_GAP, a long position sized at ALLOCATION × portfolio is opened.

    All exits are rule-based: take-profit (+1.2 %), stop-loss (−0.7 %),
    or a 2-hour time-stop.  Risk controls: COOLDOWN_MINS cooldown after
    every exit; no new entries once MAX_DAILY_SL stop-losses occur in a day.
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(5_000)
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.CASH)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # Parameters — each can be overridden via the QC parameter panel
        self._tp      = float(self.get_parameter("take_profit")   or TAKE_PROFIT)
        self._sl      = float(self.get_parameter("stop_loss")     or STOP_LOSS)
        self._toh     = float(self.get_parameter("timeout_hours") or TIMEOUT_HOURS)
        self._s_min   = float(self.get_parameter("score_min")     or SCORE_MIN)
        self._s_gap   = float(self.get_parameter("score_gap")     or SCORE_GAP)
        self._alloc   = float(self.get_parameter("allocation")    or ALLOCATION)
        self._max_sl  = int(self.get_parameter("max_daily_sl")    or MAX_DAILY_SL)
        self._cd_mins = int(self.get_parameter("cooldown_mins")   or COOLDOWN_MINS)

        # Register securities and per-symbol 15-minute history
        self._symbols = []
        self._state   = {}   # symbol -> {"closes": deque, "volumes": deque}
        for ticker in UNIVERSE:
            sym = self.add_crypto(ticker, Resolution.MINUTE, Market.KRAKEN).symbol
            self._symbols.append(sym)
            self._state[sym] = {
                "closes":  deque(maxlen=30),   # 30 × 15-min bars ≈ 7.5 h
                "volumes": deque(maxlen=30),
            }
            # 15-min consolidator to keep close/volume history fresh
            self.consolidate(
                sym,
                timedelta(minutes=15),
                lambda bar, s=sym: self._on_15m(s, bar),
            )

        # Open-position state
        self._pos_sym    = None    # currently held symbol (or None)
        self._entry_px   = 0.0    # fill price of current position
        self._entry_time = None   # datetime of entry
        self._exit_time  = None   # datetime of most recent exit
        self._daily_sl   = 0      # stop-loss hits today

        # Warm up enough 15-min history before the first scoring pass
        self.set_warm_up(timedelta(days=5))

        # Reset daily SL counter at midnight
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(0, 0),
            self._reset_daily,
        )

    # ── 15-minute bar update ──────────────────────────────────────────────────

    def _on_15m(self, sym, bar):
        """Append close and volume from a freshly closed 15-min bar."""
        st = self._state[sym]
        st["closes"].append(float(bar.close))
        st["volumes"].append(float(bar.volume))

    # ── Main data handler ─────────────────────────────────────────────────────

    def on_data(self, data):
        if self.is_warming_up:
            return

        # Exit check — runs on every incoming minute bar for finer precision
        if self._pos_sym is not None and self._pos_sym in data.bars:
            self._check_exit(float(data.bars[self._pos_sym].close))

        # Entry logic — fire only at 15-minute bar boundaries
        if self.time.minute % 15 != 0:
            return
        if self._pos_sym is not None:
            return
        if self._exit_time is not None and (
            self.time - self._exit_time < timedelta(minutes=self._cd_mins)
        ):
            return
        if self._daily_sl >= self._max_sl:
            return

        self._try_enter()

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, price):
        """Evaluate TP / SL / timeout for the current open position."""
        ret     = (price - self._entry_px) / self._entry_px
        elapsed = (self.time - self._entry_time).total_seconds() / 3600.0

        reason = None
        if ret >= self._tp:
            reason = "TP"
        elif ret <= -self._sl:
            reason = "SL"
        elif elapsed >= self._toh:
            reason = "TIMEOUT"

        if reason:
            self.liquidate(self._pos_sym, tag=reason)
            self.debug(
                f"Exit  {self._pos_sym.value}  ret={ret:.3%}  "
                f"held={elapsed:.2f}h  [{reason}]"
            )
            if reason == "SL":
                self._daily_sl += 1
            self._exit_time  = self.time
            self._pos_sym    = None
            self._entry_px   = 0.0
            self._entry_time = None

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_enter(self):
        """Score all coins and open a position in the top-ranked one."""
        scores = {}
        for sym in self._symbols:
            sc = self._score(sym)
            if sc is not None:
                scores[sym] = sc

        if not scores:
            return

        ranked    = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sym, top_sc = ranked[0]
        second_sc       = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_sc < self._s_min or (top_sc - second_sc) < self._s_gap:
            return

        self.set_holdings(top_sym, self._alloc, tag="ENTRY")
        self._pos_sym    = top_sym
        self._entry_px   = float(self.securities[top_sym].price)
        self._entry_time = self.time
        self.debug(
            f"Enter {top_sym.value}  score={top_sc:.3f}  "
            f"price={self._entry_px:.4f}"
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, sym):
        """
        Composite score in [0, 1] using three sub-signals:

        * 50% momentum  — return over last 4 intervals (4 × 15 min ≈ 1 h);
                          1 % move normalised to ±1.
        * 30% RSI(14)   — reward 55–74 (uptrend); penalise ≥ 75 (overbought)
                          or < 40 (oversold); simple non-smoothed calculation.
        * 20% vol spike — current-bar volume vs prior 15-bar mean; capped at 1.

        Returns None when there is insufficient history (< 20 bars).
        """
        st      = self._state[sym]
        closes  = st["closes"]
        volumes = st["volumes"]

        if len(closes) < 20 or len(volumes) < 2:
            return None

        c = list(closes)   # index 0 = oldest, index -1 = most recent

        # 1) Momentum: return over the last 4 intervals (4 × 15 min ≈ 1 h).
        #    c[-5] is the close 4 bar-periods before the current close c[-1].
        mom   = (c[-1] - c[-5]) / c[-5]
        mom_n = max(min(mom / 0.01, 1.0), -1.0)

        # 2) RSI(14) — simple (non-smoothed Wilder) computation
        deltas = [c[i] - c[i - 1] for i in range(-14, 0)]
        gains  = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        avg_g  = gains / 14
        avg_l  = losses / 14
        rsi    = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)

        if rsi >= 75:
            rsi_s = -0.5    # overbought — penalise entry
        elif rsi >= 55:
            rsi_s =  1.0    # healthy uptrend — reward
        elif rsi >= 40:
            rsi_s =  0.3    # neutral
        else:
            rsi_s = -0.3    # oversold — slight penalty

        # 3) Volume spike vs prior 15-bar mean (exclude current bar)
        vols      = list(volumes)
        prior_avg = float(np.mean(vols[-16:-1]))   # exactly 15 prior bars
        vol_spike = (vols[-1] / prior_avg - 1.0) if prior_avg > 0 else 0.0
        vol_s     = min(max(vol_spike, 0.0), 1.0)   # clamp to [0, 1]

        # Weighted sum in [−1, 1] mapped to [0, 1]
        raw = 0.50 * mom_n + 0.30 * rsi_s + 0.20 * vol_s
        return (raw + 1.0) / 2.0

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily(self):
        """Clear the daily stop-loss counter at midnight."""
        self._daily_sl = 0
