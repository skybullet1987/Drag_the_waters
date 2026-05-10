"""Pulse — central configuration.

All tunable constants live here. Walk-forward parameter selection (Phase 3)
will sweep a subset of these.

Capital ramp anchored to user direction:
    $100 → $500 → $2K → $5K → $55K (MG36 capacity ceiling)

Risk: max DD 25% (circuit-breaker trip at 20%, hard halt at 25%).
"""

# ─── Capital + risk ──────────────────────────────────────────────────────────
INITIAL_CASH_USD          = 100.0       # paper-trade start
TARGET_CAPACITY_USD       = 55_000.0    # MG36 capacity estimate, do not exceed
MAX_DRAWDOWN_TRIP_PCT     = 0.20        # circuit breaker arms here
MAX_DRAWDOWN_HALT_PCT     = 0.25        # hard halt — full liquidate
MAX_DD_RECOVERY_PCT       = 0.05        # equity must recover 5% above trip-low to unblock
PER_TRADE_HARD_KILL_PCT   = 0.08        # market liquidate at -8% per trade

# ─── Universe (Phase 0a) ─────────────────────────────────────────────────────
# A symbol is eligible only if all hold:
UNIV_MIN_DOLLAR_VOL_24H_USD = 5_000_000.0   # rolling 24h $vol on Kraken
UNIV_MAX_AVG_SPREAD_BPS     = 30.0          # rolling 60-bar mean spread
UNIV_MIN_PRICE_USD          = 0.01          # excludes meme dust
UNIV_MIN_DAYS_HISTORY       = 30            # excludes brand-new listings
UNIV_KRAKEN_CASH_ONLY       = True          # excludes margin-only / wrapped variants

# ─── Symbol tiers (per-tier max position USD + slippage budget) ──────────────
# Each tier defines: max_position_usd, slippage_budget_bps
TIER_MAJOR_SYMBOLS = ("BTCUSD", "ETHUSD")
TIER_LARGE_SYMBOLS = (
    "SOLUSD", "XRPUSD", "BNBUSD", "ADAUSD", "DOGEUSD",
    "LINKUSD", "AVAXUSD", "DOTUSD",
)
TIER_MID_SYMBOLS = (
    "LTCUSD", "MATICUSD", "ATOMUSD", "UNIUSD", "AAVEUSD",
    "NEARUSD", "INJUSD", "OPUSD", "ARBUSD", "BCHUSD",
    "TRXUSD", "FETUSD", "ICPUSD", "RENDERUSD", "HBARUSD",
)
# everything else passing the universe gate falls into MICRO

TIER_LIMITS = {
    "major": {"max_pos_usd": 5_000.0, "slip_budget_bps": 10.0},
    "large": {"max_pos_usd": 1_500.0, "slip_budget_bps": 25.0},
    "mid":   {"max_pos_usd":   500.0, "slip_budget_bps": 50.0},
    "micro": {"max_pos_usd":   100.0, "slip_budget_bps": 100.0},
}

# ─── Slippage tier — auto demote/eject ───────────────────────────────────────
TIER_SLIP_LOOKBACK_TRADES   = 5
TIER_DEMOTE_BUDGET_MULT     = 1.5    # if avg recent slip > 1.5× budget, demote tier
TIER_EJECT_BUDGET_MULT      = 3.0    # if avg recent slip > 3× budget, eject 7d
TIER_EJECT_DURATION_HOURS   = 24 * 7

# ─── Engine entry thresholds (will be walk-forward tuned in Phase 3) ─────────
SCALP_ENTRY_THRESHOLD       = 0.55
SCALP_HIGH_CONVICTION_THRES = 0.70

# ─── Exit ────────────────────────────────────────────────────────────────────
QUICK_TAKE_PROFIT_PCT       = 0.12
TIGHT_STOP_LOSS_PCT         = 0.035
ATR_TP_MULT                 = 4.0
ATR_SL_MULT                 = 2.0
TRAIL_ACTIVATION_PCT        = 0.04
TRAIL_STOP_PCT              = 0.025
TIME_STOP_HOURS             = 3.0

# ─── Position sizing ─────────────────────────────────────────────────────────
MAX_POSITIONS               = 6
TARGET_POSITION_ANN_VOL     = 0.35
PORTFOLIO_VOL_CAP           = 0.80

# ─── Maker/taker assumption (calibrated to live evidence) ────────────────────
# MG36 paper trade: 0/2 maker limits filled within 30s timeout → assume taker reality
LIVE_MAKER_FILL_RATE_EXPECTATION = 0.10   # 10%, not 60%; informs harsh sim
LIMIT_ORDER_TTL_SECONDS          = 30

# ─── Fear & Greed regime layer (Tier D.1 — properly implemented this time) ───
FG_GREED_EXTREME_THRESHOLD  = 75    # halve max_positions above this
FG_FEAR_EXTREME_THRESHOLD   = 25    # bias bounce setups below this
FG_SIZE_MULT_GREED          = 0.5   # at extreme greed
FG_SIZE_MULT_FEAR           = 1.2   # at extreme fear (slight up-size for capitulation buys)

# ─── Audit / harsh simulator defaults ────────────────────────────────────────
HARSH_SLIPPAGE_BASE_BPS     = 100.0   # calibrated to MG36 paper ⚠️ HIGH SLIPPAGE 65-197bp
HARSH_TAKER_FEE_PCT         = 0.0040  # 100% taker assumption
HARSH_LIMIT_TTL_SECONDS     = 30
HARSH_FILL_DELAY_BARS       = 1       # T+1 fill (act on bar T's signal at T+1's open)
HARSH_MULTI_ORDER_SPREAD_BPS = 10.0   # +10bp per concurrent open order
HARSH_REJECT_RATE_NORMAL    = 0.02
HARSH_REJECT_RATE_VOL_SPIKE = 0.05
HARSH_FORCE_OBI_TO_ZERO     = True    # OBI is fake in backtest (live evidence)
