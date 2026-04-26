# ── Vox Configuration ─────────────────────────────────────────────────────────
#
# All tunable strategy constants live here.  Each constant can be overridden
# at runtime via the QuantConnect parameter panel (see VoxAlgorithm.initialize).
# ─────────────────────────────────────────────────────────────────────────────

# Exit thresholds
TAKE_PROFIT       = 0.020   # +2.0 %  close long on gain
STOP_LOSS         = 0.012   # −1.2 %  close long on loss
TIMEOUT_HOURS     = 3.0     # close after this many hours regardless

# ATR-based dynamic exit multipliers
ATR_TP_MULT       = 2.0     # TP = entry + ATR_TP_MULT × ATR
ATR_SL_MULT       = 1.2     # SL = entry − ATR_SL_MULT × ATR

# Ensemble confidence gates
SCORE_MIN         = 0.60    # minimum mean_proba to open a position
SCORE_GAP         = 0.05    # required lead of top coin over runner-up
MAX_DISPERSION    = 0.15    # max std_proba across models
MIN_AGREE         = 4       # min number of models with proba >= 0.5

# Position sizing
ALLOCATION        = 0.50    # fallback fraction of portfolio if Kelly disabled
KELLY_FRAC        = 0.25    # fractional-Kelly multiplier
MAX_ALLOC         = 0.80    # hard ceiling on any single trade allocation
USE_KELLY         = True    # set False to use flat ALLOCATION

# Risk management
MAX_DAILY_SL      = 2       # halt new entries after this many SL hits per day
COOLDOWN_MINS     = 15      # minutes to wait after any exit before re-entering
SL_COOLDOWN_MINS  = 60      # per-coin cooldown specifically after an SL exit
MAX_DD_PCT        = 0.08    # drawdown circuit-breaker: halt if equity drops > 8 %
CASH_BUFFER       = 0.99    # keep 1 % cash headroom for fees/rounding

# Data cadence
RESOLUTION_MINUTES     = 5   # subscribe at 5-min bars, consolidate internally
DECISION_INTERVAL_MIN  = 15  # only evaluate entries at 15-min boundaries

# Warm-up and retraining
WARMUP_DAYS       = 90       # bars of history needed before trading
RETRAIN_DAY       = "monday" # day-of-week label used for scheduled retrain
