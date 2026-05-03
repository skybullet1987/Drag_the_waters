# Vox v2 — ML Ensemble Kraken Rotation Strategy

## What is Vox?

**Vox** is a machine-learning overlay strategy that lives alongside the
rule-based `main.py` (KrakenTopCoinAlgorithm) in this repository.  Where
`main.py` uses a hand-crafted composite score (momentum + RSI + volume spike),
Vox replaces the scoring layer with a four-model **heterogeneous soft-voting
classifier ensemble** combined with an **expected-return regression ensemble**,
trained on cost-aware triple-barrier labels derived from the same market data.

Both strategies share the same fill-driven state machine pattern (position
state updated only in `on_order_event`), the same Kraken brokerage model, and
the same pre-trade validation philosophy.  They are designed to be run and
compared independently — Vox does **not** modify `main.py`.

---

## Architecture (Vox v2)

```
  ┌─────────────┐   5-min bars    ┌──────────────────────────┐
  │  Kraken /   │ ──────────────► │  State Deques            │
  │  QC Feed    │                 │  (closes, highs, lows,   │
  └─────────────┘                 │   volumes — per symbol)  │
                                  └──────────┬───────────────┘
                                             │  every 15 min
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Feature Builder         │
                                  │  (models.py)             │
                                  │  ret×4, RSI, ATR, vol,   │
                                  │  BTC-rel, hour           │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  VoxEnsemble (v2)        │
                                  │  Classifiers (weighted)  │
                                  │  + Return Regressors     │
                                  └──────────┬───────────────┘
                                             │  class_proba, std_proba,
                                             │  n_agree, pred_return
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Vox v2 Decision Gate    │
                                  │  EV + pred_return + cost │
                                  │  penalty cooldown        │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Regime Gate             │
                                  │  (risk.py)               │
                                  │  4h BTC SMA(20) + slope  │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Kelly Sizer             │
                                  │  (risk.py)               │
                                  │  Fractional-Kelly or     │
                                  │  flat allocation         │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Pre-trade Validation    │
                                  │  price>0, cash check,    │
                                  │  lot-size, min-order     │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Execution               │
                                  │  market_order("ENTRY")   │
                                  │  fill-driven state FSM   │
                                  └──────────────────────────┘
```

---

## Vox v2 Model Stack

### Classifier ensemble (weighted soft voting)

| Name | Class | Weight | Notes |
|------|-------|--------|-------|
| `hgbc` | `HistGradientBoostingClassifier` | **0.35** | Strong sklearn-native boosted trees; well-calibrated; no external deps |
| `et` | `ExtraTreesClassifier` + isotonic calib | **0.25** | Randomly split trees; uncorrelated from RF |
| `lr` | `LogisticRegression` | **0.20** | Linear baseline; fast; well-calibrated by design |
| `rf` | `RandomForestClassifier` + isotonic calib | **0.20** | Bagged trees |

> **GaussianNB is intentionally removed.** At typical positive rates of 1–5%
> it dominates the soft vote with extreme probabilities and degrades calibration.
> It is not retained even at low weight.

### Regression ensemble (weighted mean predicted return)

| Name | Class | Weight | Notes |
|------|-------|--------|-------|
| `hgbr` | `HistGradientBoostingRegressor` | **0.40** | Strong sklearn-native gradient booster |
| `etr` | `ExtraTreesRegressor` | **0.35** | Non-linear, diverse from HGBR |
| `ridge` | `Ridge` | **0.25** | Stable linear baseline; handles small datasets |

Trained on cost-aware realised trade returns from `triple_barrier_outcome()`.
Only used when enough training samples are available (≥ 50 samples).

---

## Vox v2 Decision Equation

For each candidate symbol that passes the confidence gates, Vox computes:

```text
class_proba  = weighted VotingClassifier P(class=1)
pred_return  = weighted regression prediction of net forward return

ev = class_proba × tp_use − (1 − class_proba) × sl_use − cost_fraction

final_score = 0.6 × ev + 0.4 × pred_return   (when regressors trained)
            = ev × (1 − std_proba)            (fallback, before regressor training)
```

Where:
- `tp_use`, `sl_use` — ATR-based TP/SL per candidate (or fixed fallback)
- `cost_fraction` — `COST_BPS × 1e-4` (estimated round-trip fee + slippage)
- `std_proba` — std-dev of per-model probabilities (model disagreement)

Candidates are **ranked by `final_score`** and the top candidate is selected
only if it passes all gates (see entry-gate pipeline below).

---

## Entry Gate Pipeline (Vox v2)

| # | Gate | Description |
|---|------|-------------|
| 1 | Feature history | `build_features()` returns a valid feature vector |
| 2 | **Penalty cooldown** | Symbol not in post-repeated-loss penalty cooldown |
| 3 | Class probability | `class_proba >= score_min_eff` (base-rate-aware) |
| 4 | Dispersion | `std_proba <= MAX_DISPERSION (0.15)` |
| 5 | Agreement | `n_agree >= MIN_AGREE (3)` — 3 of 4 models agree |
| 6 | **EV gate** | `ev_after_costs > MIN_EV (0.004)` — 0.4 % expected edge |
| 7 | **Predicted return** | `pred_return >= PRED_RETURN_MIN (0.003)` (if regressors trained) |
| 8 | Score gap | `top_final_score − second_final_score >= EV_GAP (0.00025)` |
| 9 | Regime | 4h BTC SMA(20) above price + positive slope |
| 10 | Risk manager | Global cooldown, daily SL cap, drawdown circuit-breaker |
| 11 | Pre-trade | Price > 0, sufficient cash, lot-size ≥ min order |

---

## Trade-Quality Controls (Vox v2)

### Minimum hold time

Ordinary TP/SL/timeout exits are suppressed for the first `MIN_HOLD_MINUTES`
(default 15) minutes.  The only exception is an **emergency SL**: if the loss
exceeds `EMERGENCY_SL` (default 3.0 %) before the minimum hold time, the
position is closed regardless.

```python
# Logic in _check_exit():
if elapsed_minutes < min_hold_minutes:
    if ret <= -emergency_sl:
        exit(reason="EXIT_SL")   # emergency stop
    else:
        return                   # suppress normal exit — hold
else:
    # Normal TP/SL/timeout logic
```

This prevents the algorithm from entering and immediately being shaken out by
microstructure noise within 1–5 bars.

### Per-symbol penalty cooldown (repeated losses)

After `PENALTY_COOLDOWN_LOSSES` (default 3) consecutive SL exits on the same
symbol, that symbol is blocked for `PENALTY_COOLDOWN_HOURS` (default 48 hours).
Entry logic skips symbols in this penalty window.

The last 10 trade outcomes are tracked per symbol in a rolling deque.
Penalty cooldown is independent of the shorter per-coin SL cooldown in
`RiskManager` (default 60 minutes after a single SL).

### Risk profiles

Vox supports four risk profiles, selectable via the `risk_profile` QC parameter
(or convenience boolean aliases).  Profiles are applied in this priority order:

```text
ruthless_mode=true  >  aggressive_mode=true  >  conservative_mode=true
  >  risk_profile parameter  >  default (balanced)
```

| Profile | `risk_profile` value | Alias | Character |
|---------|----------------------|-------|-----------|
| Conservative | `"conservative"` | `conservative_mode=true` | Strict research-grade gates, rarely trades |
| **Balanced** (default) | `"balanced"` | — | Tradable defaults, controlled drawdown |
| Aggressive | `"aggressive"` | `aggressive_mode=true` | Looser gates, larger sizing, wider TP/SL |
| Ruthless | `"ruthless"` | `ruthless_mode=true` | Maximum aggression — **extreme risk** |

#### Balanced (default)

Normal mode — intentionally tradable.  See the parameter table below.

#### Conservative mode

Setting `conservative_mode=True` (or `risk_profile=conservative`) applies
stricter, research-grade defaults in one switch.

| Parameter | Conservative override | Balanced default |
|-----------|-----------------------|-----------------|
| `score_min` | ≥ 0.55 | 0.50 |
| `max_dispersion` | ≤ 0.15 | 0.22 |
| `min_agree` | ≥ 3 | 2 |
| `min_ev` | ≥ 0.004 (0.4 %) | 0.001 (0.1 %) |
| `cost_bps` | ≥ 50 (0.50 %) | 35 (0.35 %) |
| `pred_return_min` | ≥ 0.003 (+0.30 %) | −0.0005 (soft veto) |
| `max_daily_sl` | ≤ 1 | 2 |
| `cooldown_mins` | ≥ 30 | 20 |
| `take_profit` | ≥ 0.035 | 0.030 |
| `stop_loss` | ≥ 0.020 | 0.015 |
| `min_hold_minutes` | ≥ 20 | 15 |

**Use conservative mode for:** research validation, high-quality trade
selection, and comparing filtered vs unfiltered sets.

**Do not use conservative mode in normal operation** — it will block most trades
because `pred_return_min >= 0.003` requires a strong positive predicted return,
which the ensemble rarely exceeds in the first months of data.

#### Aggressive mode (`risk_profile=aggressive` or `aggressive_mode=true`)

Looser signal gates, larger position sizing, wider TP/SL, faster re-entry.
Suitable for users who want more trades and higher upside, and accept higher
drawdown risk than balanced mode.

| Parameter | Aggressive | Balanced default |
|-----------|-----------|-----------------|
| `score_min` | 0.48 | 0.50 |
| `max_dispersion` | 0.28 | 0.22 |
| `min_agree` | 1 | 2 |
| `min_ev` | 0.0005 | 0.001 |
| `pred_return_min` | −0.0010 | −0.0005 |
| `ev_gap` | 0.0 | 0.0001 |
| `cost_bps` | 25 | 35 |
| `allocation` | 0.75 | 0.50 |
| `max_alloc` | 0.95 | 0.80 |
| `kelly_frac` | 0.50 | 0.25 |
| `take_profit` | 0.045 (+4.5%) | 0.030 |
| `stop_loss` | 0.020 (−2.0%) | 0.015 |
| `timeout_hours` | 8 | 6 |
| `min_hold_minutes` | 5 | 15 |
| `emergency_sl` | 0.025 | 0.030 |
| `max_daily_sl` | 3 | 2 |
| `cooldown_mins` | 5 | 20 |
| `sl_cooldown_mins` | 20 | 60 |
| `max_dd_pct` | 0.20 (20%) | 0.08 |

Aggressive mode also enables:
- **Momentum score boost** in the final ranking formula (see below).
- **Momentum breakout override** (see below).

#### Ruthless v2 mode (`risk_profile=ruthless` or `ruthless_mode=true`)

> ⚠ **VERY HIGH RISK — use for experimentation only.**
> Ruthless v2 targets genuinely high-upside, high-risk trading.  Expect larger
> position sizes, slower (but bigger) winners, and **significantly larger
> drawdowns and losses than balanced or aggressive mode**.  Ruthless mode can
> draw down 35 % from peak before the circuit-breaker engages.
> **Never use ruthless mode for live trading without careful validation on a
> small account first.**

Ruthless v2 is designed to shift away from fee-sensitive scalping toward
asymmetric, larger winners:

- **Wider TP/SL** (9 % / 3 %) — P/L ratio ≈ 3.0 vs. 1.56 in the old ruthless
- **24 h timeout** — winners have a full day to run vs. 12 h previously
- **Kelly disabled by default** — flat 90 % allocation per trade instead of
  Kelly potentially shrinking orders to 5–8 % of portfolio
- **Allocation floor** — when Kelly is enabled, it cannot shrink allocations
  below 75 % (`min_alloc=0.75`)
- **Runner mode** — trailing stop replaces the instant TP exit (see below)
- **Very loose pred_return gate** — regressor veto is nearly disabled

Ruthless **v3** adds trend-confirmation and chop-protection mechanics:

- **TP/SL floors** — ATR-derived exits cannot shrink below the configured ruthless
  9 % TP and 3 % SL thresholds.
- **Ruthless confirmation gate** — entries require passing one of three confirmation
  paths (see below).
- **Anti-chop same-symbol cooldown** — 120-min block after every SL exit;
  24-hour block after 2 SL exits within any 24-hour window.
- **Portfolio loss-streak brake** — all entries pause for 6 hours after 4
  consecutive losing trades.
- **Profile-aligned labels** — training labels use 9 % TP / 3 % SL / 96-bar
  horizon for ruthless mode to align the model with actual ruthless targets.
- **SMA slope feature** — feature index 9 now provides a short-SMA slope
  proxy (trend direction) instead of the reserved zero slot.

| Parameter | Ruthless v2/v3 | Balanced default |
|-----------|---------------|-----------------|
| `score_min` | 0.45 | 0.50 |
| `max_dispersion` | 0.35 | 0.22 |
| `min_agree` | 1 | 2 |
| `min_ev` | 0.0 | 0.001 |
| `pred_return_min` | **−0.004** | −0.0005 |
| `ev_gap` | 0.0 | 0.0001 |
| `cost_bps` | 20 | 35 |
| `allocation` | **0.90** | 0.50 |
| `max_alloc` | **1.00 (100 %)** | 0.80 |
| `use_kelly` | **False** (flat 90 %) | True |
| `min_alloc` | **0.75** | 0.0 |
| `kelly_frac` | 0.75 | 0.25 |
| `take_profit` | **0.09 (+9.0 %)** | 0.030 |
| `stop_loss` | **0.03 (−3.0 %)** | 0.015 |
| `timeout_hours` | **24** | 6 |
| `min_hold_minutes` | 10 | 15 |
| `emergency_sl` | 0.05 (−5.0 %) | 0.030 |
| `max_daily_sl` | 5 | 2 |
| `cooldown_mins` | 0 | 20 |
| `sl_cooldown_mins` | **120** *(v3: was 5)* | 60 |
| `penalty_cooldown_losses` | 5 | 3 |
| `penalty_cooldown_hours` | 12 | 48 |
| `max_dd_pct` | 0.35 (35 %) | 0.08 |
| `runner_mode` | **True** | False |
| `trail_after_tp` | **0.04 (+4 %)** | — |
| `trail_pct` | **0.025 (2.5 %)** | — |
| `label_tp` | **0.09** *(v3 override)* | 0.012 |
| `label_sl` | **0.03** *(v3 override)* | 0.010 |
| `label_horizon_bars` | **96** *(v3 override)* | 72 |

Ruthless v2/v3 also enables:
- **Momentum score boost** in the final ranking formula.
- **Momentum breakout override** (see below).
- **Ruthless confirmation gate** (see below).

> ⚠️ **Warning**: Even with v3 improvements, **large drawdowns (up to 35 %)
> are expected** in ruthless mode.  The strategy deliberately accepts large
> losses in pursuit of large winners.  Ruthless mode is not appropriate for
> production live trading without extensive validation.

#### Ruthless TP/SL floors (v3)

ATR-derived TP/SL are computed per-candidate, but in ruthless mode they
cannot shrink below the profile's configured thresholds:

```python
tp_use = max(atr_tp, self._tp)   # at least 9%
sl_use = max(atr_sl, self._sl)   # at least 3%
```

This prevents the common failure mode where tight ATR conditions produce
0.7 %–1.5 % effective stops despite a 3 % configured SL.

Entry logs indicate when floors are applied:
```
[ruthless] ENTRY BCHUSD  tp=0.0900(floor=True)  sl=0.0300(floor=True)  ...
```

#### Ruthless confirmation gate (v3)

Ruthless mode requires candidates to pass at least one of three confirmation
paths before a large (≈90 %) allocation entry is placed:

| Path | Condition |
|------|-----------|
| `momentum_override` | Entry path already is `momentum_override` |
| `strong_ml` | `ev_score >= 0.006 AND class_proba >= 0.60 AND n_agree >= 2` |
| `trend_momentum` | `ret_4 >= 0.010 AND ret_16 >= 0.020 AND vol_r >= 1.5` |

Candidates that fail all three paths are skipped.  Entry logs include the
confirmation reason:
```
[ruthless] ENTRY SOLUSD  confirm=strong_ml  proba=0.65  n_agree=3  ev=0.00821  ...
[ruthless] ENTRY ADAUSD  confirm=trend_momentum  ret4=0.0143  ret16=0.0261  vol_r=2.1 ...
[ruthless] ENTRY INJUSD  confirm=momentum_override  ...
```

#### Anti-chop same-symbol cooldown (v3)

After any SL exit, the same symbol is blocked for **120 minutes** (up from 5
minutes in v2).

Additionally, if the same symbol has **2 or more SL exits within any 24-hour
window**, it is blocked for a full **24 hours**.  This prevents BCH/ARB/NEAR/INJ-
style loss spirals where the same coin is re-entered minutes after a stop-out.

```
[ruthless] ANTI-CHOP BLOCK: BCHUSD had 2 SL exits in 24h — blocked until 2024-01-02 14:30:00
```

This cooldown is independent of the existing penalty cooldown mechanism.

#### Portfolio loss-streak brake (v3)

After **4 consecutive losing trades** (across all symbols), all new entries are
paused for **6 hours**.

```
[ruthless] LOSS-STREAK PAUSE: 4 consecutive losses — all entries paused until 2024-01-07 18:00:00
```

The streak counter resets to zero:
- When any trade is a **winner** (positive return), OR
- When the pause is triggered (streak resets to prevent a second immediate pause)

The pause only affects new entries.  Open positions are managed normally (exits
still fire as usual).

#### Runner / trailing-profit mode (`runner_mode=true`)

Ruthless v2 enables `runner_mode` by default.  Instead of exiting immediately
when the return crosses the TP threshold, the strategy activates a **trailing
stop** and lets winners run.

**Behavior:**

1. When `ret >= trail_after_tp` (default +4 %) **or** `ret >= tp_use`
   (whichever comes first), trailing is activated instead of exiting.
2. The highest price seen since activation is tracked as `trail_high_px`.
3. The position exits with tag **`EXIT_TRAIL`** when:
   ```
   price <= trail_high_px × (1 − trail_pct)
   ```
   i.e. the price falls 2.5 % from the trailing high.
4. Hard SL (`EXIT_SL`), emergency SL, and timeout (`EXIT_TIMEOUT`) remain
   active throughout — the trail never overrides catastrophic protection.

**Example with defaults (take_profit=0.09, trail_after_tp=0.04, trail_pct=0.025):**

```
Entry at $100.
+4 % → $104 → trailing activated, trail_high=104
Price rises to $115 → trail_high=115, trail_stop=115×0.975=$112.12
Price drops to $112 → EXIT_TRAIL at ~+12 %
```

**`EXIT_TRAIL` exit tag** appears in order logs and trade records.  It is
*not* treated as a stop-loss for penalty-cooldown accounting.

**Recommended ruthless v3 setup:**

```text
risk_profile = ruthless
# Optional overrides:
use_kelly    = false     # flat 90 % allocation (already default in ruthless)
min_alloc    = 0.75      # Kelly floor if re-enabling Kelly
runner_mode  = true      # trailing stop — already default in ruthless
```

---

## Ruthless v4 Improvements

### 1 — Delayed trailing activation

Trailing now activates at **+7 %** (was +4 %) and trails **3 %** from the
high-water mark (was 2.5 %).  This gives winners more room to develop before
clipping, addressing the common pattern where the trailing stop was triggered
at +1–3 % gains while losses ran to the full −3 %.

```
trail_after_tp = 0.07   # was 0.04
trail_pct      = 0.03   # was 0.025
```

Balanced, conservative, and aggressive profiles are not affected.

### 2 — Breakeven stop

Once a ruthless trade reaches **+3 %** peak return, the effective stop is
raised to entry **+0.3 %**.  If the price then falls back to that level the
trade exits as **`EXIT_SL`** (keeping the minor profit) instead of running to
the full −3 % stop-loss.

```
breakeven_after  = 0.03    # arm at +3 % peak return
breakeven_buffer = 0.003   # effective stop = +0.3 %
```

The `breakeven_active` field is logged on every exit.  The breakeven state
is fully reset on entry, exit, reconcile, and dust-clearing paths.

### 3 — Momentum-failure early exit

If a ruthless position has been held for at least **30 minutes**, is down
**−1.2 %** or more, and **both** `ret_4` and `ret_16` are negative, the trade
is cut immediately with tag **`EXIT_MOM_FAIL`**.

```
mom_fail_enabled          = true
mom_fail_min_hold_minutes = 30
mom_fail_loss             = -0.012   # -1.2%
```

This exits via **market order** to guarantee reliability.  Non-ruthless
profiles are not affected.

### 4 — Smarter timeout extension

When the timeout is reached on a ruthless runner trade:

| Condition | Action |
|-----------|--------|
| `ret >= +3 %` | Allow normal EXIT_TIMEOUT |
| `ret > -1 %` AND `ret_4 > 0` AND extension budget remaining | Extend hold by 12 h |
| All other cases | Exit immediately |

```
timeout_min_profit   = 0.03    # allow exit only if profitable
timeout_extend_hours = 12      # extend window by 12 h
max_timeout_hours    = 48      # hard cap on total hold time
```

The extension is logged once per position:
```
[ruthless] TIMEOUT EXTENDED: SOLUSD ret=1.5% new_timeout=36.0h
```

### 5 — Market mode / regime detection

A lightweight `MarketModeDetector` (in `Vox/market_mode.py`) classifies the
BTC 4-hour trend into one of five modes:

| Mode | Description |
|------|-------------|
| `risk_on_trend` | BTC trending up, moderate volatility |
| `pump` | BTC rising fast with volume expansion |
| `chop` | Oscillating, low directional momentum |
| `selloff` | BTC trending down |
| `high_vol_reversal` | Extreme volatility, conflicting signals |

Ruthless entries are permitted only in **`risk_on_trend`** and **`pump`**
modes (configurable via `RUTHLESS_ALLOWED_MODES`).  If the BTC data detector
detects a non-allowed mode, it becomes an additional `market_mode` confirmation
path rather than a hard block — ensuring the confirmation gate logic still
allows strong-ML or trend-momentum entries to proceed.

```
market_mode_enabled   = true
ruthless_allowed_modes = ["risk_on_trend", "pump"]
```

### 6 — Ruthless-specific labels / horizon

Label parameters for ruthless training are now wider, targeting larger 24–48 h
moves instead of the default 6 h balanced labels:

```
ruthless_label_tp           = 0.06     # vs. balanced 1.2 %
ruthless_label_sl           = 0.03     # vs. balanced 1.0 %
ruthless_label_horizon_bars = 96       # 8 h at 5-min bars (≈24 h window)
```

When `risk_profile=ruthless`, the retrain pipeline uses these wider barriers
so the model is optimised for the actual ruthless trade geometry.

### 7 — Richer trend/chop features (20 total)

`build_features()` now returns **20 features** (was 10).  Ten new
dependency-free features distinguish clean trends from chop:

| Feature | Description |
|---------|-------------|
| `range_efficiency` | Net move / sum of absolute moves; near 1.0 = clean trend |
| `sma_fast_slope` | 4-bar SMA slope vs. prior 4 bars |
| `price_vs_sma_fast` | Close relative to 4-bar SMA |
| `price_vs_sma_slow` | Close relative to 8-bar SMA |
| `recent_high_breakout` | Distance above 16-bar prior high |
| `vol_zscore` | Volume z-score over 16 bars |
| `reversal_frac` | Fraction of sign changes in last 8 bars |
| `green_bar_ratio` | Fraction of up-bars in last 8 bars |
| `atr_expansion` | Recent ATR / lagged ATR − 1 |
| `btc_ret_1` | 1-bar BTC return |

**Backward compatibility:** if an older 10-feature model is loaded from the
ObjectStore and prediction fails due to feature count mismatch, the ensemble
automatically marks itself as unfitted and retrains on the next cycle.

### 8 — Meta-filter / veto model

A rules-based `MetaFilter` (in `Vox/meta_model.py`) computes a conviction
score from multiple signals and vetoes low-confidence ruthless entries:

```
meta_filter_enabled = true   (ruthless only)
meta_min_proba      = 0.55
```

The meta-score combines model confidence, model agreement, EV, momentum,
volume, and market mode alignment.  Entry logs show the meta decision:

```
[ruthless] META-FILTER VETO: INJUSD meta_score=0.42 < 0.55
[ruthless] meta_filter approved SOLUSD score=0.73
```

Non-ruthless profiles are not affected.

### 9 — Optional entry limit orders with TTL

Entry limit orders can optionally reduce slippage by buying slightly below the
current price.  Disabled by default.

```
ruthless_use_entry_limit_orders = false   # set true to enable
entry_limit_offset              = 0.001   # buy limit at price × (1 − 0.001)
entry_limit_ttl_minutes         = 3       # cancel unfilled limits after 3 min
```

Unfilled limits are always cancelled after the TTL — stale signals are never
chased.  **Stop / emergency / momentum-failure exits always remain market orders
regardless of this setting.**

### 10 — Optional LightGBM / XGBoost ensemble members

Import guards allow adding optional gradient boosting members if installed:

```python
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
```

Enable via config flags:
```
use_lightgbm = false   # set true if lightgbm is installed
use_xgboost  = false   # set true if xgboost is installed
use_catboost = false   # keep false: CatBoost may not be in QC cloud
```

The **sklearn-native fallback** (LogisticRegression, HistGradientBoosting,
ExtraTreesClassifier, RandomForestClassifier) always works and is used by
default.  The algorithm logs availability at startup.

### 11 — Optional exit limit orders (non-urgent only)

A scaffold for passive limit exits on non-urgent tags (`EXIT_TRAIL`,
`EXIT_TIMEOUT`, `EXIT_TP`) is wired via config:

```
use_exit_limit_orders      = false   # disabled by default
exit_limit_offset          = 0.0005
exit_limit_ttl_minutes     = 1
exit_limit_fallback_to_market = true
```

**`EXIT_SL`, `EXIT_EMERGENCY_SL`, and `EXIT_MOM_FAIL` always use market orders.**
Exit limit orders are disabled by default since the risk of leaving a position
open indefinitely outweighs the marginal slippage saving.

### ⚠️ Ruthless remains high-risk

Even with all v4 improvements, ruthless mode:
- Takes near-full portfolio allocations on single positions
- Can suffer large drawdowns during trend failures or black-swan events
- Is **not suitable** as a primary strategy without careful out-of-sample validation
- Requires robust anti-chop controls, breakeven protection, and meta-filtering
  to avoid the "high-size chop scalping" failure mode

### File-size constraint

All Python files in the `Vox/` package must remain **under 63,000 characters**
for QuantConnect compatibility.  Large new code is placed in dedicated modules
(`execution.py`, `market_mode.py`, `meta_model.py`) rather than `main.py`.

---

### Momentum breakout override (aggressive/ruthless)

In aggressive and ruthless profiles, a candidate that fails normal ML gates
(class probability / dispersion / agreement) may still be entered if strong
momentum conditions are present.  This allows Vox to participate in crypto
pump events that occur before the ML ensemble fully detects the move.

**Momentum override conditions (all must be satisfied):**

| Feature | Default threshold | Parameter |
|---------|------------------|-----------|
| `ret_4` (4-bar return) | ≥ 0.015 (+1.5%) | `momentum_ret4_min` |
| `ret_16` (16-bar return) | ≥ 0.025 (+2.5%) | `momentum_ret16_min` |
| `vol_r` (volume ratio) | ≥ 2.0× | `momentum_volume_min` |
| `btc_rel` (BTC outperformance) | ≥ 0.005 | `momentum_btc_rel_min` |

Additionally, the candidate is blocked if its expected value after costs is
worse than `momentum_override_min_ev` (default −0.002, i.e. −0.2 %).  This
prevents obviously catastrophic entries from slipping through.

The regime filter still applies to momentum override candidates.  All execution
safety, risk manager, lot-size, and cash guards are fully enforced.

To enable momentum override explicitly in balanced mode:
```text
momentum_override = true
```

To disable it even in ruthless mode:
```text
momentum_override = false
```

When a trade is entered via momentum override, it is identified in logs:
```text
[vox] MOMENTUM OVERRIDE candidate=SOLUSD ret4=... ret16=... vol_r=... btc_rel=...
      ev=... pred_ret=... proba=...
```

And in the persistence log record:
```json
"entry_path": "momentum_override"
```

Normal ML entries have `"entry_path": "ml"`.

### Momentum score boost (aggressive/ruthless)

For aggressive and ruthless profiles, the final ranking formula adds a momentum
contribution to raise strong breakout candidates to the top:

```text
momentum_score = 0.40 × ret_4 + 0.30 × ret_16
               + 0.20 × normalised_volume_excess + 0.10 × btc_rel

  (capped to [−0.05, 0.10] to avoid raw-momentum explosion)

final_score = 0.50 × ev + 0.25 × pred_return + 0.25 × momentum_score
```

The balanced/conservative formula is unchanged:
```text
final_score = 0.6 × ev + 0.4 × pred_return
```

### Conservative mode

(See the conservative row in the [Risk profiles](#risk-profiles) table above.)

**Use conservative mode for:**
- Research validation where you want strict selectivity to ensure high quality.
- Evaluating whether a smaller set of high-confidence trades is profitable.
- Comparing filtered vs unfiltered trade sets.

**Do not use conservative mode in normal operation** — it will block most trades
because the regressor (`pred_return_min >= 0.003`) requires a strong positive
predicted return, which the ensemble rarely exceeds in the first months of data.

### Realized EV logging

At entry, Vox stores predicted `class_proba`, `pred_return`, `ev`,
`final_score`, and `entry_path` (`"ml"` or `"momentum_override"`).  At exit,
the realized return and exit reason are logged to `ObjectStore` alongside the
entry predictions via `PersistenceManager.log_trade`.

This allows post-hoc evaluation of:
- Does predicted EV correlate with realized return?
- Which symbols/hours have positive realized EV?
- Are the model probabilities calibrated (e.g. `class_proba=0.60` → ~60 % win rate)?
- How do ML-path vs momentum-override-path trades compare in realized return?

---

## Cost-Aware Labels

Training labels use `triple_barrier_outcome()` instead of `triple_barrier_label()`.
The new function applies a cost adjustment:

```text
label  = 1  if TP is hit before SL/timeout AND (tp - cost_fraction) > 0
realized_net_return:
  TP hit    →  tp_use − cost_fraction
  SL hit    →  −sl_use − cost_fraction
  Timeout   →  (final_price − entry_price) / entry_price − cost_fraction
```

The **regression targets** (`y_return`) are these realized net returns.
The **classification labels** (`y_class`) are 1 only when the TP barrier is hit
and the net profit is positive.  This ensures the model learns to identify trades
that are *actually profitable after costs*, not just raw price movements.

Label cost is controlled by `LABEL_COST_BPS` (default 30 bps), which can be
overridden via the QC parameter `label_cost_bps`.

> **Triple-barrier semantics preserved:** Labels are still generated by the
> triple-barrier method (TP before SL/timeout = 1, otherwise 0).  The only
> change is the net-of-costs requirement.  Next-bar up/down labels are not used.

---

## Triple-Barrier Labeling

Vox uses the **triple-barrier method** from Marcos López de Prado's
*Advances in Financial Machine Learning*:

```
                         upper barrier  ──  entry × (1 + tp)
                                         ↑ label = 1
entry price ──────────────────────────────────────────────────
                                         ↓ label = 0
                         lower barrier  ──  entry × (1 − sl)

   |← timeout_bars →|  vertical barrier: label = 0 if neither hit
```

A bar is labelled **1** if the price reaches the upper barrier before the lower
barrier within `timeout_bars` steps; **0** otherwise.

## Parameters and Defaults

All parameters are defined at the top of `main.py` as module-level constants
and can be overridden at runtime via the QuantConnect parameter panel.

### Execution parameters (Vox v2 defaults)

> **Normal mode is intentionally balanced/tradable.**  The defaults below allow
> candidates to pass the gates under reasonable market conditions.  Set
> `conservative_mode=True` (or `risk_profile=conservative`) to restore stricter
> research-grade thresholds.  See [Risk profiles](#risk-profiles) for all options.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risk_profile` | `"balanced"` | Risk/reward profile: `conservative`, `balanced`, `aggressive`, `ruthless` |
| `aggressive_mode` | `False` | Convenience alias for `risk_profile=aggressive` |
| `ruthless_mode` | `False` | Convenience alias for `risk_profile=ruthless` |
| `take_profit` | `0.030` | Take-profit fraction (+3 %) — wider than v1 to reduce fee drag |
| `stop_loss` | `0.015` | Stop-loss fraction (−1.5 %) — wider to avoid noise chop |
| `timeout_hours` | `6.0` | Max hold time in hours |
| `atr_tp_mult` | `2.0` | ATR multiplier for dynamic TP |
| `atr_sl_mult` | `1.2` | ATR multiplier for dynamic SL |
| `score_min` | `0.50` | Upper clamp on the effective score threshold (class_proba gate) |
| `score_gap` | `0.02` | Required probability gap to runner-up (probability units) |
| `max_dispersion` | `0.22` | Max std_proba across models — relaxed to allow trading |
| `min_agree` | `2` | Min models with proba ≥ agree_thr — relaxed to allow trading |
| `allocation` | `0.50` | Flat allocation fallback fraction |
| `kelly_frac` | `0.25` | Fractional-Kelly multiplier |
| `max_alloc` | `0.80` | Hard ceiling on allocation |
| `use_kelly` | `True` | Use Kelly sizing; False = flat |
| `use_calibration` | `True` | Wrap tree models in CalibratedClassifierCV |
| `max_daily_sl` | `2` | Daily SL cap |
| `cooldown_mins` | `20` | Global post-exit cooldown (min) |
| `sl_cooldown_mins` | `60` | Per-coin SL cooldown (min) |
| `max_dd_pct` | `0.08` | Drawdown circuit-breaker (8 %) |
| `cash_buffer` | `0.99` | Cash headroom multiplier |
| `cost_bps` | `35` | Estimated round-trip cost in basis points (0.35 %) |
| `min_ev` | `0.001` | Minimum EV after costs to enter (0.1 %) — relaxed default |
| `ev_gap` | `0.0001` | Required final_score gap to runner-up (return-fraction units) |
| `pred_return_min` | `-0.0005` | Regression veto: blocks only clearly bad predicted returns (−0.05 %) |
| `min_hold_minutes` | `15` | Suppress ordinary exits before this many minutes |
| `emergency_sl` | `0.030` | Allow early exit during min-hold if loss exceeds this (3.0 %) |
| `conservative_mode` | `False` | Legacy alias for `risk_profile=conservative` |
| `penalty_cooldown_losses` | `3` | Consecutive SL exits triggering penalty cooldown |
| `penalty_cooldown_hours` | `48` | Hours a symbol is blocked after penalty trigger |
| `momentum_override` | profile default | Enable momentum breakout override (`true`/`false`; auto-enabled for aggressive/ruthless) |
| `momentum_ret4_min` | `0.015` | Minimum 4-bar return for momentum override |
| `momentum_ret16_min` | `0.025` | Minimum 16-bar return for momentum override |
| `momentum_volume_min` | `2.0` | Minimum volume ratio for momentum override |
| `momentum_btc_rel_min` | `0.005` | Minimum BTC-relative outperformance for momentum override |
| `momentum_override_min_ev` | `-0.002` | Block momentum override if EV is below this threshold |

### Label / training parameters

| Constant / Parameter | Default | Description |
|----------------------|---------|-------------|
| `SCORE_MIN_FLOOR` | `0.15` | Floor for the base-rate-aware effective score threshold at runtime |
| `LABEL_TP` / `label_tp` | `0.012` | Take-profit fraction for training labels (+1.2 %) |
| `LABEL_SL` / `label_sl` | `0.010` | Stop-loss fraction for training labels (−1.0 %) |
| `LABEL_HORIZON_BARS` / `label_horizon_bars` | `72` | Timeout bars for training labels (≈6h at 5-min bars) |
| `LABEL_COST_BPS` / `label_cost_bps` | `30` | Round-trip cost for cost-aware label generation |

`LABEL_*` constants are defined in `models.py` and re-imported by `main.py`.
The looser barriers increase the positive rate from ~1–5 % to a more balanced
range, improving ensemble calibration without changing live execution behavior.

### Alignment Constraint

> **Note:** Training labels are intentionally decoupled from execution
> barriers.  `LABEL_TP`, `LABEL_SL`, `LABEL_HORIZON_BARS` govern what is
> labelled "1" at training time, while `TAKE_PROFIT`, `STOP_LOSS`, and
> `TIMEOUT_HOURS` govern when positions are closed at execution time.
> `LABEL_COST_BPS` adds a cost deduction to labels so the model learns trades
> that are profitable *after costs*, not merely raw price moves.

### Base-rate-aware confidence gate

At each decision tick the strategy derives two effective thresholds from the
most recent training `positive_rate` (`pr`):

| Threshold | Formula | Notes |
|-----------|---------|-------|
| `agree_thr` | `clip(2 × pr, 0.15, 0.55)` | Replaces the hard-coded `0.5` in the model agreement count |
| `score_min_eff` | `clip(max(SCORE_MIN_FLOOR, 3 × pr), SCORE_MIN_FLOOR, SCORE_MIN)` | Replaces the raw `SCORE_MIN` in the class_proba gate |

Both values are logged every decision tick in the `[diag]` line.

---

## Backtest Setup

```
Start date    : 2024-01-01
End date      : 2025-12-31
Initial cash  : $10,000 USD
Brokerage     : Kraken (CASH account)
Resolution    : 5-min subscription, 15-min decisions
Slippage      : ConstantSlippageModel(0.001)  — 0.1 % per side
Warm-up       : 90 calendar days
```

---

## How to Enable Kraken API Universe Fetch (Live Only)

By default Vox uses the static 20-pair list in `universe.KRAKEN_PAIRS`.

For live trading you can switch to a dynamic top-20 by volume:

```python
# In VoxAlgorithm.initialize(), replace:
self._symbols = add_universe(self)

# With:
from infra import fetch_kraken_top20_usd
live_pairs = fetch_kraken_top20_usd(self)
self._symbols = []
for ticker in live_pairs:
    try:
        sym = self.add_crypto(ticker, Resolution.MINUTE, Market.KRAKEN).symbol
        self._symbols.append(sym)
    except Exception:
        pass
```

> **Warning:** Do NOT use `fetch_kraken_top20_usd` in a backtest.  The Kraken
> API returns the current universe, which introduces look-ahead bias (you would
> be trading 2024 data with 2025 knowledge of which coins became top-20).

---

## Kill Switch

To immediately halt new position entries without stopping the algorithm:

**Activate** (from QC Research Notebook or ObjectStore API):
```python
qb.object_store.save("vox/kill_switch", "1")
```

**Deactivate**:
```python
qb.object_store.delete("vox/kill_switch")
```

The kill switch is checked on every decision tick.  Open positions continue to
be managed (exits still fire); only new entries are blocked.

---

## Module Reference

| File | Purpose |
|------|---------|
| `main.py` | `VoxAlgorithm` — QCAlgorithm entry point + all strategy constants |
| `config.py` | Risk profile constants + `setup_risk_profile()` |
| `momentum.py` | Momentum override and score helpers |
| `models.py` | Feature engineering, triple-barrier labeling, `VoxEnsemble`, training pipeline |
| `risk.py` | `RegimeFilter` (4h BTC gate), Kelly sizing + `min_alloc` floor, `RiskManager` (cooldowns, drawdown CB) |
| `infra.py` | Universe list + `add_universe()`, `OrderHelper`, `PartialFillTracker`, `PersistenceManager` |

### Exit reason tags

| Tag | Meaning |
|-----|---------|
| `EXIT_TP` | Take-profit threshold reached (normal mode) |
| `EXIT_SL` | Stop-loss threshold reached (also used for emergency SL) |
| `EXIT_TIMEOUT` | Position held for `timeout_hours` without a TP/SL trigger |
| `EXIT_TRAIL` | Trailing stop activated at `trail_after_tp` triggered at `trail_pct` pullback from high (ruthless runner mode) |

---

## Known Limitations

1. **Survivorship bias** — The static universe contains coins that are
   prominent *today*.  Coins that delisted between 2024-2025 are excluded.
2. **Single position** — Vox holds at most one coin at a time.  This
   concentrates risk and leaves capital idle when no clear winner exists.
3. **No shorting** — Only long entries are considered.  Bear markets are
   partially addressed by the regime filter but the strategy is still
   directionally long-only.
4. **Close-only ATR proxy** — `build_features` uses close-to-close differences
   as a true-range proxy when high/low data is unavailable in the feature
   buffer.  The main.py execution path uses proper OHLC data.
5. **ObjectStore append cost** — `log_trade` reads + rewrites the full JSONL
   file on every entry attempt.  This is acceptable for low-frequency trading
   but would need batching for high-frequency strategies.

---

## Synchronous Market-Order State-Machine Caveat

> **Critical implementation note for QuantConnect/LEAN backtests.**

In QuantConnect's default backtest environment, `market_order()` uses
`ImmediateFillModel`, which resolves the fill *synchronously* — meaning
`on_order_event()` fires **inside** the `market_order()` call, before it
returns.

This creates a subtle race condition in the entry state machine:

```
# WRONG — _pending_sym is None when on_order_event fires:
order             = self.market_order(sym, qty, tag="ENTRY")   # ← fills here
self._pending_sym = sym     # ← too late; on_order_event already checked this

# RIGHT — _pending_sym is set before market_order() is called:
self._pending_sym = sym
order             = self.market_order(sym, qty, tag="ENTRY")   # ← fires correctly
self._pending_oid = order.order_id   # order_id only available after the call
```

The same issue affects `_check_exit()`: if `market_order()` fills synchronously,
`on_order_event` may clear `self._pos_sym` before `_check_exit()` continues,
causing `AttributeError: 'NoneType' object has no attribute 'value'` on the
logging line that references `self._pos_sym.value`.

**Fixes applied:**

1. `_try_enter()` — `_pending_sym`, `_tp_dyn`, and `_sl_dyn` are set **before**
   `market_order()`.  Only `_pending_oid` is assigned after (it requires the
   returned order ticket).

2. `_check_exit()` — Local immutable copies (`sym`, `entry_px`, `entry_time`)
   are captured at the top of the function.  All subsequent code — including
   portfolio lookup, logging, and `market_order()` — uses these locals instead
   of `self._pos_sym`.  Logging happens **before** `market_order()`.

3. `_reconcile()` — Safety net upgraded to recover from a missed synchronous
   fill: if the pending order is `FILLED` and `_pos_sym` is still `None`, and
   the portfolio actually holds the coin, position state is reconstructed.
   When stale position state is cleared, `_exit_time` is updated and
   `_risk.record_exit()` is called so cooldown accounting is not bypassed.

4. `on_data()` — Fallback exit path added: when the held symbol has no bar on
   the current tick (illiquid pair) but the timeout has elapsed, `_check_exit()`
   is called with `self.securities[sym].price` as the price input.

---

## Soft-Voting Ensemble Design (Vox v2)

The `VoxEnsemble` (in `models.py`) implements a **weighted heterogeneous soft-voting**
classifier ensemble combined with a **regression ensemble** for expected return
prediction.

### Why weighted soft voting?

With positive rates of 1–5 % (typical for triple-barrier labeling on short
time horizons), hard majority voting nearly always produces the majority class.
Averaging probabilities lets a confident minority of models pull the weighted mean
above the adaptive threshold even when most models output sub-0.5 probabilities.

HistGradientBoostingClassifier receives the highest weight (0.35) because it is
typically the strongest tabular model in this ensemble and has good built-in
probability calibration.

### Confidence metrics used

| Metric | How computed | Used for |
|--------|-------------|----------|
| `class_proba` | VotingClassifier's weighted P(class=1) | Primary entry gate (`score_min_eff`) |
| `mean_proba` | Same as `class_proba` (backward-compat alias) | — |
| `std_proba` | Std-dev of per-model probabilities | Dispersion gate: high std → uncertain → skip |
| `n_agree` | Count of models with P ≥ `agree_thr` | Agreement gate: require ≥ `min_agree` |
| `pred_return` | Weighted regression ensemble prediction | Predicted-return gate + final_score blend |
| `return_dispersion` | Std-dev of regressor predictions | Informational; not gated |

The adaptive thresholds are:

| Threshold | Formula | Purpose |
|-----------|---------|---------|
| `agree_thr` | `clip(2 × positive_rate, 0.15, 0.55)` | Scales the "agreeing" bar proportionally to class frequency |
| `score_min_eff` | `clip(max(SCORE_MIN_FLOOR, 3 × positive_rate), SCORE_MIN_FLOOR, SCORE_MIN)` | Avoids rejecting every signal when positive_rate is very low |

### Calibration

Tree-based models (RF, ET) are wrapped in
`CalibratedClassifierCV(method="isotonic", cv=2)` to convert raw scores into
reliable probability estimates.  LogisticRegression is well-calibrated by design.
`HistGradientBoostingClassifier` has good built-in calibration and is used directly
to avoid cv-split issues on small datasets.  Calibration can be disabled via the
`use_calibration=False` parameter for faster iteration.

---

## Safe Crypto Exit Quantity (CashBook Fix)

### Problem

In Kraken cash-mode backtests (and live), `portfolio[sym].quantity` can be
**slightly larger** than the actual base-currency `CashBook` balance after
fees and rounding (e.g. `portfolio.quantity = 202.509` while
`cash_book["OP"].amount = 201.962`).  Selling the raw portfolio quantity
submits an order larger than the exchangeable balance and QuantConnect rejects
it with:

```
Order Error: Insufficient buying power to complete orders
Reason: Your portfolio holds 201.96198993 OP … but your Sell order is for
202.50936 OP.  Cash Modeling trading does not permit short holdings …
```

Without a fix, `on_order_event(INVALID)` clears `_exiting = False` and the
algo retries every minute with the *same invalid quantity*, spamming order
errors for the rest of the backtest.

### Fix — `OrderHelper.safe_crypto_sell_qty()` (Vox/infra.py)

```python
qty = OrderHelper.safe_crypto_sell_qty(
    self, sym, lot_size, min_order_size,
    exit_qty_buffer_lots=1,
)
```

Logic:
1. Read `portfolio[sym].quantity` (the tracked holding).
2. Determine the **base currency** via `OrderHelper.get_crypto_base_currency()`:
   - QuantConnect `SymbolProperties` does **not** expose `base_currency`, so
     that attribute is never accessed.
   - Instead, the resolver reads `symbol_properties.quote_currency` (which QC
     does expose); if `sym.value` ends with that quote string, the leading
     portion is the base (e.g. `OPUSD` with quote `USD` → `OP`).
   - Fallback: strip the longest matching quote suffix from `sym.value` in
     order `USDT`, `USDC`, `USD`, `EUR`, `GBP`, `BTC`, `ETH`.
   - Returns `None` when the base cannot be determined; in that case the
     CashBook lookup is skipped and the portfolio quantity is used directly.
3. Read the actual `portfolio.cash_book[base_ccy].amount` (the real balance).
4. Take `min(portfolio_qty, cash_qty)` — never sell more than actually held.
5. Floor to `lot_size`, then subtract `exit_qty_buffer_lots × lot_size` as an
   extra precision/fee margin (default: 1 lot).
6. If the result is zero or below `min_order_size`, return `0.0` (dust).

When `safe_crypto_sell_qty` returns `0.0`, `_check_exit()` clears position
state immediately (`_risk.record_exit()` is called so cooldowns apply) instead
of submitting an invalid order.

### INVALID exit retry throttling

`on_order_event(INVALID)` for an EXIT order now:
1. Increments `_exit_retry_count`.
2. Recomputes `safe_crypto_sell_qty`.
3. If `safe_qty == 0` **or** `retry_count >= MAX_EXIT_RETRY_COUNT (3)`,
   records the exit via `_risk.record_exit()` and clears all position state.
4. Otherwise clears `_exiting = False` to allow one more retry.

This eliminates the unbounded retry/spam loop described in the problem report.

The same `_safe_sell_qty()` helper is applied to the root `main.py`
`_check_exit()` (inline, since `main.py` does not import `infra`).

---

## Profit-Aware EV Ranking

### Why raw probability is insufficient

A trade with `mean_proba = 0.30` but a large ATR-derived TP (`tp = 0.05`)
and a tight SL (`sl = 0.015`) has **positive expected value** even though the
classifier "expects" a loss 70 % of the time.  Conversely, a trade with
`mean_proba = 0.55` but `tp < sl` can have *negative* edge.  Ranking purely
on `mean_proba` ignores this asymmetry.

### EV scoring formula

For each candidate passing the confidence gates, Vox v2 computes:

```text
class_proba  = weighted VotingClassifier P(class=1)
pred_return  = weighted regression ensemble prediction
ev           = class_proba × tp_use − (1 − class_proba) × sl_use − cost_fraction
final_score  = 0.6 × ev + 0.4 × pred_return   (if regressors trained)
             = ev × (1 − std_proba)             (fallback before regressor training)
```

where:

| Term | Description |
|------|-------------|
| `tp_use`, `sl_use` | Per-candidate ATR-based TP/SL (or fixed fallback) |
| `cost_fraction` | `COST_BPS × 1e-4` — estimated round-trip fee + slippage |
| `pred_return` | Regression ensemble predicted net return (0.0 if not trained) |

All EV-related values (`ev`, `final_score`) are **return fractions**, not probabilities.
A value of `0.001` means a 0.1 % expected return; `0.01` means 1 %.  This is important
when setting thresholds: `MIN_EV = 0.004` requires 0.4 % expected return after costs.

Candidates are ranked by `final_score` and only candidates with
`ev > min_ev` (and `pred_return >= pred_return_min` when regressors are trained)
are considered.

### EV gap selectivity

After ranking, Vox requires the top candidate to lead the second-best by at
least `ev_gap` in `final_score`.  This controls **selectivity**, not trade
validity.

> **Important:** `ev_gap` is in **return-fraction units** (same as `final_score`),
> NOT probability units.  Using the probability gap threshold (`score_gap = 0.02`)
> as an EV gap would require a 2 percentage-point EV advantage, blocking nearly
> all trades since typical EV scores are in the `0.001–0.02` range.

### Parameters

| Parameter / Constant | Default | Units | Description |
|----------------------|---------|-------|-------------|
| `COST_BPS` / `cost_bps` | `35` | basis points | Estimated round-trip fee+slippage (0.35 %) |
| `MIN_EV` / `min_ev` | `0.001` | return fraction | Minimum EV after costs to enter (0.1 %) — relaxed default |
| `EV_GAP` / `ev_gap` | `0.0001` | return fraction | Required score lead of top over second-best (0.01 %) |
| `PRED_RETURN_MIN` / `pred_return_min` | `-0.0005` | return fraction | Regression veto: blocks only clearly bad predicted returns (−0.05 %) |
| `SCORE_GAP` / `score_gap` | `0.02` | probability (0–1) | Probability gap between top and runner-up — **not** used for EV comparisons |
| `EXIT_QTY_BUFFER_LOTS` / `exit_qty_buffer_lots` | `1` | lots | Safety lot buffer on exits |

#### `pred_return_min` tuning guidance

`PRED_RETURN_MIN = -0.0005` means the regressor acts as a **soft veto** — it
only blocks candidates where the regression ensemble predicts a clearly negative
return (< −0.05 %).  This is intentional: early in the backtest, regression
targets are sparse and predictions typically lie in the range `−0.002` to
`+0.001`.  Requiring a positive predicted return at this stage would block
*every* candidate.

Use `pred_return_min = 0.003` (or higher) via `conservative_mode=True` when you
want strict research validation and the regressors have sufficient training data
(e.g. after several months of live data with realized returns logged).

### Diagnostics

Every time candidates pass all gates the following is logged:

```
[diag] candidates=3 top=OPUSD final_score=0.00412 ev_score=0.00395
       pred_ret=0.00441 gap=0.00201 class_proba=0.287 std_proba=0.081
       n_agree=3 tp=0.0245 sl=0.0147
```

When no candidate passes, the diagnostic summary includes `best_ev` (the best
EV-after-costs seen among candidates that passed the preliminary gates) so you
can understand whether the EV gate or the pred_return gate is the binding
constraint:

```
[diag] eval=18 pass_disp=12 pass_agree=10 pass_score=3 pass_ev=0 pass_pred_ret=0
       best_proba=0.183 best_agree=3 best_disp=0.073 best_pred_ret=-0.00203
       best_ev=-0.00120 (thresh: score>=0.150 agree>=2 disp<=0.22
       ev>0.00100 pred_ret>=-0.00050 cost=0.0035)
```

**Diagnostic throttling** — routine skip messages are intentionally rate-limited
to avoid QuantConnect's 100 KB log cap during multi-year backtests:

| Log type | Throttle |
|----------|---------|
| No-candidate summary (`[diag]`) | At most once every **6 hours** (`DIAG_INTERVAL_HOURS = 6`) |
| Routine skip (EV gap, regime, risk block) | At most once every **6 hours** (`SKIP_DIAG_INTERVAL_SECS = 21600`) |
| Entry fills, exit fills, errors | **Unthrottled** — always logged |
| Retrain summary | **Unthrottled** — always logged |

---

## Tuning Guide and Overfitting Warnings

### Metrics to validate

Before declaring Vox "working", validate the following metrics on **out-of-sample**
data (never tune to in-sample backtest results):

| Metric | Target | How to compute |
|--------|--------|----------------|
| Total return | > 0 on OOS period | QuantConnect backtest |
| Max drawdown | < 15 % | QuantConnect backtest |
| Sharpe/Sortino | > 0.5 | QuantConnect backtest |
| Profit factor | > 1.2 | `gross_wins / gross_losses` |
| Win rate | > 45 % | `wins / total_trades` |
| Average win/loss ratio | > 1.0 | `avg_win / abs(avg_loss)` |
| Fee drag | < 3 % of equity over period | `total_fees / starting_equity` |
| Turnover | < 20 % portfolio/day | QuantConnect backtest |
| Calibration / Brier score | Brier < 0.25 | `sklearn.metrics.brier_score_loss` |
| Realized EV by predicted EV bucket | Positive slope | Use `PersistenceManager` logs |

### Overfitting warnings

- **Do not tune parameters to a single backtest period.** Use at least 3 distinct
  out-of-sample windows (e.g. 2023, 2024, 2025 separately).
- **Tighter gates do not always mean better OOS performance.** Very tight EV
  thresholds may select only 5–10 trades in a year — not enough for statistical
  significance.
- **Model weights are reasonable defaults, not optimized values.** Do not
  auto-optimize the classifier/regressor weights to a specific backtest period.
- **Cost estimates should be conservative.** Kraken fees are typically 0.10–0.26 %
  per side; `COST_BPS = 50` (0.50 % round-trip) already includes slippage margin.
  Lowering this aggressively may cause the model to learn trades that look
  profitable in simulation but lose money live.
- **The penalty cooldown is not a symbol blacklist.** It is a dynamic cooling
  period that resets after 48 hours.  Do not tune `penalty_cooldown_losses` to
  match a specific set of bad symbols in one backtest.

### Recommended parameter validation sequence

1. Run with default v2 parameters; observe diagnostics (`[diag]` lines).
2. If trade count is < 10/month: lower `min_ev`, `pred_return_min`, or relax
   `min_agree` / `max_dispersion`.
3. If trade count is > 50/month with negative expectancy: raise `min_ev`,
   `score_min`, `min_hold_minutes`.
4. Enable `conservative_mode=True` for a simple "tighter" configuration.
5. Compare OOS win rate, profit factor, and fee drag across 3 separate years.
6. Only adjust parameters where the OOS distribution clearly supports the change.
flooding logs.

### Overfitting warning

> **Maximising backtest profit can overfit.**  The EV scoring parameters
> (`COST_BPS`, `MIN_EV`, `EV_GAP`, `SCORE_GAP`) and the ensemble hyper-parameters
> should be validated on **out-of-sample** data or via walk-forward testing.
> Conservative defaults are provided; tighten them only when out-of-sample
> metrics support it.

---

## Tuning Guidance and Validation Metrics

### Recommended validation metrics

| Metric | Target / Notes |
|--------|---------------|
| **Total return** | Primary objective; compare to buy-and-hold BTC |
| **Maximum drawdown** | Keep below 15–20 % for comfort |
| **Sharpe ratio** | > 1.0 is reasonable for crypto daily; annualised |
| **Sortino ratio** | Penalises downside vol only; more relevant than Sharpe for skewed crypto returns |
| **Profit factor** | Gross profit / gross loss; > 1.5 is healthy |
| **Trade count / turnover** | Low trade count → unreliable stats; aim for > 30 trades per test window |
| **Fees/slippage drag** | Compare net vs gross returns; if drag > 20 % of gross, reduce `COST_BPS` default |
| **Precision @ top-K** | Fraction of entries that hit TP; should beat the positive_rate base-rate |
| **Brier score / calibration** | If `mean_proba` is well-calibrated, 0.3 should mean ~30 % win rate |

### Walk-forward validation

1. Split the data: e.g. train on 2024, test on 2025.
2. Re-train the ensemble on the train window only (set `VOX_ENABLE_CV=True`
   for fold-level diagnostics).
3. Run the full backtest on the test window with the pre-trained model
   (load via ObjectStore or inline).
4. Only accept parameter changes that improve **test-window** metrics.

### Parameter tuning order

1. `LABEL_TP`, `LABEL_SL`, `LABEL_HORIZON_BARS` — control the training
   target.  Looser barriers increase positive rate but may not align with
   live execution TP/SL.
2. `SCORE_MIN`, `MIN_AGREE`, `MAX_DISPERSION` — confidence gates; loosen if
   trade count is very low; tighten if precision is poor.
3. `COST_BPS`, `MIN_EV`, `EV_GAP` — EV filter; increase `COST_BPS` to account for
   actual Kraken fees (0.16–0.26 % maker/taker per side = 32–52 bps round trip).
   `MIN_EV` and `EV_GAP` are **return fractions** — see the EV Ranking section for
   units and tuning ranges.
4. `ATR_TP_MULT`, `ATR_SL_MULT` — trade geometry; a higher ratio improves
   Kelly edge but reduces win rate.
5. `KELLY_FRAC`, `MAX_ALLOC` — position sizing; use quarter-Kelly or lower.

---

## Future Work

- **Meta-labeling** — Train a secondary binary classifier on the primary
  model's signals to filter false positives (López de Prado chapter 4).
- **Top-N positions** — Extend the state machine to hold up to N concurrent
  positions with proportional Kelly sizing across the portfolio.
- **Dynamic universe rotation** — Replace the static 20-pair list with a
  rolling universe ranked by liquidity and volatility, refreshed weekly.
- **Online learning** — Use incremental model updates (e.g. `partial_fit` on
  PassiveAggressiveClassifier) to adapt to regime shifts without a full weekly
  retrain.
- **Calibration evaluation** — Per-bucket realized win rate vs. predicted
  `class_proba` using the trade logs from `PersistenceManager`.  If buckets
  diverge significantly, re-calibrate or adjust the `score_min` gate.
- **Regime-aware regressors** — Train separate regression ensembles for
  bull/bear/sideways regimes identified by the BTC regime filter.

---

## Vox v5 — Model Attribution & Observability

### Stable Model IDs

Each classifier in the ensemble has a stable string ID used in logs and journal records:

| ID       | Model                                   | Notes            |
|----------|-----------------------------------------|------------------|
| `lr`     | LogisticRegression                      | linear baseline  |
| `hgbc`   | HistGradientBoostingClassifier          | primary booster  |
| `et`     | ExtraTreesClassifier (calibrated)       | diverse trees    |
| `rf`     | RandomForestClassifier (calibrated)     | bagged trees     |
| `lgbm`   | LGBMClassifier (optional)               | needs lightgbm   |
| `xgb`    | XGBClassifier (optional)                | needs xgboost    |
| `catboost` | CatBoostClassifier (optional)         | never required   |

Enabled IDs are logged at startup:
```
[model_registry] enabled=lr(w=1.0),hgbc(w=1.0),et(w=1.0),rf(w=1.0)
```

### Per-Model Vote Attribution

Each entry prediction now includes per-model probabilities:
```python
conf = {
    "class_proba":   0.64,   # VotingClassifier result (unchanged)
    "weighted_mean": 0.63,   # user-weight-adjusted mean
    "std_proba":     0.05,
    "n_agree":       3,
    "vote_threshold": 0.5,
    "per_model": {
        "lr":   0.55,
        "hgbc": 0.70,
        "et":   0.67,
        "rf":   0.61,
    },
    "votes": { ... }  # alias for per_model
}
```

Enable compact vote logging per trade (disabled by default — protects QC log cap):
```python
LOG_MODEL_VOTES = True   # in config.py, or via QC parameter log_model_votes=true
```

Example vote log line:
```
[vote] ADAUSD mean=0.64 std=0.05 agree=3/4 mode=pump votes=lr:0.55,hgbc:0.70,et:0.67,rf:0.61
```

### Optional Weighted Ensemble

Adjust per-model influence via `config.py`:
```python
MODEL_WEIGHT_LR   = 1.0   # reduce to 0.5 to halve LR's influence
MODEL_WEIGHT_HGBC = 1.5   # increase to boost HGBC
MODEL_WEIGHT_ET   = 0.0   # set 0 to exclude a model
MODEL_WEIGHT_RF   = 1.0
```

Default weights (all 1.0) preserve current behaviour.  The `weighted_mean`
field in confidence dicts reflects user weights; `class_proba` is unchanged.

### Trade Journal

Every trade persistence log record now includes:
- **Entry fields**: `model_votes`, `market_mode`, `confirm` path, ensemble metrics.
- **Exit fields**: `max_return_seen`, `trail_active`, `breakeven_active`, `model_votes`.

Records are emitted to `PersistenceManager` at both entry and exit and can be
retrieved from the ObjectStore JSON log for offline analysis.

**Per-model accuracy analysis** (offline, from journal records):
```python
from trade_journal import TradeJournal
j = TradeJournal()
j.load_json(your_json_log_string)
attr = j.compute_model_attribution()
# attr["hgbc"]["win_rate_when_yes"] — win rate when HGBC voted yes
# attr["lr"]["avg_return_when_yes"]  — avg return when LR voted yes
```

⚠️ **Warning**: with only 10–20 trades the per-model statistics are extremely
noisy.  Run attribution over multiple windows to get reliable signal:
- 2023, 2024, 2025, 2026 YTD
- Bull periods, chop periods, selloffs

### Good-Market-Mode Relaxation (ruthless only)

In `pump` / `risk_on_trend` modes, confirmation thresholds are slightly relaxed
to increase sample size without returning to chop overtrading:

| Parameter | Strict (default) | Relaxed (pump/trend) |
|-----------|-----------------|----------------------|
| `ruthless_confirm_ev_min` | 0.006 | 0.004 |
| `ruthless_confirm_volr_min` | 1.5 | 1.3 |
| `meta_min_proba` (future) | 0.55 | 0.52 |

Configurable in `config.py`:
```python
RUTHLESS_GOOD_MODE_RELAXATION     = True    # master switch
RUTHLESS_GOOD_MODE_MIN_EV         = 0.004
RUTHLESS_GOOD_MODE_VOLUME_MIN     = 1.3
```

Confirm reason is tagged `strong_ml_relax` / `trend_momentum_relax` when
relaxation is active, making it easy to filter the effect in backtest analysis.

### Exit Diagnostics

The ruthless exit log now includes entry/fill prices and high-water mark:
```
[exit_diag] SOLUSD  tag=EXIT_SL  entry=268.23000  fill=268.34000
            ret=+0.04%  max=+0.04%  sl=0.0300  elapsed_min=12.5
            (warn:ret>=0_tagged_sl)
```

This clarifies suspicious exits: e.g. an `EXIT_SL` on a positive fill means
the stop was a breakeven stop or fee-adjusted loss, not a directional loss.

### Entry Limit Order Observability

Startup log shows current limit-order configuration:
```
[vox] RUTHLESS entry_limit_orders: use=True offset=0.001 ttl_min=3
```

### QuantConnect File-Size Rule

Every individual Python code file must remain **under 63,000 characters**.
Check with `wc -c Vox/*.py` before deploying.  Current sizes:
- `main.py` — ~62,600 chars (critical; monitor closely)
- `models.py` — ~46,000 chars
- `config.py` — ~27,000 chars
- `execution.py` — ~11,000 chars
- `trade_journal.py`, `diagnostics.py`, `model_registry.py`, `tuning.py` — < 15,000 each

---

## Ruthless Profit-Voting Mode

### Overview

`RUTHLESS_PROFIT_VOTING_MODE = True` (set in `config.py`) activates a distinct
vote-score/ranking-based entry gate for the `ruthless` risk profile.  Balanced,
conservative, and aggressive profiles are **not affected**.

In profit-voting mode, candidates must pass:
1. The standard ruthless confirmation gate (strong_ml / trend_momentum / market_mode).
2. An additional **vote-score gate** requiring a minimum `vote_yes_fraction`
   and `top3_mean` across active-role models.
3. For `chop` market mode: a stricter **supermajority gate**.

Startup log shows whether profit-voting is active:
```
[profile] risk_profile=ruthless profit_voting=True
[profile] chop_rule=supermajority_only vote_threshold=0.55 yes_frac_min=0.50 top3_mean_min=0.62
```

### Vote-Score Fields

Each candidate evaluation produces:

| Field | Description |
|-------|-------------|
| `active_model_count` | Number of active-role models |
| `vote_yes_fraction` | Fraction of active models with P ≥ `vote_threshold` (0.55) |
| `top3_mean` | Mean of the top-3 active model probabilities |
| `vote_score` | Weighted composite: `0.4×active_mean + 0.3×vote_yes_fraction + 0.3×top3_mean` |

Entry requires (trend / pump): `vote_yes_fraction >= 0.50` AND `top3_mean >= 0.62`.

Entry requires (chop): `vote_yes_fraction >= 0.70` AND `top3_mean >= 0.75`
AND `pred_return >= 0.01` AND `ev_score >= 0.01`.

### Active vs Shadow vs Diagnostic Model Roles

| Role | Affects trading? | Persisted/logged? | Purpose |
|------|-----------------|-------------------|---------|
| `active` | **Yes** | Yes | Ensemble decision (trading confidence) |
| `shadow` | No | Yes | Safe test-bench for future models |
| `diagnostic` | No | Yes | Veto/risk/debug signals |
| `disabled` | No | No | Off entirely |

Backward-compatible fields (`class_proba`, `std_proba`, `n_agree`) **always map
to active-role values only**, so non-ruthless code paths remain unaffected.

### Default Model Roles

| Model | Default role | Notes |
|-------|-------------|-------|
| `rf` | active | Core ensemble |
| `et` | active | Core ensemble |
| `hgbc` | active | Core ensemble |
| `gnb` | **diagnostic** | Always-bullish (vote_gnb=1.0); never inflates n_agree |
| `lr` | **diagnostic** | Always-bearish (0.006–0.023); used as veto signal only |
| `lgbm` | shadow | Optional; only if USE_LIGHTGBM=True |
| `xgb` | shadow | Optional; only if USE_XGBOOST=True |

**GNB quarantine rationale:** live data showed `vote_gnb=1.0` on every trade.
This would inflate `n_agree` and `class_proba` regardless of real signal quality.
It is now `MODEL_ROLE_GNB = "diagnostic"` with `MODEL_WEIGHT_GNB = 0.0`.

**LR quarantine rationale:** live data showed `vote_lr ≈ 0.006–0.023` on every
trade.  As an always-bearish model, it was suppressing `active_mean` without
adding discrimination.  It is `MODEL_ROLE_LR = "diagnostic"` with
`MODEL_WEIGHT_LR = 0.0` by default.

### Extended Shadow Lab Models

The `shadow_lab.py` module adds additional shadow and diagnostic models when
`ENABLE_SHADOW_LAB_EXTENDED = True`:

**Buy-probability shadow models** (role: `shadow`):
- `gbc` — GradientBoostingClassifier (compact, QC-friendly)
- `ada` — AdaBoostClassifier (compact, QC-friendly)

**Regime/risk diagnostic models** (role: `diagnostic`):
- `markov_regime` — Logistic-regression-based regime diagnostic
  (uptrend / downtrend / chop detection from ret_4, ret_16, vol_ratio)
- `hmm_regime` — Optional GaussianHMM regime model (requires `hmmlearn`);
  falls back to `markov_regime` if unavailable
- `kmeans_regime` — KMeans-based regime clustering (4 clusters)
- `isoforest_risk` — IsolationForest anomaly score (unusual market setups)

These models output a probability-like score but are **never used as direct buy
votes**.  Their outputs are persisted under `shadow_votes` and `diagnostic_votes`
in the trade journal for post-hoc attribution.

### Candidate Journal

`CandidateJournal` records the top-N candidates (configurable) from each decision
cycle — including **skipped candidates** — for post-hoc analysis.

Enabled by `PERSIST_CANDIDATE_JOURNAL = True` (default: `True`).

Each record contains:
```json
{
  "time": "2025-03-01 12:00:00",
  "symbol": "SOLUSD",
  "rank": 1,
  "selected": true,
  "reject_reason": null,
  "market_mode": "risk_on_trend",
  "vote_score": 0.712,
  "active_mean": 0.68,
  "active_std": 0.05,
  "active_n_agree": 3,
  "vote_yes_fraction": 0.67,
  "top3_mean": 0.75,
  "pred_return": 0.025,
  "ev_score": 0.0082,
  "active_votes": {"rf": 0.72, "et": 0.68, "hgbc": 0.71},
  "shadow_votes": {"gbc": 0.65, "ada": 0.60},
  "diagnostic_votes": {"gnb": 1.0, "lr": 0.01, "markov_regime": 0.70},
  "entry_path": "ml"
}
```

Access records from a Research notebook:
```python
from QuantConnect.Research import QuantBook
qb = QuantBook()
import json
if qb.object_store.contains_key("vox/trade_log.jsonl"):
    lines = qb.object_store.read("vox/trade_log.jsonl").splitlines()
    # Filter entry_attempt records
    entries = [json.loads(l) for l in lines if '"event": "entry_attempt"' in l]
```

### Ruthless Payoff Floor

To prevent the "tiny scalp" failure mode (`avg_win = +0.24%`), ruthless mode
enforces a minimum TP floor:

```python
RUTHLESS_MIN_TP = 0.04   # do not target wins < +4%
```

The ATR-based TP may be larger; this is a floor, not a ceiling.  Average wins
should be in the +3–10% range for ruthless mode.

---

## Clearing ObjectStore Logs Before Comparing Windows

**Always clear old logs before switching between date windows or profiles.**
Stale logs from a prior run mix with new results and produce confusing stats.

Run this in a QuantConnect Research notebook:
```python
from QuantConnect.Research import QuantBook
qb = QuantBook()

for key in ["vox/trade_log.jsonl"]:
    if qb.object_store.contains_key(key):
        qb.object_store.delete(key)
        print("deleted", key)

# Optionally delete model for fresh retrain (useful when comparing profiles):
# if qb.object_store.contains_key("vox/model.pkl"):
#     qb.object_store.delete("vox/model.pkl")
```

**Never compare balanced vs ruthless results in the same ObjectStore session.**
Each test window needs a clean log.

---

## Reproducible Tuning Workflow

1. Clear ObjectStore logs (see above).
2. Set `risk_profile = ruthless` (or `balanced`) in QC parameters.
3. Run backtest for a defined date window (e.g. Jan 2025 – May 2026).
4. Export trade log from Research: filter `event = entry_attempt`.
5. Inspect `vote_score`, `vote_yes_fraction`, `top3_mean` on taken trades.
6. Inspect skipped-candidate journal to see what the model was ranking.
7. Check `shadow_votes` to see if `gbc` / `ada` would have agreed.
8. **Do not tune thresholds** based on < 30 trades — the sample is too small.

### How to Promote a Shadow Model

Before promoting any shadow model to active role:
- Shadow model must have ≥ 30 yes-votes in the journal across multiple windows.
- `avg_return_when_yes > 0` across multiple periods.
- `win_rate_when_yes > 0.50` across multiple periods.
- Not degenerate (not always-bullish or always-bearish).
- Not redundant with existing active models (check correlation of votes).

To promote:
1. Add model ID to the `RUTHLESS_ACTIVE_MODELS` list in `config.py`.
2. Change its `MODEL_ROLE_xxx` constant from `"shadow"` to `"active"`.
3. Set a non-zero `MODEL_WEIGHT_xxx` (start with 0.75 to weight it below peers).
4. Retrain model (delete `vox/model.pkl`, run fresh backtest).
5. Validate on a held-out window before enabling for live trading.

---

## QuantConnect File-Size Constraint

Every individual Python file must remain **under 63,000 characters**.
Check with:
```bash
wc -c Vox/*.py
```

If a file is over the limit, move helper logic to a new module.  Current files:
- `main.py` — ~61,800 chars
- `models.py` — ~62,700 chars
- `config.py` — ~37,000 chars
- `execution.py` — ~12,900 chars
- `shadow_lab.py` — ~12,900 chars
- `model_registry.py`, `diagnostics.py`, `market_mode.py`, `meta_model.py` — < 13,000 each
- `candidate_journal.py`, `profit_voting.py` — < 8,000 each

---

## Ruthless V2 — Aggressive Multi-Position Opportunity Engine

### Overview

Ruthless V2 is a **separate, opt-in** aggressive opportunity engine built on
top of the existing ruthless V1 profile.  V1 behavior is completely preserved
unless V2 is explicitly activated.

**Activation** (either method works):
```
risk_profile = ruthless_v2
```
or:
```
risk_profile = ruthless
ruthless_v2_mode = true
```

**Startup log confirms V2 is active:**
```
[profile] risk_profile=ruthless v2=True max_positions=4 active_models=rf,et,hgbc_l2,lgbm_bal,gbc,ada
[v2_weights] ada=0.700 et=0.800 gbc=0.850 hgbc_l2=1.100 lgbm_bal=1.000 rf=1.350
```

### V2 vs V1 Differences

| Feature | V1 (ruthless) | V2 (ruthless_v2) |
|---------|---------------|------------------|
| Max concurrent positions | 1 | 4 |
| Max entries/day | 3 (informational) | 8 (enforced) |
| Active voter pool | rf, et, hgbc_l2, lgbm_bal | rf, et, hgbc_l2, lgbm_bal, gbc, ada |
| Diagnostic models | gnb, lr, lr_bal | gnb, lr, lr_bal, cal_et, cal_rf |
| Dynamic voter weighting | No | Yes (contextual bandit) |
| Multi-horizon lanes | No | Yes (scalp/continuation/runner) |
| Cross-sectional ranking | No | Yes (top-N by v2_opportunity_score) |
| Pump continuation/exhaustion | No | Yes |
| Split exits (partial TP + runner) | No | Yes |
| True vote audit | No | Yes (trade_vote_audit.jsonl) |

### V2 Config Parameters

```python
# ruthless_v2.py defaults (importable from ruthless_v2)
RUTHLESS_V2_MAX_CONCURRENT_POSITIONS   = 4
RUTHLESS_V2_MAX_NEW_ENTRIES_PER_DAY    = 8
RUTHLESS_V2_MAX_ENTRIES_PER_SYMBOL_PER_DAY = 2
RUTHLESS_V2_MAX_SYMBOL_ALLOCATION      = 0.30
RUTHLESS_V2_MIN_SYMBOL_ALLOCATION      = 0.08
RUTHLESS_V2_MAX_TOTAL_EXPOSURE         = 1.25
RUTHLESS_V2_REENTRY_COOLDOWN_MIN       = 30

# Partial TP / split-exit
RUTHLESS_V2_PARTIAL_TP_ENABLED         = True
RUTHLESS_V2_PARTIAL_TP_FRACTION        = 0.50
RUTHLESS_V2_SCALP_TP                   = 0.015   # scalp lane TP target
RUTHLESS_V2_CONTINUATION_TP            = 0.04    # continuation lane TP
RUTHLESS_V2_RUNNER_INITIAL_TP          = 0.06    # runner initial partial TP
RUTHLESS_V2_RUNNER_TRAIL_PCT           = 0.04    # runner remainder trail
RUTHLESS_V2_PUMP_RUNNER_TRAIL_PCT      = 0.06    # wider trail in pump mode
```

### V2 Active Voter Pool

V2 uses a more aggressive active voter pool. Conservative/degenerate models
are **diagnostic-only** and never affect trading:

| Model | V2 Role | Base Weight | Notes |
|-------|---------|------------|-------|
| `rf` | active | 1.35 | Core tree model |
| `hgbc_l2` | active | 1.10 | HistGradientBoosting with L2 |
| `lgbm_bal` | active | 1.00 | LightGBM balanced (if available) |
| `gbc` | active | 0.85 | GradientBoostingClassifier |
| `et` | active | 0.80 | ExtraTreesClassifier |
| `ada` | active | 0.70 | AdaBoost |
| `xgb_bal` | optional active | 0.75 | XGBoost (if available) |
| `catboost_bal` | optional active | 0.75 | CatBoost (if available) |
| `gnb` | diagnostic | — | Always-bullish; degenerate |
| `lr` / `lr_bal` | diagnostic | — | Always-bearish; unreliable |
| `cal_et` / `cal_rf` | diagnostic | — | Calibration overlays only |

### V2 Opportunity Score Formula

```python
ruthless_v2_score = (
    0.25 * dynamic_vote_score       # weighted model votes
    + 0.20 * continuation_score     # 2-8h momentum lane
    + 0.20 * runner_score           # 12-48h breakout lane
    + 0.15 * breakout_score         # price/volume breakout signal
    + 0.10 * volume_expansion_score # volume surge above rolling avg
    + 0.10 * regime_score           # BTC market regime quality
    - cost_penalty                  # proportional to cost_bps
    - exhaustion_penalty            # pump exhaustion (late-chase penalty)
)
```

### Multi-Horizon Lanes

Each V2 entry is assigned to a lane based on the dominant score:

| Lane | Target Hold | TP Target | Trail Stop |
|------|-------------|-----------|-----------|
| scalp | 30–90 min | +1.5% | — |
| continuation | 2–8 h | +4.0% | — |
| runner | 12–48 h | +6.0% initial | 4% (6% in pump mode) |

Split exits: sell `PARTIAL_TP_FRACTION` (default 50%) at initial TP, hold
remainder as runner with trailing stop.

### Dynamic Voter Weighting

After each selected trade closes, voter weights are updated using an
exponential-decay payoff tracker:

```python
# Reward: model voted yes → trade won
# Penalty: model voted yes → trade lost
effective_weight = base_weight * perf_score
# perf_score ∈ [MIN_WEIGHT_MULTIPLIER, MAX_WEIGHT_MULTIPLIER] = [0.25, 2.0]
# Minimum 5 observations before weight changes (min_obs guard)
```

### Pump Continuation / Exhaustion System

V2 tracks same-symbol activity to distinguish real pump continuations from
late-chase exhaustion:

- **Continuation**: High volume, intact momentum, strong RF/hgbc_l2 — re-entry allowed
- **Exhaustion**: Repeated entries in 2h, many trail wins, fading volume — re-entry blocked or allocation reduced
- `exhaustion_override_allowed()`: Returns True if continuation is strong enough (≥0.55) to override the `reentry_cooldown` block

---

## Trade Vote Audit — Research Analysis

**File**: `vox/trade_vote_audit.jsonl` (ObjectStore)

Unlike `trade_log.jsonl` which logs all *attempts*, the vote audit records only
**confirmed filled selected trades**, paired entry + exit, with a unique `trade_id`.
Use this for accurate model accuracy analysis — no duplicate attempt pollution.

### Copy-Paste Research Snippets

#### Load and inspect selected trades

```python
import json
import pandas as pd

# Read from ObjectStore in Research
audit_raw = qb.object_store.read("vox/trade_vote_audit.jsonl")
records = [json.loads(l) for l in audit_raw.split("\n") if l.strip()]

entries = pd.DataFrame([r for r in records if r.get("entry_type") == "entry"])
exits   = pd.DataFrame([r for r in records if r.get("entry_type") == "exit"])

print(f"Selected entries: {len(entries)}, Closed trades: {len(exits)}")
print(entries[["trade_id","symbol","entry_time","entry_price","allocation",
               "risk_profile","market_mode","confirm","vote_score","lane_selected"]].to_string())
```

#### Model accuracy by closed selected trade

```python
# Pair entries with exits
paired = exits.merge(entries[["trade_id","active_votes","shadow_votes",
                               "diagnostic_votes","effective_model_weights"]],
                     on="trade_id", how="left")

# Expand per-model votes
model_rows = []
for _, row in paired.iterrows():
    ret   = row.get("realized_return")
    winner = row.get("winner")
    for vote_col in ["active_votes", "shadow_votes", "diagnostic_votes"]:
        votes = row.get(vote_col)
        if not isinstance(votes, dict):
            continue
        for model_id, proba in votes.items():
            model_rows.append({
                "trade_id":    row["trade_id"],
                "model_id":    model_id,
                "vote_source": vote_col.replace("_votes",""),
                "proba":       float(proba),
                "voted_yes":   float(proba) >= 0.50,
                "realized_return": ret,
                "winner":      winner,
            })

mdf = pd.DataFrame(model_rows)
yes_only = mdf[mdf["voted_yes"]]
attr = yes_only.groupby("model_id").agg(
    yes_count      = ("voted_yes","sum"),
    win_rate       = ("winner","mean"),
    avg_return     = ("realized_return","mean"),
).sort_values("win_rate", ascending=False)
print(attr.to_string())
```

#### Symbol performance summary

```python
sym_perf = exits.groupby("symbol").agg(
    trades        = ("trade_id", "count"),
    wins          = ("winner",   "sum"),
    avg_return    = ("realized_return", "mean"),
    total_pnl     = ("realized_pnl",    "sum"),
    avg_hold_min  = ("hold_minutes",    "mean"),
).assign(win_rate=lambda x: x["wins"] / x["trades"])
print(sym_perf.sort_values("total_pnl", ascending=False).to_string())
```

#### Lane performance (scalp / continuation / runner)

```python
# Join lane info from entries onto exits
lane_df = exits.merge(entries[["trade_id","lane_selected","scalp_score",
                                "continuation_score","runner_score"]],
                      on="trade_id", how="left")
lane_perf = lane_df.groupby("lane_selected").agg(
    trades    = ("trade_id", "count"),
    wins      = ("winner",   "sum"),
    avg_ret   = ("realized_return", "mean"),
    avg_hold  = ("hold_minutes",    "mean"),
).assign(win_rate=lambda x: x["wins"] / x["trades"])
print(lane_perf.to_string())
```

#### Threshold analysis (find optimal vote_score cutoff)

```python
full = exits.merge(entries[["trade_id","vote_score","dynamic_vote_score",
                              "v2_opportunity_score","meta_entry_score"]],
                   on="trade_id", how="left")

for col in ["vote_score","v2_opportunity_score"]:
    for thr in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        subset = full[full[col] >= thr]
        if len(subset) == 0:
            continue
        print(f"{col}>={thr:.2f}: n={len(subset)} wr={subset['winner'].mean():.2f} "
              f"avg_ret={subset['realized_return'].mean():.4f}")
```

#### Runner capture analysis

```python
runner_df = exits.merge(
    entries[["trade_id","runner_score","lane_selected","pump_continuation_score"]],
    on="trade_id", how="left"
)
print(runner_df[runner_df["lane_selected"]=="runner"][
    ["symbol","entry_time","exit_time","realized_return","exit_reason",
     "runner_score","pump_continuation_score","hold_minutes"]
].to_string())
```

#### Same-symbol pump / exhaustion stats

```python
sym_history = exits.merge(
    entries[["trade_id","symbol","pump_continuation_score","pump_exhaustion_score",
             "exhaustion_score"]],
    on="trade_id", how="left"
)
pump_agg = sym_history.groupby("symbol").agg(
    trades       = ("trade_id","count"),
    avg_exh      = ("pump_exhaustion_score","mean"),
    avg_cont     = ("pump_continuation_score","mean"),
    avg_ret      = ("realized_return","mean"),
    win_rate     = ("winner","mean"),
).sort_values("avg_ret", ascending=False)
print(pump_agg.to_string())
```

---

## Module Size Tracking

Every Python file must remain under 63,000 characters.
Check before deploying to QC Cloud:

```bash
wc -c Vox/*.py | sort -rn | head -15
```

Current approximate sizes (after Ruthless V2 addition):
- `main.py` — ~62,600 chars (critical; monitor closely)
- `models.py` — ~62,700 chars (critical)
- `config.py` — ~42,000 chars
- `ruthless_v2.py` — ~33,500 chars (new)
- `trade_vote_audit.py` — ~22,000 chars (new)
- `execution.py` — ~14,000 chars
- `shadow_lab.py` — ~13,600 chars
- `profit_voting.py` — ~14,200 chars
- `trade_journal.py` — ~14,000 chars
- `infra.py`, `model_registry.py`, `diagnostics.py` — < 21,000 each
