# Vox — ML Ensemble Kraken Rotation Strategy

## What is Vox?

**Vox** is a machine-learning overlay strategy that lives alongside the
rule-based `main.py` (KrakenTopCoinAlgorithm) in this repository.  Where
`main.py` uses a hand-crafted composite score (momentum + RSI + volume spike),
Vox replaces the scoring layer with a five-model **heterogeneous soft-voting
ensemble** trained on triple-barrier labels derived from the same market data.

Both strategies share the same fill-driven state machine pattern (position
state updated only in `on_order_event`), the same Kraken brokerage model, and
the same pre-trade validation philosophy.  They are designed to be run and
compared independently — Vox does **not** modify `main.py`.

---

## Architecture

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
                                  │  (models.py)           │
                                  │  ret×4, RSI, ATR, vol,   │
                                  │  BTC-rel, hour           │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  VoxEnsemble             │
                                  │  (models.py)           │
                                  │  Soft vote of 5 models   │
                                  └──────────┬───────────────┘
                                             │  mean_proba, std_proba,
                                             │  n_agree, per_model
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Confidence Gate         │
                                  │  score_min / score_gap   │
                                  │  max_dispersion / min_agree│
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Regime Gate             │
                                  │  (risk.py)             │
                                  │  4h BTC SMA(20) + slope  │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │  Kelly Sizer             │
                                  │  (risk.py)             │
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

## Ensemble Models

| # | Name | Class | Why |
|---|------|-------|-----|
| 1 | `lr` | `LogisticRegression(C=1.0)` | Linear baseline; fast, interpretable, well-calibrated out-of-box |
| 2 | `rf` | `RandomForestClassifier(n_estimators=200, max_depth=5)` + isotonic calibration | Bagged trees capture non-linear interactions; decorrelated from LGBM |
| 3 | `lgbm` | `LGBMClassifier(n_estimators=200, lr=0.05, …)` + isotonic calibration | Gradient-boosted trees; high signal-to-noise on tabular financial data |
| 4 | `et` | `ExtraTreesClassifier(n_estimators=200, max_depth=5)` + isotonic calibration | Extremely randomised splits create low correlation with RF; adds diversity |
| 5 | `gnb` | `GaussianNB()` | Probabilistic baseline with no hyper-parameters; anchors the soft vote |

If LightGBM is not installed, `lgbm` falls back to `GradientBoostingClassifier`
with matching depth/learning-rate settings and a log warning.

All tree models are wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`
so that `predict_proba` outputs are reliable probability estimates rather than
raw decision-function scores.

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

### Alignment Constraint

> **Note:** Training labels are now intentionally decoupled from execution
> barriers.  The dedicated `LABEL_TP`, `LABEL_SL`, `LABEL_HORIZON_BARS`
> constants (see below) govern what is labelled "1" at training time, while
> `TAKE_PROFIT`, `STOP_LOSS`, and `TIMEOUT_HOURS` govern when positions are
> closed at execution time.  Looser training barriers increase the positive
> rate, improving model calibration without changing the live trading behaviour.

---

## Parameters and Defaults

All parameters are defined at the top of `main.py` as module-level constants
and can be overridden at runtime via the QuantConnect parameter panel.

### Execution parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `take_profit` | `0.020` | Take-profit fraction (+2 %) |
| `stop_loss` | `0.012` | Stop-loss fraction (−1.2 %) |
| `timeout_hours` | `3.0` | Max hold time in hours |
| `atr_tp_mult` | `2.0` | ATR multiplier for dynamic TP |
| `atr_sl_mult` | `1.2` | ATR multiplier for dynamic SL |
| `score_min` | `0.25` | Upper clamp on the effective score threshold |
| `score_gap` | `0.02` | Required probability gap to runner-up |
| `max_dispersion` | `0.30` | Max std_proba across models |
| `min_agree` | `1` | Min models with proba ≥ agree_thr |
| `allocation` | `0.50` | Flat allocation fallback fraction |
| `kelly_frac` | `0.25` | Fractional-Kelly multiplier |
| `max_alloc` | `0.80` | Hard ceiling on allocation |
| `use_kelly` | `True` | Use Kelly sizing; False = flat |
| `use_calibration` | `True` | Wrap tree models in CalibratedClassifierCV |
| `max_daily_sl` | `2` | Daily SL cap |
| `cooldown_mins` | `15` | Global post-exit cooldown (min) |
| `sl_cooldown_mins` | `60` | Per-coin SL cooldown (min) |
| `max_dd_pct` | `0.08` | Drawdown circuit-breaker (8 %) |
| `cash_buffer` | `0.99` | Cash headroom multiplier |

### Label / training parameters

| Constant / Parameter | Default | Description |
|----------------------|---------|-------------|
| `SCORE_MIN_FLOOR` | `0.15` | Floor for the base-rate-aware effective score threshold at runtime |
| `LABEL_TP` / `label_tp` | `0.012` | Take-profit fraction for training labels (+1.2 %) |
| `LABEL_SL` / `label_sl` | `0.010` | Stop-loss fraction for training labels (−1.0 %) |
| `LABEL_HORIZON_BARS` / `label_horizon_bars` | `72` | Timeout bars for training labels (≈6h at 5-min bars) |

`LABEL_*` constants are defined in `models.py` and re-imported by `main.py`.
The looser barriers increase the positive rate from ~1–5 % to a more balanced
range, improving ensemble calibration without changing live execution behaviour.

### Base-rate-aware confidence gate

At each decision tick the strategy derives two effective thresholds from the
most recent training `positive_rate` (`pr`):

| Threshold | Formula | Notes |
|-----------|---------|-------|
| `agree_thr` | `clip(2 × pr, 0.15, 0.55)` | Replaces the hard-coded `0.5` in the model agreement count |
| `score_min_eff` | `clip(max(SCORE_MIN_FLOOR, 3 × pr), SCORE_MIN_FLOOR, SCORE_MIN)` | Replaces the raw `SCORE_MIN` in the mean_proba gate |

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
| `models.py` | Feature engineering, triple-barrier labeling, `VoxEnsemble`, training pipeline |
| `risk.py` | `RegimeFilter` (4h BTC gate), Kelly sizing, `RiskManager` (cooldowns, drawdown CB) |
| `infra.py` | Universe list + `add_universe()`, `OrderHelper`, `PartialFillTracker`, `PersistenceManager` |

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

## Soft-Voting Ensemble Design

The `VoxEnsemble` (in `models.py`) implements a **heterogeneous soft-voting**
ensemble rather than hard majority voting.  Soft voting averages the
`predict_proba` output of each model and is more robust to class imbalance and
miscalibrated individual models.

### Why soft voting?

With positive rates of 1–5 % (typical for triple-barrier labeling on short
time horizons), hard majority voting nearly always produces the majority class.
Averaging probabilities lets a confident minority of models pull the mean above
the adaptive threshold even when most models output sub-0.5 probabilities.

### Confidence metrics used

| Metric | How computed | Used for |
|--------|-------------|----------|
| `mean_proba` | Average P(class=1) across all models | Primary entry gate (`score_min_eff`) |
| `std_proba` | Std-dev of per-model probabilities | Dispersion gate: high std → uncertain → skip |
| `n_agree` | Count of models with P ≥ `agree_thr` | Agreement gate: require ≥ `min_agree` |

The adaptive thresholds are:

| Threshold | Formula | Purpose |
|-----------|---------|---------|
| `agree_thr` | `clip(2 × positive_rate, 0.15, 0.55)` | Scales the "agreeing" bar proportionally to class frequency |
| `score_min_eff` | `clip(max(SCORE_MIN_FLOOR, 3 × positive_rate), SCORE_MIN_FLOOR, SCORE_MIN)` | Avoids rejecting every signal when positive_rate is very low |

### Calibration

Tree-based models (RF, LGBM/GB, ET) are wrapped in
`CalibratedClassifierCV(method="isotonic", cv=2)` to convert raw scores into
reliable probability estimates.  LogisticRegression and GaussianNB are
well-calibrated out-of-the-box.  Calibration can be disabled via the
`use_calibration=False` parameter for faster iteration.

GaussianNB is re-fit with class-frequency-derived sample weights after the
main ensemble fit to prevent the majority-class imbalance from dominating its
naive likelihood.

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

For each candidate passing the confidence gates, Vox computes:

```text
gross_ev       = mean_proba × tp_use − (1 − mean_proba) × sl_use
ev_after_costs = gross_ev − COST_BPS × 1e-4
confidence_adj = max(0, 1 − std_proba)
entry_score    = ev_after_costs × confidence_adj
```

where:

| Term | Description |
|------|-------------|
| `tp_use`, `sl_use` | Per-candidate ATR-based TP/SL (or fixed fallback) |
| `COST_BPS` | Estimated round-trip fee + slippage in basis points (default 20 bps) |
| `confidence_adj` | Multiplier in [0, 1]: reduces score when models strongly disagree |

All EV-related values (`gross_ev`, `ev_after_costs`, `entry_score`) are
**return fractions**, not probabilities.  A value of `0.001` means a 0.1 %
expected return; `0.01` means 1 %.  This is important when setting the EV
thresholds: `MIN_EV = 0.0005` requires only a 0.05 % expected return, while
`SCORE_GAP = 0.02` in probability units represents a 2 percentage-point
probability lead — a very different magnitude.

Candidates are then **ranked by `entry_score`** rather than raw `mean_proba`,
and only candidates with `ev_after_costs > min_ev` are considered.

### EV gap selectivity

After ranking, Vox requires the top candidate to lead the second-best by at
least `ev_gap` in `entry_score`.  This controls **selectivity**, not trade
validity: if the top EV is strongly positive and the second EV is nearly as
good, both are acceptable trades — use a small `ev_gap` (or `0.0`) when you
want the best available trade rather than holding out for a runaway winner.

> **Important:** `ev_gap` is in **return-fraction units** (same as `entry_score`),
> NOT probability units.  Using the probability gap threshold (`score_gap = 0.02`)
> as an EV gap would require a 2 percentage-point EV advantage, which blocks
> nearly all trades since typical EV scores are in the `0.001–0.02` range.

### Parameters

| Parameter / Constant | Default | Units | Description |
|----------------------|---------|-------|-------------|
| `COST_BPS` / `cost_bps` | `20` | basis points | Estimated round-trip fee+slippage |
| `MIN_EV` / `min_ev` | `0.0005` | return fraction | Minimum EV after costs to enter (0.0005 = 0.05 %) |
| `EV_GAP` / `ev_gap` | `0.00025` | return fraction | Required EV lead of top over second-best (0.00025 = 0.025 %) |
| `SCORE_GAP` / `score_gap` | `0.02` | probability (0–1) | Probability gap between top and runner-up — **not** used for EV comparisons |
| `EXIT_QTY_BUFFER_LOTS` / `exit_qty_buffer_lots` | `1` | lots | Safety lot buffer on exits |

**Tuning ranges:**

| Parameter | Conservative | Aggressive | Notes |
|-----------|-------------|------------|-------|
| `min_ev` | `0.00025` – `0.001` | `0.0` | Lower allows more trades; raise if entering on noise |
| `ev_gap` | `0.0000` – `0.001` | `0.0` | `0.0` = no gap required; raise only if top coin is often ambiguous |
| `score_gap` | `0.01` – `0.05` | `0.005` | Probability gap only; `0.02` is a reasonable default |

### Diagnostics

Every time candidates pass all gates the following is logged:

```
[diag] candidates=3 top=OPUSD ev_score=0.00412 ev_gap=0.00201
       mean_p=0.287 std_p=0.081 n_agree=3 tp=0.0245 sl=0.0147
```

When no candidate passes, the hourly diagnostic log shows how many failed
each gate:

```
[diag] eval=18 pass_disp=12 pass_agree=10 pass_score=3 pass_ev=0
       best_mean=0.183 ... (thresh: score>=0.150 agree>=1 disp<=0.30 ev>0.00050 cost=0.0020)
```

Routine skip messages (EV gap too small, regime block, risk block) are
throttled to **at most once per hour** (`SKIP_DIAG_INTERVAL_SECS = 3600`) to
prevent QuantConnect log rate-limiting during normal backtests.

This helps tune `MIN_EV`, `EV_GAP`, `COST_BPS`, and the confidence gates without
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
- **Regression head** — A lightweight forward-return regressor combined with
  the classifier EV could further improve entry quality.  The current EV
  scoring layer (using classifier probability and ATR TP/SL) serves as the
  profit-aware substitute until a regression head is feasible without a large
  rewrite.
