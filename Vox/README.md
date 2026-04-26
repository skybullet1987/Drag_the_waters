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
| `score_min` | `0.55` | Upper clamp on the effective score threshold |
| `score_gap` | `0.01` | Required probability gap to runner-up |
| `max_dispersion` | `0.25` | Max std_proba across models |
| `min_agree` | `3` | Min models with proba ≥ agree_thr |
| `allocation` | `0.50` | Flat allocation fallback fraction |
| `kelly_frac` | `0.25` | Fractional-Kelly multiplier |
| `max_alloc` | `0.80` | Hard ceiling on allocation |
| `use_kelly` | `True` | Use Kelly sizing; False = flat |
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
- **Transaction cost model** — Incorporate Kraken's maker/taker fee schedule
  into the sizing calculation to avoid entering trades where expected value
  is negative after fees.
