# Pulse — Best-of-breed crypto scalper for QuantConnect

Pulse is the production-grade trading engine that emerged from the **Path A**
plan after analyzing every one of the user's prior QC strategies (see
`backtest_audit/fixtures/mg36_paper_2026-03-16.txt` for the live evidence
that anchored every design decision).

> ⚠️  **Existing strategies untouched.** `Vox/` and root `main.py` are kept
> as research artifacts. Pulse is a fresh, parallel package.

---

## Quick start (1 command per phase)

| Phase | Command | What it does |
|---|---|---|
| **Deploy** | `python3 backtest_audit/qc_runner.py --project-id <PID> --backtest-name "smoke"` | Push Pulse files + run a single backtest |
| **Phase 2** | `python3 backtest_audit/run_phase2_validation.py --project-id <PID>` | Standard vs Harsh-sim gap report |
| **Phase 3** | `python3 backtest_audit/qc_sweep_runner.py --project-id <PID> --n-samples 50 --use-harsh-sim` | 50 LHS param sets × 6 windows |
| **Local sanity** | `python3 Pulse/sanity_check.py` | Exercise engine on synthetic data (no QC) |

Set credentials once:
```bash
export QC_USER_ID=330252
export QC_API_TOKEN=<your_token>
```

---

## Module map

| Module | Lines | What it does |
|---|---|---|
| `config.py` | ~100 | All tunable constants in one place |
| `universe.py` | ~280 | `UniverseGate` (capacity + spread + history) + `SymbolTierClassifier` (4 tiers, auto-demote/eject) |
| `slippage.py` | ~140 | `RealisticCryptoSlippage` (Sweet Water + calibration upgrades) |
| `fees.py` | ~160 | `KrakenTieredFeeModel` + `MakerTakerFeeModel` + `VolumeTracker` |
| `circuit.py` | ~360 | `DrawdownCircuitBreaker` + `RollingMaxDD` + `PerTradeKill` + `EquityCurveStop` |
| `features.py` | ~500 | CVD, Kyle's λ, Yang-Zhang RV, trade-rate z-score, VWAP±σ, spillover, liquidity clusters |
| `regime.py` | ~270 | 5-mode market regime + golden cross + BTC.D + composed sizing |
| `alt_data.py` | ~330 | `FearGreedData` + `FGSignal` + **`BybitFundingData` + `FundingSignal`** |
| `scalp_engine.py` | ~370 | **`MicroScalpEngine v8`** — OBI replaced with CVD + spillover + funding |
| `execution.py` | ~340 | `safe_sell_quantity` (CashBook fix), `round_to_lot`, slippage logging |
| `events.py` | ~360 | Pure-Python order audit state machine + cash-mode trigger |
| `main.py` | ~700 | `PulseAlgorithm` QC entry point + `use_harsh_sim` flag + `runtime_overrides` |
| `trend_engine.py` | ~290 | **HYDRA done right** — dual-confirmation + ATR SL + correlation cap |
| `mr_engine.py` | ~200 | Mean-reversion (RSI<30 + VWAP-σ + bounce capitulation) |
| `portfolio.py` | ~310 | `StrategyAllocator` — Sharpe-weighted multi-strategy capital allocation |
| `sizing.py` | ~340 | Bayesian per-symbol Beta posterior + `PyramidSizer` + `WinStreakSizer` |

All files **< 30 KB** (QC limit is 63 KB).

---

## Engine architecture

```
                         ┌───────────────────────────┐
                         │  Universe (Phase 0a gate) │
                         │   - capacity (>$5M/24h)   │
                         │   - spread (<30bp)        │
                         │   - history (≥30d)        │
                         │   - 4 tiers + auto-eject  │
                         └────────────┬──────────────┘
                                      │ eligible symbols
                                      ▼
                ┌───────────────────────────────────────────┐
                │       MicroScalpEngine v8 (per symbol)    │
                │                                           │
                │   5 SCORE COMPONENTS (each 0.0-0.20):     │
                │     1. CVD slope                          │
                │     2. Volume ignition (z-score)          │
                │     3. EMA5/EMA20 micro-trend             │
                │     4. RSI hybrid                         │
                │     5. VWAP±σ band                        │
                │                                           │
                │   BOOSTS:                                 │
                │     +0.10 cross-symbol spillover          │
                │     +0.10 liquidity-cluster magnet        │
                │                                           │
                │   GATE: score ≥ 0.55 (entry) / 0.70 (HC)  │
                └────────────┬──────────────────────────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────────────┐
       │  5-WAY SIZE MULTIPLIER (applied AFTER score):    │
       │    composed_mult = kyle × rv × regime × fg ×     │
       │                    funding                       │
       │                                                  │
       │  Plus per-symbol Bayesian win-rate scaler        │
       │  Plus PyramidSizer adds at +3% / +6% MFE         │
       │  Plus WinStreakSizer +20% per consec. win        │
       └────────────┬─────────────────────────────────────┘
                    │
                    ▼
       ┌──────────────────────────────────────────────────┐
       │     CIRCUIT BREAKERS (always-on safety):         │
       │       - DD: trip at -20%, halt at -25%           │
       │       - Per-trade hard kill at -8%               │
       │       - Equity curve stop (14d no high → 7d off) │
       │       - F&G ≥ 90 → block all new entries         │
       │       - Funding ≥ +0.10%/8h → block long entries │
       │       - Selloff regime → 0× alt size             │
       └──────────────────────────────────────────────────┘
```

---

## Phase 2 — Harsh-sim validation

Goal: prove Pulse's edge survives live-realistic assumptions.

```bash
python3 backtest_audit/run_phase2_validation.py --project-id <PID>
```

What happens:
1. Pushes all 16 Pulse modules to your QC project
2. Pushes `runtime_overrides.py` with `use_harsh_sim=False` → STANDARD backtest
3. Pulls the standard result
4. Pushes `runtime_overrides.py` with `use_harsh_sim=True` → HARSH backtest
   - Slippage 100bp base (calibrated to MG36 evidence: mean 111bp, max 197bp)
   - 100% taker (Kraken 0.40%) — assumes maker limits don't fill
   - T+1 fill (no synchronous bar-close magic)
   - Random rejection rate (2% normal, 5% during vol spikes)
5. Builds gap report comparing the two
6. Classifies:
   - **PASS** if `harsh_return / standard_return` ∈ [0.30, 0.70]
   - **FAIL** if < 0.30 (too much edge erodes — iterate signals)
   - **WARN** if > 0.70 (suspiciously close — check for residual look-ahead)

Cost: ~20 min QC cloud time.

If FAIL: re-tune the signal weights in `Pulse/scalp_engine.py` and re-run.

---

## Phase 3 — Walk-forward parameter sweep

Goal: avoid single-period selection bias by testing 50 parameter sets across
6 historical regime windows (2022 H1 → 2026 YTD).

```bash
# Cost-bounded first run (recommended for first attempt):
python3 backtest_audit/qc_sweep_runner.py \
    --project-id <PID> \
    --n-samples 20 \
    --max-windows 4 \
    --use-harsh-sim

# Full sweep (cloud-time intensive — 50 × 6 = 300 backtests, ~50 hours):
python3 backtest_audit/qc_sweep_runner.py \
    --project-id <PID> \
    --n-samples 50 \
    --use-harsh-sim
```

What happens:
1. Generate `n_samples` Latin-hypercube param sets across 12 dimensions
2. For each (params, window) pair:
   - Push `runtime_overrides.py` with the params + window dates
   - Run a backtest, parse stats into `WindowResult`
3. Score each param set by **survivability**:
   - `mean_return − consistency_penalty × std − worst_case × max_DD`
   - "Blessed" requires: score > 0, no catastrophic windows, ≥4/6 positive
4. Write HTML report ranked by score; blessed sets highlighted

Output: `backtest_audit/reports/phase3_sweep_<timestamp>.html`

The blessed parameter set goes to Phase 5.

---

## Phase 5 — Paper trading + capital ramp

Goal: confirm live PnL matches the harsh-sim backtest before risking real money.

1. Pick a blessed parameter set from Phase 3
2. Push to QC live paper trading at $100
3. Monitor for 30 days
4. Acceptance: live PnL within ±20% of same-period harsh-sim backtest
5. Capital ramp:

| Week | Capital | Trigger to advance |
|---|---|---|
| 1-4 | $100 | 30d paper match |
| 5-8 | $500 | 30d real-money match |
| 9-12 | $2,000 | 60d real-money positive PnL |
| 13-16 | $5,000 | 90d + DD < 15% |
| 17+ | scale to ~$55K capacity ceiling | 120d positive |

---

## Local development

### Run the test suite
```bash
python3 -m pytest                            # ALL tests (Pulse + audit + Vox)
python3 -m pytest Pulse/tests                # Pulse only
python3 -m pytest backtest_audit/tests       # Audit harness only
```

### Run the local sanity check (no QC required)
```bash
python3 Pulse/sanity_check.py
```
This exercises every component on synthetic OHLCV — proves end-to-end
imports, scoring, and risk circuits work before you spend QC cloud time.

### File-size discipline
Every Pulse `.py` must stay under **63,000 chars** (QC compatibility).
Internal target: **30 KB** per file.
```bash
wc -c Pulse/*.py | sort -rn
```

---

## How Pulse is different from Vox / HYDRA

| Concern | HYDRA-100x (lost −47%) | VOX (won 1.5%) | Pulse |
|---|---|---|---|
| Universe | 30 hard-coded daily | 20 hard-coded 5-min | **Dynamic gate**: capacity + spread + history; 4 tiers, auto-eject |
| Primary signal | Daily momentum + golden cross | 16-model ML ensemble | **5-component microstructure** (CVD/vol/trend/RSI/VWAP) |
| OBI from QuoteBars | n/a | n/a | **Removed** — was noise in live |
| Live-survivable feature replacements | n/a | n/a | **CVD + Kyle's λ + Yang-Zhang RV + spillover** |
| Single position | Yes (3-coin basket) | Yes (1 at a time) | **Up to 6 concurrent** with multi-strategy allocator |
| Risk circuits | None | Per-coin SL only | **DD breaker + Per-trade kill + Equity stop + F&G + Funding** |
| Per-symbol skill | n/a | n/a | **Bayesian Beta posterior** |
| Sweep methodology | Parameter brute force | Single backtest | **Latin hypercube × 6 regime windows** |

---

## Where the live evidence lives
- `backtest_audit/fixtures/mg36_paper_2026-03-16.txt` — the real losing-case
  paper-trade log that anchored every design decision
- `backtest_audit/fixtures/synthetic_winner_paper.txt` — synthetic winning case

Run the audit:
```bash
python3 backtest_audit/run_mg36_audit.py
```
Output: `backtest_audit/reports/mg36_audit.html`
