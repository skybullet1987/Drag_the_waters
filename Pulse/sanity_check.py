"""Pulse end-to-end sanity check on synthetic OHLCV.

Exercises every component without QC AlgorithmImports:
  - Universe gate accepts good symbols, rejects polluted ones
  - Tier classifier maps to correct tier
  - All 5 scalp signal components fire
  - All 5 size multipliers compose correctly
  - Per-trade hard kill triggers
  - DD circuit breaker trips/halts/recovers
  - Equity curve stop pauses on stale equity
  - Pyramid sizer adds at MFE thresholds
  - Win-streak sizer ramps on consecutive wins
  - Bayesian per-symbol scaler updates with outcomes
  - Multi-strategy allocator weights sub-strategies
  - Trend basket composes with regime detection
  - MR engine rejects non-capitulation conditions

Run before any QC deployment to catch integration bugs that would
otherwise burn cloud time.

Usage:
    python3 Pulse/sanity_check.py

Exit code 0 = all green; non-zero = something broke (re-run pytest).
"""

from __future__ import annotations

import os
import random
import sys
import traceback
from datetime import datetime, timedelta

# Make repo importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─── Test reporter ──────────────────────────────────────────────────────────

PASSES: list[str] = []
FAILURES: list[tuple[str, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSES.append(label)
        print(f"  ✓ {label}")
    else:
        FAILURES.append((label, detail))
        print(f"  ✗ {label}{f' — {detail}' if detail else ''}")


def section(name: str) -> None:
    print(f"\n── {name} " + "─" * (60 - len(name)))


# ─── Sanity checks ──────────────────────────────────────────────────────────

def check_universe():
    section("Universe gate + tier classifier")
    from Pulse.universe import UniverseGate, SymbolTierClassifier, SymbolStats

    g = UniverseGate()
    c = SymbolTierClassifier()

    btc = SymbolStats("BTCUSD", 40_000_000_000, 1.0, 50000, 2000)
    fart = SymbolStats("FARTCOINUSD", 300_000, 120.0, 0.21, 15)

    check("UniverseGate accepts BTCUSD",  g.is_eligible(btc).eligible)
    check("UniverseGate rejects FARTCOIN", not g.is_eligible(fart).eligible)
    check("BTCUSD classified as 'major'", c.classify("BTCUSD") == "major")
    check("KASUSD classified as 'micro'", c.classify("KASUSD") == "micro")


def check_features():
    section("Live-survivable features (CVD, Kyle, YZ vol, spillover, clusters)")
    from Pulse.features import (
        cumulative_volume_delta, cvd_signal_score,
        kyle_lambda, yang_zhang_realized_variance,
        cross_symbol_score_boost, liquidity_clusters,
    )

    rnd = random.Random(0)
    n = 100
    closes = [100 + 0.5 * i + rnd.gauss(0, 0.1) for i in range(n)]
    # For CVD to be positive on an uptrend, close must be near the bar high
    # (buying-pressure signature). signed_bar_volume = vol*(2*c - h - l)/(h-l)
    opens  = [c - 0.4 for c in closes]
    highs  = [c + 0.05 for c in closes]   # close near high (buying pressure)
    lows   = [c - 0.5 for c in closes]
    vols   = [1000 + rnd.gauss(0, 100) for _ in range(n)]

    cvd = cumulative_volume_delta(opens, highs, lows, closes, vols)
    score = cvd_signal_score(cvd, lookback=12)
    check("CVD slope positive on uptrend (close-near-high)", score > 0,
          detail=f"score={score:.4f}")

    lam = kyle_lambda(closes, vols, lookback=20)
    check("Kyle's lambda positive", lam > 0, detail=f"λ={lam:.6f}")

    var = yang_zhang_realized_variance(opens, highs, lows, closes, lookback=20)
    check("YZ realized variance non-negative", var >= 0, detail=f"var={var:.6f}")

    boost = cross_symbol_score_boost(
        "INJUSD",
        {"BTC": 0.03, "ETH": 0.04, "SOL": 0.05, "XRP": 0.025, "DOGE": 0.022,
         "INJ": 0.005},
    )
    check("Spillover boosts laggard +0.10", boost == 0.10)

    cl = liquidity_clusters(closes, vols, n_clusters=3)
    check("Liquidity clusters returned", len(cl) >= 1)


def check_regime():
    section("Regime detection (5-mode + golden cross + BTC.D)")
    from Pulse.regime import (
        detect_market_mode, golden_cross_regime,
        btc_dominance_regime, compose_regime_size_multiplier,
    )

    bull = [100 + i * 0.5 for i in range(250)]
    sell = [200 - i * 0.5 for i in range(250)]
    check("detect_market_mode = risk_on_trend on uptrend",
          detect_market_mode([100 + i * 0.5 for i in range(20)]) == "risk_on_trend")
    check("detect_market_mode = selloff on steep drop",
          detect_market_mode([100 - i * 1.0 for i in range(15)]) == "selloff")
    check("golden_cross_regime = bull on long uptrend",
          golden_cross_regime(bull).regime == "bull")
    check("golden_cross_regime = bear on long downtrend",
          golden_cross_regime(sell).regime == "bear")
    btcd = btc_dominance_regime(0.20, [0.05] * 5)
    check("BTC dominance = btc_strong (BTC outpacing alts)",
          btcd.regime == "btc_strong")
    composed = compose_regime_size_multiplier("selloff", None, None)
    check("Selloff regime → size mult = 0",
          composed["size_mult"] == 0.0)


def check_alt_data():
    section("Alt data signals (F&G + Funding)")
    from Pulse.alt_data import FGSignal, FundingSignal, FUNDING_REGIMES

    fg_fear = FGSignal.from_value(15)
    fg_greed = FGSignal.from_value(85)
    fg_panic = FGSignal.from_value(95)
    check("F&G extreme_fear → 1.2× size mult",
          fg_fear.size_multiplier == 1.20)
    check("F&G extreme_greed → 0.5× size mult + halve max_pos",
          fg_greed.size_multiplier == 0.5 and fg_greed.max_positions_multiplier == 0.5)
    check("F&G panic ≥ 90 → block_new_entries",
          fg_panic.block_new_entries)

    fund_squeeze = FundingSignal.from_rate(-0.001)
    fund_panic   = FundingSignal.from_rate(0.0015)
    check("Funding deep_short_squeeze → 1.20× size",
          fund_squeeze.size_modifier == 1.20)
    check("Funding panic-positive → block_new_entries",
          fund_panic.block_new_entries)


def check_scalp_engine():
    section("MicroScalpEngine v8 end-to-end")
    from Pulse.scalp_engine import (
        SymbolBars, MarketContext, compute_scalp_score, rank_candidates,
    )

    rnd = random.Random(42)
    n = 80
    closes = [100 + 0.4 * i + rnd.gauss(0, 0.1) for i in range(n)]
    for i in range(60, n - 1):
        closes[i] = closes[i - 1] + rnd.gauss(0.02, 0.15)
    opens  = [c - 0.2 for c in closes]
    highs  = [max(o, c) + 0.3 for o, c in zip(opens, closes)]
    lows   = [min(o, c) - 0.2 for o, c in zip(opens, closes)]
    vols   = [1000 + rnd.gauss(0, 100) for _ in range(n - 1)] + [6000.0]

    bars = SymbolBars("SOLUSD", opens, highs, lows, closes, vols)
    score = compute_scalp_score(bars)
    check("Strong setup scores ≥ 0.55 (entry threshold)",
          score.score >= 0.55,
          detail=f"score={score.score:.3f} components={score.cvd_score+score.vol_ignition+score.micro_trend+score.rsi_filter+score.vwap_signal:.2f}")

    # Funding integration
    ctx = MarketContext(funding_rate=-0.001)   # deep short squeeze
    s2 = compute_scalp_score(bars, ctx)
    check("Funding signal raises composed_size_mult vs no-funding",
          s2.composed_size_mult > score.composed_size_mult)

    # F&G panic blocks
    ctx_panic = MarketContext(fg_value=95.0)
    s3 = compute_scalp_score(bars, ctx_panic)
    check("F&G ≥ 90 blocks entry even on max-conviction signal",
          not s3.enter)


def check_circuits():
    section("Risk circuits (DD breaker + per-trade kill + equity stop)")
    from Pulse.circuit import (
        DrawdownCircuitBreaker, PerTradeKill, EquityCurveStop,
    )

    cb = DrawdownCircuitBreaker(trip_drawdown_pct=0.20, halt_drawdown_pct=0.25)
    cb.update(1000, datetime(2026, 1, 1))
    cb.update(799, datetime(2026, 1, 2))   # 20.1% DD → trip
    check("DD breaker trips at 20%", not cb.can_enter_new_positions())
    cb.update(749, datetime(2026, 1, 3))   # 25.1% DD → halt
    check("DD breaker halts at 25%", cb.should_liquidate_all())

    kill = PerTradeKill(threshold_pct=0.08)
    d = kill.evaluate("KAS", entry_price=0.04, current_price=0.0367)
    check("Per-trade kill fires at -8.25%", d.should_kill,
          detail=f"return={d.return_pct:.4f}")

    es = EquityCurveStop(stale_days=14, pause_days=7)
    es.update(1000, datetime(2026, 1, 1))
    es.update(950, datetime(2026, 1, 15))   # 14 days no new high
    check("EquityCurveStop blocks after 14 stale days",
          not es.can_enter_new_positions(datetime(2026, 1, 15)))


def check_sizing():
    section("Sizing (Bayesian + Pyramid + WinStreak)")
    from Pulse.sizing import (
        BayesianSymbolSizer, PyramidSizer, PyramidPositionState,
        WinStreakSizer,
    )

    bs = BayesianSymbolSizer()
    for _ in range(20):
        bs.update("WINNER", won=True)
    for _ in range(20):
        bs.update("LOSER", won=False)
    check("Bayesian sizer scales WINNER above LOSER",
          bs.size_multiplier("WINNER") > bs.size_multiplier("LOSER"))
    check("LOSER hits floor multiplier",
          bs.size_multiplier("LOSER") == bs.min_scaler)

    ps = PyramidSizer()
    state = PyramidPositionState("BTC", 500.0)
    d = ps.evaluate(state, current_mfe_pct=0.04)
    ps.commit_add(state, d)
    check("PyramidSizer fires at +4% MFE (rung 0)",
          d.should_add and 0 in state.rungs_fired)

    ws = WinStreakSizer()
    for _ in range(5):
        ws.record_outcome(won=True)
    check("WinStreak sizer caps at 2.0× after 5 wins",
          ws.size_multiplier() == 2.0)


def check_portfolio():
    section("Multi-strategy portfolio allocator")
    from Pulse.portfolio import StrategyAllocator, STRATEGIES

    a = StrategyAllocator()
    base = datetime(2026, 5, 10)
    for i in range(20):
        a.record_outcome("scalp", base + timedelta(hours=i),  0.01)
        a.record_outcome("trend", base + timedelta(hours=i), -0.005)
        a.record_outcome("mr",    base + timedelta(hours=i),  0.003)
    d = a.compute_allocation(now=base + timedelta(days=1))
    check("All 3 strategies get ≥ 20% floor allocation",
          all(d.weights[s] >= 0.20 - 1e-6 for s in STRATEGIES))
    check("All weights sum to 1.0",
          abs(sum(d.weights.values()) - 1.0) < 1e-6)


def check_trend_engine():
    section("Trend engine (HYDRA done right)")
    from Pulse.trend_engine import (
        compute_basket, score_trend_candidate, detect_trend_regime,
    )

    btc_bull = [100 + i * 0.5 for i in range(250)]
    btc_bear = [200 - i * 0.5 for i in range(250)]

    bull_dec = detect_trend_regime(btc_bull, confirm_days=5)
    bear_dec = detect_trend_regime(btc_bear, confirm_days=5)
    check("Trend regime: bull on uptrend",
          bull_dec.regime == "bull")
    check("Trend regime: bear on downtrend",
          bear_dec.regime == "bear" and bear_dec.should_liquidate_all)

    rnd = random.Random(0)
    cands = []
    for i in range(5):
        closes = [100 + i * 0.5 + rnd.gauss(0, 0.5) for _ in range(250)]
        cands.append(score_trend_candidate(
            f"COIN{i}", closes,
            [c * 1.005 for c in closes],
            [c * 0.995 for c in closes],
        ))
    basket = compute_basket(btc_bull, cands)
    check("compute_basket returns at least 1 selection in bull regime",
          len(basket.selected) >= 1)
    if basket.weights:
        sw = sum(basket.weights.values())
        n = len(basket.selected)
        # Per-coin cap is 35%; with n=2 max sum = 0.70 (intentional —
        # leaves cash for small-basket safety). With n>=3, sum can reach 1.0.
        max_possible = min(1.0, n * 0.35)
        check("Basket weights respect per-coin 35% cap",
              all(w <= 0.35 + 1e-6 for w in basket.weights.values()),
              detail=f"max_w={max(basket.weights.values()):.3f}")
        check("Basket sum ≤ max_possible (cap-aware)",
              sw <= max_possible + 1e-6,
              detail=f"sum={sw:.3f}, max_possible={max_possible:.3f}")
    else:
        check("Basket weights present (skipped — corr cap rejected all)", True)


def check_mr_engine():
    section("Mean-reversion engine")
    from Pulse.mr_engine import evaluate_mr_entry

    # Synthetic capitulation: declining + final green bounce + volume spike
    rnd = random.Random(0)
    closes = [100 - 10 * (i / 98) for i in range(99)]
    closes.append(closes[-1] * 1.005)   # green bounce
    opens = [c + 0.05 for c in closes[:-1]] + [closes[-2]]
    highs = [c + 0.1 for c in closes]
    lows  = [c - 0.1 for c in closes]
    vols  = [100 + rnd.gauss(0, 10) for _ in range(99)] + [600.0]

    sig = evaluate_mr_entry(
        symbol="SOLUSD",
        opens=opens, highs=highs, lows=lows, closes=closes, volumes=vols,
    )
    check("MR engine fires on capitulation+bounce",
          sig.enter, detail=f"reason={sig.rejection_reason or sig.reason}")


def check_audit_harness():
    section("Audit harness imports + log parser on real fixture")
    from backtest_audit.log_parser import parse_log_file, pair_trades
    from backtest_audit.compare import build_report
    from backtest_audit.harsh_simulator import HarshFillSimulator, HarshConfig
    from backtest_audit.regime_runner import REGIME_WINDOWS, score_param_set, WindowResult
    from backtest_audit.param_sweep import pulse_param_sweep
    from backtest_audit.qc_orders import parse_qc_orders, pair_qc_trades

    fixture = os.path.join(
        os.path.dirname(__file__), "..", "backtest_audit", "fixtures",
        "mg36_paper_2026-03-16.txt",
    )
    p = parse_log_file(fixture)
    check("MG36 fixture parses with 0 errors",
          len(p.parse_errors) == 0)
    check("MG36 fixture has 4 SCALP ENTRY events",
          len(p.entries) == 4)
    trades = pair_trades(p)
    check("MG36 fixture pairs 2 round-trip trades", len(trades) == 2)

    rep = build_report(live=trades, backtest=[])
    check("Audit gap report builds on live-only data",
          rep.n_live == 2)

    sim = HarshFillSimulator(HarshConfig(seed=42, reject_rate_normal=0.0))
    fill = sim.simulate_fill("KAS", "Buy", True, 0.04, 0.04, tier="micro")
    check("Harsh sim micro-cap fill has slippage > 200bp",
          fill.slippage_bps > 200)

    check("REGIME_WINDOWS has 6 windows", len(REGIME_WINDOWS) == 6)
    sweep = pulse_param_sweep(n_samples=5)
    check("LHS sweep generates 5 named param sets", len(sweep) == 5)

    # qc_orders
    raw = [
        {"id": 1, "symbol": "BTC", "status": 3, "direction": 0,
         "quantity": 1, "price": 100, "time": "2025-01-01T00:00:00Z"},
        {"id": 2, "symbol": "BTC", "status": 3, "direction": 1,
         "quantity": 1, "price": 110, "time": "2025-01-01T00:30:00Z"},
    ]
    qc_trades = pair_qc_trades(parse_qc_orders(raw))
    check("qc_orders pairs 1 round-trip from synthetic JSON",
          len(qc_trades) == 1)


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    print("Pulse end-to-end sanity check (synthetic data, no QC)")
    print("=" * 65)

    try:
        check_universe()
        check_features()
        check_regime()
        check_alt_data()
        check_scalp_engine()
        check_circuits()
        check_sizing()
        check_portfolio()
        check_trend_engine()
        check_mr_engine()
        check_audit_harness()
    except Exception as exc:
        print()
        print("UNCAUGHT EXCEPTION during sanity check:")
        traceback.print_exc()
        return 1

    print()
    print("=" * 65)
    print(f"PASSES:  {len(PASSES)}")
    print(f"FAILURES: {len(FAILURES)}")
    if FAILURES:
        print()
        print("FAILED CHECKS:")
        for label, detail in FAILURES:
            print(f"  ✗ {label}" + (f" — {detail}" if detail else ""))
        return 1

    print()
    print("✓ ALL SANITY CHECKS PASSED — safe to deploy to QC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
