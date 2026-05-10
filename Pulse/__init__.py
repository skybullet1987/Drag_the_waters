"""Pulse — best-of-breed crypto scalping engine.

Built on Path A: audit-first, walk-forward-validated, multi-strategy.
See PLAN.md (Cursor artifacts) for design rationale.

Modules planned (filled in across phases):
- universe.py    — capacity gate + 4-tier classifier (Phase 0a)
- config.py      — all tunable constants
- features.py    — CVD, Kyle's λ, realized vol, trade-rate, VWAP-σ (Phase 1)
- regime.py      — 5-mode market regime + BTC.D (Phase 1)
- scalp_engine.py — MicroScalpEngine v8 (OBI removed) (Phase 1)
- trend_engine.py — daily trend basket sub-strategy (Phase 4)
- mr_engine.py   — mean-reversion sub-strategy (Phase 4)
- portfolio.py   — multi-strategy capital allocator (Phase 4)
- execution.py   — order helpers
- slippage.py    — RealisticCryptoSlippage (Sweet Water variant)
- fees.py        — KrakenTieredFeeModel + MakerTakerFeeModel
- circuit.py     — DrawdownCircuitBreaker + per-trade kill
- events.py      — order audit handlers
- alt_data.py    — Fear & Greed (properly used this time)
- main.py        — PulseAlgorithm QC entry point
"""

__version__ = "0.0.1"
