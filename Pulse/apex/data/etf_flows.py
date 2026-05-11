"""Pulse.apex.data.etf_flows — Spot BTC/ETH ETF net inflow signal.

Source: farside.co.uk (free, public CSV per-ETF, daily).
Endpoints (BTC):
  https://farside.co.uk/wp-content/uploads/2024/01/IBIT.csv
  https://farside.co.uk/wp-content/uploads/2024/01/FBTC.csv
  ... (10 ETFs)

These return a CSV with columns: Date, Inflow_USD_M (millions).

Signal logic
------------
Compute total daily net flow across all tracked ETFs (sum). Then
score the recent 3-day mean against the 30-day distribution:

  z > +2  → strong inflow regime  → +1
  z < -2  → strong outflow regime → -1

For BTC symbols this score is used directly; alts ride the BTC bid
with a 0.6× attenuation (less direct exposure to ETF flows).

Replay: a static CSV `apex_etf_flows.csv` is bundled with the QC
project (refreshed daily via apex_data/fetch_etf_flows.py). The
PythonData class reads from the project root.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from Pulse.apex.registry import SignalScore, SignalRegistry


# ─── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_SHORT_LOOKBACK   = 3      # days
DEFAULT_LONG_LOOKBACK    = 30
DEFAULT_BULLISH_Z        = 2.0
DEFAULT_ALT_ATTENUATION  = 0.6    # alts get 60% of the ETF signal magnitude


# ─── Pure-Python core ────────────────────────────────────────────────────────


def parse_etf_csv(content: str) -> list[tuple[str, float]]:
    """Parse a farside-style CSV into [(date, total_inflow_usd_m), ...].

    Handles two layouts:
      A. Date, Inflow_USD_M                               (single-ETF feed)
      B. Date, IBIT, FBTC, ARKB, ...                      (consolidated)
    """
    out: list[tuple[str, float]] = []
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return out
    header = [c.strip() for c in lines[0].split(",")]
    if len(header) < 2:
        return out
    for raw in lines[1:]:
        parts = [c.strip() for c in raw.split(",")]
        if not parts or len(parts) < 2:
            continue
        date = parts[0]
        try:
            # Sum every numeric column except the date
            vals = [float(p.replace("$", "").replace(",", ""))
                    for p in parts[1:] if p and p not in ("-", "—")]
            total = sum(vals)
        except ValueError:
            continue
        out.append((date, total))
    return out


def compute_etf_flow_score(
    flows: Sequence[float],
    *,
    is_btc: bool,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    bullish_z:      float = DEFAULT_BULLISH_Z,
    alt_attenuation: float = DEFAULT_ALT_ATTENUATION,
) -> tuple[float, dict]:
    """Returns (score ∈ [-1, +1], meta).

    `flows` is the daily net-flow series in USD millions (oldest → newest).
    Positive = inflow, negative = outflow.
    """
    if len(flows) < long_lookback:
        return 0.0, {"error": "insufficient_history",
                     "have": len(flows), "need": long_lookback}
    short_mean = sum(flows[-short_lookback:]) / short_lookback
    window     = list(flows[-long_lookback:])
    m = sum(window) / long_lookback
    var = sum((x - m) ** 2 for x in window) / long_lookback
    if var <= 0:
        return 0.0, {"error": "no_variance"}
    sd = math.sqrt(var)
    z = (short_mean - m) / sd
    raw = max(-1.0, min(1.0, z / bullish_z))
    score = raw if is_btc else raw * alt_attenuation
    return score, {"z": z, "short_mean": short_mean, "long_mean": m,
                   "is_btc": is_btc, "raw": raw}


# ─── Registry callable factory ──────────────────────────────────────────────

def _is_btc(symbol: str) -> bool:
    s = symbol.upper()
    return s.startswith("BTC") or s in {"XBTUSD", "WBTCUSD"}


def make_etf_flow_signal_fn(
    flows_provider,
    *,
    short_lookback: int = DEFAULT_SHORT_LOOKBACK,
    long_lookback:  int = DEFAULT_LONG_LOOKBACK,
    bullish_z:      float = DEFAULT_BULLISH_Z,
    alt_attenuation: float = DEFAULT_ALT_ATTENUATION,
):
    """`flows_provider(context)` → list[float] daily net flows in USD M."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            flows = flows_provider(context) or []
        except Exception as exc:   # noqa: BLE001
            return SignalScore("etf_flow", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_etf_flow_score(
            flows, is_btc=_is_btc(symbol),
            short_lookback=short_lookback, long_lookback=long_lookback,
            bullish_z=bullish_z, alt_attenuation=alt_attenuation,
        )
        valid = "error" not in meta
        return SignalScore("etf_flow", symbol, score, valid=valid, meta=meta)
    return _fn


def register_etf_flow(registry: SignalRegistry, flows_provider, **kwargs):
    registry.register("etf_flow",
                      make_etf_flow_signal_fn(flows_provider, **kwargs))


# ─── In-process flow store ──────────────────────────────────────────────────


@dataclass
class ETFFlowStore:
    """Append daily-totals into a rolling buffer of last N days."""
    keep_days: int = 60
    flows: list[float] = None

    def __post_init__(self) -> None:
        if self.flows is None:
            self.flows = []

    def record(self, daily_total_usd_m: float) -> None:
        if daily_total_usd_m is None:
            return
        self.flows.append(float(daily_total_usd_m))
        if len(self.flows) > self.keep_days:
            del self.flows[0]

    def load_csv(self, content: str) -> None:
        """Bulk-load from a farside-style CSV string."""
        rows = parse_etf_csv(content)
        for _, total in rows:
            self.record(total)

    def get(self, context: dict | None = None) -> list[float]:
        return list(self.flows)


# ─── QC PythonData class (loads bundled CSV on first tick) ──────────────────

try:
    from AlgorithmImports import (   # type: ignore  # noqa: F401
        PythonData,
        SubscriptionDataSource,
        SubscriptionTransportMedium,
    )
    from datetime import datetime, timedelta
    HAS_QC = True
except Exception:
    HAS_QC = False
    PythonData = object


if HAS_QC:

    class ETFFlowsData(PythonData):
        """Daily ETF net-flow data, served from a CSV bundled in the QC project.

        File expected at project root: ``apex_etf_flows.csv``.
        Format: ``YYYY-MM-DD,inflow_usd_millions``
        """

        def GetSource(self, config, date, isLiveMode):
            return SubscriptionDataSource(
                "apex_etf_flows.csv",
                SubscriptionTransportMedium.LocalFile,
            )

        def Reader(self, config, line, date, isLiveMode):
            if not line or line.startswith("Date") or line.startswith("#"):
                return None
            try:
                parts = [p.strip() for p in line.split(",")]
                d = datetime.strptime(parts[0], "%Y-%m-%d")
                value = float(parts[1])
                result = ETFFlowsData()
                result.Symbol = config.Symbol
                result.Time = d
                result.EndTime = d + timedelta(days=1)
                result.Value = value
                return result
            except Exception:
                return None
else:
    ETFFlowsData = None  # type: ignore
