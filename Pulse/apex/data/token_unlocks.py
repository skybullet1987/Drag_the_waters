"""Pulse.apex.data.token_unlocks — predictable supply-shock detector.

Source: pre-computed via apex_data/fetch_token_unlocks.py from the
public token.unlocks.app calendar. CSV format:

  Date,Symbol,UnlockUSD,UnlockPctOfSupply

Signal logic (BEARISH GATE)
----------------------------
For each symbol, look ahead `look_ahead_days` (default 7).
If an unlock event ≥ `pct_threshold` (default 1% of circulating supply)
is scheduled, emit a NEGATIVE score scaled by:

  score = -min(1.0, unlock_pct / 5.0)   # 5% = full -1

Otherwise score is 0 (NOT bullish — just no information).

This signal asymmetrically biases AGAINST entries with imminent supply
shocks. The ML ensemble can still override on other strong signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from Pulse.apex.registry import SignalScore, SignalRegistry


DEFAULT_LOOK_AHEAD_DAYS = 7
DEFAULT_PCT_THRESHOLD = 0.01     # 1% of supply
DEFAULT_FULL_PENALTY_PCT = 0.05  # 5% unlock = full -1


@dataclass(frozen=True)
class UnlockEvent:
    date:    datetime
    symbol:  str
    usd:     float
    pct_of_supply: float


def parse_unlock_csv(content: str) -> list[UnlockEvent]:
    out: list[UnlockEvent] = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("Date") or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split(",")]
        if len(parts) < 4:
            continue
        try:
            d = datetime.strptime(parts[0], "%Y-%m-%d")
            sym = parts[1].upper()
            usd = float(parts[2]) if parts[2] else 0.0
            pct = float(parts[3]) if parts[3] else 0.0
            out.append(UnlockEvent(date=d, symbol=sym, usd=usd,
                                   pct_of_supply=pct))
        except ValueError:
            continue
    return out


def upcoming_unlocks(
    events: Iterable[UnlockEvent],
    symbol: str,
    now: datetime,
    look_ahead_days: int = DEFAULT_LOOK_AHEAD_DAYS,
    pct_threshold: float = DEFAULT_PCT_THRESHOLD,
) -> list[UnlockEvent]:
    sym_clean = symbol.upper().replace("USD", "").replace("USDT", "")
    horizon = now + timedelta(days=look_ahead_days)
    out: list[UnlockEvent] = []
    for ev in events:
        if ev.pct_of_supply < pct_threshold:
            continue
        ev_sym_clean = ev.symbol.replace("USD", "").replace("USDT", "")
        if ev_sym_clean != sym_clean:
            continue
        if now <= ev.date <= horizon:
            out.append(ev)
    return out


def compute_token_unlock_score(
    events: Iterable[UnlockEvent],
    symbol: str,
    now: datetime,
    *,
    look_ahead_days: int = DEFAULT_LOOK_AHEAD_DAYS,
    pct_threshold: float = DEFAULT_PCT_THRESHOLD,
    full_penalty_pct: float = DEFAULT_FULL_PENALTY_PCT,
) -> tuple[float, dict]:
    upcoming = upcoming_unlocks(events, symbol, now, look_ahead_days,
                                 pct_threshold)
    if not upcoming:
        return 0.0, {"upcoming_count": 0}
    biggest = max(upcoming, key=lambda e: e.pct_of_supply)
    score = -min(1.0, biggest.pct_of_supply / full_penalty_pct)
    return score, {
        "upcoming_count": len(upcoming),
        "biggest_pct": biggest.pct_of_supply,
        "biggest_date": biggest.date.isoformat(),
    }


def make_token_unlock_signal_fn(events_provider, now_provider, **kwargs):
    """events_provider(ctx) → list[UnlockEvent].
       now_provider(ctx)    → datetime."""
    def _fn(symbol: str, context: dict) -> SignalScore:
        try:
            events = events_provider(context) or []
            now = now_provider(context)
        except Exception as exc:   # noqa: BLE001
            return SignalScore("token_unlock", symbol, 0.0, valid=False,
                               meta={"error": str(exc)[:120]})
        score, meta = compute_token_unlock_score(events, symbol, now, **kwargs)
        return SignalScore("token_unlock", symbol, score, valid=True, meta=meta)
    return _fn


def register_token_unlock(registry: SignalRegistry, events_provider,
                           now_provider, **kwargs):
    registry.register("token_unlock",
                      make_token_unlock_signal_fn(events_provider,
                                                  now_provider, **kwargs))


@dataclass
class UnlockCalendarStore:
    events: list[UnlockEvent] = field(default_factory=list)

    def load_csv(self, content: str) -> None:
        self.events.extend(parse_unlock_csv(content))

    def get(self, context: dict | None = None) -> list[UnlockEvent]:
        return list(self.events)
