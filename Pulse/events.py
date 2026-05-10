"""events — order audit + state machine for Pulse.

Ported from Sweet Water v3-2's events.py — the cleanest order-event handler
in the user's portfolio. Tracks per-order intent (entry/exit), accumulates
realized PnL, maintains rolling-win windows for adaptive risk, and emits
slippage warnings.

Pure-Python state machine (unit-testable):
- ``OrderAuditState``  — full mutable state container
- ``handle_order_event()`` — single transition function

QC adapter (gated):
- ``on_order_event()`` — wraps QC's OrderEvent objects and dispatches

Why split:
  Sweet Water's monolithic on_order_event() was untestable without QC. By
  factoring the state machine into a pure transition function we can verify
  every status transition (Submitted, PartiallyFilled, Filled, Canceled,
  Invalid) against synthetic events.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from Pulse.execution import OrderIntent, compute_slippage_pct

try:
    from AlgorithmImports import (   # type: ignore  # noqa: F401
        OrderStatus, OrderDirection,
    )
    HAS_QC = True
except Exception:
    HAS_QC = False


# ─── Pure-Python types ───────────────────────────────────────────────────────

class OrderStatusName(Enum):
    """Local enum mirroring QC's OrderStatus values (for testing without QC).

    NOTE: NOT ``(str, Enum)`` because QC's Python.NET wrapper rejects
    multiple inheritance with managed classes. Compare via
    ``OrderStatusName.FILLED.value == "Filled"`` when string compare needed.
    """
    SUBMITTED         = "Submitted"
    PARTIALLY_FILLED  = "PartiallyFilled"
    FILLED            = "Filled"
    CANCELED          = "Canceled"
    INVALID           = "Invalid"
    NEW               = "New"


@dataclass(frozen=True)
class OrderEventDTO:
    """QC OrderEvent simplified for pure-Python testing."""
    order_id:       str
    symbol:         str
    status:         str            # OrderStatusName values
    direction:      str            # "Buy" | "Sell"
    quantity:       float          # signed; positive = buy, negative = sell
    fill_quantity:  float          # signed
    fill_price:     float          # 0 for non-fill events
    timestamp:      datetime
    intent:         OrderIntent | None = None


@dataclass
class OrderAuditState:
    """Mutable container — caller passes one of these into handle_order_event."""
    # Per-symbol tracking
    entry_prices:     dict = field(default_factory=dict)   # sym -> float
    highest_prices:   dict = field(default_factory=dict)   # sym -> float
    entry_times:      dict = field(default_factory=dict)   # sym -> datetime
    entry_volumes:    dict = field(default_factory=dict)   # sym -> float

    # Pending order tracking
    pending_orders:   dict = field(default_factory=dict)   # sym -> qty in flight
    submitted_orders: dict = field(default_factory=dict)   # sym -> {oid, time, qty, intent}

    # PnL accumulators (after-fee)
    total_pnl:           float = 0.0
    winning_trades:      int = 0
    losing_trades:       int = 0
    consecutive_losses:  int = 0
    daily_trade_count:   int = 0

    # Rolling outcome window for cash-mode pause logic
    rolling_window_size: int = 16
    recent_outcomes:     deque = field(
        default_factory=lambda: deque(maxlen=16)
    )
    rolling_win_sizes:   deque = field(
        default_factory=lambda: deque(maxlen=50)
    )
    rolling_loss_sizes:  deque = field(
        default_factory=lambda: deque(maxlen=50)
    )

    # Slippage audit
    slippage_log:        list = field(default_factory=list)

    # Failed-exit retry counts
    failed_exit_counts:  dict = field(default_factory=dict)

    # Logical pause flag (set by caller when WR drops below threshold)
    cash_mode_until:     datetime | None = None

    # Estimated round-trip fee for net-PnL calc (default Kraken taker x2)
    estimated_round_trip_fee: float = 0.0080


@dataclass
class HandleResult:
    """Side-effect summary returned by handle_order_event for testing/logging."""
    action: str                       # "noop", "entry_recorded", "exit_recorded", etc.
    pnl_pct:        float | None = None
    is_winner:      bool | None  = None
    new_consecutive_losses: int | None = None
    cleared_state:  bool = False
    slippage_logged: bool = False


# ─── Pure transition function ────────────────────────────────────────────────

def handle_order_event(
    state: OrderAuditState,
    event: OrderEventDTO,
    *,
    reference_price: float | None = None,
    on_cash_mode_trigger: Callable[[OrderAuditState, datetime, float], None] | None = None,
    cash_mode_wr_threshold: float = 0.15,
    cash_mode_pause_minutes: int = 45,
) -> HandleResult:
    """Apply one OrderEvent to OrderAuditState.

    Returns a HandleResult describing what changed (for logging + tests).

    Caller is responsible for:
      - passing the right `reference_price` (mid or signal price at submit time)
        so we can compute slippage on fills;
      - implementing on_cash_mode_trigger if they want pause-on-bad-WR logic.
    """
    sym = event.symbol
    s = state

    # ── SUBMITTED: bookkeeping only ─────────────────────────────────────────
    if event.status == OrderStatusName.SUBMITTED.value:
        s.pending_orders[sym] = s.pending_orders.get(sym, 0) + abs(event.quantity)
        s.submitted_orders[sym] = {
            "order_id":  event.order_id,
            "time":      event.timestamp,
            "quantity":  event.quantity,
            "intent":    event.intent or OrderIntent.ENTRY,
        }
        return HandleResult(action="submitted")

    # ── PARTIALLY_FILLED: track fill but keep pending until full or canceled ─
    if event.status == OrderStatusName.PARTIALLY_FILLED.value:
        if sym in s.pending_orders:
            s.pending_orders[sym] = max(
                0.0, s.pending_orders[sym] - abs(event.fill_quantity),
            )
            if s.pending_orders[sym] <= 0:
                s.pending_orders.pop(sym, None)
        # If this is a buy and we don't have an entry yet, record from partial
        if event.direction == "Buy" and sym not in s.entry_prices:
            s.entry_prices[sym]   = event.fill_price
            s.highest_prices[sym] = event.fill_price
            s.entry_times[sym]    = event.timestamp
        slip_logged = False
        if reference_price:
            s.slippage_log.append(_make_slip_log(event, reference_price))
            slip_logged = True
        return HandleResult(action="partial_fill", slippage_logged=slip_logged)

    # ── FILLED ──────────────────────────────────────────────────────────────
    if event.status == OrderStatusName.FILLED.value:
        s.pending_orders.pop(sym, None)
        s.submitted_orders.pop(sym, None)

        slip_logged = False
        if reference_price:
            s.slippage_log.append(_make_slip_log(event, reference_price))
            slip_logged = True

        if event.direction == "Buy":
            # Entry fill
            s.entry_prices[sym]   = event.fill_price
            s.highest_prices[sym] = event.fill_price
            s.entry_times[sym]    = event.timestamp
            s.daily_trade_count += 1
            return HandleResult(
                action="entry_recorded",
                slippage_logged=slip_logged,
            )

        # Sell fill — close position, record PnL net of estimated round-trip fee
        entry_px = s.entry_prices.get(sym)
        if entry_px is None or entry_px <= 0:
            # Couldn't pair; just clear state
            _clear_position_state(s, sym)
            return HandleResult(
                action="exit_unpaired",
                cleared_state=True,
                slippage_logged=slip_logged,
            )

        gross = (event.fill_price - entry_px) / entry_px
        net   = gross - s.estimated_round_trip_fee
        s.total_pnl += net
        is_win = net > 0
        if is_win:
            s.winning_trades += 1
            s.consecutive_losses = 0
            s.rolling_win_sizes.append(net)
        else:
            s.losing_trades += 1
            s.consecutive_losses += 1
            s.rolling_loss_sizes.append(abs(net))
        s.recent_outcomes.append(1 if is_win else 0)

        # Cash-mode pause hook
        if (len(s.recent_outcomes) >= s.rolling_window_size
                and on_cash_mode_trigger is not None):
            wr = sum(s.recent_outcomes) / len(s.recent_outcomes)
            if wr < cash_mode_wr_threshold:
                on_cash_mode_trigger(s, event.timestamp, wr)

        _clear_position_state(s, sym)
        return HandleResult(
            action="exit_recorded",
            pnl_pct=net,
            is_winner=is_win,
            new_consecutive_losses=s.consecutive_losses,
            cleared_state=True,
            slippage_logged=slip_logged,
        )

    # ── CANCELED ────────────────────────────────────────────────────────────
    if event.status == OrderStatusName.CANCELED.value:
        s.pending_orders.pop(sym, None)
        s.submitted_orders.pop(sym, None)
        return HandleResult(action="canceled")

    # ── INVALID ─────────────────────────────────────────────────────────────
    if event.status == OrderStatusName.INVALID.value:
        s.pending_orders.pop(sym, None)
        s.submitted_orders.pop(sym, None)
        if event.direction == "Sell":
            cnt = s.failed_exit_counts.get(sym, 0) + 1
            s.failed_exit_counts[sym] = cnt
            # Was 3 — too eager to abandon the position after a transient
            # data gap. With 10, we keep retrying (typically the next bar
            # with valid data succeeds). The 'force_cleanup' branch is now
            # a true safety valve, not a routine reaction to noisy data.
            if cnt >= 10:
                _clear_position_state(s, sym)
                return HandleResult(action="invalid_force_cleanup",
                                    cleared_state=True)
        return HandleResult(action="invalid")

    # Unknown status
    return HandleResult(action="noop")


def _clear_position_state(s: OrderAuditState, sym) -> None:
    s.entry_prices.pop(sym, None)
    s.highest_prices.pop(sym, None)
    s.entry_times.pop(sym, None)
    s.entry_volumes.pop(sym, None)
    s.failed_exit_counts.pop(sym, None)


def _make_slip_log(event: OrderEventDTO, reference_price: float) -> dict:
    pct = compute_slippage_pct(event.fill_price, reference_price, event.direction)
    return {
        "time":             event.timestamp.isoformat(),
        "symbol":           event.symbol,
        "direction":        event.direction,
        "fill_price":       event.fill_price,
        "reference_price":  reference_price,
        "slippage_bps":     round(pct * 10_000, 2),
        "intent":           event.intent.value if event.intent else "unknown",
    }


# ─── Stat helpers ────────────────────────────────────────────────────────────

def rolling_win_rate(state: OrderAuditState) -> float | None:
    if not state.recent_outcomes:
        return None
    return sum(state.recent_outcomes) / len(state.recent_outcomes)


def avg_win_size(state: OrderAuditState) -> float:
    return (sum(state.rolling_win_sizes) / len(state.rolling_win_sizes)
            if state.rolling_win_sizes else 0.0)


def avg_loss_size(state: OrderAuditState) -> float:
    return (sum(state.rolling_loss_sizes) / len(state.rolling_loss_sizes)
            if state.rolling_loss_sizes else 0.0)


def expectancy(state: OrderAuditState) -> float:
    """E[trade] = WR × avg_win − (1-WR) × avg_loss."""
    wr = rolling_win_rate(state)
    if wr is None:
        return 0.0
    return wr * avg_win_size(state) - (1 - wr) * avg_loss_size(state)


# ─── QC adapter ──────────────────────────────────────────────────────────────

if HAS_QC:

    def on_order_event(algo, event, state: OrderAuditState,
                       reference_price: float | None = None) -> HandleResult:
        """QC OrderEvent → DTO → handle_order_event.

        Caller (PulseAlgorithm) should pass:
          - algo: the QCAlgorithm instance (used for .Debug logging)
          - event: a QC OrderEvent
          - state: the OrderAuditState held on the algo
          - reference_price: optional mid/signal price at submit time
        """
        try:
            dto = OrderEventDTO(
                order_id=str(event.OrderId),
                symbol=event.Symbol.Value if hasattr(event.Symbol, "Value")
                       else str(event.Symbol),
                status=str(event.Status).split(".")[-1],
                direction=("Buy" if event.Direction == OrderDirection.Buy
                           else "Sell"),
                quantity=float(event.Quantity or 0),
                fill_quantity=float(event.FillQuantity or 0),
                fill_price=float(event.FillPrice or 0),
                timestamp=event.UtcTime if hasattr(event, "UtcTime") else algo.Time,
            )
            res = handle_order_event(state, dto, reference_price=reference_price)
            if res.action != "noop":
                algo.Debug(f"[order_audit] {dto.symbol} {dto.status} {dto.direction} "
                          f"qty={dto.quantity} fill={dto.fill_price} → {res.action}")
            return res
        except Exception as exc:
            algo.Debug(f"on_order_event handler error: {exc}")
            return HandleResult(action="error")
else:
    on_order_event = None  # type: ignore
