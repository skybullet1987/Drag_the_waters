"""Pulse.apex.ml.dataset — build (X, y) supervised-learning datasets.

Inputs:
  feature_matrix : list of (timestamp, symbol, feature_vector) tuples
  price_series   : dict {symbol: list[(timestamp, close_price)]} sorted asc

Outputs:
  X : 2D list of feature_vector rows
  y : 1D list of binary labels (1 if forward_return > pos_threshold else 0)
  meta : list of (timestamp, symbol, forward_return) for diagnostics

Pure-Python — no numpy hard dependency. Uses bisect for fast price
lookup at (timestamp + horizon).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence


# ─── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_FORWARD_HOURS    = 24
DEFAULT_POS_THRESHOLD    = 0.015   # +1.5% in 24h
DEFAULT_NEG_THRESHOLD    = -0.015  # for 3-class extension


# ─── Types ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sample:
    timestamp:  datetime
    symbol:     str
    features:   list[float]
    forward_return: float
    label_pos:  int     # 1 if forward_return > pos_threshold
    label_neg:  int     # 1 if forward_return < neg_threshold (optional 3-class)


# ─── Forward-return computation ──────────────────────────────────────────────


def lookup_forward_price(price_pairs: Sequence[tuple[datetime, float]],
                         start_ts: datetime,
                         forward_hours: int) -> float | None:
    """Find the close price closest to (start_ts + forward_hours).

    `price_pairs` MUST be sorted by timestamp ascending. Returns None if
    we don't have a price within ±60min of the target time (data gap).
    """
    target = start_ts + timedelta(hours=forward_hours)
    timestamps = [p[0] for p in price_pairs]
    idx = bisect.bisect_left(timestamps, target)
    candidates = []
    if idx < len(timestamps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs((timestamps[i] - target).total_seconds()))
    if abs((timestamps[best] - target).total_seconds()) > 3600:
        return None
    return price_pairs[best][1]


def compute_forward_return(price_pairs: Sequence[tuple[datetime, float]],
                            start_ts: datetime,
                            start_price: float,
                            forward_hours: int) -> float | None:
    fwd = lookup_forward_price(price_pairs, start_ts, forward_hours)
    if fwd is None or start_price <= 0:
        return None
    return (fwd - start_price) / start_price


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_dataset(
    feature_records: Iterable[tuple[datetime, str, Sequence[float], float]],
    price_series: dict[str, Sequence[tuple[datetime, float]]],
    *,
    forward_hours:  int   = DEFAULT_FORWARD_HOURS,
    pos_threshold:  float = DEFAULT_POS_THRESHOLD,
    neg_threshold:  float = DEFAULT_NEG_THRESHOLD,
    drop_missing:   bool  = True,
) -> list[Sample]:
    """Assemble a list of Sample objects ready for sklearn-style fitting.

    Args
    ----
    feature_records : iterable of (timestamp, symbol, feature_vec, current_price)
    price_series    : per-symbol sorted [(ts, close), ...]
    forward_hours   : horizon for the label
    pos_threshold   : forward_return > pos_threshold → label_pos = 1
    neg_threshold   : forward_return < neg_threshold → label_neg = 1
    drop_missing    : if True, skip records with missing forward price
    """
    out: list[Sample] = []
    for ts, sym, feats, cur_price in feature_records:
        prices = price_series.get(sym)
        if prices is None:
            if not drop_missing:
                out.append(Sample(ts, sym, list(feats), 0.0, 0, 0))
            continue
        fwd_ret = compute_forward_return(prices, ts, cur_price, forward_hours)
        if fwd_ret is None:
            if not drop_missing:
                out.append(Sample(ts, sym, list(feats), 0.0, 0, 0))
            continue
        out.append(Sample(
            timestamp=ts, symbol=sym, features=list(feats),
            forward_return=fwd_ret,
            label_pos=int(fwd_ret > pos_threshold),
            label_neg=int(fwd_ret < neg_threshold),
        ))
    return out


def to_xy(samples: Iterable[Sample]) -> tuple[list[list[float]], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    for s in samples:
        X.append(list(s.features))
        y.append(s.label_pos)
    return X, y


def to_xy_returns(samples: Iterable[Sample]
                  ) -> tuple[list[list[float]], list[float]]:
    X: list[list[float]] = []
    y: list[float] = []
    for s in samples:
        X.append(list(s.features))
        y.append(s.forward_return)
    return X, y


# ─── Walk-forward splitter ───────────────────────────────────────────────────


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: datetime
    train_end:   datetime
    test_start:  datetime
    test_end:    datetime
    train_idx:   list[int]
    test_idx:    list[int]


def walk_forward_splits(
    samples: Sequence[Sample],
    *,
    train_window_days: int = 365,
    test_window_days:  int = 30,
    step_days:         int = 30,
    min_test_size:     int = 50,
) -> list[WalkForwardFold]:
    """Generate non-overlapping forward-rolling train/test folds.

    The training window slides forward by `step_days` between folds.
    Folds with fewer than `min_test_size` test rows are dropped.
    """
    if not samples:
        return []
    sorted_samples = sorted(enumerate(samples), key=lambda p: p[1].timestamp)
    timestamps = [p[1].timestamp for p in sorted_samples]
    start = timestamps[0]
    end = timestamps[-1]
    folds: list[WalkForwardFold] = []
    cur_train_start = start
    while True:
        train_end  = cur_train_start + timedelta(days=train_window_days)
        test_start = train_end
        test_end   = test_start + timedelta(days=test_window_days)
        if test_end > end + timedelta(days=1):
            break
        train_idx, test_idx = [], []
        for orig_idx, samp in sorted_samples:
            if cur_train_start <= samp.timestamp < train_end:
                train_idx.append(orig_idx)
            elif test_start <= samp.timestamp < test_end:
                test_idx.append(orig_idx)
        if len(test_idx) >= min_test_size and len(train_idx) >= min_test_size:
            folds.append(WalkForwardFold(
                train_start=cur_train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                train_idx=train_idx, test_idx=test_idx,
            ))
        cur_train_start = cur_train_start + timedelta(days=step_days)
    return folds
