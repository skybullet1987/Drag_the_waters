# ── Vox Infrastructure ────────────────────────────────────────────────────────
#
# Consolidated module for universe management, order-execution helpers, and
# ObjectStore persistence.  Previously split across universe.py, execution.py,
# and persistence.py.
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import pickle
try:
    from AlgorithmImports import *
except ImportError:
    pass
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════

KRAKEN_PAIRS = [
    "BTCUSD",  "ETHUSD",   "SOLUSD",   "XRPUSD",  "XDGUSD",
    "ADAUSD",  "AVAXUSD",  "LINKUSD",  "DOTUSD",  "LTCUSD",
    "TRXUSD",  "BCHUSD",   "MATICUSD", "ATOMUSD", "UNIUSD",
    "AAVEUSD", "ARBUSD",   "OPUSD",    "INJUSD",  "NEARUSD",
]


def add_universe(algorithm):
    """
    Register every ticker in KRAKEN_PAIRS with *algorithm* at minute resolution
    on the Kraken market.

    Any pair that Kraken / QuantConnect does not support is logged and silently
    skipped so the backtest can still run with the remaining symbols.

    Parameters
    ----------
    algorithm : QCAlgorithm

    Returns
    -------
    list[Symbol]
        Successfully added Symbol objects.
    """
    symbols = []
    for ticker in KRAKEN_PAIRS:
        try:
            sym = algorithm.add_crypto(
                ticker, Resolution.MINUTE, Market.KRAKEN
            ).symbol
            symbols.append(sym)
        except Exception as exc:
            algorithm.log(f"[universe] Skipping {ticker} — not supported: {exc}")
    return symbols


def fetch_kraken_top20_usd(algorithm):
    """
    Query the Kraken public Ticker endpoint and return the top-20 USD pairs
    ranked by 24-hour quote volume.

    .. warning::
        **LIVE USE ONLY.**  Calling this during a backtest introduces
        look-ahead bias because the live Kraken snapshot reflects the current
        universe composition, not the composition at the backtest date.

        This function automatically returns the static KRAKEN_PAIRS list when
        ``algorithm.live_mode`` is False (i.e., during backtests).

    Parameters
    ----------
    algorithm : QCAlgorithm

    Returns
    -------
    list[str]
        Up to 20 ticker strings.  Falls back to the static KRAKEN_PAIRS list
        on any error or during backtests.
    """
    # ── Backtest guard: never call the live API in a backtest ─────────────────
    # algorithm.live_mode is False in backtests and paper-trading; True only
    # when running against a live brokerage.  Fetching the current Kraken
    # universe during a historical backtest would introduce look-ahead bias
    # (2025 knowledge used when replaying 2024 data).
    if not getattr(algorithm, "live_mode", False):
        algorithm.log(
            "[universe] fetch_kraken_top20_usd: not live_mode — "
            "returning static KRAKEN_PAIRS to avoid look-ahead bias"
        )
        return list(KRAKEN_PAIRS)

    url = "https://api.kraken.com/0/public/Ticker"
    try:
        raw = algorithm.download(url)
        data = json.loads(raw)
        if data.get("error"):
            algorithm.log(f"[universe] Kraken Ticker error: {data['error']}")
            return list(KRAKEN_PAIRS)

        pairs = []
        for pair, info in data.get("result", {}).items():
            if not pair.endswith("USD"):
                continue
            if any(ch in pair for ch in (".", "_")):
                continue
            try:
                volume_24h = float(info["v"][1])
                pairs.append((pair, volume_24h))
            except (KeyError, ValueError, IndexError):
                continue

        pairs.sort(key=lambda x: x[1], reverse=True)
        top20 = [p[0] for p in pairs[:20]]
        algorithm.log(f"[universe] Kraken live top-20: {top20}")
        return top20 if top20 else list(KRAKEN_PAIRS)

    except Exception as exc:
        algorithm.log(
            f"[universe] fetch_kraken_top20_usd failed ({exc}); "
            "falling back to static list"
        )
        return list(KRAKEN_PAIRS)


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class OrderHelper:
    """
    Static helpers for exchange lot-size arithmetic and order validation.

    All methods are stateless and may be called without instantiation.
    """

    # Quote suffixes used to derive the base currency from a symbol string.
    # Listed longest-first so the correct one is matched before any prefix.
    _QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH")

    @staticmethod
    def get_lot_size(security):
        """
        Read the lot size from *security.symbol_properties*.

        Returns
        -------
        float
            Lot size (e.g. 0.001 for BTC on Kraken), or 1e-8 as a safe minimum.
        """
        try:
            return float(security.symbol_properties.lot_size)
        except Exception:
            return 1e-8

    @staticmethod
    def get_min_order_size(security):
        """
        Read the minimum order size from *security.symbol_properties*.

        Returns
        -------
        float
            Minimum order size, or 0.0 if not available.
        """
        try:
            return float(security.symbol_properties.minimum_order_size)
        except Exception:
            return 0.0

    @staticmethod
    def round_qty(qty, lot_size):
        """
        Floor *qty* to the nearest multiple of *lot_size*.

        Uses integer arithmetic to avoid floating-point rounding surprises.
        """
        if lot_size <= 0:
            return qty
        return math.floor(qty / lot_size) * lot_size

    @staticmethod
    def validate_qty(qty, min_order_size):
        """Return False if *qty* is below the exchange minimum order size."""
        return qty >= min_order_size

    @staticmethod
    def get_crypto_base_currency(algorithm, sym):
        """
        Derive the base currency for a crypto symbol without relying on the
        nonexistent ``SymbolProperties.base_currency`` attribute.

        QuantConnect ``SymbolProperties`` does **not** expose ``base_currency``.
        This method derives it by:

        1. Reading ``symbol_properties.quote_currency`` (which QC does expose);
           if ``sym.value`` ends with that quote string, the base is the
           leading portion (e.g. ``OPUSD`` with quote ``USD`` → ``OP``).
        2. Falling back to stripping common quote suffixes from ``sym.value``
           in longest-first order: ``USDT``, ``USDC``, ``USD``, ``EUR``,
           ``GBP``, ``BTC``, ``ETH``.

        Returns ``None`` when the base currency cannot be determined.
        """
        sym_val = sym.value.upper()

        # 1. Prefer quote_currency from symbol_properties (QC-supported field).
        try:
            quote = str(
                algorithm.securities[sym].symbol_properties.quote_currency
            ).upper()
            if quote and sym_val.endswith(quote):
                base = sym_val[: -len(quote)]
                if base:
                    return base
        except Exception:
            pass

        # 2. Fallback: strip well-known quote suffixes (longest first).
        for suffix in OrderHelper._QUOTE_SUFFIXES:
            if sym_val.endswith(suffix):
                base = sym_val[: -len(suffix)]
                if base:
                    return base

        return None

    @staticmethod
    def safe_crypto_sell_qty(algorithm, sym, lot_size, min_order_size,
                             exit_qty_buffer_lots=1):
        """
        Compute a safe sell quantity for a crypto cash-mode position.

        In Kraken cash-mode the ``portfolio[sym].quantity`` can exceed the
        actual base-currency ``CashBook`` balance after fees/rounding.
        Submitting a sell for ``portfolio.quantity`` therefore causes an
        ``INVALID`` order ("insufficient buying power / short not allowed").

        This helper returns the *minimum* of the portfolio holding and the
        CashBook balance, floored to ``lot_size``, then optionally reduced by
        ``exit_qty_buffer_lots`` lots as a precision/fee safety margin.

        Parameters
        ----------
        algorithm             : QCAlgorithm
        sym                   : Symbol
        lot_size              : float  — from OrderHelper.get_lot_size()
        min_order_size        : float  — from OrderHelper.get_min_order_size()
        exit_qty_buffer_lots  : int    — lot units reserved as dust buffer
                                         (default 1).

        Returns
        -------
        float
            Safe sell quantity ≥ min_order_size, or 0.0 when the remaining
            position is too small to sell (treated as non-actionable dust).
        """
        # Current portfolio holding
        portfolio_qty = float(algorithm.portfolio[sym].quantity)
        if portfolio_qty <= 0:
            return 0.0

        # ── Determine base currency ───────────────────────────────────────────
        # QC SymbolProperties does not expose base_currency; use the dedicated
        # resolver which reads quote_currency and falls back to suffix parsing.
        base_ccy = OrderHelper.get_crypto_base_currency(algorithm, sym)

        # ── CashBook balance for the base currency ────────────────────────────
        cash_qty = portfolio_qty   # default: trust portfolio qty
        if base_ccy:
            try:
                cash_qty = float(
                    algorithm.portfolio.cash_book[base_ccy].amount
                )
            except Exception:
                pass   # CashBook lookup failed — fall back to portfolio qty

        # Sellable = min(portfolio holding, actual cash balance)
        sellable = min(portfolio_qty, cash_qty)

        # ── Floor to lot_size ─────────────────────────────────────────────────
        if lot_size > 0:
            sellable = math.floor(sellable / lot_size) * lot_size
            # Subtract safety buffer (one or more lots) to absorb fee/rounding
            buffer_qty = exit_qty_buffer_lots * lot_size
            sellable   = sellable - buffer_qty

        # Must be actionable
        if sellable <= 0 or sellable < min_order_size:
            return 0.0

        return float(sellable)


class PartialFillTracker:
    """
    Accumulate partial fills for a single open order.

    The Vox strategy uses market orders which should fill immediately on
    Kraken, but partial fills can occur during low-liquidity windows.
    This tracker accumulates filled quantities and marks the order complete
    only when the cumulative fill meets or exceeds the target.

    Usage
    -----
    >>> tracker = PartialFillTracker()
    >>> tracker.start_order(order_id=101, target_qty=0.5)
    >>> tracker.on_fill(order_id=101, filled_qty=0.3)
    >>> tracker.on_fill(order_id=101, filled_qty=0.2)
    >>> tracker.is_complete(101)
    True
    """

    def __init__(self):
        self._orders = {}   # order_id -> {"target": float, "filled": float}

    def start_order(self, order_id, target_qty):
        """Begin tracking a new order."""
        self._orders[order_id] = {"target": target_qty, "filled": 0.0}

    def on_fill(self, order_id, filled_qty):
        """Accumulate a (partial) fill."""
        if order_id in self._orders:
            self._orders[order_id]["filled"] += abs(filled_qty)

    def is_complete(self, order_id):
        """Return True if the cumulative fill meets or exceeds the target."""
        if order_id not in self._orders:
            return True   # unknown → assume closed
        rec = self._orders[order_id]
        return rec["filled"] >= rec["target"]

    def get_filled(self, order_id):
        """Return the cumulative filled quantity for *order_id* (0.0 if not tracked)."""
        if order_id not in self._orders:
            return 0.0
        return self._orders[order_id]["filled"]

    def clear(self, order_id):
        """Remove *order_id* from tracking."""
        self._orders.pop(order_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE  (ObjectStore model, trade log, kill switch)
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceManager:
    """
    Persist ML models and trade logs to the QuantConnect ObjectStore.

    All write/read operations are wrapped in try/except so that any storage
    failure degrades gracefully (logs a warning) rather than crashing the algo.

    Kill switch
    -----------
    If the key ``kill_key`` exists in the ObjectStore, the strategy will refuse
    to open new positions.  To activate from the QC Research environment::

        qb.object_store.save("vox/kill_switch", "1")

    To deactivate::

        qb.object_store.delete("vox/kill_switch")

    Parameters
    ----------
    algorithm  : QCAlgorithm
    model_key  : str — ObjectStore key for the pickled model.
    log_key    : str — ObjectStore key for the JSONL trade log.
    kill_key   : str — ObjectStore key for the kill switch flag.
    """

    def __init__(
        self,
        algorithm,
        model_key  = "vox/model.pkl",
        log_key    = "vox/trade_log.jsonl",
        kill_key   = "vox/kill_switch",
        flush_every = 50,
    ):
        self._algo        = algorithm
        self._model_key   = model_key
        self._log_key     = log_key
        self._kill_key    = kill_key
        self._flush_every = flush_every
        self._buffer      = []   # list[str] of pre-serialised JSON lines

    # ── Model serialisation ───────────────────────────────────────────────────

    def save_model(self, model):
        """Pickle *model* and write the bytes to the ObjectStore."""
        try:
            # Defensive: strip any non-picklable logger before serialisation.
            if hasattr(model, "set_logger"):
                try:
                    model.set_logger(None)
                except Exception as set_exc:
                    self._algo.log(f"[persistence] set_logger(None) failed: {set_exc}")
            data = pickle.dumps(model)
            self._algo.object_store.save_bytes(self._model_key, data)
            self._algo.log(
                f"[persistence] Model saved to ObjectStore key={self._model_key}"
            )
            # Re-attach logger so subsequent in-memory use still logs.
            if hasattr(model, "set_logger"):
                try:
                    model.set_logger(self._algo.log)
                except Exception as set_exc:
                    self._algo.log(f"[persistence] set_logger(log) failed: {set_exc}")
        except Exception as exc:
            self._algo.log(f"[persistence] save_model failed: {exc}")

    def load_model(self):
        """
        Load and unpickle the model from the ObjectStore.

        Returns
        -------
        object or None
        """
        try:
            if not self._algo.object_store.contains_key(self._model_key):
                return None
            data = self._algo.object_store.read_bytes(self._model_key)
            model = pickle.loads(data)
            self._algo.log(
                f"[persistence] Model loaded from ObjectStore key={self._model_key}"
            )
            return model
        except Exception as exc:
            self._algo.log(f"[persistence] load_model failed: {exc}")
            return None

    # ── Trade logging ─────────────────────────────────────────────────────────

    def log_trade(self, entry_dict):
        """
        Buffer *entry_dict* as a JSON line and flush to ObjectStore when the
        buffer reaches *flush_every* entries.
        """
        try:
            self._buffer.append(json.dumps(entry_dict, default=str))
            if len(self._buffer) >= self._flush_every:
                self.flush_trade_log()
        except Exception as exc:
            self._algo.log(f"[persistence] log_trade buffer failed: {exc}")

    def flush_trade_log(self):
        """Write all buffered trade lines to the ObjectStore and clear the buffer."""
        if not self._buffer:
            return
        try:
            existing = ""
            if self._algo.object_store.contains_key(self._log_key):
                existing = self._algo.object_store.read(self._log_key)
            payload = existing + "\n".join(self._buffer) + "\n"
            self._algo.object_store.save(self._log_key, payload)
            self._buffer.clear()
        except Exception as exc:
            self._algo.log(f"[persistence] flush_trade_log failed: {exc}")

    # ── Kill switch ───────────────────────────────────────────────────────────

    def is_kill_switch_active(self):
        """Return True if the kill switch key exists in the ObjectStore."""
        try:
            return self._algo.object_store.contains_key(self._kill_key)
        except Exception as exc:
            self._algo.log(
                f"[persistence] is_kill_switch_active check failed: {exc}"
            )
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY HYDRATION
# ═══════════════════════════════════════════════════════════════════════════════

def hydrate_state_from_history(algorithm, state, symbols,
                               days, resolution_minutes, max_history_bars):
    """Fetch history and populate per-symbol state deques.

    Standalone so it can be called from main.py without duplicating logic.

    Parameters
    ----------
    algorithm          : QCAlgorithm — provides .history() and .log()
    state              : dict sym → {"closes", "highs", "lows", "volumes"}
    symbols            : list[Symbol]
    days               : int — calendar days of history to request
    resolution_minutes : int — bar resolution (e.g. 5)
    max_history_bars   : int — safety cap on bar count
    """
    bars_needed = int(days * 24 * 60 / resolution_minutes)
    bars_needed = min(bars_needed, max_history_bars)
    for sym in symbols:
        try:
            hist = algorithm.history(sym, bars_needed, Resolution.MINUTE)
            if hist is None or hist.empty:
                continue
            if hasattr(hist.index, "levels") and len(hist.index.levels) > 1:
                df = hist.loc[sym] if sym in hist.index.get_level_values(0) else hist
            else:
                df = hist
            df = df.resample(f"{resolution_minutes}min").agg({
                "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            st = state[sym]
            st["closes"].clear(); st["highs"].clear()
            st["lows"].clear();   st["volumes"].clear()
            for _, row in df.iterrows():
                st["closes"].append(float(row["close"]))
                st["highs"].append(float(row["high"]))
                st["lows"].append(float(row["low"]))
                st["volumes"].append(float(row["volume"]))
        except Exception as exc:
            algorithm.log(f"[vox] history hydrate failed for {sym.value}: {exc}")


# ===============================================================================
# model_registry
# ===============================================================================

# ── Vox Model Registry ────────────────────────────────────────────────────────
#
# Stable model identifiers, role constants, and weighted ensemble helpers.
#
# Model IDs are intentionally short for compact log output.
# The core sklearn stack (lr, hgbc, et, rf) is always present.
# Optional external models (lgbm, xgb, catboost) are guarded by import checks.
# ─────────────────────────────────────────────────────────────────────────────

# ── Core model IDs ────────────────────────────────────────────────────────────
MODEL_ID_LR       = "lr"
MODEL_ID_HGBC     = "hgbc"     # HistGradientBoostingClassifier
MODEL_ID_ET       = "et"       # ExtraTreesClassifier
MODEL_ID_RF       = "rf"       # RandomForestClassifier
MODEL_ID_GNB      = "gnb"      # GaussianNB (diagnostic-only by default)
MODEL_ID_LGBM     = "lgbm"
MODEL_ID_XGB      = "xgb"
MODEL_ID_CATBOOST = "catboost"

# Ordered list of all possible model IDs (core + optional)
ALL_MODEL_IDS = [
    MODEL_ID_LR,
    MODEL_ID_HGBC,
    MODEL_ID_ET,
    MODEL_ID_RF,
    MODEL_ID_LGBM,
    MODEL_ID_XGB,
    MODEL_ID_CATBOOST,
]

# Human-readable descriptions for startup logging
MODEL_DESCRIPTIONS = {
    MODEL_ID_LR:       "LogisticRegression (linear baseline)",
    MODEL_ID_HGBC:     "HistGradientBoostingClassifier (strong sklearn booster)",
    MODEL_ID_ET:       "ExtraTreesClassifier (randomised trees, diverse)",
    MODEL_ID_RF:       "RandomForestClassifier (bagged trees)",
    MODEL_ID_GNB:      "GaussianNB (diagnostic-only; degenerate on crypto)",
    MODEL_ID_LGBM:     "LGBMClassifier (external, optional)",
    MODEL_ID_XGB:      "XGBClassifier (external, optional)",
    MODEL_ID_CATBOOST: "CatBoostClassifier (external, optional)",
}

# ── Model role constants ───────────────────────────────────────────────────────
# active     — contributes to ensemble vote; affects trading confidence.
# shadow     — predicted and logged but NEVER affects trading decisions.
# diagnostic — predicted and logged for risk/veto/debug only.
# disabled   — skipped entirely (not trained or predicted).
ROLE_ACTIVE     = "active"
ROLE_SHADOW     = "shadow"
ROLE_DIAGNOSTIC = "diagnostic"
ROLE_DISABLED   = "disabled"

# Default role for each core model ID
DEFAULT_MODEL_ROLES = {
    MODEL_ID_LR:       ROLE_DIAGNOSTIC,  # was always-bearish; diagnostic only
    MODEL_ID_HGBC:     ROLE_ACTIVE,
    MODEL_ID_ET:       ROLE_ACTIVE,
    MODEL_ID_RF:       ROLE_ACTIVE,
    MODEL_ID_GNB:      ROLE_DIAGNOSTIC,  # always-bullish (vote_gnb=1.0); diagnostic
    MODEL_ID_LGBM:     ROLE_SHADOW,
    MODEL_ID_XGB:      ROLE_SHADOW,
    MODEL_ID_CATBOOST: ROLE_SHADOW,
}


# ── Registry entry structure ──────────────────────────────────────────────────
#
# Each entry is a dict with:
#   id       : str   — stable model ID
#   model    : estimator or None
#   enabled  : bool
#   weight   : float — used in weighted mean computation
#   role     : str   — one of ROLE_* constants

def make_registry_entry(model_id, model, enabled=True, weight=1.0, role=ROLE_ACTIVE):
    """Create a model registry entry dict.

    Parameters
    ----------
    model_id : str
    model    : estimator or None
    enabled  : bool
    weight   : float
    role     : str — one of ROLE_ACTIVE / ROLE_SHADOW / ROLE_DIAGNOSTIC / ROLE_DISABLED
    """
    return {
        "id":      model_id,
        "model":   model,
        "enabled": enabled,
        "weight":  float(weight),
        "role":    role,
    }


def build_registry_from_estimators(estimators, weights_dict=None, roles_dict=None):
    """Build a registry list from VotingClassifier estimators.

    Parameters
    ----------
    estimators   : list of (name, estimator) — from VotingClassifier
    weights_dict : dict[str, float] or None — optional per-model weights
    roles_dict   : dict[str, str] or None   — optional per-model roles

    Returns
    -------
    list of registry entry dicts
    """
    registry = []
    for name, model in estimators:
        w = (weights_dict or {}).get(name, 1.0)
        r = (roles_dict or {}).get(name, ROLE_ACTIVE)
        registry.append(make_registry_entry(
            model_id=name,
            model=model,
            enabled=True,
            weight=w,
            role=r,
        ))
    return registry


# ── Weighted mean computation ──────────────────────────────────────────────────

def compute_weighted_mean(votes, weights_dict=None):
    """Compute weighted mean probability from a per-model votes dict.

    Parameters
    ----------
    votes        : dict[str, float] — model_id -> P(class=1)
    weights_dict : dict[str, float] or None — model_id -> weight.
                   If None or empty, falls back to unweighted mean.

    Returns
    -------
    float — weighted (or unweighted) mean in [0, 1].
    """
    if not votes:
        return 0.5

    if not weights_dict:
        # Unweighted mean
        return sum(votes.values()) / len(votes)

    total_w = 0.0
    weighted_sum = 0.0
    for model_id, proba in votes.items():
        w = weights_dict.get(model_id, 1.0)
        if w > 0:
            weighted_sum += w * proba
            total_w += w

    if total_w <= 0:
        # Fallback: unweighted mean
        return sum(votes.values()) / len(votes)

    return weighted_sum / total_w


def weights_are_uniform(weights_dict):
    """Return True if all weights in the dict are equal (no custom weighting)."""
    if not weights_dict:
        return True
    vals = list(weights_dict.values())
    if len(vals) <= 1:
        return True
    return all(abs(v - vals[0]) < 1e-9 for v in vals)


# ── Role-aware vote splitting ──────────────────────────────────────────────────

def split_votes_by_role(votes, roles_dict):
    """Split a per-model votes dict into role-separated sub-dicts.

    Parameters
    ----------
    votes      : dict[str, float] — model_id -> P(class=1)
    roles_dict : dict[str, str]   — model_id -> role string

    Returns
    -------
    tuple of (active_votes, shadow_votes, diagnostic_votes)
        Each is a dict[str, float] containing only models of that role.
    """
    active     = {}
    shadow     = {}
    diagnostic = {}
    for mid, proba in votes.items():
        role = roles_dict.get(mid, ROLE_ACTIVE)
        if role == ROLE_ACTIVE:
            active[mid] = proba
        elif role == ROLE_SHADOW:
            shadow[mid] = proba
        elif role == ROLE_DIAGNOSTIC:
            diagnostic[mid] = proba
        # ROLE_DISABLED models should not appear in votes at all
    return active, shadow, diagnostic


def compute_vote_score(active_votes, vote_thr=0.55):
    """Compute profit-voting score fields from active-model votes.

    Parameters
    ----------
    active_votes : dict[str, float]
        Model-id -> P(class=1) for active-role models only.
    vote_thr : float
        Per-model yes/no threshold.

    Returns
    -------
    dict with keys:
        active_model_count  — int
        vote_yes_fraction   — float in [0, 1]
        top3_mean           — float: mean of top-3 active probabilities
        vote_score          — float: weighted composite
    """
    import numpy as _np
    if not active_votes:
        return {"active_model_count": 0, "vote_yes_fraction": 0.0,
                "top3_mean": 0.0, "vote_score": 0.0}
    vals = sorted(active_votes.values(), reverse=True)
    n    = len(vals)
    am   = float(_np.mean(vals))
    yf   = sum(1 for v in vals if v >= vote_thr) / n
    t3   = float(_np.mean(vals[:3])) if vals else 0.0
    vs   = 0.40 * am + 0.30 * yf + 0.30 * t3
    return {"active_model_count": n, "vote_yes_fraction": yf,
            "top3_mean": t3, "vote_score": vs}


def compute_active_stats(active_votes, agree_thr=0.5):
    """Compute mean / std / n_agree from active-role votes only.

    Parameters
    ----------
    active_votes : dict[str, float] — active-model votes
    agree_thr    : float            — threshold for agreement counting

    Returns
    -------
    dict with keys: active_mean, active_std, active_n_agree
    """
    import numpy as _np
    if not active_votes:
        return {"active_mean": 0.5, "active_std": 0.0, "active_n_agree": 0}
    vals = list(active_votes.values())
    return {
        "active_mean":    float(_np.mean(vals)),
        "active_std":     float(_np.std(vals)),
        "active_n_agree": int(sum(1 for v in vals if v >= agree_thr)),
    }


# ── Startup log helper ────────────────────────────────────────────────────────

def format_model_registry_log(estimators, weights_dict=None, roles_dict=None):
    """Format a startup log line listing enabled model IDs, weights, and roles.

    Example output:
        [model_registry] active=hgbc(w=1.0),et(w=1.0),rf(w=1.0) diag=lr
    """
    active_parts = []
    shadow_parts = []
    diag_parts   = []
    for name, _ in estimators:
        w    = (weights_dict or {}).get(name, 1.0)
        role = (roles_dict or {}).get(name, ROLE_ACTIVE)
        tag  = f"{name}(w={w:.2g})"
        if role == ROLE_SHADOW:
            shadow_parts.append(tag)
        elif role == ROLE_DIAGNOSTIC:
            diag_parts.append(tag)
        else:
            active_parts.append(tag)
    parts = []
    if active_parts:
        parts.append("active=" + ",".join(active_parts))
    if shadow_parts:
        parts.append("shadow=" + ",".join(shadow_parts))
    if diag_parts:
        parts.append("diag=" + ",".join(diag_parts))
    return "[model_registry] " + " ".join(parts) if parts else "[model_registry] (none)"


def format_vote_summary(votes, vote_threshold=0.5):
    """Format a compact per-model vote string.

    Example: lr:0.55,hgbc:0.62,et:0.70,rf:0.58
    """
    if not votes:
        return ""
    return ",".join(f"{mid}:{v:.2f}" for mid, v in votes.items())


# ── Default model weights from config ─────────────────────────────────────────

def build_weights_dict_from_config(config_module):
    """Extract per-model weights from a config module.

    Looks for MODEL_WEIGHT_LR, MODEL_WEIGHT_HGBC, etc. constants.
    Returns a dict only containing models with non-default (!=1.0) weights,
    or all weights if any differ.
    """
    mapping = {
        MODEL_ID_LR:       "MODEL_WEIGHT_LR",
        MODEL_ID_HGBC:     "MODEL_WEIGHT_HGBC",
        MODEL_ID_ET:       "MODEL_WEIGHT_ET",
        MODEL_ID_RF:       "MODEL_WEIGHT_RF",
        MODEL_ID_GNB:      "MODEL_WEIGHT_GNB",
        MODEL_ID_LGBM:     "MODEL_WEIGHT_LGBM",
        MODEL_ID_XGB:      "MODEL_WEIGHT_XGB",
        MODEL_ID_CATBOOST: "MODEL_WEIGHT_CATBOOST",
        # Shadow/promoted models
        "hgbc_l2":         "MODEL_WEIGHT_HGBC_L2",
        "cal_et":          "MODEL_WEIGHT_CAL_ET",
        "cal_rf":          "MODEL_WEIGHT_CAL_RF",
        "lgbm_bal":        "MODEL_WEIGHT_LGBM_BAL",
    }
    result = {}
    for model_id, attr in mapping.items():
        val = getattr(config_module, attr, 1.0)
        result[model_id] = float(val)
    return result


# ── Default model roles from config ───────────────────────────────────────────

def build_roles_dict_from_config(config_module):
    """Extract per-model roles from a config module.

    Looks for MODEL_ROLE_LR, MODEL_ROLE_HGBC, etc. constants.
    Falls back to DEFAULT_MODEL_ROLES if the constant is missing.
    """
    mapping = {
        MODEL_ID_LR:       "MODEL_ROLE_LR",
        MODEL_ID_HGBC:     "MODEL_ROLE_HGBC",
        MODEL_ID_ET:       "MODEL_ROLE_ET",
        MODEL_ID_RF:       "MODEL_ROLE_RF",
        MODEL_ID_GNB:      "MODEL_ROLE_GNB",
        MODEL_ID_LGBM:     "MODEL_ROLE_LGBM",
        MODEL_ID_XGB:      "MODEL_ROLE_XGB",
        MODEL_ID_CATBOOST: "MODEL_ROLE_CATBOOST",
    }
    result = {}
    for model_id, attr in mapping.items():
        val = getattr(config_module, attr, DEFAULT_MODEL_ROLES.get(model_id, ROLE_ACTIVE))
        result[model_id] = str(val)
    return result


# ===============================================================================
# model_health
# ===============================================================================

# ── Vox Model Health Diagnostics ──────────────────────────────────────────────
#
# Tracks rolling per-model probability statistics to detect degenerate models.
#
# Flags emitted:
#   degenerate_bullish  — model votes >= extreme_proba on >= degenerate_frac of obs
#   degenerate_bearish  — model votes <= (1 - extreme_proba) on >= degenerate_frac
#   low_variance        — rolling std of model probabilities < low_std threshold
#
# Usage:
#   tracker = ModelHealthTracker()
#   tracker.update_batch({"hgbc": 0.62, "lr": 0.01, "gnb": 1.0})  # per-prediction
#   flags = tracker.get_all_flags()
#   summary = tracker.format_log_summary(roles_dict={"gnb": "diagnostic"})
# ─────────────────────────────────────────────────────────────────────────────



# Default thresholds (overridable via ModelHealthTracker constructor)
DEFAULT_MIN_OBS         = 20
DEFAULT_EXTREME_PROBA   = 0.95
DEFAULT_DEGENERATE_FRAC = 0.90
DEFAULT_LOW_STD         = 0.01


class ModelHealthTracker:
    """Tracks rolling probability statistics per model for health diagnostics.

    Parameters
    ----------
    min_obs          : int   — minimum observations before emitting flags
    extreme_proba    : float — threshold defining "extreme" probability (e.g. 0.95)
    degenerate_frac  : float — fraction of obs above/below extreme that triggers flag
    low_std          : float — rolling std below this → low_variance flag
    window           : int   — rolling window size (0 = unbounded)
    """

    def __init__(
        self,
        min_obs         = DEFAULT_MIN_OBS,
        extreme_proba   = DEFAULT_EXTREME_PROBA,
        degenerate_frac = DEFAULT_DEGENERATE_FRAC,
        low_std         = DEFAULT_LOW_STD,
        window          = 200,
    ):
        self._min_obs         = int(min_obs)
        self._extreme_proba   = float(extreme_proba)
        self._degenerate_frac = float(degenerate_frac)
        self._low_std         = float(low_std)
        self._window          = int(window)
        # model_id -> deque of float probabilities
        self._history = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, model_id, proba):
        """Record a single probability prediction for a model.

        Parameters
        ----------
        model_id : str
        proba    : float in [0, 1]
        """
        if model_id not in self._history:
            maxlen = self._window if self._window > 0 else None
            self._history[model_id] = deque(maxlen=maxlen)
        try:
            self._history[model_id].append(float(proba))
        except (ValueError, TypeError):
            pass

    def update_batch(self, votes_dict):
        """Record predictions for multiple models at once.

        Parameters
        ----------
        votes_dict : dict[str, float] — model_id -> probability
        """
        for mid, proba in votes_dict.items():
            self.update(mid, proba)

    # ── Flag computation ──────────────────────────────────────────────────────

    def get_flags(self, model_id):
        """Return health flags for a single model.

        Parameters
        ----------
        model_id : str

        Returns
        -------
        dict with keys:
            n_obs           — int:   number of observations recorded
            mean_proba      — float: rolling mean probability
            std_proba       — float: rolling std probability
            pct_above_thr   — float: fraction of obs >= extreme_proba
            pct_below_thr   — float: fraction of obs <= (1 - extreme_proba)
            degenerate_bullish  — bool
            degenerate_bearish  — bool
            low_variance        — bool
            flags           — list[str]: active flag names
        """
        hist = list(self._history.get(model_id, []))
        n = len(hist)
        if n == 0:
            return {
                "n_obs": 0, "mean_proba": None, "std_proba": None,
                "pct_above_thr": None, "pct_below_thr": None,
                "degenerate_bullish": False, "degenerate_bearish": False,
                "low_variance": False, "flags": [],
            }

        mean_p = sum(hist) / n
        std_p  = _population_std(hist)
        low_thr = 1.0 - self._extreme_proba
        pct_above = sum(1 for p in hist if p >= self._extreme_proba) / n
        pct_below = sum(1 for p in hist if p <= low_thr) / n

        deg_bull = (n >= self._min_obs) and (pct_above >= self._degenerate_frac)
        deg_bear = (n >= self._min_obs) and (pct_below >= self._degenerate_frac)
        low_var  = (n >= self._min_obs) and (std_p < self._low_std)

        active_flags = []
        if deg_bull:
            active_flags.append("degenerate_bullish")
        if deg_bear:
            active_flags.append("degenerate_bearish")
        if low_var:
            active_flags.append("low_variance")

        return {
            "n_obs":               n,
            "mean_proba":          mean_p,
            "std_proba":           std_p,
            "pct_above_thr":       pct_above,
            "pct_below_thr":       pct_below,
            "degenerate_bullish":  deg_bull,
            "degenerate_bearish":  deg_bear,
            "low_variance":        low_var,
            "flags":               active_flags,
        }

    def get_all_flags(self):
        """Return health flags for all tracked models.

        Returns
        -------
        dict[str, dict] — model_id -> flags dict (same shape as get_flags)
        """
        return {mid: self.get_flags(mid) for mid in self._history}

    # ── Log formatting ────────────────────────────────────────────────────────

    def format_log_summary(self, roles_dict=None):
        """Format a compact multi-line health summary for logging.

        Parameters
        ----------
        roles_dict : dict[str, str] or None — model_id -> role string

        Returns
        -------
        str — newline-joined log lines, one per model
        """
        lines = []
        for mid in sorted(self._history.keys()):
            f = self.get_flags(mid)
            if f["n_obs"] == 0:
                continue
            role = (roles_dict or {}).get(mid, "?")
            mean_s = f"{f['mean_proba']:.3f}" if f["mean_proba"] is not None else "?"
            std_s  = f"{f['std_proba']:.3f}"  if f["std_proba"]  is not None else "?"
            pct_s  = f"{f['pct_above_thr']:.0%}" if f["pct_above_thr"] is not None else "?"
            flag_s = f["flags"][0] if f["flags"] else "ok"
            lines.append(
                f"[model_health] {mid} role={role}"
                f" n={f['n_obs']} mean={mean_s} std={std_s} yes={pct_s}"
                f" flag={flag_s}"
            )
        return "\n".join(lines) if lines else "[model_health] no data"

    # ── State management ──────────────────────────────────────────────────────

    def model_ids(self):
        """Return list of tracked model IDs."""
        return list(self._history.keys())

    def reset(self, model_id=None):
        """Clear history for one model or all models.

        Parameters
        ----------
        model_id : str or None — if None, clears all
        """
        if model_id is None:
            self._history.clear()
        else:
            self._history.pop(model_id, None)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _population_std(vals):
    """Population std-dev of a list of floats (pure-Python, no numpy)."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
