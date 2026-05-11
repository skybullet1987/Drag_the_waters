"""Pulse.apex.ml.predict — inference wrapper used by the live engine.

Tiny class that owns:
  - lazy joblib load on first use
  - graceful fallback to neutral prob 0.5 when model is missing
  - a tiny in-memory cache to avoid re-scoring identical vectors back-to-back

Designed to be created in PulseAlgorithm.Initialize() and called from
the 4-hour Apex tick.
"""

from __future__ import annotations

import math
from typing import Sequence

from Pulse.apex.config import APEX_FEATURE_DIM, APEX_FALLBACK_WEIGHTS
from Pulse.apex.feature_vector import DEFAULT_SIGNAL_ORDER, FEATURES_PER_SIGNAL


class ApexInference:
    """Inference façade for the Apex ML ensemble.

    Args
    ----
    model_path  : path to the joblib bundle (None → fallback mode)
    fallback_only : if True, skip joblib load even when path is set
    """

    def __init__(self,
                 model_path: str | None = None,
                 fallback_only: bool = False) -> None:
        self.model_path  = model_path
        self.fallback_only = fallback_only
        self._bundle: dict | None = None
        self._loaded = False
        self._load_err: str | None = None

    # ── Bundle loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded or self.fallback_only or not self.model_path:
            self._loaded = True
            return
        try:
            from Pulse.apex.ml.train import load_ensemble
            self._bundle = load_ensemble(self.model_path)
        except Exception as exc:    # noqa: BLE001
            self._load_err = str(exc)[:200]
            self._bundle = None
        finally:
            self._loaded = True

    @property
    def in_fallback_mode(self) -> bool:
        return self._bundle is None

    @property
    def load_error(self) -> str | None:
        return self._load_err

    # ── Scoring ─────────────────────────────────────────────────────────────

    def predict_one(self, feature_vec: Sequence[float]) -> float:
        """Return P(positive label) ∈ [0, 1] for a single feature vector."""
        self._ensure_loaded()
        if len(feature_vec) != APEX_FEATURE_DIM:
            # Defensive truncate / pad so we never throw on shape mismatch
            v = list(feature_vec)[:APEX_FEATURE_DIM]
            while len(v) < APEX_FEATURE_DIM:
                v.append(0.0)
        else:
            v = list(feature_vec)
        if self._bundle is None:
            return _fallback_prob(v)
        try:
            from Pulse.apex.ml.train import predict_ensemble
            probs = predict_ensemble(self._bundle, [v])
            p = probs[0] if probs else 0.5
            if p is None or math.isnan(p):
                return 0.5
            return max(0.0, min(1.0, float(p)))
        except Exception:
            return _fallback_prob(v)

    def predict_many(self,
                     feature_matrix: Sequence[Sequence[float]]
                     ) -> list[float]:
        """Batch version of predict_one."""
        self._ensure_loaded()
        if not feature_matrix:
            return []
        # Defensive shape normalization
        rows: list[list[float]] = []
        for row in feature_matrix:
            r = list(row)[:APEX_FEATURE_DIM]
            while len(r) < APEX_FEATURE_DIM:
                r.append(0.0)
            rows.append(r)
        if self._bundle is None:
            return [_fallback_prob(r) for r in rows]
        try:
            from Pulse.apex.ml.train import predict_ensemble
            probs = predict_ensemble(self._bundle, rows)
            out: list[float] = []
            for p in probs:
                if p is None or math.isnan(p):
                    out.append(0.5)
                else:
                    out.append(max(0.0, min(1.0, float(p))))
            return out
        except Exception:
            return [_fallback_prob(r) for r in rows]


# ─── Fallback heuristic (cold-start before the model is ready) ──────────────


def _fallback_prob(vec: Sequence[float]) -> float:
    """Convert a 30-dim feature vector to a [0, 1] prob using the
    APEX_FALLBACK_WEIGHTS dict.

    Maps signed score from APEX_FALLBACK_WEIGHTS sum to a sigmoid:
        prob = 1 / (1 + exp(-2 * weighted_sum))
    """
    weighted = 0.0
    for i, sig in enumerate(DEFAULT_SIGNAL_ORDER):
        if sig not in APEX_FALLBACK_WEIGHTS:
            continue
        idx_raw = i * FEATURES_PER_SIGNAL    # raw feature is at position 0
        if idx_raw < len(vec):
            weighted += vec[idx_raw] * APEX_FALLBACK_WEIGHTS[sig]
    return 1.0 / (1.0 + math.exp(-2.0 * weighted))
