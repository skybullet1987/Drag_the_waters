from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _safe_round_trip_cost(default=0.0052):
    try:
        from execution import ESTIMATED_ROUND_TRIP_FEE  # type: ignore
        return float(ESTIMATED_ROUND_TRIP_FEE)
    except Exception:
        return float(default)


def make_cost_aware_labels(close, horizon=1, k=2.0, round_trip_cost=None):
    round_trip_cost = _safe_round_trip_cost() if round_trip_cost is None else float(round_trip_cost)
    close = pd.Series(close).astype(float)
    fwd_ret = close.shift(-horizon) / close - 1.0
    upper = k * round_trip_cost
    lower = -k * round_trip_cost
    y = pd.Series(0, index=close.index, dtype=int)
    y[fwd_ret > upper] = 1
    y[fwd_ret < lower] = -1
    return y


@dataclass
class WalkForwardConfig:
    mode: str = "expanding"  # expanding or rolling
    min_train_size: int = 500
    train_window: int = 2000
    retrain_every: int = 24
    drop_flat: bool = True


class SvmSignalModel:
    def __init__(self, C=1.0, gamma="scale", probability=True, neutral_threshold=0.55, max_train_samples=50000):
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.neutral_threshold = neutral_threshold
        self.max_train_samples = int(max_train_samples)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=C, gamma=gamma, probability=probability)),
        ])

    def _prepare_training_data(self, X, y, drop_flat=True):
        X_df = pd.DataFrame(X).copy()
        y_sr = pd.Series(y).copy()
        valid = (~X_df.isna().any(axis=1)) & (~y_sr.isna())
        if drop_flat:
            valid &= y_sr != 0
        X_clean = X_df.loc[valid]
        y_clean = y_sr.loc[valid]
        if len(X_clean) > self.max_train_samples:
            step = int(np.ceil(len(X_clean) / self.max_train_samples))
            X_clean = X_clean.iloc[::step]
            y_clean = y_clean.iloc[::step]
        return X_clean, y_clean

    def fit(self, X, y, drop_flat=True):
        X_clean, y_clean = self._prepare_training_data(X, y, drop_flat=drop_flat)
        if y_clean.nunique() < 2:
            raise ValueError("Need at least two classes to train SVM")
        self.pipeline.fit(X_clean, y_clean)
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X)
        probs = self.pipeline.predict_proba(X_df)
        classes = list(self.pipeline.named_steps["svc"].classes_)
        out = pd.DataFrame(index=X_df.index)
        out["prob_up"] = probs[:, classes.index(1)] if 1 in classes else 0.0
        out["prob_down"] = probs[:, classes.index(-1)] if -1 in classes else 0.0
        out["prob_flat"] = probs[:, classes.index(0)] if 0 in classes else 0.0
        return out

    def predict(self, X):
        prob = self.predict_proba(X)
        confidence = prob[["prob_up", "prob_down"]].max(axis=1)
        edge = (prob["prob_up"] - prob["prob_down"]).abs()
        signal = np.where(
            confidence >= self.neutral_threshold,
            np.where(prob["prob_up"] >= prob["prob_down"], 1, -1),
            0,
        )
        return pd.DataFrame(
            {
                "signal": signal.astype(int),
                "confidence": confidence.astype(float),
                "edge": edge.astype(float),
            },
            index=prob.index,
        )

    def walk_forward_predict(self, X, y, config=None):
        X_df = pd.DataFrame(X).copy()
        y_sr = pd.Series(y).copy()
        cfg = config or WalkForwardConfig()

        n = len(X_df)
        preds = pd.DataFrame(index=X_df.index, columns=["signal", "confidence", "edge"], dtype=float)

        start = max(cfg.min_train_size, 2)
        for i in range(start, n, cfg.retrain_every):
            train_end = i
            if cfg.mode == "rolling":
                train_start = max(0, train_end - cfg.train_window)
            else:
                train_start = 0

            X_train = X_df.iloc[train_start:train_end]
            y_train = y_sr.iloc[train_start:train_end]
            X_test = X_df.iloc[i:min(i + cfg.retrain_every, n)]
            if X_test.empty:
                continue

            try:
                self.fit(X_train, y_train, drop_flat=cfg.drop_flat)
                p = self.predict(X_test)
                preds.loc[p.index, :] = p[["signal", "confidence", "edge"]].values
            except ValueError:
                preds.loc[X_test.index, :] = [0, 0.0, 0.0]

        return preds

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)
