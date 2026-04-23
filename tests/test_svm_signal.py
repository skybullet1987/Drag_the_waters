import numpy as np
import pandas as pd

from models.svm_signal import SvmSignalModel, WalkForwardConfig, make_cost_aware_labels


def test_svm_walk_forward_smoke():
    rng = np.random.default_rng(7)
    n = 600
    r = np.zeros(n)
    noise = rng.normal(0, 0.001, n)
    for t in range(1, n):
        r[t] = 0.85 * r[t - 1] + noise[t]
    close = 100 * np.exp(np.cumsum(r))

    close = pd.Series(close)
    X = pd.DataFrame({
        "ret1": close.pct_change(1),
        "ret2": close.pct_change(2),
        "ret3": close.pct_change(3),
    })
    y = make_cost_aware_labels(close, horizon=1, k=0.0, round_trip_cost=0.0)

    model = SvmSignalModel(C=1.0, gamma="scale", neutral_threshold=0.52)
    wf = WalkForwardConfig(mode="expanding", min_train_size=120, retrain_every=10, drop_flat=True)
    preds = model.walk_forward_predict(X, y, config=wf).dropna()

    y_true = y.loc[preds.index]
    y_pred = preds["signal"].astype(int)

    active = y_pred != 0
    hit_rate = (y_true[active] == y_pred[active]).mean()

    assert hit_rate > 0.5
    assert set(y_pred.unique()).issubset({-1, 0, 1})
