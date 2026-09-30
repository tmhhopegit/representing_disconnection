"""Cross-validation with fixed, balanced, reusable folds.

The original scripts drew folds with randi(K, N, R) (unequal, sometimes empty
folds), and many drew new folds for every model, so differences between models
included fold noise. Here folds are made once per dataset from a seed and
shared by every model being compared (a paired design).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(n: int, k: int = 10, repeats: int = 1, seed: int = 0, stratify=None):
    """[(repeat, train_idx, test_idx)]; k >= n gives leave-one-out (one repeat)."""
    idx = np.arange(n)
    if k >= n:
        return [(0, np.delete(idx, i), np.array([i])) for i in range(n)]
    out = []
    rng = np.random.default_rng(seed)
    for r in range(repeats):
        rs = int(rng.integers(2 ** 31))
        if stratify is None:
            split = KFold(k, shuffle=True, random_state=rs).split(idx)
        else:
            split = StratifiedKFold(k, shuffle=True, random_state=rs).split(idx, stratify)
        out += [(r, tr, te) for tr, te in split]
    return out


def regression_metrics(y, pred) -> dict:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    n = len(y)
    if n > 2 and pred.std() > 0 and y.std() > 0:
        r, p = stats.pearsonr(pred, y)
    else:
        r, p = np.nan, np.nan
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": n, "r": float(r), "p": float(p), "R2": float(1 - sse / sst) if sst > 0 else np.nan,
            "rmse": float(np.sqrt(sse / n)), "mae": float(np.abs(y - pred).mean())}


@dataclass
class CVPredictions:
    pred: np.ndarray            # (n,) held-out predictions averaged over repeats
    pred_repeats: np.ndarray    # (n, repeats)
    fold_mse: np.ndarray        # (repeats, k) mean squared error of each test fold
    fold_n: np.ndarray          # (repeats, k) test-fold sizes

    def metrics(self, y):
        return regression_metrics(y, self.pred)


def _fit_predict(est, X, y, tr, te):
    m = clone(est).fit(X[tr], y[tr])
    return m.predict(X[te])


def cross_val_predict_repeated(estimator, X, y, folds, n_jobs: int = 1) -> CVPredictions:
    """Held-out predictions for every row from every repeat of the given folds."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    R = max(f[0] for f in folds) + 1
    if n_jobs == 1:
        outs = [_fit_predict(estimator, X, y, tr, te) for _, tr, te in folds]
    else:
        outs = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(estimator, X, y, tr, te) for _, tr, te in folds)
    P = np.full((n, R), np.nan)
    per_rep = {}
    for (r, tr, te), p in zip(folds, outs):
        P[te, r] = p
        per_rep.setdefault(r, []).append((np.mean((p - y[te]) ** 2), len(te)))
    K = max(len(v) for v in per_rep.values())
    mse = np.full((R, K), np.nan); nn = np.zeros((R, K), int)
    for r, v in per_rep.items():
        for j, (e, m) in enumerate(v):
            mse[r, j] = e; nn[r, j] = m
    return CVPredictions(np.nanmean(P, 1), P, mse, nn)
