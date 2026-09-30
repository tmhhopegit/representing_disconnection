"""How much data do the models need?

learning_curve       held-out accuracy as the training set grows (repeated random
                     subsamples of each outer training fold)
pls_score_stability  PLS_LearnCurve.m's question: how close are the PLS scores of
                     held-out patients, from a model fitted to a subsample, to
                     their scores from the model fitted to all training patients?

PLS_LearnCurve.m compared scores without aligning the components. PLS
components are only defined up to sign (and near-ties can swap order), so a
flipped component counted as a large error however stable the model was; here
each component is sign-aligned (and matched by absolute correlation) before
comparison. It also loaded the reference scores from a file computed on all
patients (test patients included), and saved one file per fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression

from .cv import make_folds, regression_metrics


def learning_curve(estimator, X, y, sizes, repeats=10, k=5, seed=0) -> pd.DataFrame:
    X = np.asarray(X, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rows = []
    for _, tr, te in make_folds(len(y), k, 1, seed):
        for size in sizes:
            if size > len(tr):
                continue
            for rep in range(repeats):
                sub = rng.choice(tr, size, replace=False)
                pred = clone(estimator).fit(X[sub], y[sub]).predict(X[te])
                m = regression_metrics(y[te], pred)
                rows.append({"size": size, "repeat": rep, "rmse": m["rmse"], "r": m["r"]})
    return pd.DataFrame(rows).groupby("size")[["rmse", "r"]].agg(["mean", "std"])


def _aligned_error(ref, est):
    C = np.abs(np.corrcoef(ref.T, est.T)[:ref.shape[1], ref.shape[1]:])
    C = np.nan_to_num(C)
    rows, cols = linear_sum_assignment(-C)
    est = est[:, cols]
    signs = np.sign(np.sum(ref * est, 0))
    signs[signs == 0] = 1
    return float(np.mean((ref - est * signs) ** 2))


def pls_score_stability(X, Y, n_components, sizes, repeats=10, k=5, seed=0) -> pd.DataFrame:
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    rng = np.random.default_rng(seed)
    rows = []
    for _, tr, te in make_folds(len(X), k, 1, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        Z = (X - mu) / sd
        full = PLSRegression(n_components, scale=False).fit(Z[tr], Y[tr])
        ref = full.transform(Z[te])
        for size in sizes:
            for rep in range(repeats):
                sub = rng.choice(tr, size, replace=False)
                m = PLSRegression(n_components, scale=False).fit(Z[sub], Y[sub])
                rows.append({"size": size, "repeat": rep, "error": _aligned_error(ref, m.transform(Z[te]))})
    return pd.DataFrame(rows).groupby("size")["error"].agg(["mean", "std"])
