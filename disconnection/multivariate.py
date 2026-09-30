"""Predicting behaviour (change) from a whole-brain map via its principal components
(BigMultiPredict2.m; trainClassifier_PCA.m used PCA keeping 95% of the variance).

pca_predict adjusts for covariates, reduces the map to principal components and
fits a model (Gaussian process or least squares), all inside each training
fold. Optionally, components are chosen by forward or backward sequential
selection within the training fold (BigMultiPredict2.m's sequentialfs).

Fixes relative to BigMultiPredict2.m:
- covariates were regressed out of the outcome and every voxel using all
  patients before cross-validation;
- folds were drawn with randi (unequal and possibly empty);
- all principal components of the training set were used (as many as training
  patients), which lets the model fit the training data exactly.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

from .cv import make_folds, regression_metrics
from .models import make_regressor


def _residualise(Y, C, coef=None):
    A = np.column_stack([np.ones(len(Y)), C]) if C is not None and np.size(C) else np.ones((len(Y), 1))
    if coef is None:
        coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    return Y - A @ coef, coef


def pca_predict(X, y, covariates=None, model="gp_squared_exponential", variance=0.95, max_components=20,
                select=None, k=10, repeats=1, seed=0) -> dict:
    """Cross-validated prediction of y from the principal components of X.

    variance:        keep the components explaining this fraction of the training
                     variance (capped at max_components)
    select:          None, "forward" or "backward": sequential selection of
                     components by inner 5-fold CV with the chosen model
    Returns metrics (against the in-fold covariate-adjusted y), predictions, the
    adjusted outcome and the number of components per fold."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    C = None if covariates is None else np.asarray(covariates, float)
    base = make_regressor(model) if isinstance(model, (str, int)) else model
    folds = make_folds(len(y), k, repeats, seed)
    R = max(f[0] for f in folds) + 1
    P = np.full((len(y), R), np.nan)
    A = np.full((len(y), R), np.nan)
    ncomp = []
    for r, tr, te in folds:
        Ctr = None if C is None else C[tr]
        Cte = None if C is None else C[te]
        ytr, cy = _residualise(y[tr], Ctr)
        yte, _ = _residualise(y[te], Cte, cy)
        Xtr, cx = _residualise(X[tr], Ctr)
        Xte, _ = _residualise(X[te], Cte, cx)
        mu = Xtr.mean(0)
        U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        frac = np.cumsum(S ** 2) / np.sum(S ** 2)
        c = int(min(np.searchsorted(frac, variance) + 1, max_components, len(tr) - 2))
        Ttr = (Xtr - mu) @ Vt[:c].T
        Tte = (Xte - mu) @ Vt[:c].T
        cols = np.arange(c)
        if select in ("forward", "backward") and c > 1:
            sfs = SequentialFeatureSelector(clone(base), direction=select,
                                            n_features_to_select="auto", tol=1e-6 if select == "forward" else -1e-6,
                                            cv=5, scoring="neg_mean_squared_error").fit(Ttr, ytr)
            cols = np.flatnonzero(sfs.get_support())
        m = clone(base).fit(Ttr[:, cols], ytr)
        P[te, r] = m.predict(Tte[:, cols])
        A[te, r] = yte
        ncomp.append(len(cols))
    pred, adj = np.nanmean(P, 1), np.nanmean(A, 1)
    return {**regression_metrics(adj, pred), "pred": pred, "adjusted": adj, "ncomp": ncomp}
