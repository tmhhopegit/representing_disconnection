"""Choosing predictors.

    stepwise_cv            forward/backward selection by cross-validated error
                           (OneStepSelect.m)
    best_subsets           every subset of predictors, ranked by AIC, keeping
                           models whose terms are all significant (bestregress.m)
    pls_backward           backward elimination of predictors for a PLS model by
                           cross-validated error (PLORAS_PLS.m)
    nested_selection_cv    honest accuracy of any "select, then fit" procedure

Any selection scored on the same patients it was chosen on is optimistic;
nested_selection_cv repeats the selection inside each outer training fold.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from .cv import make_folds, regression_metrics


def _cv_mse(X, y, folds, model):
    pred = np.full((len(y), max(f[0] for f in folds) + 1), np.nan)
    for r, tr, te in folds:
        if X.shape[1] == 0:
            pred[te, r] = y[tr].mean()
        else:
            pred[te, r] = clone(model).fit(X[tr], y[tr]).predict(X[te]).ravel()
    return float(np.mean((np.nanmean(pred, 1) - y) ** 2))


def stepwise_cv(X, y, model=None, k=10, repeats=10, seed=0, start=()):
    """Alternating forward and backward steps, keeping a change only if it lowers the
    cross-validated mean squared error (OneStepSelect.m, which maximised 1/MSE).

    The same folds are used for every candidate. The original drew new random
    folds (randi) inside every evaluation, so candidates were compared on
    different splits and the search partly followed fold noise.
    Returns (selected columns, CV MSE of the selected model)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    model = model or LinearRegression()
    folds = make_folds(len(y), k, repeats, seed)
    cur = list(start)
    best = _cv_mse(X[:, cur], y, folds, model)
    while True:
        improved = False
        while True:                                   # forward
            cands = [j for j in range(X.shape[1]) if j not in cur]
            if not cands:
                break
            scores = [_cv_mse(X[:, cur + [j]], y, folds, model) for j in cands]
            j = int(np.argmin(scores))
            if scores[j] < best:
                best, cur, improved = scores[j], cur + [cands[j]], True
            else:
                break
        while len(cur) > 0:                            # backward
            scores = [_cv_mse(X[:, [c for c in cur if c != j]], y, folds, model) for j in cur]
            j = int(np.argmin(scores))
            if scores[j] <= best:
                best, cur, improved = scores[j], [c for c in cur if c != cur[j]], True
            else:
                break
        if not improved:
            break
    return sorted(cur), best


def best_subsets(X, y, labels=None, alpha=0.05, max_size=None):
    """All-subsets regression (bestregress.m).

    Fits every subset of columns (up to max_size), keeps the models in which every
    predictor is significant at alpha, and ranks them by AIC. Returns a list of
    dicts sorted by AIC, each with the columns, labels, coefficients, p-values,
    SSE, R^2, AIC, delta-AIC and Akaike weight.
    (The original labelled its second statistic "number of factors" but stored R^2
    there.)"""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    labels = labels or [f"x{j}" for j in range(p)]
    max_size = max_size or p
    out = []
    sst = ((y - y.mean()) ** 2).sum()
    for size in range(1, max_size + 1):
        for cols in combinations(range(p), size):
            A = np.column_stack([np.ones(n), X[:, cols]])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ b
            sse = (resid ** 2).sum()
            df = n - A.shape[1]
            if df <= 0:
                continue
            cov = sse / df * np.linalg.pinv(A.T @ A)
            se = np.sqrt(np.diag(cov))[1:]
            pvals = 2 * stats.t.sf(np.abs(b[1:] / se), df)
            if np.all(pvals < alpha):
                out.append({"columns": cols, "labels": [labels[c] for c in cols], "intercept": b[0],
                            "coef": b[1:], "p": pvals, "sse": sse, "R2": 1 - sse / sst,
                            "aic": n * np.log(sse / n) + 2 * (size + 1)})
    out.sort(key=lambda m: m["aic"])
    if out:
        a = np.array([m["aic"] for m in out])
        d = a - a.min()
        w = np.exp(-d / 2) / np.exp(-d / 2).sum()
        for m, di, wi in zip(out, d, w):
            m["delta_aic"], m["weight"] = di, wi
    return out


def pls_backward(X, Y, n_components=None, k=10, repeats=10, seed=0, min_features=1):
    """Backward elimination for PLS regression by cross-validated MSE (PLORAS_PLS.m).

    At each step the predictor whose removal lowers the MSE most is removed; stop
    when no removal helps. n_components: fixed number of components (default: the
    number chosen by CV on all predictors, from 1..min(10, p)).
    Differences from the original:
    - the best removal is taken, not the first one that helps (the original
      scanned from the last predictor and stopped at the first improvement);
    - the baseline and the candidates use the same folds and repeats (the
      original used 100 repeats for the baseline and 10 for candidates);
    - the number of components is fixed rather than "number of predictors - 1"
      (which made PLS almost identical to least squares).
    Returns (selected columns, MSE path list)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    folds = make_folds(len(Y), k, repeats, seed)

    def mse(cols, nc):
        P = np.zeros((len(Y), Y.shape[1], max(f[0] for f in folds) + 1))
        for r, tr, te in folds:
            Xtr, Xte = X[tr][:, cols], X[te][:, cols]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1
            ym, ys = Y[tr].mean(0), Y[tr].std(0)
            ys[ys == 0] = 1
            m = PLSRegression(min(nc, len(cols)), scale=False).fit((Xtr - mu) / sd, (Y[tr] - ym) / ys)
            P[te, :, r] = m.predict((Xte - mu) / sd) * ys + ym
        return float(np.mean((P.mean(2) - Y) ** 2))

    cols = list(range(X.shape[1]))
    if n_components is None:
        cands = range(1, min(10, X.shape[1]) + 1)
        n_components = min(cands, key=lambda c: mse(cols, c))
    best = mse(cols, n_components)
    path = [(list(cols), best)]
    while len(cols) > min_features:
        scores = [mse([c for c in cols if c != j], n_components) for j in cols]
        j = int(np.argmin(scores))
        if scores[j] >= best:
            break
        best = scores[j]
        cols = [c for c in cols if c != cols[j]]
        path.append((list(cols), best))
    return cols, path


def nested_selection_cv(select, X, y, model=None, k=10, seed=0):
    """Outer-CV accuracy of: columns = select(Xtrain, ytrain); fit model on them.

    select: function (X, y) -> list of column indices. Returns metrics, held-out
    predictions and the columns chosen in each outer fold."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    model = model or LinearRegression()
    pred = np.full(len(y), np.nan)
    chosen = []
    for _, tr, te in make_folds(len(y), k, 1, seed):
        cols = list(select(X[tr], y[tr]))
        chosen.append(cols)
        if cols:
            pred[te] = clone(model).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])
        else:
            pred[te] = y[tr].mean()
    return {**regression_metrics(y, pred), "pred": pred, "selected": chosen}
