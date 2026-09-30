"""Predictor blocks and feature selection inside cross-validation folds.

The analyses combine blocks of predictors, e.g. demographics ("dem"), regional
lesion load ("ll"), regional disconnection ("disc", ChaCo) and edge-wise
disconnection ("conn"). A *feature-set spec* says which blocks to use and how
to filter each one, e.g.

    {"dem": "all", "ll": "bonferroni", "conn": ("match", "ll")}

Rules (all computed from the training patients only):
    "all"                 every column
    "bonferroni"          correlation with the target at p < 0.05 / (number of
                          non-constant columns)       (cv_component_parallel)
    "bonferroni_or_p05"   as above, falling back to p < 0.05 when nothing
                          survives                    (cv_parallel)
    "p05"                 p < 0.05 uncorrected
    ("top", k)            the k columns with the smallest p-values
    ("match", block)      as many columns (smallest p) as were selected from
                          another block - the "restricted" connectivity sets,
                          matched in size to the lesion-load set
    "perm"                p below the 5th percentile of the smallest p-value
                          over 1000 permutations of the target (FeatureSelect
                          in MakeRegMatrices.m)

BlockSelector does the selection as the first step of a scikit-learn pipeline,
so wrapping it in cross-validation keeps selection inside the folds. The
original MakeRegMatrices.m and TestConnectivity.m selected features using all
patients before cross-validation, which inflates accuracy (see README).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .models import make_regressor

STANDARD_SETS = {
    # the five sets of cv_component_parallel (train_models_stacked.m etc.)
    "ll": {"dem": "all", "ll": "bonferroni"},
    "conn_matched": {"dem": "all", "conn": ("match", "ll")},
    "conn": {"dem": "all", "conn": "bonferroni"},
    "ll+conn_matched": {"dem": "all", "ll": "bonferroni", "conn": ("match", "ll")},
    "ll+conn": {"dem": "all", "ll": "bonferroni", "conn": "bonferroni"},
    # regional (ChaCo) disconnection, as TestConnectivity.m / RunDisconnVsLesionLoad.m
    "disc": {"dem": "all", "disc": "bonferroni"},
    "ll+disc": {"dem": "all", "ll": "bonferroni", "disc": "bonferroni"},
}


def correlation_p(X, y):
    """Two-sided p-values of the Pearson correlation of each column with y (1 for
    constant columns). Vectorised."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    sx = np.sqrt((Xc ** 2).sum(0))
    sy = np.sqrt((yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc.T @ yc) / (sx * sy)
    r = np.clip(np.nan_to_num(r, nan=0.0), -1, 1)
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / np.maximum(1 - r ** 2, 1e-300))
    p = 2 * stats.t.sf(np.abs(t), df)
    p[sx == 0] = 1.0
    return p


def permutation_threshold(X, y, n_perm=1000, alpha=0.05, seed=0):
    """p-value threshold controlling the family-wise error: the alpha-quantile of
    the smallest p-value across columns when y is permuted."""
    rng = np.random.default_rng(seed)
    mins = np.array([correlation_p(X, rng.permutation(y)).min() for _ in range(n_perm)])
    return np.quantile(mins, alpha)


class BlockSelector(TransformerMixin, BaseEstimator):
    """Select columns block by block, from training data only.

    slices: {block name: (start, stop)} column ranges of each block in X.
    spec:   {block name: rule} (see module docstring); blocks not in spec are dropped.
    """

    def __init__(self, slices=None, spec=None, alpha=0.05, n_perm=1000, seed=0):
        self.slices, self.spec, self.alpha, self.n_perm, self.seed = slices, spec, alpha, n_perm, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        chosen, counts = {}, {}
        order = sorted(self.spec, key=lambda b: isinstance(self.spec[b], tuple) and self.spec[b][0] == "match")
        for block in order:
            rule = self.spec[block]
            a, b = self.slices[block]
            Xb = X[:, a:b]
            p = correlation_p(Xb, y) if rule != "all" else None
            testable = int((Xb.std(0) > 0).sum())
            if rule == "all":
                idx = np.arange(b - a)
            elif rule in ("bonferroni", "bonferroni_or_p05"):
                idx = np.flatnonzero(p < self.alpha / max(testable, 1))
                if rule == "bonferroni_or_p05" and len(idx) == 0:
                    idx = np.flatnonzero(p < self.alpha)
            elif rule == "p05":
                idx = np.flatnonzero(p < self.alpha)
            elif rule == "perm":
                thr = permutation_threshold(Xb, y, self.n_perm, self.alpha, self.seed)
                idx = np.flatnonzero(p < thr)
            elif isinstance(rule, tuple) and rule[0] == "top":
                idx = np.argsort(p, kind="stable")[:rule[1]]
            elif isinstance(rule, tuple) and rule[0] == "match":
                idx = np.argsort(p, kind="stable")[:counts.get(rule[1], 0)]
            else:
                raise ValueError(f"unknown selection rule {rule!r}")
            chosen[block] = a + np.sort(idx)
            counts[block] = len(idx)
        self.selected_ = {k: chosen[k] for k in self.spec}
        self.columns_ = np.concatenate([self.selected_[k] for k in self.spec]).astype(int)
        self.counts_ = counts
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if len(self.columns_) == 0:
            return np.zeros((len(X), 1))
        return X[:, self.columns_]


def stack_blocks(blocks: dict, names=None):
    """Horizontally stack named blocks; returns (X, slices)."""
    names = list(names or blocks)
    mats, slices, start = [], {}, 0
    for nme in names:
        M = np.asarray(blocks[nme], float)
        M = M[:, None] if M.ndim == 1 else M
        mats.append(M)
        slices[nme] = (start, start + M.shape[1])
        start += M.shape[1]
    return np.hstack(mats), slices


def make_pipeline_for(spec, slices, model="linear", **selector_kw):
    """Pipeline: in-fold block selection, then the regression model."""
    est = make_regressor(model) if isinstance(model, (str, int, np.integer)) else model
    return Pipeline([("select", BlockSelector(slices, spec, **selector_kw)), ("model", est)])
