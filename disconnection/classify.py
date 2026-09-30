"""Classifying patients as impaired or not (TrainAllEnsembleKNN.m, trainClassifier*.m,
classfun_tree*.m).

classify_cv cross-validates any classifier from models.CLASSIFIERS, with the
classes balanced inside each training fold by random over-sampling of the
smaller class (as TrainAllEnsembleKNN.m did), and returns the AUC of the
held-out scores.

Fixes relative to the originals:
- Folds are balanced and stratified (the original used randi, so a fold could
  contain no impaired patients).
- The auto-generated trainClassifier*.m files hard-coded the number of columns
  (164, 124, 56), so they only worked for one dataset; TrainAllEnsembleKNN.m
  called trainClassifier with a test set and four outputs, which the later
  version of trainClassifier.m (one input, three outputs) does not accept.
- trainClassifier_Any.m's RUSBoost branch replaced the predictors with a fixed
  56-column table before cross-validation.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .cv import make_folds
from .models import make_classifier, scores_for_auc


def oversample(X, y, rng):
    """Resample the smaller class with replacement up to the size of the larger one."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    big = counts.max()
    idx = np.concatenate([np.flatnonzero(y == c) if n == big else
                          np.concatenate([np.flatnonzero(y == c), rng.choice(np.flatnonzero(y == c), big - n)])
                          for c, n in zip(classes, counts)])
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def classify_cv(X, labels, model="svm_gaussian_medium", k=10, repeats=10, balance=True, seed=0) -> dict:
    """Cross-validated classification. labels: 0/1. Returns AUC, accuracy, balanced
    accuracy, and the held-out scores (averaged over repeats) and predicted labels
    (majority over repeats)."""
    X = np.asarray(X, float); y = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    folds = make_folds(len(y), k, repeats, seed, stratify=y if k < len(y) else None)
    R = max(f[0] for f in folds) + 1
    S = np.full((len(y), R), np.nan)
    L = np.full((len(y), R), np.nan)
    base = make_classifier(model) if isinstance(model, (str, int)) else model
    for r, tr, te in folds:
        Xtr, ytr = (oversample(X[tr], y[tr], rng) if balance else (X[tr], y[tr]))
        m = clone(base).fit(Xtr, ytr)
        S[te, r] = scores_for_auc(m, X[te])
        L[te, r] = m.predict(X[te])
    s = np.nanmean(S, 1)
    lab = (np.nanmean(L, 1) >= 0.5).astype(int)
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return {"auc": float(auc), "accuracy": float((lab == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, lab)), "scores": s, "predicted": lab}


def misclassification_probability(model, X_train, y_train, X_test, y_test) -> float:
    """Sum over test patients of 1 - P(true class) (classfun_tree1/2/3.m, a criterion
    for sequential feature selection)."""
    m = clone(model).fit(X_train, y_train)
    proba = m.predict_proba(X_test)
    col = np.searchsorted(m.classes_, y_test)
    return float((1 - proba[np.arange(len(y_test)), col]).sum())
