"""Does disconnection predict language outcome better than lesion load?

compare_feature_sets() is the core experiment of train_models_stacked.m,
train_models_stacked_backwards.m, train_models_hierarchical.m,
TestConnectivity.m, RunDisconnVsLesionLoad.m, RunAllRegs.m, run_new_comparison.m
and train_all_models_fuzzy_bin.m: for each task score and each model
("inducer"), cross-validate several feature sets on the same folds and compare
their held-out predictions.

Feature sets are specs (see features.py) or stacks of them:
    {"ll": STANDARD_SETS["ll"], "conn": STANDARD_SETS["conn"],
     "stack(ll,conn)": ("stack", "ll", "conn")}

A stack is fitted properly: inside each outer training fold, the level-1
models' predictions for the training patients come from an inner
cross-validation (scikit-learn's StackingRegressor). The original
(cv_stacked_parallel) trained the stacking model on outer-CV predictions that
had been produced, for the training patients, by models trained with the test
patients included.

ResidualChain replaces train_models_hierarchical.m's "levels": each level
predicts what the previous levels got wrong, with inner cross-validation so the
residuals it learns from are honest. The original reported the correlation of
the final level's predictions with the final residuals rather than with the
task scores, and never combined the levels into one prediction.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from .cv import cross_val_predict_repeated, make_folds, regression_metrics
from .features import STANDARD_SETS, make_pipeline_for, stack_blocks
from .models import make_regressor
from .stats import corrected_resampled_ttest, holm, paired_error_test, pitman_morgan


class ResidualChain(RegressorMixin, BaseEstimator):
    """levels > 1: level l is trained on [X, previous levels' predictions] to predict
    the residual y - (sum of previous levels), using inner-CV predictions for the
    training rows. The prediction is the sum over levels."""

    def __init__(self, estimator=None, levels=2, inner_k=5, seed=0):
        self.estimator, self.levels, self.inner_k, self.seed = estimator, levels, inner_k, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.models_ = []
        feats = X
        target = y.copy()
        for _ in range(self.levels):
            est = clone(self.estimator)
            k = min(self.inner_k, len(y))
            oof = cross_val_predict(est, feats, target, cv=KFold(k, shuffle=True, random_state=self.seed))
            self.models_.append(clone(self.estimator).fit(feats, target))
            feats = np.column_stack([feats, oof])
            target = target - oof
        return self

    def predict(self, X):
        feats = np.asarray(X, float)
        total = np.zeros(len(feats))
        for m in self.models_:
            p = m.predict(feats)
            total += p
            feats = np.column_stack([feats, p])
        return total


def build_estimator(set_def, slices, model, sets=None, final_model="linear", inner_k=5, levels=1, seed=0):
    """Estimator for one feature set: a selection pipeline, or a stack of them."""
    if isinstance(set_def, tuple) and set_def[0] == "stack":
        members = [(nm, make_pipeline_for(sets[nm], slices, model)) for nm in set_def[1:]]
        est = StackingRegressor(members, final_estimator=make_regressor(final_model),
                                cv=KFold(inner_k, shuffle=True, random_state=seed))
    else:
        est = make_pipeline_for(set_def, slices, model)
    if levels > 1:
        est = ResidualChain(est, levels=levels, inner_k=inner_k, seed=seed)
    return est


class ComparisonResults:
    def __init__(self, table, predictions, targets, folds_by_task, fold_mse, n_by_task):
        self.table = table                  # one row per task x model x feature set
        self.predictions = predictions      # {(task, model, set): (rows, pred)}
        self.targets = targets              # {task: (rows, y)}
        self.folds_by_task = folds_by_task
        self.fold_mse = fold_mse            # {(task, model, set): (repeats, k) fold MSEs}
        self.n_by_task = n_by_task

    def compare(self, baseline: str, other: str, holm_over_tasks: bool = True) -> pd.DataFrame:
        """Paired comparisons of `other` against `baseline` for every task and model.

        diff / relative / ci:  mean per-patient reduction in squared error (positive
                               = other better) with a bootstrap 95% CI
        p_t, p_wilcoxon:       paired one-sided tests on those per-patient differences
        p_pitman_morgan:       one-sided test that other's errors have smaller variance
                               (the paired version of the original vartest2)
        p_corrected_cv:        Nadeau-Bengio corrected resampled t-test on fold MSEs
        Holm-adjusted versions over tasks (for each model) are added."""
        rows = []
        for (task, model, st), (idx, pa) in self.predictions.items():
            if st != baseline:
                continue
            key_b = (task, model, other)
            if key_b not in self.predictions:
                continue
            _, pb = self.predictions[key_b]
            _, y = self.targets[task]
            res = paired_error_test(y, pa, pb)
            ok = np.isfinite(pa) & np.isfinite(pb)
            _, p_pm = pitman_morgan((pa - y)[ok], (pb - y)[ok])
            fa, fb = self.fold_mse[(task, model, baseline)], self.fold_mse[key_b]
            n = self.n_by_task[task]
            k = fa.shape[1]
            n_test = n / k
            _, p_cv = corrected_resampled_ttest(fa, fb, n - n_test, n_test) if k < n else (np.nan, np.nan)
            rows.append({"task": task, "model": model, "baseline": baseline, "other": other, **res,
                         "p_pitman_morgan": p_pm, "p_corrected_cv": p_cv})
        out = pd.DataFrame(rows)
        if holm_over_tasks and len(out):
            for col in ("p_t", "p_wilcoxon", "p_pitman_morgan", "p_corrected_cv"):
                out[col + "_holm"] = out.groupby("model")[col].transform(lambda s: holm(s.to_numpy()))
        return out


def compare_feature_sets(blocks: dict, Y, sets: dict | None = None, models=("linear",), k: int = 10,
                         repeats: int = 1, seed: int = 0, final_model: str = "linear", inner_k: int = 5,
                         levels: int = 1, task_names=None, n_jobs: int = 1, verbose: bool = False,
                         ) -> ComparisonResults:
    """Cross-validate every feature set with every model for every task.

    blocks: {name: (patients x features)}; must contain every block the sets use.
    Y:      (patients,) or (patients x tasks) scores; NaN = missing.
    sets:   {name: spec or ("stack", set1, set2, ...)}; default: the lesion-load,
            connectivity and combined sets of the original stacking scripts.
    Rows missing any used block or the task score are dropped for that task;
    every model and set then uses the same folds."""
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    sets = sets or {n: STANDARD_SETS[n] for n in ("ll", "conn_matched", "conn", "ll+conn_matched", "ll+conn")}
    used = set()
    for s in sets.values():
        if isinstance(s, tuple):
            for m in s[1:]:
                used |= set(sets[m])
        else:
            used |= set(s)
    X, slices = stack_blocks({b: blocks[b] for b in blocks if b in used})
    names = task_names or [f"task_{j}" for j in range(Y.shape[1])]
    rows, preds, targets, folds_by_task, fold_mse, n_by = [], {}, {}, {}, {}, {}
    for t in range(Y.shape[1]):
        ok = np.isfinite(Y[:, t]) & np.isfinite(X).all(1)
        idx = np.flatnonzero(ok)
        y = Y[idx, t]
        Xt = X[idx]
        folds = make_folds(len(y), k, repeats, seed + t)
        folds_by_task[names[t]] = folds
        targets[names[t]] = (idx, y)
        n_by[names[t]] = len(y)
        for model in models:
            for sname, sdef in sets.items():
                est = build_estimator(sdef, slices, model, sets, final_model, inner_k, levels, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cvp = cross_val_predict_repeated(est, Xt, y, folds, n_jobs=n_jobs)
                m = regression_metrics(y, cvp.pred)
                rows.append({"task": names[t], "model": model, "set": sname, **m})
                preds[(names[t], model, sname)] = (idx, cvp.pred)
                fold_mse[(names[t], model, sname)] = cvp.fold_mse
                if verbose:
                    print(f"{names[t]:>10} {model:>22} {sname:>18}  r = {m['r']:.3f}  RMSE = {m['rmse']:.3f}",
                          flush=True)
    return ComparisonResults(pd.DataFrame(rows), preds, targets, folds_by_task, fold_mse, n_by)
