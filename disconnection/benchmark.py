"""Synthetic benchmarks.

representation_benchmark
    Does the comparison framework pick the right representation? Behaviour is
    simulated from lesion load ("load"), from disconnection ("disconnection"),
    from both, or from neither; every feature set is cross-validated with several
    models, and each set is compared with lesion load alone.

pitfalls
    Quantifies the analysis choices in the original code that could mislead:
    1. selecting features on all patients before cross-validation;
    2. comparing two models' prediction errors with an F-test for independent
       samples (vartest2) instead of a paired test;
    3. comparing models with a Wilcoxon test on the 10 fold scores of 5x2
       cross-validation (untitled4.m, compile_fuzbin_res*.m);
    4. searching over subsets of CV repeats for the most favourable result
       (compile_fuzbin_res2.m);
    5. training a stacking model on outer-CV predictions (cv_stacked_parallel).
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from .compare import compare_feature_sets
from .cv import cross_val_predict_repeated, make_folds, regression_metrics
from .features import STANDARD_SETS, correlation_p, make_pipeline_for, stack_blocks
from .stats import corrected_resampled_ttest, cv5x2_f_test, f_test_variances, pitman_morgan
from .synthetic import make_brain, simulate

DEFAULT_SETS = {**{n: STANDARD_SETS[n] for n in ("ll", "disc", "conn_matched", "conn", "ll+conn")},
                "stack(ll,conn)": ("stack", "ll", "conn")}
DEFAULT_MODELS = ("linear", "ridge", "bagged_trees", "svm_gaussian_10")
TRUTH_SEEDS = {"load": 11, "disconnection": 23, "both": 37, "none": 51}


def representation_benchmark(truths=("load", "disconnection", "both", "none"), datasets=3, n=200, n_tasks=2,
                             effect=0.8, models=DEFAULT_MODELS, sets=None, k=10, repeats=1, n_jobs=1,
                             verbose=True):
    sets = sets or DEFAULT_SETS
    brain = make_brain(seed=7)
    scores, comps = [], []
    for truth in truths:
        for d in range(datasets):
            t0 = time.time()
            p = simulate(n, n_tasks, truth=truth, effect=effect, brain=brain, seed=1000 * d + TRUTH_SEEDS.get(truth, 0))
            res = compare_feature_sets(p.feature_sets(), p.behaviour, sets, models=models, k=k, repeats=repeats,
                                       seed=d, n_jobs=n_jobs)
            tab = res.table.assign(truth=truth, dataset=d)
            scores.append(tab)
            for other in sets:
                if other != "ll":
                    comps.append(res.compare("ll", other).assign(truth=truth, dataset=d))
            if verbose:
                print(f"{truth} dataset {d}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(scores, ignore_index=True), pd.concat(comps, ignore_index=True)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby(["truth", "model", "set"]).r.mean().unstack("set").round(3)


def summarise_comparisons(comps: pd.DataFrame, p_col="p_t_holm") -> pd.DataFrame:
    """Per truth and feature set: mean relative MSE reduction vs lesion load, and the
    share of (dataset, task, model) comparisons significantly better / worse."""
    g = comps.groupby(["truth", "other"])
    worse = comps.assign(w=comps["diff"] < 0).groupby(["truth", "other"])
    return pd.DataFrame({"relative_gain": g["relative"].mean(),
                         "share_better": g[p_col].apply(lambda s: float((s < 0.05).mean())),
                         "share_worse_point": worse["w"].mean()}).round(3)


# ----------------------------------------------------------------------------- pitfalls
def pit_selection_leakage(datasets=30, n=150, seed=0):
    """Mean CV r with connection features selected (p < .05) on all patients before
    CV, versus inside each training fold. Behaviour depends on demographics only."""
    brain = make_brain(seed=5)
    pre, inside = [], []
    for d in range(datasets):
        p = simulate(n, 1, truth="none", brain=brain, seed=seed + d)
        y = p.behaviour[:, 0]
        folds = make_folds(n, 10, 1, d)
        keep = np.flatnonzero(correlation_p(p.edge_disconnection, y) < 0.05)
        Xp = np.hstack([p.demographics, p.edge_disconnection[:, keep]])
        pre.append(regression_metrics(y, cross_val_predict_repeated(Ridge(1.0), Xp, y, folds).pred)["r"])
        Xs, sl = stack_blocks({"dem": p.demographics, "conn": p.edge_disconnection})
        pipe = make_pipeline_for({"dem": "all", "conn": "p05"}, sl, Ridge(1.0))
        inside.append(regression_metrics(y, cross_val_predict_repeated(pipe, Xs, y, folds).pred)["r"])
    dem_only = []
    for d in range(datasets):
        p = simulate(n, 1, truth="none", brain=brain, seed=seed + d)
        y = p.behaviour[:, 0]
        dem_only.append(regression_metrics(y, cross_val_predict_repeated(Ridge(1.0), p.demographics, y,
                                                                          make_folds(n, 10, 1, d)).pred)["r"])
    return {"r_selected_before_cv": float(np.mean(pre)), "r_selected_in_fold": float(np.mean(inside)),
            "r_demographics_only": float(np.mean(dem_only))}


def pit_paired_variance_tests(sims=4000, n=150, error_corr=0.8, ratio=1.15, seed=0):
    """Type-I error (equal accuracy) and power (model 2's error variance smaller by
    `ratio`) of the independent-samples F-test (vartest2) and the paired
    Pitman-Morgan test, for errors correlated as two models' errors on the same
    patients are."""
    rng = np.random.default_rng(seed)
    s = np.sqrt((1 - error_corr) / error_corr)
    out = {}
    for label, rt in (("null", 1.0), ("alternative", ratio)):
        f = pm = 0
        for _ in range(sims):
            c = rng.normal(size=n)
            e1 = c + s * rng.normal(size=n)
            e2 = (c + s * rng.normal(size=n)) / np.sqrt(rt)
            f += f_test_variances(e1, e2)[1] < 0.05
            pm += pitman_morgan(e1, e2)[1] < 0.05
        out[f"{label}_f_test"] = f / sims
        out[f"{label}_pitman_morgan"] = pm / sims
    return out


def _equal_models(rng, n):
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + X[:, 1] + 1.5 * rng.normal(size=n)
    return X, y


def pit_fold_score_tests(datasets=200, n=150, seed=0):
    """Two equally good models (each uses one of two equally informative predictors),
    compared by 5x2 cross-validation. Share of datasets with p < .05 (one-sided):
    Wilcoxon on the 10 fold scores (the original), Alpaydin's 5x2cv F-test, and
    the Nadeau-Bengio corrected t-test."""
    rng = np.random.default_rng(seed)
    rej = {"wilcoxon_fold_scores": 0, "cv5x2_f": 0, "corrected_t": 0}
    for d in range(datasets):
        X, y = _equal_models(rng, n)
        folds = make_folds(n, 2, 5, d)
        fa = cross_val_predict_repeated(LinearRegression(), X[:, [0]], y, folds).fold_mse
        fb = cross_val_predict_repeated(LinearRegression(), X[:, [1]], y, folds).fold_mse
        rej["wilcoxon_fold_scores"] += stats.wilcoxon(fa.ravel(), fb.ravel(), alternative="greater").pvalue < 0.05
        rej["cv5x2_f"] += cv5x2_f_test(fa - fb)[1] < 0.05
        rej["corrected_t"] += corrected_resampled_ttest(fa, fb, n / 2, n / 2)[1] < 0.05
    return {k: v / datasets for k, v in rej.items()}


def pit_cherry_picking(datasets=10, n_tasks=13, n=150, seed=0):
    """13 tasks, two equally good models, 10 repeats of 2-fold CV. Number of tasks
    'significant' (Wilcoxon on fold scores) using the first 5 repeats, versus the
    best of all 252 choices of 5 repeats (the search of compile_fuzbin_res2.m)."""
    rng = np.random.default_rng(seed)
    first, best = [], []
    for d in range(datasets):
        pairs = []
        for t in range(n_tasks):
            X, y = _equal_models(rng, n)
            folds = make_folds(n, 2, 10, d * n_tasks + t)
            fa = cross_val_predict_repeated(LinearRegression(), X[:, [0]], y, folds).fold_mse
            fb = cross_val_predict_repeated(LinearRegression(), X[:, [1]], y, folds).fold_mse
            pairs.append((fa, fb))

        def nsig(reps):
            reps = list(reps)
            return sum(stats.wilcoxon(fa[reps].ravel(), fb[reps].ravel(), alternative="greater").pvalue < 0.05
                       for fa, fb in pairs)
        first.append(nsig(range(5)))
        best.append(max(nsig(c) for c in itertools.combinations(range(10), 5)))
    return {"significant_first_5_repeats": float(np.mean(first)), "significant_best_subset": float(np.mean(best)),
            "tasks": n_tasks, "expected_by_chance": 0.05 * n_tasks}


def pit_stacking_leak(datasets=40, n=120, seed=0):
    """Pure noise, two blocks of 60 predictors, tree base models. CV r of a stacking
    model trained on outer-CV predictions (the original) vs properly nested."""
    rng = np.random.default_rng(seed)
    orig, nested = [], []
    base = lambda: DecisionTreeRegressor(min_samples_leaf=4, random_state=0)
    for d in range(datasets):
        A, B, y = rng.normal(size=(n, 60)), rng.normal(size=(n, 60)), rng.normal(size=n)
        folds = make_folds(n, 10, 5, d)
        pa = cross_val_predict_repeated(base(), A, y, folds).pred
        pb = cross_val_predict_repeated(base(), B, y, folds).pred
        orig.append(regression_metrics(y, cross_val_predict_repeated(base(), np.column_stack([pa, pb]), y,
                                                                      folds).pred)["r"])
        X, sl = stack_blocks({"a": A, "b": B})
        st = StackingRegressor([("a", make_pipeline_for({"a": "all"}, sl, base())),
                                ("b", make_pipeline_for({"b": "all"}, sl, base()))], final_estimator=base(),
                               cv=KFold(5, shuffle=True, random_state=0))
        nested.append(regression_metrics(y, cross_val_predict_repeated(st, X, y, make_folds(n, 10, 1, d)).pred)["r"])
    return {"r_original_stacking": float(np.mean(orig)), "r_nested_stacking": float(np.mean(nested))}


def pitfalls(quick=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = 0.25 if quick else 1.0
        return {"selection_leakage": pit_selection_leakage(int(30 * f) or 3),
                "paired_variance_tests": pit_paired_variance_tests(int(4000 * f)),
                "fold_score_tests": pit_fold_score_tests(int(200 * f)),
                "cherry_picking": pit_cherry_picking(max(2, int(10 * f))),
                "stacking_leak": pit_stacking_leak(max(4, int(40 * f)))}


def figure(scores: pd.DataFrame, path=None):
    """Small multiples: one panel per ground truth; mean CV r of each feature set
    (rows) for each model (dots), datasets and tasks averaged."""
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0", "#c9a227"]
    truths = [t for t in ("load", "disconnection", "both", "none") if t in set(scores.truth)]
    sets = list(dict.fromkeys(scores.set))
    models = list(dict.fromkeys(scores.model))
    per = scores.groupby(["truth", "model", "set"]).r.mean().reset_index()
    fig, axes = plt.subplots(1, len(truths), figsize=(3.3 * len(truths), 0.45 * len(sets) + 1.9), sharey=True,
                             sharex=True, facecolor=surface, squeeze=False)
    for ax, truth in zip(axes[0], truths):
        ax.set_facecolor(surface)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.grid(True, axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink2, labelsize=8)
        for j, m in enumerate(models):
            sub = per[(per.truth == truth) & (per.model == m)].set_index("set").reindex(sets)
            off = (j - (len(models) - 1) / 2) * 0.13
            ax.scatter(sub.r, np.arange(len(sets)) + off, s=20, color=palette[j % len(palette)], zorder=3,
                       edgecolor=surface, lw=0.8, label=m)
        ax.set_title(f"behaviour from: {truth}", fontsize=9, color=ink)
        ax.set_xlabel("cross-validated r", fontsize=8, color=ink2)
    axes[0, 0].set_yticks(range(len(sets)))
    axes[0, 0].set_yticklabels(sets, fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Which predictors recover the simulated truth? (mean over datasets and tasks)", fontsize=10,
                 color=ink, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=surface, bbox_inches="tight")
    return fig
