"""Tests on synthetic data. Run: pytest tests/"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from disconnection import benchmark
from disconnection.classify import classify_cv, misclassification_probability, oversample
from disconnection.compare import ResidualChain, compare_feature_sets
from disconnection.cv import cross_val_predict_repeated, make_folds, regression_metrics
from disconnection.data import long_format, select_cohort
from disconnection.features import (STANDARD_SETS, BlockSelector, correlation_p, make_pipeline_for,
                                    permutation_threshold, stack_blocks)
from disconnection.learning import learning_curve, pls_score_stability
from disconnection.models import (CLASSIFIER_MODES, CLASSIFIERS, REGRESSION_MODES, REGRESSORS, BisquareRegressor,
                                  StepwiseLinear, make_classifier, make_regressor, scores_for_auc)
from disconnection.multivariate import pca_predict
from disconnection.selection import best_subsets, nested_selection_cv, pls_backward, stepwise_cv
from disconnection.stats import (corrected_resampled_ttest, cv5x2_f_test, cv5x2_paired_t,
                                 dependent_correlation_difference, fisher_z_independent, holm, meng_contrast,
                                 meng_heterogeneity, meng_z, paired_error_test, pitman_morgan)
from disconnection.synthetic import make_brain, simulate

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
BRAIN = make_brain(seed=3)


# ------------------------------------------------------------------ synthetic data
def test_synthetic_structure():
    p = simulate(120, n_tasks=2, truth="disconnection", brain=BRAIN, seed=1)
    assert p.lesion_load.shape == (120, BRAIN.n_regions)
    assert p.edge_disconnection.shape == (120, len(BRAIN.edges))
    assert ((p.lesion_load >= 0) & (p.lesion_load <= 1)).all()
    assert ((p.disconnection >= 0) & (p.disconnection <= 1)).all()
    # load and disconnection are related but not the same
    r = np.corrcoef(p.lesion_load.sum(1), p.disconnection.sum(1))[0, 1]
    assert 0.3 < r < 0.99
    # every lesioned voxel of a region raises that region's load
    assert (p.lesion_load_fuzzy.sum(1) > 0).all()


def test_truth_none_has_no_brain_signal():
    p = simulate(400, 1, truth="none", brain=BRAIN, seed=2)
    Xb = np.hstack([p.lesion_load, p.edge_disconnection])
    pv = correlation_p(Xb, p.behaviour[:, 0])
    assert (pv < 0.05 / Xb.shape[1]).sum() <= 1


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("name", list(REGRESSORS))
def test_every_regressor_learns(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6))
    y = X[:, 0] - X[:, 1] + 0.5 * rng.normal(size=150)
    pred = make_regressor(name).fit(X[:110], y[:110]).predict(X[110:])
    floor = {"tree_coarse": 0.15}.get(name, 0.5 if "tree" in name or name == "svm_gaussian_fine" else 0.8)
    assert np.corrcoef(pred, y[110:])[0, 1] > floor


@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_every_classifier_separates(name):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(160, 5))
    y = (X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=160) > 0).astype(int)
    m = make_classifier(name).fit(X[:120], y[:120])
    from sklearn.metrics import roc_auc_score
    floor = 0.5 if name == "knn_coarse" else 0.7          # 100 neighbours of 120 patients
    assert roc_auc_score(y[120:], scores_for_auc(m, X[120:])) >= floor


def test_original_mode_numbers():
    assert len(REGRESSION_MODES) == 19 and REGRESSION_MODES[4] == "stepwise" and REGRESSION_MODES[15] == "bagged_trees"
    assert len(CLASSIFIER_MODES) == 20 and CLASSIFIER_MODES[20] == "rusboost"
    assert type(make_regressor(16)).__name__ == "Pipeline"


def test_bisquare_resists_outliers():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    y = 2 * x + 0.1 * rng.normal(size=100)
    y[:5] += 30
    assert abs(BisquareRegressor().fit(x[:, None], y).coef_[0] - 2) < 0.1


def test_stepwise_linear_picks_true_terms():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 8))
    y = X[:, 2] - 2 * X[:, 5] + rng.normal(size=200)
    assert {2, 5} <= set(StepwiseLinear().fit(X, y).selected_)


def test_svr_box_constraint_uses_training_responses_only():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60)
    a = make_regressor("svm_gaussian_10").fit(X[:40], y[:40]).predict(X[40:])
    y2 = y.copy()
    y2[40:] *= 100                                   # test responses must not matter
    b = make_regressor("svm_gaussian_10").fit(X[:40], y2[:40]).predict(X[40:])
    assert np.allclose(a, b)


# ------------------------------------------------------------------ features and CV
def test_block_selector_rules():
    rng = np.random.default_rng(5)
    n = 200
    dem = rng.normal(size=(n, 2))
    ll = rng.normal(size=(n, 20))
    conn = rng.normal(size=(n, 50))
    y = ll[:, 3] + ll[:, 7] + conn[:, 10] + 0.5 * rng.normal(size=n)
    X, sl = stack_blocks({"dem": dem, "ll": ll, "conn": conn})
    s = BlockSelector(sl, {"dem": "all", "ll": "bonferroni", "conn": ("match", "ll")}).fit(X, y)
    ll_cols = s.selected_["ll"] - sl["ll"][0]
    assert {3, 7} <= set(ll_cols)
    assert len(s.selected_["conn"]) == len(ll_cols)
    assert 10 in s.selected_["conn"] - sl["conn"][0]
    assert s.transform(X).shape[1] == 2 + 2 * len(ll_cols)
    top = BlockSelector(sl, {"ll": ("top", 2)}).fit(X, y)
    assert set(top.selected_["ll"] - sl["ll"][0]) == {3, 7}


def test_block_selector_fallback_and_empty():
    rng = np.random.default_rng(6)
    X, sl = stack_blocks({"ll": rng.normal(size=(60, 30))})
    y = rng.normal(size=60)
    s = BlockSelector(sl, {"ll": "bonferroni"}).fit(X, y)
    assert s.transform(X).shape == (60, max(1, len(s.columns_)))
    s2 = BlockSelector(sl, {"ll": "bonferroni_or_p05"}).fit(X, y)
    assert len(s2.columns_) >= len(s.columns_)


def test_permutation_threshold_controls_fwe():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 40))
    thr = permutation_threshold(X, rng.normal(size=80), n_perm=200)
    assert 0.0001 < thr < 0.01                        # much stricter than 0.05 for 40 tests


def test_folds_balanced_and_loo():
    folds = make_folds(23, 5, 3, seed=1)
    for r in range(3):
        te = np.concatenate([t for rr, _, t in folds if rr == r])
        assert sorted(te) == list(range(23))
        sizes = [len(t) for rr, _, t in folds if rr == r]
        assert max(sizes) - min(sizes) <= 1
    assert len(make_folds(7, 10)) == 7
    y = np.array([0] * 30 + [1] * 10)
    for _, _, te in make_folds(40, 5, 1, 0, stratify=y):
        assert y[te].sum() == 2


def test_selection_inside_folds_ignores_test_rows():
    rng = np.random.default_rng(8)
    X, sl = stack_blocks({"dem": rng.normal(size=(60, 2)), "conn": rng.normal(size=(60, 40))})
    y = rng.normal(size=60)
    folds = make_folds(60, 5, 1, 0)
    pipe = make_pipeline_for({"dem": "all", "conn": "p05"}, sl, "linear")
    a = cross_val_predict_repeated(pipe, X, y, folds).pred
    te = folds[0][2]
    y2 = y.copy()
    y2[te] += 50
    b = cross_val_predict_repeated(pipe, X, y2, folds).pred
    assert np.allclose(a[te], b[te])


def test_cv_fold_mse_matches_predictions():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] + rng.normal(size=50)
    folds = make_folds(50, 5, 2, 0)
    from sklearn.linear_model import LinearRegression
    cvp = cross_val_predict_repeated(LinearRegression(), X, y, folds, n_jobs=2)
    assert cvp.fold_mse.shape == (2, 5) and cvp.pred_repeats.shape == (50, 2)
    r0 = [(tr, te) for r, tr, te in folds if r == 0]
    assert np.isclose(cvp.fold_mse[0, 0], np.mean((cvp.pred_repeats[r0[0][1], 0] - y[r0[0][1]]) ** 2))
    m = regression_metrics(y, cvp.pred)
    assert m["r"] > 0.5 and m["n"] == 50


# ------------------------------------------------------------------ the comparison
def test_compare_detects_disconnection_advantage():
    p = simulate(250, 2, truth="disconnection", effect=1.0, brain=BRAIN, seed=11)
    sets = {n: STANDARD_SETS[n] for n in ("ll", "conn")}
    res = compare_feature_sets(p.feature_sets(), p.behaviour, sets, models=["bagged_trees"], k=5)
    c = res.compare("ll", "conn")
    assert (c["diff"] > 0).all()
    assert res.table.query("set == 'conn'").r.mean() > res.table.query("set == 'll'").r.mean()


def test_compare_no_false_advantage_when_load_is_truth():
    # With a linear model, connection features do not beat lesion load when behaviour
    # depends on lesion load. (Flexible models such as bagged trees can do about as well
    # with either representation, because each is a proxy for the other; see README.)
    p = simulate(250, 2, truth="load", effect=1.0, brain=BRAIN, seed=12)
    sets = {n: STANDARD_SETS[n] for n in ("ll", "conn")}
    res = compare_feature_sets(p.feature_sets(), p.behaviour, sets, models=["ridge"], k=5)
    c = res.compare("ll", "conn")
    assert (c["p_t_holm"] > 0.05).all()


def test_compare_stack_and_missing_rows():
    p = simulate(160, 2, truth="both", brain=BRAIN, seed=13)
    Y = p.behaviour.copy()
    Y[:10, 1] = np.nan
    sets = {"ll": STANDARD_SETS["ll"], "conn": STANDARD_SETS["conn"], "stack": ("stack", "ll", "conn")}
    res = compare_feature_sets(p.feature_sets(), Y, sets, models=["ridge"], k=5, task_names=["a", "b"])
    assert res.n_by_task == {"a": 160, "b": 150}
    assert set(res.table.set) == {"ll", "conn", "stack"}
    assert res.table.r.min() > 0.2
    c = res.compare("ll", "stack")
    assert {"p_t_holm", "p_pitman_morgan", "p_corrected_cv", "ci_low"} <= set(c.columns)


def test_residual_chain_improves_underfit_model():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(300, 3))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.2 * rng.normal(size=300)
    from sklearn.tree import DecisionTreeRegressor
    base = DecisionTreeRegressor(max_depth=2, random_state=0)
    folds = make_folds(300, 5, 1, 0)
    one = regression_metrics(y, cross_val_predict_repeated(base, X, y, folds).pred)["r"]
    three = regression_metrics(y, cross_val_predict_repeated(ResidualChain(base, levels=3), X, y, folds).pred)["r"]
    assert three > one + 0.05


# ------------------------------------------------------------------ statistics
def test_pitman_morgan_calibrated_and_more_powerful_than_f_test():
    out = benchmark.pit_paired_variance_tests(sims=600, seed=1)
    assert 0.02 < out["null_pitman_morgan"] < 0.09
    assert out["alternative_pitman_morgan"] > 2 * out["alternative_f_test"]


def test_paired_error_test_direction():
    rng = np.random.default_rng(15)
    y = rng.normal(size=200)
    good = y + 0.3 * rng.normal(size=200)
    bad = y + 1.0 * rng.normal(size=200)
    r = paired_error_test(y, bad, good)
    assert r["diff"] > 0 and r["p_t"] < 0.001 and r["ci_low"] > 0
    assert paired_error_test(y, good, bad)["p_t"] > 0.99


def test_fold_score_tests_under_null():
    out = benchmark.pit_fold_score_tests(datasets=60, seed=2)
    assert out["cv5x2_f"] <= 0.12 and out["corrected_t"] <= 0.15
    assert out["wilcoxon_fold_scores"] > out["cv5x2_f"]


def test_5x2_tests_detect_real_difference():
    rng = np.random.default_rng(16)
    from sklearn.linear_model import LinearRegression
    X = rng.normal(size=(150, 2))
    y = 2 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(size=150)
    folds = make_folds(150, 2, 5, 0)
    fa = cross_val_predict_repeated(LinearRegression(), X[:, [1]], y, folds).fold_mse
    fb = cross_val_predict_repeated(LinearRegression(), X[:, [0]], y, folds).fold_mse
    assert cv5x2_f_test(fa - fb)[1] < 0.01
    assert cv5x2_paired_t(fa - fb)[1] < 0.05
    assert corrected_resampled_ttest(fa, fb, 75, 75)[1] < 0.01


def test_correlation_tests_reference_values():
    # Fisher z (compare_correlation_coefficients.m)
    z = (np.arctanh(0.5) - np.arctanh(0.2)) / np.sqrt(1 / 47 + 1 / 57)
    assert np.isclose(fisher_z_independent(0.5, 0.2, 50, 60), 2 * stats.norm.sf(abs(z)))
    # Meng et al. (1992) worked example: r1 = .63, r2 = .30, rx = .38 (Meng's table uses n = 15 -> z ~ 1.3?)
    zm, pm = meng_z(0.5, 0.2, 0.3, 100)
    assert zm > 0 and 0 < pm < 0.05
    R = np.array([[1, .5, .4, .1], [.5, 1, .3, .2], [.4, .3, 1, .25], [.1, .2, .25, 1]])
    chi2, p = meng_heterogeneity(R, 0, 100)
    assert chi2 > 0 and 0 <= p <= 1
    zc, pc = meng_contrast(R, 0, 100, [1, 0, -1])
    assert zc > 0
    # dependent correlations: CI must be symmetric in which variable is which
    d1 = dependent_correlation_difference(0.6, 0.3, 0.4, 100)
    d2 = dependent_correlation_difference(0.3, 0.6, 0.4, 100)
    assert np.isclose(d1["ci"][1] - d1["ci"][0], d2["ci"][1] - d2["ci"][0])
    assert np.isclose(d1["p"], d2["p"]) and d1["p"] < 0.01


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


# ------------------------------------------------------------------ selection, classification, learning
def test_stepwise_cv_finds_true_predictors():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(150, 8))
    y = X[:, 1] + X[:, 4] + rng.normal(size=150)
    cols, mse = stepwise_cv(X, y, k=5, repeats=2)
    assert {1, 4} <= set(cols) and len(cols) <= 4


def test_best_subsets():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(100, 4))
    y = 3 * X[:, 0] + 2 * X[:, 2] + rng.normal(size=100)
    models = best_subsets(X, y, labels=list("abcd"))
    assert models[0]["labels"] == ["a", "c"] and models[0]["delta_aic"] == 0
    assert np.isclose(sum(m["weight"] for m in models), 1)
    assert all((m["p"] < 0.05).all() for m in models)


def test_pls_backward_removes_noise():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(120, 10))
    y = X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=120)
    cols, path = pls_backward(X, y, n_components=2, k=5, repeats=2)
    assert {0, 1} <= set(cols) and len(cols) < 10
    assert path[-1][1] <= path[0][1]


def test_nested_selection_less_optimistic():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(60, 30))
    y = rng.normal(size=60)
    sel = lambda Xt, yt: list(np.argsort(correlation_p(Xt, yt))[:5])
    nested = nested_selection_cv(sel, X, y, k=5)
    cols = sel(X, y)
    from sklearn.linear_model import LinearRegression
    naive = regression_metrics(y, cross_val_predict_repeated(LinearRegression(), X[:, cols], y,
                                                             make_folds(60, 5, 1, 0)).pred)
    assert naive["r"] > nested["r"] + 0.2


def test_classify_cv_and_oversampling():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=120) > 0.8).astype(int)
    Xo, yo = oversample(X, y, rng)
    assert yo.mean() == 0.5
    out = classify_cv(X, y, "logistic", k=5, repeats=2)
    assert out["auc"] > 0.8 and 0 <= out["balanced_accuracy"] <= 1
    tree = make_classifier("tree_coarse")
    assert misclassification_probability(tree, X[:80], y[:80], X[80:], y[80:]) < 40


def test_learning_curve_improves_with_size():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(200, 10))
    y = X @ rng.normal(size=10) + rng.normal(size=200)
    lc = learning_curve(make_regressor("linear"), X, y, sizes=[15, 120], repeats=5)
    assert lc.loc[120, ("rmse", "mean")] < lc.loc[15, ("rmse", "mean")]


def test_pls_stability_sign_invariant():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(150, 6))
    Y = X[:, :2] @ rng.normal(size=(2, 2)) + 0.1 * rng.normal(size=(150, 2))
    st = pls_score_stability(X, Y, 2, sizes=[30, 110], repeats=3)
    assert st.loc[110, "mean"] < st.loc[30, "mean"]
    assert st.loc[110, "mean"] < 0.2


def test_pca_predict_in_fold():
    rng = np.random.default_rng(24)
    lat = rng.normal(size=80)
    X = np.outer(lat, rng.normal(size=300)) + rng.normal(size=(80, 300))
    C = rng.normal(size=(80, 2))
    y = lat + C[:, 0] + 0.3 * rng.normal(size=80)
    out = pca_predict(X, y, covariates=C, model="linear", k=5)
    assert out["r"] > 0.7 and max(out["ncomp"]) <= 20


# ------------------------------------------------------------------ data helpers
def test_select_cohort_and_long_format():
    df = pd.DataFrame({"order": [1, 1, 2, 1], "age": [60, 85, 60, 50], "months_post_stroke": [12, 12, 12, 3],
                       "lesion_volume": [5, 5, 5, 5], "right_volume": [0, 0, 0, 0], "handedness_left": [0, 0, 0, 0],
                       "language_native": [1, 1, 1, 1]})
    assert list(select_cohort(df, "big")) == [True, True, False, True]
    assert list(select_cohort(df, "small")) == [True, False, False, False]
    a = pd.DataFrame({"id": ["p1", "p1", "p2", "p3"], "time": [3, 12, 5, 6], "naming": [1, 2, 3, 4]})
    img = pd.DataFrame({"id": ["p1", "p1", "p2"], "vol": [10, 99, 20]})
    lf = long_format(a, img)
    assert len(lf) == 3 and list(lf.vol) == [10, 10, 20]


# ------------------------------------------------------------------ pitfalls and CLI
def test_pitfall_selection_leakage():
    out = benchmark.pit_selection_leakage(datasets=6)
    assert out["r_selected_before_cv"] > out["r_selected_in_fold"] + 0.03


def test_cli_compare(tmp_path):
    p = simulate(120, 1, truth="disconnection", brain=BRAIN, seed=30)
    cols = {f"dem_{i}": p.demographics[:, i] for i in range(5)}
    cols.update({f"ll_{i}": p.lesion_load[:, i] for i in range(p.lesion_load.shape[1])})
    cols.update({f"conn_{i}": p.edge_disconnection[:, i] for i in range(p.edge_disconnection.shape[1])})
    cols["naming"] = p.behaviour[:, 0]
    pd.DataFrame(cols).to_csv(tmp_path / "d.csv", index=False)
    r = subprocess.run([sys.executable, "-m", "disconnection", "compare", str(tmp_path / "d.csv"), "--blocks",
                        "dem=dem_", "ll=ll_", "conn=conn_", "--targets", "naming", "--models", "ridge", "--k", "5",
                        "--out", str(tmp_path / "o")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    scores = pd.read_csv(tmp_path / "o" / "scores.csv")
    assert set(scores.set) >= {"ll", "conn", "stack(ll,conn)"}
    assert (tmp_path / "o" / "comparisons.csv").exists() and (tmp_path / "o" / "predictions.csv").exists()
