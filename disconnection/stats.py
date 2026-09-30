"""Statistical comparisons of models and correlations.

Comparing two models' predictions for the SAME patients (paired):
    pitman_morgan           equal variance of two paired error vectors
                            (the paired version of the F-test, vartest2, which the
                            original used as if the two error vectors were
                            independent)
    paired_error_test       paired t-test / Wilcoxon / bootstrap CI on the
                            difference in squared error, patient by patient
Comparing models across repeated cross-validation folds:
    corrected_resampled_ttest   Nadeau & Bengio (2003): the t-test on fold scores,
                            corrected for the overlap between training sets
    cv5x2_paired_t          Dietterich (1998), for 5 repeats of 2-fold CV
    cv5x2_f_test            Alpaydin (1999), the more robust version
                            (the original compared 5x2 fold scores with a
                            Wilcoxon signed-rank test, which treats the 10 overlapping
                            folds as independent)
Correlations:
    fisher_z_independent    two correlations from independent samples
                            (compare_correlation_coefficients.m)
    meng_z                  correlated correlations (Meng, Rubin & Rosenthal,
                            1992; mengz.m by E. Spaak)
    dependent_correlation_difference   two correlations sharing a variable:
                            Zou (2007) CI and Steiger (1980) z (rddiffci.m by
                            M. Stuttgen; its CI used r13 in place of r23)
    holm                    Holm step-down adjustment of p-values
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def f_test_variances(e1, e2, alternative="greater"):
    """The original analysis (vartest2): F = var(e1)/var(e2) with independent-sample
    degrees of freedom. Kept only to reproduce old results; for errors from the
    same patients use pitman_morgan."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    F = e1.var(ddof=1) / e2.var(ddof=1)
    d1, d2 = len(e1) - 1, len(e2) - 1
    p = stats.f.sf(F, d1, d2) if alternative == "greater" else (
        stats.f.cdf(F, d1, d2) if alternative == "less" else 2 * min(stats.f.sf(F, d1, d2), stats.f.cdf(F, d1, d2)))
    return F, float(p)


def pitman_morgan(e1, e2, alternative="greater"):
    """Pitman-Morgan test of var(e1) = var(e2) for paired e1, e2.

    alternative "greater": var(e1) > var(e2) (model 2 predicts better).
    Uses corr(e1 + e2, e1 - e2), which is zero exactly when the variances are
    equal. Returns (variance ratio, p)."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    n = len(e1)
    r = np.corrcoef(e1 + e2, e1 - e2)[0, 1]
    t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-300))
    if alternative == "greater":
        p = stats.t.sf(t, n - 2)
    elif alternative == "less":
        p = stats.t.cdf(t, n - 2)
    else:
        p = 2 * stats.t.sf(abs(t), n - 2)
    return float(e1.var(ddof=1) / e2.var(ddof=1)), float(p)


def paired_error_test(y, pred_a, pred_b, n_boot=2000, seed=0) -> dict:
    """Is model b more accurate than model a for the same patients?

    d_i = (y_i - a_i)^2 - (y_i - b_i)^2 (positive = b better). Returns the mean
    difference, its proportion of a's MSE, a bootstrap 95% CI, and one-sided
    p-values from a paired t-test and a Wilcoxon signed-rank test."""
    y = np.asarray(y, float); a = np.asarray(pred_a, float); b = np.asarray(pred_b, float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    d = (y[ok] - a[ok]) ** 2 - (y[ok] - b[ok]) ** 2
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    t = stats.ttest_1samp(d, 0, alternative="greater")
    try:
        w = stats.wilcoxon(d, alternative="greater").pvalue
    except ValueError:
        w = np.nan
    mse_a = ((y[ok] - a[ok]) ** 2).mean()
    return {"n": int(ok.sum()), "mse_a": float(mse_a), "mse_b": float(((y[ok] - b[ok]) ** 2).mean()),
            "diff": float(d.mean()), "relative": float(d.mean() / mse_a) if mse_a > 0 else np.nan,
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "p_t": float(t.pvalue), "p_wilcoxon": float(w)}


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test, alternative="greater"):
    """Nadeau & Bengio's corrected resampled t-test for repeated K-fold CV.

    scores_*: per-fold losses (e.g. MSE), any shape, same order for both models.
    alternative "greater": model b has lower loss than a. Returns (t, p)."""
    d = (np.asarray(scores_a, float) - np.asarray(scores_b, float)).ravel()
    d = d[np.isfinite(d)]
    J = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else np.inf * np.sign(d.mean())
    p = stats.t.sf(t, J - 1) if alternative == "greater" else 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)


def cv5x2_paired_t(diff):
    """Dietterich's 5x2cv paired t-test. diff: (5, 2) loss differences. Two-sided p."""
    d = np.asarray(diff, float)
    s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
    t = d[0, 0] / np.sqrt(s2.mean())
    return float(t), float(2 * stats.t.sf(abs(t), 5))


def cv5x2_f_test(diff):
    """Alpaydin's combined 5x2cv F-test. diff: (5, 2) loss differences. Returns (F, p)."""
    d = np.asarray(diff, float)
    s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
    F = (d ** 2).sum() / (2 * s2.sum())
    return float(F), float(stats.f.sf(F, 10, 5))


def fisher_z_independent(r1, r2, n1, n2):
    """Two-sided p for H0: rho1 = rho2 (independent samples)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(2 * stats.norm.sf(abs(z)))


def meng_z(r1, r2, rx, n):
    """Meng, Rubin & Rosenthal (1992): is corr(X,Y)=r1 larger than corr(X,Z)=r2,
    where corr(Y,Z)=rx? One-tailed. Returns (z, p)."""
    rsq = (r1 ** 2 + r2 ** 2) / 2
    f = min((1 - rx) / (2 * (1 - rsq)), 1.0)
    h = (1 - f * rsq) / (1 - rsq)
    z = (np.arctanh(r1) - np.arctanh(r2)) * np.sqrt((n - 3) / (2 * (1 - rx) * h))
    return float(z), float(stats.norm.sf(z))


def meng_heterogeneity(R, k, n):
    """Meng et al.'s chi-square test that variable k correlates equally with all the
    others in correlation matrix R. Returns (chi2, p)."""
    R = np.asarray(R, float)
    x = np.delete(R[k], k)
    Rr = np.delete(np.delete(R, k, 0), k, 1)
    rx = np.median(Rr[np.tril_indices_from(Rr, -1)])
    rsq = np.mean(x ** 2)
    f = min((1 - rx) / (2 * (1 - rsq)), 1.0)
    h = (1 - f * rsq) / (1 - rsq)
    zx = np.arctanh(x)
    chi2 = (n - 3) * ((zx - zx.mean()) ** 2).sum() / ((1 - rx) * h)
    return float(chi2), float(stats.chi2.sf(chi2, len(x) - 1))


def meng_contrast(R, k, n, lam):
    """Meng et al.'s contrast test (one-tailed). Returns (z, p)."""
    chi2, _ = meng_heterogeneity(R, k, n)
    zx = np.arctanh(np.delete(np.asarray(R, float)[k], k))
    z = np.corrcoef(zx, np.asarray(lam, float))[0, 1] * np.sqrt(chi2)
    return float(z), float(stats.norm.sf(z))


def dependent_correlation_difference(r13, r23, r12, n, alpha=0.05) -> dict:
    """Do variables 1 and 2 predict variable 3 equally well? (rddiffci.m)

    Returns r13 - r23, Zou's (2007) confidence interval and Steiger's (1980)
    z-test p-value. The original computed the variance of r23 with r13
    ((1-r13^2)^2/n for both), so its interval was wrong whenever r13 != r23."""
    for r in (r12, r13, r23):
        if not -1 <= r <= 1:
            raise ValueError("correlations must be in [-1, 1]")
    z = stats.norm.ppf(1 - alpha / 2)
    cov = ((r12 - 0.5 * r13 * r23) * (1 - r13 ** 2 - r23 ** 2 - r12 ** 2) + r12 ** 3) / n
    var13 = (1 - r13 ** 2) ** 2 / n
    var23 = (1 - r23 ** 2) ** 2 / n
    half = z * np.sqrt(var13 + var23 - 2 * cov)
    ra = (r13 + r23) / 2
    cv1 = (1 / (1 - ra ** 2) ** 2) * (r12 * (1 - 2 * ra ** 2) - 0.5 * ra ** 2 * (1 - 2 * ra ** 2 - r12 ** 2))
    zs = np.sqrt(n - 3) * (np.arctanh(r13) - np.arctanh(r23)) / np.sqrt(2 - 2 * cv1)
    return {"diff": r13 - r23, "ci": (r13 - r23 - half, r13 - r23 + half), "z": float(zs),
            "p": float(2 * stats.norm.sf(abs(zs)))}


def holm(pvals):
    """Holm-adjusted p-values (NaNs left as NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        out[j] = min(running, 1.0)
    return out
