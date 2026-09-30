"""The regression and classification models ("inducers") of the original code.

trainRegression_Any.m / trainRegression_cv.m / trainRegModel.m chose among 19
regression models by number ("mode"); trainClassifier_Any.m among 20
classifiers. The same models are available here by name (or by the original
number, via REGRESSION_MODES / CLASSIFIER_MODES), built from scikit-learn.

Translation notes (MATLAB -> scikit-learn):
- Predictors are standardised inside every model that standardised in MATLAB
  ('Standardize', true), always with the training data only.
- SVM kernel scale s: MATLAB divides predictors by s, so a Gaussian kernel with
  scale s is exp(-|x-z|^2 / s^2), i.e. gamma = 1/s^2; a polynomial kernel is
  (1 + x.z / s^2)^q. KernelScale 'auto' (a MATLAB heuristic) is approximated by
  s = sqrt(number of predictors), the typical distance between standardised
  points.
- SVR box constraint and epsilon follow Regression Learner's defaults,
  IQR(y)/1.349 and IQR(y)/13.49, computed from the TRAINING responses (the
  original computed them from all responses before cross-validation).
- Trees: MinLeafSize -> min_samples_leaf; MaxNumSplits m -> max_leaf_nodes m+1.
- LSBoost (30 cycles, rate 0.1, leaf size 8, MATLAB's default 10 splits per
  tree) -> GradientBoostingRegressor; Bag (30 trees, leaf size 8, a third of the
  predictors per split, MATLAB's default for regression) -> RandomForestRegressor.
- Robust fitlm (bisquare weights) -> BisquareRegressor below (IRLS, as MATLAB).
- stepwiselm -> StepwiseLinear below (forward/backward by BIC on linear terms;
  the original allowed interactions, which with ~100 predictors was very slow
  and was skipped in the later scripts).
- GP (constant basis; squared-exponential / Matern 5/2 / exponential / rational
  quadratic kernels, standardised) -> GaussianProcessRegressor with a constant
  mean (normalize_y) and a fitted noise term.
- Random-subspace ensembles -> bagging without bootstrap over random feature
  subsets; RUSBoost -> RUSBoostClassifier below (AdaBoost with random
  under-sampling of the majority class at every round).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------- helper estimators
class BisquareRegressor(RegressorMixin, BaseEstimator):
    """Robust linear regression with Tukey bisquare weights (MATLAB fitlm 'RobustOpts','on')."""

    def __init__(self, tune: float = 4.685, max_iter: int = 50, tol: float = 1e-8):
        self.tune, self.max_iter, self.tol = tune, max_iter, tol

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        A = np.column_stack([np.ones(len(y)), X])
        w = np.ones(len(y))
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        p = A.shape[1]
        # leverage adjustment, as MATLAB robustfit
        Q, _ = np.linalg.qr(A)
        h = np.minimum((Q ** 2).sum(1), 1 - 1e-8)
        adj = 1 / np.sqrt(1 - h)
        for _ in range(self.max_iter):
            r = (y - A @ b) * adj
            rs = np.sort(np.abs(r))[p - 1:] if len(r) > p else np.abs(r)
            s = np.median(rs) / 0.6745
            if s <= 0:
                break
            u = r / (self.tune * s)
            w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
            sw = np.sqrt(w)
            b_new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
            if np.max(np.abs(b_new - b)) < self.tol * max(1, np.max(np.abs(b))):
                b = b_new
                break
            b = b_new
        self.intercept_, self.coef_, self.weights_ = b[0], b[1:], w
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, float) @ self.coef_


class StepwiseLinear(RegressorMixin, BaseEstimator):
    """Bidirectional stepwise least squares on linear terms, by BIC (stepwiselm with
    'Criterion','bic', as the fallback in trainRegression_Any.m)."""

    def __init__(self, criterion: str = "bic", max_steps: int = 1000):
        self.criterion, self.max_steps = criterion, max_steps

    def _score(self, X, y, cols):
        n = len(y)
        A = np.column_stack([np.ones(n), X[:, cols]]) if cols else np.ones((n, 1))
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        rss = max(((y - A @ b) ** 2).sum(), 1e-300)
        pen = np.log(n) if self.criterion == "bic" else 2.0
        return n * np.log(rss / n) + pen * (len(cols) + 1)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        cur, best = [], self._score(X, y, [])
        for _ in range(self.max_steps):
            moves = [(self._score(X, y, cur + [j]), cur + [j]) for j in range(X.shape[1]) if j not in cur]
            moves += [(self._score(X, y, [c for c in cur if c != j]), [c for c in cur if c != j]) for j in cur]
            if not moves:
                break
            s, cols = min(moves, key=lambda m: m[0])
            if s >= best - 1e-12:
                break
            best, cur = s, cols
        self.selected_ = sorted(cur)
        self.model_ = LinearRegression().fit(X[:, self.selected_], y) if cur else None
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.model_.predict(X[:, self.selected_]) if self.model_ is not None else np.full(len(X), self.mean_)


class AutoSVR(RegressorMixin, BaseEstimator):
    """SVR with Regression Learner's box constraint and epsilon, set from the training
    responses: C = IQR(y)/1.349, epsilon = IQR(y)/13.49."""

    def __init__(self, kernel="rbf", scale=None, degree=3, standardize=True):
        self.kernel, self.scale, self.degree, self.standardize = kernel, scale, degree, standardize

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        q75, q25 = np.percentile(y, [75, 25])
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr == 0:
            iqr = 1.0
        s = self.scale if self.scale is not None else np.sqrt(max(X.shape[1], 1))
        if self.kernel == "linear":
            svr = SVR(kernel="linear", C=iqr / 1.349, epsilon=iqr / 13.49)
            steps = [FunctionTransformer(lambda Z, s=s: Z / s)]
        elif self.kernel == "poly":
            svr = SVR(kernel="poly", degree=self.degree, gamma=1 / s ** 2, coef0=1.0, C=iqr / 1.349,
                      epsilon=iqr / 13.49)
            steps = []
        else:
            svr = SVR(kernel="rbf", gamma=1 / s ** 2, C=iqr / 1.349, epsilon=iqr / 13.49)
            steps = []
        pre = [StandardScaler()] if self.standardize else []
        self.model_ = make_pipeline(*pre, *steps, svr).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


class RUSBoostClassifier(ClassifierMixin, BaseEstimator):
    """AdaBoost (SAMME) where each round's learner is fitted on a random under-sample
    that balances the classes (Seiffert et al., 2010), as MATLAB's 'RUSBoost'."""

    def __init__(self, n_estimators=30, learning_rate=0.1, max_leaf_nodes=21, random_state=0):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_leaf_nodes, self.random_state = max_leaf_nodes, random_state

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yi = np.searchsorted(self.classes_, y)
        w = np.full(len(y), 1 / len(y))
        m = np.bincount(yi).min()
        self.estimators_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            idx = np.concatenate([rng.choice(np.flatnonzero(yi == k), m, replace=False) for k in range(K)])
            tree = DecisionTreeClassifier(max_leaf_nodes=self.max_leaf_nodes, random_state=int(rng.integers(1e9)))
            tree.fit(X[idx], yi[idx], sample_weight=w[idx] / w[idx].sum())
            miss = tree.predict(X) != yi
            err = np.clip(w[miss].sum() / w.sum(), 1e-10, 1 - 1e-10)
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(K - 1))
            if alpha <= 0:
                continue
            w = w * np.exp(alpha * miss)
            w /= w.sum()
            self.estimators_.append(tree); self.alphas_.append(alpha)
        if not self.estimators_:
            self.estimators_.append(DecisionTreeClassifier(max_depth=1).fit(X, yi)); self.alphas_.append(1.0)
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        votes = np.zeros((len(X), len(self.classes_)))
        for a, t in zip(self.alphas_, self.estimators_):
            proba = np.zeros((len(X), len(self.classes_)))
            proba[np.arange(len(X)), t.predict(X)] = 1
            votes += a * proba
        return votes / sum(self.alphas_)

    def predict_proba(self, X):
        d = self.decision_function(X)
        return d / np.maximum(d.sum(1, keepdims=True), 1e-12)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(1)]


def _gp(kernel):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * kernel + WhiteKernel(0.5, (1e-5, 1e2))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0,
                                                                     random_state=0))


# ---------------------------------------------------------------- registries
REGRESSORS = {
    "tree_fine": lambda: DecisionTreeRegressor(min_samples_leaf=4, random_state=0),
    "tree_medium": lambda: DecisionTreeRegressor(min_samples_leaf=12, random_state=0),
    "tree_coarse": lambda: DecisionTreeRegressor(min_samples_leaf=36, random_state=0),
    "stepwise": lambda: StepwiseLinear(),
    "robust_linear": lambda: BisquareRegressor(),
    "linear": lambda: LinearRegression(),
    "svm_linear": lambda: AutoSVR("linear", scale=0.1),
    "svm_quadratic": lambda: AutoSVR("poly", degree=2),
    "svm_cubic": lambda: AutoSVR("poly", degree=3),
    "svm_gaussian_fine": lambda: AutoSVR("rbf", scale=1.3),
    "svm_gaussian_5": lambda: AutoSVR("rbf", scale=5.0),
    "svm_gaussian_10": lambda: AutoSVR("rbf", scale=10.0),
    "svm_gaussian_coarse": lambda: AutoSVR("rbf", scale=20.0),
    "boosted_trees": lambda: GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, min_samples_leaf=8,
                                                       max_leaf_nodes=11, max_depth=None, random_state=0),
    "bagged_trees": lambda: RandomForestRegressor(n_estimators=30, min_samples_leaf=8, max_features=1 / 3,
                                                  random_state=0, n_jobs=1),
    "gp_squared_exponential": lambda: _gp(RBF(1.0, (1e-2, 1e3))),
    "gp_matern52": lambda: _gp(Matern(1.0, (1e-2, 1e3), nu=2.5)),
    "gp_exponential": lambda: _gp(Matern(1.0, (1e-2, 1e3), nu=0.5)),
    "gp_rational_quadratic": lambda: _gp(RationalQuadratic(1.0, 1.0, (1e-2, 1e3), (1e-2, 1e3))),
}
REGRESSION_MODES = dict(enumerate(REGRESSORS, start=1))     # original mode number -> name
# Not in the original: a regularised linear model, which copes with many
# correlated disconnection features far better than least squares.
REGRESSORS["ridge"] = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))

CLASSIFIERS = {
    "tree_fine": lambda: DecisionTreeClassifier(max_leaf_nodes=101, random_state=0),
    "tree_medium": lambda: DecisionTreeClassifier(max_leaf_nodes=21, random_state=0),
    "tree_coarse": lambda: DecisionTreeClassifier(max_leaf_nodes=5, random_state=0),
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)),
    "svm_linear": lambda: _svc("linear"),
    "svm_quadratic": lambda: _svc("poly", degree=2),
    "svm_cubic": lambda: _svc("poly", degree=3),
    "svm_gaussian_fine": lambda: _svc("rbf", scale=1.9),
    "svm_gaussian_medium": lambda: _svc("rbf", scale=7.5),
    "svm_gaussian_coarse": lambda: _svc("rbf", scale=30.0),
    "knn_fine": lambda: CappedKNN(1),
    "knn_medium": lambda: CappedKNN(10),
    "knn_coarse": lambda: CappedKNN(100),
    "knn_cosine": lambda: CappedKNN(10, metric="cosine"),
    "knn_cubic": lambda: CappedKNN(10, p=3),
    "knn_weighted": lambda: CappedKNN(10, weights=_inverse_square),
    "boosted_trees": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes=21), n_estimators=30,
                                                learning_rate=0.1, random_state=0),
    "subspace_discriminant": lambda: _subspace(LinearDiscriminantAnalysis()),
    "subspace_knn": lambda: _subspace(make_pipeline(StandardScaler(), KNeighborsClassifier(1))),
    "rusboost": lambda: RUSBoostClassifier(),
}
CLASSIFIER_MODES = dict(enumerate(CLASSIFIERS, start=1))


class CappedKNN(ClassifierMixin, BaseEstimator):
    """k-nearest neighbours on standardised predictors; k is capped at the number of
    training patients (MATLAB's 100-neighbour 'coarse KNN' otherwise fails on
    small training folds)."""

    def __init__(self, n_neighbors=10, metric="minkowski", p=2, weights="uniform"):
        self.n_neighbors, self.metric, self.p, self.weights = n_neighbors, metric, p, weights

    def fit(self, X, y):
        k = min(self.n_neighbors, len(y))
        self.model_ = make_pipeline(StandardScaler(), KNeighborsClassifier(k, metric=self.metric, p=self.p,
                                                                           weights=self.weights)).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _inverse_square(d):
    return 1.0 / np.maximum(d, 1e-12) ** 2


class _Subspace(ClassifierMixin, BaseEstimator):
    """Random-subspace ensemble: 30 learners, each on max(1, min(28, p-1)) random predictors."""

    def __init__(self, base=None, n_estimators=30, random_state=0):
        self.base, self.n_estimators, self.random_state = base, n_estimators, random_state

    def fit(self, X, y):
        p = np.asarray(X).shape[1]
        k = max(1, min(28, p - 1))
        self.model_ = BaggingClassifier(clone(self.base), n_estimators=self.n_estimators, bootstrap=False,
                                        max_features=k, random_state=self.random_state).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _subspace(base):
    return _Subspace(base)


class AutoSVC(ClassifierMixin, BaseEstimator):
    """SVM classifier on standardised predictors with MATLAB-style kernel scale (box constraint 1)."""

    def __init__(self, kernel="rbf", degree=3, scale=None):
        self.kernel, self.degree, self.scale = kernel, degree, scale

    def fit(self, X, y):
        s = self.scale if self.scale is not None else np.sqrt(max(np.asarray(X).shape[1], 1))
        if self.kernel == "linear":
            steps = [StandardScaler(), FunctionTransformer(lambda Z, s=s: Z / s), SVC(kernel="linear", C=1.0)]
        elif self.kernel == "poly":
            steps = [StandardScaler(), SVC(kernel="poly", degree=self.degree, gamma=1 / s ** 2, coef0=1.0, C=1.0)]
        else:
            steps = [StandardScaler(), SVC(kernel="rbf", gamma=1 / s ** 2, C=1.0)]
        self.model_ = make_pipeline(*steps).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)


def _svc(kernel, degree=3, scale=None):
    return AutoSVC(kernel, degree, scale)


def make_regressor(name):
    """A fresh regression model by name, or by the original mode number (1-19)."""
    if isinstance(name, (int, np.integer)):
        name = REGRESSION_MODES[int(name)]
    if name not in REGRESSORS:
        raise ValueError(f"unknown regressor '{name}'; choose from {', '.join(REGRESSORS)}")
    return REGRESSORS[name]()


def make_classifier(name):
    """A fresh classifier by name, or by the original mode number (1-20)."""
    if isinstance(name, (int, np.integer)):
        name = CLASSIFIER_MODES[int(name)]
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier '{name}'; choose from {', '.join(CLASSIFIERS)}")
    return CLASSIFIERS[name]()


def scores_for_auc(model, X):
    """Continuous scores for the positive (second) class, whatever the model offers."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, -1]
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return d if d.ndim == 1 else d[:, -1]
    return model.predict(X).astype(float)
