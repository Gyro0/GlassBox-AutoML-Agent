"""Unit tests for glassbox.models — all estimators.

Run with:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.models.base import BaseModel
from glassbox.models.forest import RandomForestClassifier, RandomForestRegressor
from glassbox.models.knn import KNearestNeighbors
from glassbox.models.linear import LinearRegression, LogisticRegression
from glassbox.models.naive_bayes import GaussianNaiveBayes
from glassbox.models.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _gini_impurity,
    _mse_impurity,
)


# ======================================================================
# Shared fixtures
# ======================================================================
@pytest.fixture()
def reg_data():
    """Linear relationship y = 3x1 + 2x2 + 5 + small noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 5.0 + rng.normal(scale=0.1, size=200)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture()
def clf_data():
    """Linearly separable binary dataset."""
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=2.0, scale=0.8, size=(100, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.8, size=(100, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 100 + [0] * 100)
    idx = rng.permutation(200)
    X, y = X[idx], y[idx]
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture()
def multiclass_data():
    """Three-class dataset, well-separated Gaussians."""
    rng = np.random.default_rng(42)
    centres = [(0, 0), (5, 0), (2.5, 4)]
    parts = [rng.normal(loc=c, scale=0.6, size=(60, 2)) for c in centres]
    X = np.vstack(parts)
    y = np.array([k for k in range(3) for _ in range(60)])
    idx = rng.permutation(180)
    X, y = X[idx], y[idx]
    return X[:144], y[:144], X[144:], y[144:]


# ======================================================================
# BaseModel contract
# ======================================================================
class TestBaseModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_subclass_must_implement_fit_predict(self):
        class Bad(BaseModel):
            pass
        with pytest.raises(TypeError):
            Bad()  # type: ignore[abstract]

    def test_check_is_fitted_raises(self):
        class Dummy(BaseModel):
            def fit(self, X, y): self._fitted = True; return self
            def predict(self, X): return np.zeros(X.shape[0])
        m = Dummy()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            m.score(np.array([[1.0]]), np.array([0.0]))

    def test_validate_inputs_bad_X_type(self):
        with pytest.raises(TypeError):
            BaseModel._validate_inputs([[1, 2]])  # type: ignore[arg-type]

    def test_validate_inputs_1d_X(self):
        with pytest.raises(ValueError, match="2-D"):
            BaseModel._validate_inputs(np.array([1.0, 2.0]))

    def test_validate_inputs_length_mismatch(self):
        with pytest.raises(ValueError, match="inconsistent"):
            BaseModel._validate_inputs(np.ones((3, 2)), np.ones(2))


# ======================================================================
# LinearRegression
# ======================================================================
class TestLinearRegression:
    def test_fit_returns_self(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = LinearRegression(learning_rate=0.05, n_iterations=500)
        assert m.fit(Xtr, ytr) is m

    def test_weights_shape(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = LinearRegression().fit(Xtr, ytr)
        assert m.weights_.shape == (Xtr.shape[1] + 1,)

    def test_r2_high(self, reg_data):
        Xtr, ytr, Xte, yte = reg_data
        m = LinearRegression(learning_rate=0.05, n_iterations=2000).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.95

    def test_loss_decreases(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = LinearRegression(learning_rate=0.05, n_iterations=100).fit(Xtr, ytr)
        assert m.loss_history[-1] < m.loss_history[0]

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LinearRegression().predict(np.ones((2, 2)))


# ======================================================================
# LogisticRegression
# ======================================================================
class TestLogisticRegression:
    def test_fit_returns_self(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        assert LogisticRegression(learning_rate=0.1, n_iterations=500).fit(Xtr, ytr) is not None

    def test_accuracy_high(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        m = LogisticRegression(learning_rate=0.1, n_iterations=1000).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.90

    def test_predict_proba_range(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        probas = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(Xtr, ytr).predict_proba(Xte)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_rejects_multiclass(self):
        X = np.ones((6, 2)); y = np.array([0, 1, 2, 0, 1, 2])
        with pytest.raises(ValueError, match="exactly 2 classes"):
            LogisticRegression().fit(X, y)

    def test_sigmoid_no_overflow(self):
        z = np.array([-1000.0, 0.0, 1000.0])
        result = LogisticRegression._sigmoid(z)
        assert np.all(np.isfinite(result))
        assert result[1] == pytest.approx(0.5)


# ======================================================================
# Impurity helpers
# ======================================================================
class TestImpurityFunctions:
    def test_gini_pure_node(self):
        assert _gini_impurity(np.array([1, 1, 1, 1])) == pytest.approx(0.0)

    def test_gini_balanced_binary(self):
        assert _gini_impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)

    def test_gini_empty(self):
        assert _gini_impurity(np.array([])) == pytest.approx(0.0)

    def test_mse_constant(self):
        assert _mse_impurity(np.array([3.0, 3.0, 3.0])) == pytest.approx(0.0)

    def test_mse_empty(self):
        assert _mse_impurity(np.array([])) == pytest.approx(0.0)


# ======================================================================
# DecisionTreeClassifier
# ======================================================================
class TestDecisionTreeClassifier:
    def test_fit_returns_self(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        assert DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr) is not None

    def test_predict_shape(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        preds = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr).predict(Xte)
        assert preds.shape == (Xte.shape[0],)

    def test_accuracy_high(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        m = DecisionTreeClassifier(max_depth=None).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.90

    def test_feature_importances_sum_to_one(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)
        assert m.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_feature_importances_shape(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)
        assert m.feature_importances_.shape == (Xtr.shape[1],)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DecisionTreeClassifier().predict(np.ones((2, 2)))

    def test_multiclass(self, multiclass_data):
        Xtr, ytr, Xte, yte = multiclass_data
        m = DecisionTreeClassifier(max_depth=8).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.80

    def test_pure_node_becomes_leaf(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1, 1, 1])
        assert np.all(DecisionTreeClassifier().fit(X, y).predict(X) == 1)

    def test_predict_proba_shape_and_sum(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        m = DecisionTreeClassifier(max_depth=5).fit(Xtr, ytr)
        proba = m.predict_proba(Xte)
        assert proba.shape == (Xte.shape[0], len(m.classes_))
        assert np.allclose(proba.sum(axis=1), 1.0)


# ======================================================================
# DecisionTreeRegressor
# ======================================================================
class TestDecisionTreeRegressor:
    def test_predict_shape(self, reg_data):
        Xtr, ytr, Xte, _ = reg_data
        preds = DecisionTreeRegressor(max_depth=5).fit(Xtr, ytr).predict(Xte)
        assert preds.shape == (Xte.shape[0],)

    def test_predict_is_float(self, reg_data):
        Xtr, ytr, Xte, _ = reg_data
        preds = DecisionTreeRegressor(max_depth=5).fit(Xtr, ytr).predict(Xte)
        assert preds.dtype == float

    def test_r2_reasonable(self, reg_data):
        Xtr, ytr, Xte, yte = reg_data
        assert DecisionTreeRegressor(max_depth=10).fit(Xtr, ytr).score(Xte, yte) > 0.70

    def test_feature_importances_sum_to_one(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = DecisionTreeRegressor(max_depth=5).fit(Xtr, ytr)
        assert m.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_constant_target(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([7.0, 7.0, 7.0])
        assert np.allclose(DecisionTreeRegressor().fit(X, y).predict(X), 7.0)


# ======================================================================
# RandomForestClassifier
# ======================================================================
class TestRandomForestClassifier:
    def test_n_estimators(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = RandomForestClassifier(n_estimators=7, random_state=0).fit(Xtr, ytr)
        assert len(m.estimators_) == 7

    def test_predict_shape(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        m = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
        assert m.predict(Xte).shape == (Xte.shape[0],)

    def test_accuracy_high(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        m = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.88

    def test_feature_importances(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
        assert m.feature_importances_.shape == (Xtr.shape[1],)
        assert m.feature_importances_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_predict_proba_sums_to_one(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        m = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
        proba = m.predict_proba(Xte)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_reproducibility(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        p1 = RandomForestClassifier(n_estimators=10, random_state=7).fit(Xtr, ytr).predict(Xte)
        p2 = RandomForestClassifier(n_estimators=10, random_state=7).fit(Xtr, ytr).predict(Xte)
        assert np.array_equal(p1, p2)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RandomForestClassifier().predict(np.ones((2, 2)))


# ======================================================================
# RandomForestRegressor
# ======================================================================
class TestRandomForestRegressor:
    def test_n_estimators(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = RandomForestRegressor(n_estimators=5, random_state=0).fit(Xtr, ytr)
        assert len(m.estimators_) == 5

    def test_predict_shape(self, reg_data):
        Xtr, ytr, Xte, _ = reg_data
        m = RandomForestRegressor(n_estimators=10, random_state=0).fit(Xtr, ytr)
        assert m.predict(Xte).shape == (Xte.shape[0],)

    def test_r2_high(self, reg_data):
        Xtr, ytr, Xte, yte = reg_data
        m = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=0).fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.85

    def test_predict_is_float(self, reg_data):
        Xtr, ytr, Xte, _ = reg_data
        preds = RandomForestRegressor(n_estimators=10, random_state=0).fit(Xtr, ytr).predict(Xte)
        assert preds.dtype == float


# ======================================================================
# GaussianNaiveBayes
# ======================================================================
class TestGaussianNaiveBayes:
    def test_predict_shape(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        assert GaussianNaiveBayes().fit(Xtr, ytr).predict(Xte).shape == (Xte.shape[0],)

    def test_accuracy_high(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        assert GaussianNaiveBayes().fit(Xtr, ytr).score(Xte, yte) > 0.88

    def test_multiclass(self, multiclass_data):
        Xtr, ytr, Xte, yte = multiclass_data
        assert GaussianNaiveBayes().fit(Xtr, ytr).score(Xte, yte) > 0.85

    def test_priors_sum_to_one(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = GaussianNaiveBayes().fit(Xtr, ytr)
        assert np.exp(m.class_log_priors_).sum() == pytest.approx(1.0, abs=1e-6)

    def test_theta_var_shapes(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = GaussianNaiveBayes().fit(Xtr, ytr)
        n_classes = len(np.unique(ytr))
        assert m.theta_.shape == (n_classes, Xtr.shape[1])
        assert m.var_.shape == (n_classes, Xtr.shape[1])

    def test_var_always_positive(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        assert np.all(GaussianNaiveBayes().fit(Xtr, ytr).var_ > 0)

    def test_predict_proba_sums_to_one(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        proba = GaussianNaiveBayes().fit(Xtr, ytr).predict_proba(Xte)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_zero_alpha_no_crash(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        GaussianNaiveBayes(alpha=0.0).fit(Xtr, ytr).predict(Xte)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            GaussianNaiveBayes().predict(np.ones((2, 2)))


# ======================================================================
# KNearestNeighbors
# ======================================================================
class TestKNearestNeighbors:
    def test_predict_shape_clf(self, clf_data):
        Xtr, ytr, Xte, _ = clf_data
        m = KNearestNeighbors(k=5, task="classification").fit(Xtr, ytr)
        assert m.predict(Xte).shape == (Xte.shape[0],)

    def test_predict_shape_reg(self, reg_data):
        Xtr, ytr, Xte, _ = reg_data
        m = KNearestNeighbors(k=5, task="regression").fit(Xtr, ytr)
        assert m.predict(Xte).shape == (Xte.shape[0],)

    def test_k1_memorises_training_data(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = KNearestNeighbors(k=1, task="classification").fit(Xtr, ytr)
        assert np.mean(m.predict(Xtr) == ytr) == pytest.approx(1.0)

    def test_accuracy_euclidean(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        m = KNearestNeighbors(k=5, metric="euclidean", task="classification").fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.88

    def test_accuracy_manhattan(self, clf_data):
        Xtr, ytr, Xte, yte = clf_data
        m = KNearestNeighbors(k=5, metric="manhattan", task="classification").fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.88

    def test_regression_r2(self, reg_data):
        Xtr, ytr, Xte, yte = reg_data
        m = KNearestNeighbors(k=5, task="regression").fit(Xtr, ytr)
        assert m.score(Xte, yte) > 0.80

    def test_auto_detects_classification(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = KNearestNeighbors(k=3, task="auto").fit(Xtr, ytr)
        assert m._resolved_task == "classification"

    def test_auto_detects_regression(self, reg_data):
        Xtr, ytr, _, _ = reg_data
        m = KNearestNeighbors(k=3, task="auto").fit(Xtr, ytr)
        assert m._resolved_task == "regression"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            KNearestNeighbors(k=3, metric="cosine")

    def test_k_larger_than_data_raises(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        with pytest.raises(ValueError, match="larger than the number"):
            KNearestNeighbors(k=999).fit(Xtr, ytr)

    def test_wrong_n_features_raises(self, clf_data):
        Xtr, ytr, _, _ = clf_data
        m = KNearestNeighbors(k=3, task="classification").fit(Xtr, ytr)
        with pytest.raises(ValueError, match="features"):
            m.predict(np.ones((5, 99)))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            KNearestNeighbors().predict(np.ones((2, 2)))
