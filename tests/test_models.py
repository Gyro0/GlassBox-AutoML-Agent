"""Unit tests for glassbox.models — base class, LinearRegression, LogisticRegression.

Run with:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.models.base import BaseModel
from glassbox.models.linear import LinearRegression, LogisticRegression


# ======================================================================
# Fixtures — reusable synthetic datasets
# ======================================================================
@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple linear relationship: y = 3*x1 + 2*x2 + 5 + noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 5.0 + rng.normal(scale=0.1, size=200)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly separable binary dataset."""
    rng = np.random.default_rng(0)
    n = 200
    X_pos = rng.normal(loc=2.0, scale=0.8, size=(n // 2, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.8, size=(n // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    return X[:160], y[:160], X[160:], y[160:]


# ======================================================================
# BaseModel contract tests
# ======================================================================
class TestBaseModel:
    """Verify that BaseModel cannot be instantiated and enforces interface."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_subclass_must_implement_fit_and_predict(self) -> None:
        class IncompleteModel(BaseModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()  # type: ignore[abstract]

    def test_minimal_subclass_works(self) -> None:
        class DummyModel(BaseModel):
            _task = "classification"

            def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyModel":
                self._fitted = True
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones(X.shape[0])

        model = DummyModel()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (2,)

    def test_score_raises_before_fit(self) -> None:
        class DummyModel(BaseModel):
            def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyModel":
                self._fitted = True
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(X.shape[0])

        model = DummyModel()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.score(np.array([[1]]), np.array([0]))


# ======================================================================
# Input validation tests
# ======================================================================
class TestValidation:
    """Test the _validate_inputs guard."""

    def test_rejects_non_array_X(self) -> None:
        with pytest.raises(TypeError, match="numpy array"):
            BaseModel._validate_inputs([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_rejects_1d_X(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            BaseModel._validate_inputs(np.array([1, 2, 3]))

    def test_rejects_non_array_y(self) -> None:
        X = np.array([[1, 2]])
        with pytest.raises(TypeError, match="numpy array"):
            BaseModel._validate_inputs(X, [1])  # type: ignore[arg-type]

    def test_rejects_2d_y(self) -> None:
        X = np.array([[1, 2]])
        with pytest.raises(ValueError, match="1-D"):
            BaseModel._validate_inputs(X, np.array([[1]]))

    def test_rejects_length_mismatch(self) -> None:
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="inconsistent lengths"):
            BaseModel._validate_inputs(X, y)

    def test_accepts_valid_inputs(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        BaseModel._validate_inputs(X, y)  # should not raise


# ======================================================================
# LinearRegression tests
# ======================================================================
class TestLinearRegression:
    """Verify LinearRegression learns correct weights and predicts accurately."""

    def test_fit_returns_self(self, regression_data) -> None:
        X_train, y_train, _, _ = regression_data
        model = LinearRegression(learning_rate=0.05, n_iterations=500)
        result = model.fit(X_train, y_train)
        assert result is model

    def test_weights_shape(self, regression_data) -> None:
        X_train, y_train, _, _ = regression_data
        model = LinearRegression().fit(X_train, y_train)
        # n_features + 1 (bias)
        assert model.weights_.shape == (X_train.shape[1] + 1,)

    def test_learns_correct_coefficients(self, regression_data) -> None:
        """y = 3*x1 + 2*x2 + 5. Weights should be close to [5, 3, 2]."""
        X_train, y_train, _, _ = regression_data
        model = LinearRegression(learning_rate=0.05, n_iterations=2000)
        model.fit(X_train, y_train)

        # bias ≈ 5, w1 ≈ 3, w2 ≈ 2
        assert abs(model.weights_[0] - 5.0) < 0.3, f"bias: {model.weights_[0]}"
        assert abs(model.weights_[1] - 3.0) < 0.3, f"w1: {model.weights_[1]}"
        assert abs(model.weights_[2] - 2.0) < 0.3, f"w2: {model.weights_[2]}"

    def test_predict_shape(self, regression_data) -> None:
        X_train, y_train, X_test, _ = regression_data
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_predict_raises_before_fit(self) -> None:
        model = LinearRegression()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_r2_score_high(self, regression_data) -> None:
        """On a nearly-linear dataset, R² should be close to 1.0."""
        X_train, y_train, X_test, y_test = regression_data
        model = LinearRegression(learning_rate=0.05, n_iterations=2000)
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        assert r2 > 0.95, f"R² = {r2} (expected > 0.95)"

    def test_loss_decreases(self, regression_data) -> None:
        X_train, y_train, _, _ = regression_data
        model = LinearRegression(learning_rate=0.05, n_iterations=100)
        model.fit(X_train, y_train)
        assert len(model.loss_history) > 1
        assert model.loss_history[-1] < model.loss_history[0]

    def test_early_stopping(self) -> None:
        """With very tight tol and easy data, training should stop early."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 1))
        y = 2.0 * X[:, 0] + 1.0
        model = LinearRegression(
            learning_rate=0.1, n_iterations=10000, tol=1e-12
        )
        model.fit(X, y)
        # Should converge well before 10000 iterations
        assert len(model.loss_history) < 10000

    def test_repr(self) -> None:
        model = LinearRegression(learning_rate=0.05, n_iterations=500)
        r = repr(model)
        assert "LinearRegression" in r
        assert "0.05" in r

    def test_get_params(self) -> None:
        model = LinearRegression(learning_rate=0.05, n_iterations=500, tol=1e-6)
        params = model.get_params()
        assert params["learning_rate"] == 0.05
        assert params["n_iterations"] == 500
        assert params["tol"] == 1e-6


# ======================================================================
# LogisticRegression tests
# ======================================================================
class TestLogisticRegression:
    """Verify LogisticRegression classifies binary data correctly."""

    def test_fit_returns_self(self, classification_data) -> None:
        X_train, y_train, _, _ = classification_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        result = model.fit(X_train, y_train)
        assert result is model

    def test_weights_shape(self, classification_data) -> None:
        X_train, y_train, _, _ = classification_data
        model = LogisticRegression().fit(X_train, y_train)
        assert model.weights_.shape == (X_train.shape[1] + 1,)

    def test_predict_returns_binary(self, classification_data) -> None:
        X_train, y_train, X_test, _ = classification_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_shape(self, classification_data) -> None:
        X_train, y_train, X_test, _ = classification_data
        model = LogisticRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (X_test.shape[0],)

    def test_predict_proba_range(self, classification_data) -> None:
        """Probabilities must be in [0, 1]."""
        X_train, y_train, X_test, _ = classification_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_accuracy_high(self, classification_data) -> None:
        """Linearly separable data — accuracy should exceed 90%."""
        X_train, y_train, X_test, y_test = classification_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        assert acc > 0.90, f"Accuracy = {acc} (expected > 0.90)"

    def test_loss_decreases(self, classification_data) -> None:
        X_train, y_train, _, _ = classification_data
        model = LogisticRegression(learning_rate=0.1, n_iterations=200)
        model.fit(X_train, y_train)
        assert len(model.loss_history) > 1
        assert model.loss_history[-1] < model.loss_history[0]

    def test_predict_raises_before_fit(self) -> None:
        model = LogisticRegression()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_rejects_multiclass(self) -> None:
        """LogisticRegression should only accept 2-class problems."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        model = LogisticRegression()
        with pytest.raises(ValueError, match="exactly 2 classes"):
            model.fit(X, y)

    def test_sigmoid_stability(self) -> None:
        """Sigmoid should not overflow on extreme inputs."""
        extreme = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        result = LogisticRegression._sigmoid(extreme)
        assert np.all(np.isfinite(result))
        assert result[2] == pytest.approx(0.5)

    def test_non_01_labels(self) -> None:
        """Should handle labels that aren't exactly 0 and 1."""
        rng = np.random.default_rng(42)
        X_pos = rng.normal(loc=3.0, size=(50, 2))
        X_neg = rng.normal(loc=-3.0, size=(50, 2))
        X = np.vstack([X_pos, X_neg])
        y = np.array([10] * 50 + [20] * 50)  # non-standard labels

        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({10, 20})

    def test_get_params(self) -> None:
        model = LogisticRegression(
            learning_rate=0.05, n_iterations=500, threshold=0.6
        )
        params = model.get_params()
        assert params["learning_rate"] == 0.05
        assert params["threshold"] == 0.6
