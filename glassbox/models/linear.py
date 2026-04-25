"""Linear models implemented from scratch using NumPy.

LinearRegression — batch gradient descent on MSE loss.
LogisticRegression — batch gradient descent on binary cross-entropy.

Both expose loss_history for convergence inspection.
Bias is prepended internally; the caller never sees it.
"""

from __future__ import annotations

import numpy as np

from glassbox.models.base import BaseModel


# ======================================================================
# Linear Regression
# ======================================================================
class LinearRegression(BaseModel):
    """Ordinary least-squares regression via batch gradient descent.

    Loss function:
        L(w) = (1 / 2n) * Σ (ŷᵢ - yᵢ)²

    Update rule:
        w ← w - α · (1/n) · Xᵀ · (Xw - y)

    Parameters
    ----------
    learning_rate : float, default 0.01
        Step size for gradient descent.  Reduce if training diverges;
        increase if convergence is too slow.
    n_iterations : int, default 1000
        Maximum number of gradient-descent iterations.
    tol : float, default 1e-7
        Early-stopping tolerance.  Training halts when the absolute
        change in loss between consecutive iterations falls below this
        threshold.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features + 1,)
        Learned weight vector (includes bias at index 0).
    loss_history : list[float]
        MSE loss recorded after every iteration — useful for plotting
        convergence curves.
    """

    _task: str = "regression"

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-7,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

        # Learned state (populated by fit)
        self.weights_: np.ndarray | None = None
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Train the model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape

        rng = np.random.default_rng(seed=42)
        self.weights_ = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / n_features),
            size=n_features,
        )
        self.loss_history = []

        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = X_b @ self.weights_

            # MSE loss (for history tracking)
            loss = float(np.mean((y_pred - y) ** 2) / 2.0)
            self.loss_history.append(loss)

            # Early stopping
            if len(self.loss_history) >= 2:
                if abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                    break

            # Gradient and update
            gradient = (X_b.T @ (y_pred - y)) / n_samples
            self.weights_ -= self.learning_rate * gradient

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        self._validate_inputs(X)
        X_b = self._add_bias(X)
        return X_b @ self.weights_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones to *X* for the intercept term."""
        return np.column_stack([np.ones(X.shape[0]), X])


# ======================================================================
# Logistic Regression
# ======================================================================
class LogisticRegression(BaseModel):
    """Binary logistic classifier via batch gradient descent.

    Loss function (binary cross-entropy / log-loss):
        L(w) = -(1/n) Σ [ yᵢ · log(σᵢ) + (1 - yᵢ) · log(1 - σᵢ) ]

    where σᵢ = sigmoid(Xw)ᵢ.

    Update rule:
        w ← w - α · (1/n) · Xᵀ · (σ - y)

    Parameters
    ----------
    learning_rate : float, default 0.01
        Step size for gradient descent.
    n_iterations : int, default 1000
        Maximum number of gradient-descent iterations.
    tol : float, default 1e-7
        Early-stopping tolerance on the absolute change in loss.
    threshold : float, default 0.5
        Decision boundary: samples with predicted probability ≥
        *threshold* are assigned class 1, otherwise class 0.

    Attributes
    ----------
    weights_ : np.ndarray, shape (n_features + 1,)
        Learned weight vector (includes bias at index 0).
    loss_history : list[float]
        Binary cross-entropy loss recorded every iteration.
    classes_ : np.ndarray
        Unique class labels observed during ``fit``.
    """

    _task: str = "classification"

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float = 1e-7,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.threshold = threshold

        # Learned state
        self.weights_: np.ndarray | None = None
        self.loss_history: list[float] = []
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train the model using batch gradient descent with log-loss.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Binary target labels (values must be 0 or 1).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If *y* contains values other than 0 and 1.
        """
        self._validate_inputs(X, y)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.array([0, 1])):
            if len(self.classes_) != 2:
                raise ValueError(
                    "LogisticRegression requires exactly 2 classes, "
                    f"got {len(self.classes_)}: {self.classes_}."
                )
            # Auto-encode to 0/1 using sorted class order
            y = (y == self.classes_[1]).astype(np.float64)

        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape

        rng = np.random.default_rng(seed=42)
        self.weights_ = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / n_features),
            size=n_features,
        )
        self.loss_history = []

        for _ in range(self.n_iterations):
            # Forward pass
            z = X_b @ self.weights_
            y_prob = self._sigmoid(z)

            # Binary cross-entropy loss
            # Clip probabilities to avoid log(0)
            eps = 1e-15
            y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
            loss = -float(np.mean(
                y * np.log(y_prob_clipped)
                + (1.0 - y) * np.log(1.0 - y_prob_clipped)
            ))
            self.loss_history.append(loss)

            # Early stopping
            if len(self.loss_history) >= 2:
                if abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                    break

            # Gradient and update
            gradient = (X_b.T @ (y_prob - y)) / n_samples
            self.weights_ -= self.learning_rate * gradient

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        probas = self.predict_proba(X)
        preds = (probas >= self.threshold).astype(int)

        # Map back to original class labels if they weren't 0/1
        if self.classes_ is not None and not np.array_equal(
            self.classes_, np.array([0, 1])
        ):
            preds = self.classes_[preds]

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probability estimates in [0, 1].
        """
        self._check_is_fitted()
        self._validate_inputs(X)
        X_b = self._add_bias(X)
        return self._sigmoid(X_b @ self.weights_)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid with z clipped to [-500, 500] to prevent float64 overflow."""
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones to *X* for the intercept term."""
        return np.column_stack([np.ones(X.shape[0]), X])
