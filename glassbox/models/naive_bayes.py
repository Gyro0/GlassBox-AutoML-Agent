"""Gaussian Naïve Bayes classifier implemented from scratch using NumPy.

Assumes features are conditionally independent and Gaussian given the class.
Despite those assumptions rarely holding, it works surprisingly well as a fast baseline.

Prediction uses log-space to avoid underflow:
    log P(C_k | x) ∝ log P(C_k) + Σ_j log N(x_j; μ_kj, σ²_kj)

var_smoothing adds a small floor to all variances (prevents log(0) on constant columns).
alpha applies Laplace smoothing to the class priors.
"""

from __future__ import annotations

import numpy as np

from glassbox.models.base import BaseModel


class GaussianNaiveBayes(BaseModel):
    """Gaussian Naïve Bayes classifier.

    Parameters
    ----------
    alpha : float, default 1.0
        Laplace smoothing parameter for class priors.  Set to 0.0 to
        disable smoothing (maximum-likelihood priors).
    var_smoothing : float, default 1e-9
        Variance floor added to all per-class feature variances.
        Prevents division by zero or log(0) for constant features.

    Attributes
    ----------
    classes_ : np.ndarray, shape (n_classes,)
        Unique class labels seen during ``fit``.
    class_log_priors_ : np.ndarray, shape (n_classes,)
        Log of the (smoothed) prior probability for each class.
    theta_ : np.ndarray, shape (n_classes, n_features)
        Per-class, per-feature means (μ_kj).
    var_ : np.ndarray, shape (n_classes, n_features)
        Per-class, per-feature variances (σ²_kj + var_smoothing).
    n_features_ : int
        Number of features seen during ``fit``.
    """

    _task: str = "classification"

    def __init__(
        self,
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.var_smoothing = var_smoothing

        # Learned state
        self.classes_: np.ndarray | None = None
        self.class_log_priors_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.n_features_: int | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """Learn Gaussian parameters and class priors from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.  All features must be numeric.
        y : np.ndarray, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_ = n_features

        # --- Allocate parameter arrays ---
        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_ = np.zeros((n_classes, n_features), dtype=float)

        # --- Compute per-class statistics ---
        class_counts = np.zeros(n_classes, dtype=float)
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            class_counts[idx] = len(X_cls)
            self.theta_[idx] = X_cls.mean(axis=0)
            # Use population variance (ddof=0); add var_smoothing floor
            self.var_[idx] = X_cls.var(axis=0) + self.var_smoothing

        # --- Laplace-smoothed log priors ---
        # P(C_k) = (n_k + alpha) / (n + alpha * K)
        smoothed_counts = class_counts + self.alpha
        self.class_log_priors_ = np.log(smoothed_counts / smoothed_counts.sum())

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        log_posteriors = self._compute_log_posteriors(X)
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior probability estimates for each class.

        Uses the log-sum-exp trick for numerical stability.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Each row sums to 1.
        """
        self._check_is_fitted()
        log_posteriors = self._compute_log_posteriors(X)

        # Log-sum-exp for stable normalisation
        max_log = log_posteriors.max(axis=1, keepdims=True)
        shifted = np.exp(log_posteriors - max_log)
        return shifted / shifted.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_log_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised log-posteriors, shape (n_samples, n_classes)."""
        self._validate_inputs(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        log_posteriors = np.zeros((n_samples, n_classes), dtype=float)

        for idx in range(n_classes):
            mean = self.theta_[idx]       # shape (n_features,)
            var = self.var_[idx]          # shape (n_features,)
            # Log Gaussian likelihood: -0.5 * [log(2π σ²) + (x - μ)² / σ²]
            # Summed over features — broadcasting: X is (n, p), mean/var (p,)
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var)
                + (X - mean) ** 2 / var,
                axis=1,
            )
            log_posteriors[:, idx] = (
                self.class_log_priors_[idx] + log_likelihood  # type: ignore[index]
            )

        return log_posteriors
