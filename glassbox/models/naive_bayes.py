"""Gaussian Naïve Bayes classifier implemented from scratch using NumPy.

The model assumes each feature is conditionally independent given the
class label and follows a Gaussian distribution.  These assumptions are
almost certainly violated in practice, yet the classifier is surprisingly
robust and serves as an excellent fast baseline.

Mathematics
-----------
Given a sample x, predict the class with the highest posterior:

    P(C_k | x) ∝ P(C_k) · ∏_j P(x_j | C_k)

where:

    P(C_k) = (n_k + α) / (n + α · K)          (Laplace-smoothed prior)
    P(x_j | C_k) = N(x_j; μ_kj, σ²_kj + ε)    (Gaussian likelihood)

Working in log-space avoids numerical underflow when multiplying many
small probabilities:

    log P(C_k | x) ∝ log P(C_k) + Σ_j log P(x_j | C_k)

Implementation notes
--------------------
* **Variance floor** — a small ``var_smoothing`` constant is added to
  every per-class per-feature variance.  This prevents log(0) when a
  feature has zero variance within a class (e.g. a constant column).
* **Laplace smoothing on priors** — prevents zero-probability classes on
  unseen data when ``alpha > 0`` (default 1.0).
* **Log-sum-exp trick** is not needed here because we only need the
  argmax, not the actual posterior probabilities.  For ``predict_proba``
  we subtract the max log-score before exponentiating for numerical
  stability.
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
        """Return unnormalised log-posteriors, shape (n_samples, n_classes).

        For each class k and sample i:
            log P(C_k | x_i) ∝ log P(C_k) + Σ_j log N(x_ij; μ_kj, σ²_kj)
        """
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
