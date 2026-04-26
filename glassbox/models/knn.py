"""K-Nearest Neighbours implemented from scratch using NumPy.

Lazy learner — fit() just stores the training set; predict() computes
the full distance matrix and aggregates the k nearest labels.

Euclidean uses the ‖a−b‖² = ‖a‖² − 2aᵀb + ‖b‖² identity to avoid a
3-D broadcast. Manhattan uses the explicit 3-D broadcast (fine for the
dataset sizes GlassBox targets).

When task='auto', the task is inferred from y: integer dtype or ≤20
unique values → classification, otherwise → regression.
"""

from __future__ import annotations

import numpy as np

from glassbox.models.base import BaseModel
from glassbox.models.tree import _majority_vote


# ======================================================================
# Vectorised distance functions
# ======================================================================
def _euclidean_distances(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances without an explicit 3-D broadcast.

    Uses the identity  ‖a − b‖² = ‖a‖² − 2aᵀb + ‖b‖²  for efficiency.
    Negative values caused by floating-point rounding are clipped to 0
    before the square root.

    Parameters
    ----------
    X_test  : (n_test, p)
    X_train : (n_train, p)

    Returns
    -------
    distances : (n_test, n_train)
    """
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)    # (n_test, 1)
    sq_train = np.sum(X_train ** 2, axis=1)                  # (n_train,)
    cross = X_test @ X_train.T                               # (n_test, n_train)
    dist_sq = sq_test - 2.0 * cross + sq_train
    np.clip(dist_sq, 0.0, None, out=dist_sq)                 # Numerical guard
    return np.sqrt(dist_sq)


def _manhattan_distances(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Compute pairwise Manhattan (L1) distances via a 3-D broadcast.

    Parameters
    ----------
    X_test  : (n_test, p)
    X_train : (n_train, p)

    Returns
    -------
    distances : (n_test, n_train)
    """
    # Shape: (n_test, n_train, p) → sum over last axis → (n_test, n_train)
    return np.sum(
        np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]),
        axis=2,
    )


# ======================================================================
# Public class
# ======================================================================
class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbours classifier and regressor.

    Parameters
    ----------
    k : int, default 5
        Number of neighbours to use for prediction.
    metric : {"euclidean", "manhattan"}, default "euclidean"
        Distance metric.
    task : {"classification", "regression", "auto"}, default "auto"
        Prediction task.  ``"auto"`` detects classification when *y*
        has an integer dtype or ≤ 20 unique values, regression otherwise.

    Attributes
    ----------
    X_train_ : np.ndarray
        Memorised training features.
    y_train_ : np.ndarray
        Memorised training labels.
    n_features_ : int
        Number of features seen during ``fit``.
    classes_ : np.ndarray or None
        Unique class labels (classification only; ``None`` for regression).
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        task: str = "auto",
    ) -> None:
        super().__init__()
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError(
                f"metric must be 'euclidean' or 'manhattan', got {metric!r}."
            )
        if task not in {"classification", "regression", "auto"}:
            raise ValueError(
                f"task must be 'classification', 'regression', or 'auto', got {task!r}."
            )
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")

        self.k = k
        self.metric = metric
        self.task = task

        # Learned state
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.classes_: np.ndarray | None = None
        self._resolved_task: str = "classification"  # Determined in fit

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        """Memorise the training data.

        KNN has no training phase — this method just stores X and y and
        resolves the task type.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)

        if X.shape[0] < self.k:
            raise ValueError(
                f"k={self.k} is larger than the number of training samples "
                f"({X.shape[0]}).  Reduce k or add more data."
            )

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_features_ = X.shape[1]

        # --- Task detection ---
        if self.task == "auto":
            is_int_dtype = np.issubdtype(y.dtype, np.integer)
            is_few_unique = len(np.unique(y)) <= 20
            self._resolved_task = (
                "classification" if (is_int_dtype or is_few_unique) else "regression"
            )
        else:
            self._resolved_task = self.task

        # Set _task so BaseModel.score() uses the right metric
        self._task = self._resolved_task  # type: ignore[assignment]

        if self._resolved_task == "classification":
            self.classes_ = np.unique(y)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for *X*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Class labels for classification; continuous values for
            regression.
        """
        self._check_is_fitted()
        self._validate_inputs(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_}."
            )

        distances = self._compute_distances(X)

        # Indices of the k nearest neighbours for each test sample.
        # `kth=self.k - 1` keeps the pivot in-bounds when k == n_train
        # (np.argpartition requires kth in [0, n_train - 1]).
        # Shape: (n_test, k)
        nn_indices = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]

        return self._aggregate(nn_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_distances(self, X_test: np.ndarray) -> np.ndarray:
        """Return the (n_test, n_train) distance matrix."""
        if self.metric == "euclidean":
            return _euclidean_distances(X_test, self.X_train_)  # type: ignore[arg-type]
        return _manhattan_distances(X_test, self.X_train_)  # type: ignore[arg-type]

    def _aggregate(self, nn_indices: np.ndarray) -> np.ndarray:
        """Aggregate neighbour labels into predictions.

        Classification → majority vote.
        Regression     → mean.
        """
        neighbour_labels = self.y_train_[nn_indices]  # (n_test, k) # type: ignore[index]

        if self._resolved_task == "classification":
            return np.array(
                [_majority_vote(neighbour_labels[i]) for i in range(len(nn_indices))]
            )
        # Regression
        return neighbour_labels.mean(axis=1).astype(float)
