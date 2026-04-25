"""Random forest models implemented from scratch using NumPy only.

This module provides two ensemble models built on top of ``tree.py``:

* **RandomForestClassifier** — aggregates tree predictions by majority vote.
* **RandomForestRegressor**  — aggregates tree predictions by mean.

How it works
------------
For each of the ``n_estimators`` trees:

1. **Bootstrap sampling** — draw ``n_samples`` rows *with replacement* from
   the training data.  On average ~63% of distinct rows appear in each
   bootstrap sample; the rest form the natural out-of-bag set.
2. **Feature subspace** — at *every split* inside a tree, only
   ``max_features`` randomly chosen features are considered.  By default
   this is ``ceil(sqrt(n_features))`` for classifiers and
   ``ceil(n_features / 3)`` for regressors (matching Scikit-Learn defaults).
   This decorrelates the trees and reduces overfitting.
3. **Aggregation** — classifier uses majority vote; regressor uses mean.

Feature importances are computed as the average of each tree's
``feature_importances_`` array, weighted equally.
"""

from __future__ import annotations

import numpy as np

from glassbox.models.base import BaseModel
from glassbox.models.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    _majority_vote,
)


# ======================================================================
# Helpers
# ======================================================================
def _bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bootstrap replicate of (X, y).
    Samples ``n_samples`` rows with replacement.
    """
    n = X.shape[0]
    indices = rng.integers(0, n, size=n)
    return X[indices], y[indices]


def _resolve_max_features(max_features: int | str | None, n_features: int, task: str) -> int:
    """Translate the ``max_features`` hyper-parameter to a concrete integer.

    Rules
    -----
    * ``"sqrt"``    → ``ceil(sqrt(n_features))``
    * ``"third"``   → ``ceil(n_features / 3)``
    * ``None``      → ``"sqrt"`` for classifiers, ``"third"`` for regressors
    * ``int``       → used as-is (clamped to [1, n_features])
    """
    if max_features is None:
        max_features = "sqrt" if task == "classification" else "third"

    if max_features == "sqrt":
        return max(1, int(np.ceil(np.sqrt(n_features))))
    if max_features == "third":
        return max(1, int(np.ceil(n_features / 3)))
    if isinstance(max_features, int):
        return max(1, min(max_features, n_features))

    raise ValueError(
        f"max_features must be 'sqrt', 'third', an int, or None.  Got {max_features!r}."
    )


# ======================================================================
# Public classes
# ======================================================================
class RandomForestClassifier(BaseModel):
    """Random forest classifier — ensemble of decorrelated decision trees.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees to build.
    max_depth : int or None, default None
        Maximum depth of each individual tree.  ``None`` grows trees until
        leaves are pure or ``min_samples_split`` is hit.
    min_samples_split : int, default 2
        Minimum samples required to split an internal node in each tree.
    max_features : int, "sqrt", "third", or None, default None
        Features to consider at each split.  ``None`` uses ``"sqrt"``.
    random_state : int or None, default None
        Master seed for reproducible bootstrap and feature sampling.

    Attributes
    ----------
    estimators_ : list[DecisionTreeClassifier]
        The fitted individual trees.
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    n_features_ : int
        Number of input features.
    feature_importances_ : np.ndarray, shape (n_features,)
        Mean feature importances across all trees.
    """

    _task: str = "classification"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Build ``n_estimators`` trees on bootstrap samples of (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        resolved_max_features = _resolve_max_features(
            self.max_features, self.n_features_, "classification"
        )

        master_rng = np.random.default_rng(self.random_state)
        # Pre-generate seeds so each tree is reproducible independently
        tree_seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        self.estimators_ = []
        importances_sum = np.zeros(self.n_features_, dtype=float)

        for seed in tree_seeds:
            tree_rng = np.random.default_rng(int(seed))
            X_boot, y_boot = _bootstrap_sample(X, y, tree_rng)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=resolved_max_features,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            importances_sum += tree.feature_importances_  # type: ignore[operator]

        self.feature_importances_ = importances_sum / self.n_estimators
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by majority vote across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        self._validate_inputs(X)

        # Collect predictions from every tree: shape (n_estimators, n_samples)
        all_preds = np.array(
            [tree.predict(X) for tree in self.estimators_]
        )

        # Majority vote per sample
        return np.array(
            [_majority_vote(all_preds[:, i]) for i in range(X.shape[0])]
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates (vote fractions).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        self._validate_inputs(X)

        n_classes = len(self.classes_)  # type: ignore[arg-type]
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}  # type: ignore[union-attr]
        proba = np.zeros((X.shape[0], n_classes), dtype=float)

        for tree in self.estimators_:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                if pred in class_to_idx:
                    proba[i, class_to_idx[pred]] += 1.0

        proba /= self.n_estimators
        return proba


class RandomForestRegressor(BaseModel):
    """Random forest regressor — ensemble of decorrelated regression trees.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    max_depth : int or None, default None
        Maximum depth of each tree.
    min_samples_split : int, default 2
        Minimum samples to split an internal node.
    max_features : int, "sqrt", "third", or None, default None
        Features considered at each split.  ``None`` uses ``"third"``.
    random_state : int or None, default None
        Master seed for reproducibility.

    Attributes
    ----------
    estimators_ : list[DecisionTreeRegressor]
        The fitted individual trees.
    n_features_ : int
        Number of input features.
    feature_importances_ : np.ndarray, shape (n_features,)
        Mean feature importances across all trees.
    """

    _task: str = "regression"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_: list[DecisionTreeRegressor] = []
        self.n_features_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Build ``n_estimators`` regression trees on bootstrap samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        self.n_features_ = X.shape[1]
        resolved_max_features = _resolve_max_features(
            self.max_features, self.n_features_, "regression"
        )

        master_rng = np.random.default_rng(self.random_state)
        tree_seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        self.estimators_ = []
        importances_sum = np.zeros(self.n_features_, dtype=float)

        for seed in tree_seeds:
            tree_rng = np.random.default_rng(int(seed))
            X_boot, y_boot = _bootstrap_sample(X, y, tree_rng)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=resolved_max_features,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            importances_sum += tree.feature_importances_  # type: ignore[operator]

        self.feature_importances_ = importances_sum / self.n_estimators
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values as the mean across all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        self._validate_inputs(X)

        # Shape: (n_estimators, n_samples)
        all_preds = np.array(
            [tree.predict(X) for tree in self.estimators_], dtype=float
        )
        return all_preds.mean(axis=0)
