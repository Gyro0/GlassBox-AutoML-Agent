"""Decision tree models implemented from scratch using NumPy.

DecisionTreeClassifier splits on Gini impurity; DecisionTreeRegressor on MSE.

Internal helpers (_best_split, _build_tree, _traverse) are kept separate
from the public classes so they're easy to test and reused by forest.py.
The max_features param enables feature subsampling for random forests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from glassbox.models.base import BaseModel


# ======================================================================
# Internal node representation
# ======================================================================
@dataclass
class _Node:
    """A single node in a decision tree.

    Internal nodes have ``feature_index`` and ``threshold`` set and both
    ``left`` and ``right`` children.  Leaf nodes have ``value`` set and
    all other fields left at their defaults.

    ``n_samples`` and ``impurity_gain`` are stored at every *internal*
    node so that ``feature_importances_`` can be computed in one pass
    after training without re-touching the data.
    """

    feature_index: int | None = None    # Split feature  (None for leaves)
    threshold: float | None = None      # Split threshold (None for leaves)
    left: "_Node | None" = None         # Samples ≤ threshold
    right: "_Node | None" = None        # Samples >  threshold
    value: Any = None                   # Leaf prediction (None for internal nodes)
    n_samples: int = 0                  # Training samples that reached this node
    impurity_gain: float = 0.0          # Weighted impurity reduction from this split


# ======================================================================
# Impurity / leaf-value functions
# ======================================================================
def _gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: 1 - Σ p_k².

    Returns 0.0 for an empty or perfectly pure array.
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return float(1.0 - np.dot(probs, probs))


def _mse_impurity(y: np.ndarray) -> float:
    """Mean squared deviation from the column mean (population variance).

    Used as the splitting criterion for regression trees.
    """
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


def _majority_vote(y: np.ndarray) -> Any:
    """Return the most frequent label in *y*.  Ties broken by sort order."""
    values, counts = np.unique(y, return_counts=True)
    return values[int(np.argmax(counts))]


def _mean_value(y: np.ndarray) -> float:
    """Return the mean of *y* (leaf prediction for regression)."""
    return float(np.mean(y))


# ======================================================================
# Core tree-building helpers
# ======================================================================
def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion_fn: Any,
    max_features: int | None,
    rng: np.random.Generator | None,
) -> dict[str, Any] | None:
    """Find the (feature, threshold) pair that maximises impurity reduction.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    criterion_fn : callable
        Impurity function — lower is purer.
    max_features : int or None
        Number of features to consider.  ``None`` means all features.
    rng : np.random.Generator or None
        Random state for reproducible feature subsampling.

    Returns
    -------
    dict with keys ``feature_index``, ``threshold``, ``gain``  — or
    ``None`` if no beneficial split exists.
    """
    n_samples, n_total_features = X.shape

    # Feature subspace sampling (used by RandomForest)
    if max_features is None or max_features >= n_total_features:
        feature_indices = np.arange(n_total_features)
    else:
        if rng is None:
            rng = np.random.default_rng()
        feature_indices = rng.choice(
            n_total_features, size=max_features, replace=False
        )

    parent_impurity = criterion_fn(y)
    best_gain = -np.inf
    best_feature: int | None = None
    best_threshold: float | None = None

    for feat_idx in feature_indices:
        feature_vals = X[:, feat_idx]
        unique_vals = np.unique(feature_vals)

        if len(unique_vals) < 2:
            continue  # Constant feature — skip

        # Use midpoints between adjacent sorted unique values as thresholds.
        # This is equivalent to trying every possible split while being
        # numerically cleaner than using the raw values themselves.
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = feature_vals <= threshold
            n_left = int(left_mask.sum())
            n_right = n_samples - n_left

            if n_left == 0 or n_right == 0:
                continue

            impurity_reduction = parent_impurity - (
                n_left / n_samples * criterion_fn(y[left_mask])
                + n_right / n_samples * criterion_fn(y[~left_mask])
            )

            if impurity_reduction > best_gain:
                best_gain = impurity_reduction
                best_feature = feat_idx
                best_threshold = threshold

    if best_feature is None or best_gain <= 0.0:
        return None

    return {
        "feature_index": int(best_feature),
        "threshold": float(best_threshold),  # type: ignore[arg-type]
        "gain": float(best_gain),
    }


def _build_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    max_depth: int | None,
    min_samples_split: int,
    criterion_fn: Any,
    leaf_fn: Any,
    max_features: int | None,
    rng: np.random.Generator | None,
) -> _Node:
    """Recursively build a decision tree and return the root node.

    Stopping conditions (any one triggers a leaf):
    1. Fewer than ``min_samples_split`` samples at this node.
    2. ``max_depth`` reached.
    3. All labels identical (pure node).
    4. No split yields a positive impurity gain.
    """
    n_samples = len(y)
    node = _Node(n_samples=n_samples)

    # --- Stopping conditions ---
    if n_samples < min_samples_split:
        node.value = leaf_fn(y)
        return node

    if max_depth is not None and depth >= max_depth:
        node.value = leaf_fn(y)
        return node

    if len(np.unique(y)) == 1:  # Pure node
        node.value = leaf_fn(y)
        return node

    # --- Find best split ---
    split = _best_split(X, y, criterion_fn, max_features=max_features, rng=rng)
    if split is None:
        node.value = leaf_fn(y)
        return node

    # --- Partition and recurse ---
    left_mask = X[:, split["feature_index"]] <= split["threshold"]

    node.feature_index = split["feature_index"]
    node.threshold = split["threshold"]
    node.impurity_gain = split["gain"]
    node.left = _build_tree(
        X[left_mask], y[left_mask],
        depth + 1, max_depth, min_samples_split,
        criterion_fn, leaf_fn, max_features, rng,
    )
    node.right = _build_tree(
        X[~left_mask], y[~left_mask],
        depth + 1, max_depth, min_samples_split,
        criterion_fn, leaf_fn, max_features, rng,
    )
    return node


def _traverse(node: _Node, x: np.ndarray) -> Any:
    """Walk the tree for a single sample and return the leaf value.

    Uses an *iterative* loop (not recursion) to avoid any risk of hitting
    Python's recursion limit on deep trees.
    """
    while node.value is None:
        if x[node.feature_index] <= node.threshold:  # type: ignore[index]
            node = node.left  # type: ignore[assignment]
        else:
            node = node.right  # type: ignore[assignment]
    return node.value


def _compute_feature_importances(
    root: _Node, n_features: int, n_total: int
) -> np.ndarray:
    """Compute normalised feature importances from the fitted tree.

    Importance of feature *j* is the sum of weighted impurity reductions
    across all internal nodes that split on feature *j*:

        importance_j = Σ (n_node / n_total) * gain_j

    The result is normalised to sum to 1.
    """
    importances = np.zeros(n_features, dtype=float)

    # Iterative DFS with an explicit stack
    stack = [root]
    while stack:
        node = stack.pop()
        if node.value is not None:  # Leaf — no contribution
            continue
        importances[node.feature_index] += (  # type: ignore[index]
            node.n_samples / n_total * node.impurity_gain
        )
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    total = importances.sum()
    if total > 0.0:
        importances /= total
    return importances


# ======================================================================
# Public classes
# ======================================================================
class DecisionTreeClassifier(BaseModel):
    """Decision tree classifier using Gini impurity as the splitting criterion.

    Parameters
    ----------
    max_depth : int or None, default None
        Maximum tree depth.  ``None`` grows the tree until all leaves are
        pure or contain fewer than ``min_samples_split`` samples.
    min_samples_split : int, default 2
        Minimum samples required to attempt a split at a node.  Nodes
        with fewer samples become leaves.
    max_features : int or None, default None
        Number of features considered at each split.  ``None`` uses all
        features.  Set to ``int(sqrt(n_features))`` for Random Forest
        behaviour.
    random_state : int or None, default None
        Seed for the feature-subspace RNG.  Only relevant when
        ``max_features`` is not ``None``.

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels seen during ``fit``.
    n_features_ : int
        Number of features the tree was trained on.
    feature_importances_ : np.ndarray, shape (n_features,)
        Normalised impurity-based feature importances.
    """

    _task: str = "classification"

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        # State populated by fit
        self._root: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — class labels

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state) if self.max_features else None

        self._root = _build_tree(
            X, y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion_fn=_gini_impurity,
            leaf_fn=_majority_vote,
            max_features=self.max_features,
            rng=rng,
        )
        self.feature_importances_ = _compute_feature_importances(
            self._root, self.n_features_, len(y)
        )
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
        self._validate_inputs(X)
        return np.array([_traverse(self._root, row) for row in X])  # type: ignore[arg-type]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        The leaf value is a hard class label (majority vote), so this
        method one-hot-encodes the prediction into a probability matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        predictions = self.predict(X)
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        proba = np.zeros((len(X), n_classes), dtype=float)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}  # type: ignore[union-attr]
        for i, pred in enumerate(predictions):
            proba[i, class_to_idx[pred]] = 1.0
        return proba


class DecisionTreeRegressor(BaseModel):
    """Decision tree regressor using MSE variance reduction as the splitting
    criterion.

    Parameters
    ----------
    max_depth : int or None, default None
        Maximum tree depth.
    min_samples_split : int, default 2
        Minimum samples required to attempt a split.
    max_features : int or None, default None
        Number of features considered at each split.  ``None`` uses all.
    random_state : int or None, default None
        Seed for the feature-subspace RNG.

    Attributes
    ----------
    n_features_ : int
        Number of features the tree was trained on.
    feature_importances_ : np.ndarray, shape (n_features,)
        Normalised impurity-based feature importances.
    """

    _task: str = "regression"

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self._root: _Node | None = None
        self.n_features_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Build the regression tree from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — continuous targets

        Returns
        -------
        self
        """
        self._validate_inputs(X, y)
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state) if self.max_features else None

        self._root = _build_tree(
            X, y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion_fn=_mse_impurity,
            leaf_fn=_mean_value,
            max_features=self.max_features,
            rng=rng,
        )
        self.feature_importances_ = _compute_feature_importances(
            self._root, self.n_features_, len(y)
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for *X*.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_is_fitted()
        self._validate_inputs(X)
        return np.array(
            [_traverse(self._root, row) for row in X], dtype=float  # type: ignore[arg-type]
        )
