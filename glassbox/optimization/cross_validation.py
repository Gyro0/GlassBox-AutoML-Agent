"""Cross-validation utilities for model evaluation."""

from typing import Callable, Iterator

import numpy as np


class KFoldCV:
    """Placeholder K-fold cross-validation splitter."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation index arrays."""
        raise NotImplementedError("KFoldCV.split is not implemented yet.")


def cross_val_score(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFoldCV,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Placeholder cross-validation scoring function."""
    raise NotImplementedError("cross_val_score is not implemented yet.")
