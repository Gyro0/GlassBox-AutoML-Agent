"""SimpleImputer for handling missing values."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Self

import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted

_MISSING_TEXT = {"", "na", "nan", "none", "null"}


def _is_missing_value(value: Any) -> bool:
    """Return True for NaN-like values in numeric or object arrays."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_TEXT
    return False


def _missing_mask(values: np.ndarray) -> np.ndarray:
    """Vectorized missing-value mask that works for mixed dtypes."""
    if np.issubdtype(values.dtype, np.number):
        return np.isnan(values.astype(float))
    return np.asarray([_is_missing_value(value) for value in values], dtype=bool)


def _mode(values: np.ndarray) -> Any:
    """Return the most frequent value with a deterministic tie-breaker."""
    counts = Counter(values.tolist())
    max_count = max(counts.values())
    candidates = [value for value, count in counts.items() if count == max_count]
    return sorted(candidates, key=lambda value: repr(value))[0]


class SimpleImputer(BaseTransformer):
    """
    Replace missing values (NaN) with computed statistics.
    
    This transformer computes a statistic (mean, median, or mode) for each
    column during fit(), then replaces NaN values with these statistics
    during transform().
    
    Parameters
    ----------
    strategy : str in {"mean", "median", "mode"}, default="mean"
        The imputation strategy:
        - "mean": Replace NaN with column mean (numerical columns)
        - "median": Replace NaN with column median (robust to outliers)
        - "mode": Replace NaN with most frequent value (categorical columns)
    
    Attributes
    ----------
    fill_values_ : ndarray of shape (n_features,)
        The value to use for imputation per feature (computed during fit)
    
    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing import SimpleImputer
    >>> X = np.array([[1.0, 5.0],
    ...               [2.0, np.nan],
    ...               [3.0, 7.0]])
    >>> imputer = SimpleImputer(strategy="mean")
    >>> imputer.fit(X)
    >>> imputer.transform(X)
    array([[1., 5.],
           [2., 6.],
           [3., 7.]])
    """
    
    def __init__(
        self,
        strategy: Literal["mean", "median", "mode"] = "mean",
    ) -> None:
        if strategy not in ["mean", "median", "mode"]:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Must be 'mean', 'median', or 'mode'"
            )
        self.strategy = strategy
        self.fill_values_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> Self:
        """
        Compute fill values for each column.
        
        For each column, computes the statistic specified by strategy,
        ignoring NaN values. Raises an error if an entire column is NaN.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        
        Returns
        -------
        self : SimpleImputer
            Fitted imputer
        
        Raises
        ------
        ValueError
            If an entire column is NaN (cannot compute statistic)
        """
        check_array(X)
        
        n_features = X.shape[1]
        self.fill_values_ = np.empty(n_features, dtype=object)
        
        for col_idx in range(n_features):
            col = X[:, col_idx]
            missing = _missing_mask(col)
            observed_values = col[~missing]
            
            # Check for all-missing column
            if len(observed_values) == 0:
                raise ValueError(
                    f"Cannot impute column {col_idx}: "
                    f"entire column is NaN or missing"
                )
            
            # Compute fill value based on strategy
            if self.strategy == "mean":
                fill_val = float(np.mean(observed_values.astype(float)))
            elif self.strategy == "median":
                fill_val = float(np.median(observed_values.astype(float)))
            else:
                fill_val = _mode(observed_values)
            
            self.fill_values_[col_idx] = fill_val
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaN with stored fill values.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to impute
        
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Data with NaN replaced by fill_values_
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        """
        check_array(X)
        check_is_fitted(self, 'fill_values_')
        
        X_imputed = X.copy()  # Don't modify original
        
        for col_idx in range(X.shape[1]):
            missing = _missing_mask(X_imputed[:, col_idx])
            X_imputed[missing, col_idx] = self.fill_values_[col_idx]
        
        return X_imputed
