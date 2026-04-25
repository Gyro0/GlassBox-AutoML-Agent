"""Data validation utilities for the GlassBox library."""

from __future__ import annotations

from typing import Any

import numpy as np


def check_array(X: Any) -> None:
    """
    Validate that X is a NumPy ndarray.
    
    Parameters
    ----------
    X : array-like
        Input to check
    
    Raises
    ------
    TypeError
        If X is not an np.ndarray
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> check_array(X)  # Passes silently
    
    >>> check_array([[1, 2], [3, 4]])  # Raises TypeError
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(X).__name__}"
        )


def check_is_fitted(obj: object, attributes: str | list[str]) -> None:
    """
    Check that transformer has been fitted.
    
    A fitted transformer stores attributes (like mean_, std_, categories_)
    that are computed during fit(). This function verifies they exist
    before transform() is called.
    
    Parameters
    ----------
    obj : object
        Transformer object to check
    attributes : str or list of str
        Attribute name(s) set during fit() (e.g., "mean_", "std_")
    
    Raises
    ------
    RuntimeError
        If any attribute doesn't exist on obj or is None
    
    Examples
    --------
    >>> from glassbox.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> check_is_fitted(scaler, 'mean_')  # Raises RuntimeError

    >>> scaler.fit(np.array([[1, 2], [3, 4]]))
    >>> check_is_fitted(scaler, ['mean_', 'std_'])  # Passes
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    
    for attr in attributes:
        if not hasattr(obj, attr) or getattr(obj, attr) is None:
            raise RuntimeError(
                f"Transformer not fitted. missing attribute: {attr}"
            )


def check_consistent_length(X: np.ndarray, y: np.ndarray) -> None:
    """
    Check that X and y have the same number of samples.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target vector
    
    Raises
    ------
    ValueError
        If n_samples don't match
    
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
    >>> y = np.array([0, 1, 0])                  # 3 samples
    >>> check_consistent_length(X, y)  # Passes
    
    >>> y_wrong = np.array([0, 1])  # 2 samples
    >>> check_consistent_length(X, y_wrong)  # Raises ValueError
    """
    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0]
    
    if n_samples_X != n_samples_y:
        raise ValueError(
            f"X and y have incompatible number of samples: "
            f"{n_samples_X} vs {n_samples_y}"
        )
