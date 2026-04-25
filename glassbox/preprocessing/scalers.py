"""Scaling transformers for feature normalization."""

from __future__ import annotations

from typing import Self

import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted


class MinMaxScaler(BaseTransformer):
    """
    Scale features to a fixed range [0, 1].
    
    Scales each feature by subtracting the minimum and dividing by the range
    (max - min). This ensures all features are in the [0, 1] interval.
    
    Handles the edge case where a column has no variance (min == max) by
    setting the denominator to 1, effectively returning all 0s for that column.
    
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Minimum values per feature (computed during fit)
    max_ : ndarray of shape (n_features,)
        Maximum values per feature (computed during fit)
    
    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing import MinMaxScaler
    >>> X = np.array([[1.0, 10.0],
    ...               [2.0, 20.0],
    ...               [3.0, 30.0]])
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """
    
    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> Self:
        """
        Compute min and max per column.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        
        Returns
        -------
        self : MinMaxScaler
            Fitted scaler
        """
        check_array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to [0, 1].
        
        Applies formula: (X - min) / (max - min)
        
        For constant columns (min == max), returns 0 to avoid division by zero.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to scale
        
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data in range [0, 1]
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        """
        check_array(X)
        check_is_fitted(self, ['min_', 'max_'])
        
        # Guard: compute range, where range=0 set to 1
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1  # Avoid division by zero
        
        return (X - self.min_) / range_


class StandardScaler(BaseTransformer):
    """
    Standardize features to have mean 0 and standard deviation 1.
    
    Scales each feature by subtracting the mean and dividing by the standard
    deviation. This transformation is also known as z-score normalization.
    
    Handles the edge case where a column has zero variance (std == 0) by
    setting the denominator to 1, effectively returning all 0s for that column.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Mean values per feature (computed during fit)
    std_ : ndarray of shape (n_features,)
        Standard deviation per feature (computed during fit)
    
    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing import StandardScaler
    >>> X = np.array([[1.0, 10.0],
    ...               [2.0, 20.0],
    ...               [3.0, 30.0]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)
    >>> X_scaled = scaler.transform(X)
    >>> X_scaled.mean(axis=0)  # Should be ~[0, 0]
    >>> X_scaled.std(axis=0)   # Should be ~[1, 1]
    """
    
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
    
    def fit(self, X: np.ndarray) -> Self:
        """
        Compute mean and standard deviation per column.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        
        Returns
        -------
        self : StandardScaler
            Fitted scaler
        """
        check_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)  # Population std (ddof=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to mean 0 and std 1.
        
        Applies formula: (X - mean) / std
        
        For constant columns (std == 0), returns 0 to avoid division by zero.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to standardize
        
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Standardized data with mean ~0 and std ~1
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        """
        check_array(X)
        check_is_fitted(self, ['mean_', 'std_'])
        
        # Guard: where std=0 set to 1
        std_safe = self.std_.copy()
        std_safe[std_safe == 0] = 1
        
        return (X - self.mean_) / std_safe
