"""Categorical encoding transformers."""

from __future__ import annotations

import warnings
from typing import Any, Self

import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted


class LabelEncoder(BaseTransformer):
    """
    Encode categorical features as integers.
    
    Maps each unique category to an integer (0, 1, 2, ...).
    Categories are sorted alphabetically for consistency.
    
    Useful for ordinal or binary categorical features. For nominal
    (unordered) categorical features, use OneHotEncoder instead.
    
    Attributes
    ----------
    classes_ : ndarray
        The sorted unique categories
    mapping_ : dict
        Dictionary mapping categories to integer labels
    
    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing import LabelEncoder
    >>> X = np.array([["red"], ["blue"], ["red"], ["green"]])
    >>> encoder = LabelEncoder()
    >>> encoder.fit(X)
    >>> encoder.transform(X)
    array([[2],
           [0],
           [2],
           [1]])
    """
    
    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.mapping_: dict[Any, int] | None = None
        self.unseen_values_: list[Any] = []
    
    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the categories and create mapping.
        
        For each column, finds unique values, sorts them for consistency,
        and creates a mapping from category to integer.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data with categorical features
        
        Returns
        -------
        self : LabelEncoder
            Fitted encoder
        """
        check_array(X)
        
        # Get unique categories (1D array), sorted for consistency
        self.classes_ = np.unique(X.flatten())
        
        # Create mapping dict: category → integer index
        self.mapping_ = {cat: idx for idx, cat in enumerate(self.classes_)}
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features as integers.
        
        Replaces each category with its corresponding integer label.
        Unseen categories are encoded as -1 and reported with a warning.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Categorical data to encode
        
        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_features)
            Integer-encoded data
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        """
        check_array(X)
        check_is_fitted(self, 'mapping_')
        
        X_encoded = np.zeros_like(X, dtype=int)
        self.unseen_values_ = []
        
        for i, val in enumerate(X.flatten()):
            if val in self.mapping_:
                X_encoded.flat[i] = self.mapping_[val]
            else:
                self.unseen_values_.append(val)
                X_encoded.flat[i] = -1

        if self.unseen_values_:
            unique_unseen = sorted(set(self.unseen_values_), key=lambda value: repr(value))
            warnings.warn(
                "Unseen categories encoded as -1: "
                + ", ".join(repr(value) for value in unique_unseen),
                UserWarning,
                stacklevel=2,
            )
        
        return X_encoded.reshape(X.shape)


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as one-hot vectors.
    
    Creates a binary column for each unique category value.
    For each sample, only the column corresponding to the sample's
    category is set to 1, all others are 0.
    
    Use this for nominal (unordered) categorical features.
    Categories are sorted for consistency.
    
    Attributes
    ----------
    categories_ : list of ndarray
        List of sorted unique categories per column
    
    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing import OneHotEncoder
    >>> X = np.array([["red", "small"],
    ...               ["blue", "large"],
    ...               ["red", "medium"]])
    >>> encoder = OneHotEncoder()
    >>> encoder.fit(X)
    >>> encoder.transform(X)
    array([[0, 1, 0, 1],   # red (2nd), small (3rd)
           [1, 0, 1, 0],   # blue (1st), large (2nd)
           [0, 1, 1, 0]])  # red (2nd), medium (1st)
    """
    
    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None
    
    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the unique categories per column.
        
        For each column, finds unique values and sorts them for consistency.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data with categorical features
        
        Returns
        -------
        self : OneHotEncoder
            Fitted encoder
        """
        check_array(X)
        
        n_features = X.shape[1]
        self.categories_ = []
        
        for col_idx in range(n_features):
            col = X[:, col_idx]
            # Get unique values, sorted for consistency
            unique_vals = np.unique(col)
            self.categories_.append(unique_vals)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features as one-hot vectors.
        
        Creates binary columns for each category. Each sample has a 1 in
        the column corresponding to its category, 0 everywhere else.
        
        Unseen categories are encoded as all zeros (handled gracefully).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Categorical data to encode
        
        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_output_features)
            One-hot encoded data (n_output_features = sum of unique categories)
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        """
        check_array(X)
        check_is_fitted(self, 'categories_')
        
        n_samples = X.shape[0]
        
        # Calculate output shape: one column per category across all features
        n_output_features = sum(len(cats) for cats in self.categories_)
        X_encoded = np.zeros((n_samples, n_output_features), dtype=int)
        
        # Fill in the 1s for matching categories
        col_offset = 0
        for col_idx, categories in enumerate(self.categories_):
            for row_idx in range(n_samples):
                val = X[row_idx, col_idx]
                
                # Find position of this value in categories
                if val in categories:
                    pos = np.where(categories == val)[0][0]
                    X_encoded[row_idx, col_offset + pos] = 1
                # else: leave as 0 (unseen category handled gracefully)
            
            col_offset += len(categories)
        
        return X_encoded
