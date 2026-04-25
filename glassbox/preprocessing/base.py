"""Abstract base class for all preprocessing transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class BaseTransformer(ABC):
    """
    Abstract base class for all preprocessing transformers.
    
    Every transformer must:
    1. Inherit from this class
    2. Implement fit(X) - learn and store statistics
    3. Implement transform(X) - apply learned transformation
    
    The fit_transform() method is provided automatically and combines
    both operations for convenience.
    
    Examples
    --------
    >>> from glassbox.preprocessing import StandardScaler
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)
    >>> X_scaled = scaler.transform(X)
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> Self:
        """
        Learn transformation parameters from X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data to learn from
        
        Returns
        -------
        self : object
            Fitted transformer (return self for method chaining)
        
        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Typically stores statistics as attributes with _ suffix:
        - self.mean_
        - self.std_
        - self.categories_
        - etc.
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned transformation to X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform (usually different from fit data)
        
        Returns
        -------
        X_transformed : np.ndarray
            Transformed data
        
        Raises
        ------
        RuntimeError
            If fit() hasn't been called yet
        
        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Should call check_is_fitted() to verify fit() was called first.
        """
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to X, then transform it.
        
        This is a convenience method. Equivalent to:
        
        >>> return self.fit(X).transform(X)
        
        Use this when you want to fit and transform in one call,
        but remember this means fit is called on the same data
        you're transforming, so don't use on train data for
        preprocessing separate train/test sets.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to fit and transform
        
        Returns
        -------
        X_transformed : np.ndarray
            Fitted and transformed data
        
        Examples
        --------
        >>> from glassbox.preprocessing import StandardScaler
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4]])
        >>> scaler = StandardScaler()
        >>> X_scaled = scaler.fit_transform(X)
        """
        return self.fit(X).transform(X)
