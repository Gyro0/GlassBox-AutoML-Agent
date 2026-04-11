"""SimpleImputer for handling missing values."""

import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted


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
    
    def __init__(self, strategy="mean"):
        if strategy not in ["mean", "median", "mode"]:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Must be 'mean', 'median', or 'mode'"
            )
        self.strategy = strategy
        self.fill_values_ = None
    
    def fit(self, X):
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
        self.fill_values_ = np.zeros(n_features)
        
        for col_idx in range(n_features):
            col = X[:, col_idx]
            non_nan_values = col[~np.isnan(col)]
            
            # Check for all-NaN column
            if len(non_nan_values) == 0:
                raise ValueError(
                    f"Cannot impute column {col_idx}: "
                    f"entire column is NaN"
                )
            
            # Compute fill value based on strategy
            if self.strategy == "mean":
                fill_val = np.mean(non_nan_values)
            elif self.strategy == "median":
                fill_val = np.median(non_nan_values)
            elif self.strategy == "mode":
                # Most frequent value
                unique, counts = np.unique(non_nan_values, return_counts=True)
                fill_val = unique[np.argmax(counts)]
            
            self.fill_values_[col_idx] = fill_val
        
        return self
    
    def transform(self, X):
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
            nan_mask = np.isnan(X_imputed[:, col_idx])
            X_imputed[nan_mask, col_idx] = self.fill_values_[col_idx]
        
        return X_imputed
