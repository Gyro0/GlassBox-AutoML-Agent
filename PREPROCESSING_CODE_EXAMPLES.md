# Preprocessing Implementation - Code Examples & Test Cases

## Part 1: Validation Functions Examples

### check_array() - Usage Examples

```python
# ✓ CORRECT usage
import numpy as np
from glassbox.utils.validation import check_array

X = np.array([[1, 2], [3, 4]])
check_array(X)  # Passes silently

# ✓ Works with any shape
X_1d = np.array([1, 2, 3])
check_array(X_1d)  # Passes

# ✗ INCORRECT usage
check_array([1, 2, 3])
# Raises TypeError: Expected np.ndarray, got list

check_array([[1, 2], [3, 4]])
# Raises TypeError: Expected np.ndarray, got list

check_array({"a": 1})
# Raises TypeError: Expected np.ndarray, got dict
```

### check_is_fitted() - Usage Examples

```python
from glassbox.preprocessing import StandardScaler
from glassbox.utils.validation import check_is_fitted

scaler = StandardScaler()

# ✗ Before fit()
check_is_fitted(scaler, 'mean_')
# Raises RuntimeError: Transformer not fitted. Missing attribute: mean_

# ✓ After fit()
scaler.fit(np.array([[1, 2], [3, 4]]))
check_is_fitted(scaler, 'mean_')  # Passes silently

# ✓ Check multiple attributes
check_is_fitted(scaler, ['mean_', 'std_'])  # Passes silently

# Can also use string instead of list
check_is_fitted(scaler, 'mean_')  # Works

# ✗ If one of many is missing
check_is_fitted(scaler, ['mean_', 'missing_attr_'])
# Raises RuntimeError: Transformer not fitted. Missing attribute: missing_attr_
```

### check_consistent_length() - Usage Examples

```python
from glassbox.utils.validation import check_consistent_length

X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
y = np.array([0, 1, 0])                  # 3 samples

# ✓ Matching lengths
check_consistent_length(X, y)  # Passes silently

# ✗ Mismatched lengths
y_wrong = np.array([0, 1])  # Only 2 samples
check_consistent_length(X, y_wrong)
# Raises ValueError: X and y have incompatible number of samples: 3 vs 2

# Works with different feature counts
X = np.array([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
y = np.array([0, 1])                   # 2 samples
check_consistent_length(X, y)  # ✓ Passes (only cares about row count)
```

---

## Part 2: SimpleImputer Implementation & Tests

### Implementation Reference

```python
import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array

class SimpleImputer(BaseTransformer):
    """Replace missing values (NaN) with computed statistics."""
    
    def __init__(self, strategy="mean"):
        """
        Parameters
        ----------
        strategy : str in {"mean", "median", "mode"}
            How to compute fill values
        """
        if strategy not in ["mean", "median", "mode"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.fill_values_ = None
    
    def fit(self, X):
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
        check_array(X)
        check_is_fitted(self, 'fill_values_')
        
        X_imputed = X.copy()  # Don't modify original
        
        for col_idx in range(X.shape[1]):
            nan_mask = np.isnan(X_imputed[:, col_idx])
            X_imputed[nan_mask, col_idx] = self.fill_values_[col_idx]
        
        return X_imputed
```

### Test Cases for SimpleImputer

```python
import pytest
import numpy as np
from glassbox.preprocessing import SimpleImputer

def test_imputer_mean_strategy():
    """Test mean imputation strategy"""
    X_train = np.array([
        [1.0, 5.0],
        [2.0, np.nan],
        [3.0, 7.0],
    ])
    
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    
    # mean of col 1: (5 + 7) / 2 = 6
    assert imputer.fill_values_[1] == 6.0
    
    # Transform should replace np.nan with 6.0
    X_transformed = imputer.transform(X_train)
    expected = np.array([
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 7.0],
    ])
    np.testing.assert_array_equal(X_transformed, expected)

def test_imputer_median_strategy():
    """Test median imputation strategy"""
    X_train = np.array([
        [1.0, 10.0],
        [2.0, np.nan],
        [3.0, 30.0],
    ])
    
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    
    # median of col 1: (10, 30) → 20
    assert imputer.fill_values_[1] == 20.0

def test_imputer_mode_strategy():
    """Test mode (most frequent) imputation"""
    X_train = np.array([
        [1.0, "cat"],
        [2.0, "dog"],
        [3.0, "cat"],
        [4.0, np.nan],
    ])
    
    imputer = SimpleImputer(strategy="mode")
    imputer.fit(X_train)
    
    # mode of col 1: "cat" appears 2x, "dog" 1x
    assert imputer.fill_values_[1] == "cat"

def test_imputer_all_nan_column_raises():
    """Test that all-NaN column raises ValueError"""
    X_train = np.array([
        [1.0, np.nan],
        [2.0, np.nan],
        [3.0, np.nan],
    ])
    
    imputer = SimpleImputer(strategy="mean")
    
    with pytest.raises(ValueError, match="entire column is NaN"):
        imputer.fit(X_train)

def test_imputer_doesnt_modify_input():
    """Test that transform doesn't modify original X"""
    X_train = np.array([
        [1.0, np.nan],
        [2.0, 5.0],
    ])
    X_train_original = X_train.copy()
    
    imputer = SimpleImputer()
    imputer.fit(X_train)
    _ = imputer.transform(X_train)
    
    # Original should be unchanged
    np.testing.assert_array_equal(X_train, X_train_original)

def test_imputer_fit_transform():
    """Test fit_transform convenience method"""
    X = np.array([
        [1.0, np.nan],
        [2.0, 5.0],
    ])
    
    imputer = SimpleImputer()
    result = imputer.fit_transform(X)
    
    # Should be equivalent to fit then transform
    imputer2 = SimpleImputer()
    expected = imputer2.fit(X).transform(X)
    
    np.testing.assert_array_equal(result, expected)

def test_imputer_transform_without_fit_raises():
    """Test that transform without fit raises error"""
    X = np.array([[1.0, np.nan]])
    imputer = SimpleImputer()
    
    with pytest.raises(RuntimeError, match="not fitted"):
        imputer.transform(X)
```

---

## Part 3: StandardScaler Implementation & Tests

### Implementation Reference

```python
import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted

class StandardScaler(BaseTransformer):
    """Standardize features to mean=0, std=1."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        check_array(X)
        
        # Compute per-column statistics
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)  # Population std (ddof=0)
        
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, ['mean_', 'std_'])
        
        # Guard against zero std (constant columns)
        std_safe = self.std_.copy()
        std_safe[std_safe == 0] = 1  # Replace 0 with 1 to avoid division by zero
        
        return (X - self.mean_) / std_safe
```

### Test Cases for StandardScaler

```python
import numpy as np
import pytest
from glassbox.preprocessing import StandardScaler

def test_standard_scaler_mean_std():
    """Test that scaled data has mean≈0, std≈1"""
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ], dtype=float)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Check mean is close to 0
    means = np.mean(X_scaled, axis=0)
    np.testing.assert_array_almost_equal(means, [0, 0])
    
    # Check std is close to 1
    stds = np.std(X_scaled, axis=0)
    np.testing.assert_array_almost_equal(stds, [1, 1])

def test_standard_scaler_constant_column():
    """Test handling of constant column (std=0)"""
    X = np.array([
        [5.0, 10.0],
        [5.0, 20.0],
        [5.0, 30.0],
    ], dtype=float)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Column 0 has std=0
    assert scaler.std_[0] == 0
    
    # Transform should not crash
    X_scaled = scaler.transform(X)
    
    # Constant column should become all 0s (since (5-5)/1 = 0)
    np.testing.assert_array_almost_equal(X_scaled[:, 0], [0, 0, 0])

def test_standard_scaler_fit_transform_consistency():
    """Test that fit_transform equals fit().transform()"""
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    
    scaler1 = StandardScaler()
    result1 = scaler1.fit_transform(X)
    
    scaler2 = StandardScaler()
    result2 = scaler2.fit(X).transform(X)
    
    np.testing.assert_array_almost_equal(result1, result2)

def test_standard_scaler_transform_after_fit():
    """Test that transform uses fit statistics, not recomputd"""
    X_train = np.array([[1, 10], [2, 20]], dtype=float)
    X_test = np.array([[3, 30]], dtype=float)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save fit statistics
    mean_saved = scaler.mean_.copy()
    std_saved = scaler.std_.copy()
    
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    
    # Verify it used saved statistics, not statistics of X_test
    # If it recomputed, X_test_scaled would be all zeros (since it's 1 sample)
    assert not np.all(X_test_scaled == 0)
```

---

## Part 4: OneHotEncoder Implementation & Tests

### Implementation Reference

```python
import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.utils.validation import check_array, check_is_fitted

class OneHotEncoder(BaseTransformer):
    """Convert categorical columns to binary one-hot vectors."""
    
    def __init__(self):
        self.categories_ = None  # List of unique categories per column
    
    def fit(self, X):
        check_array(X)
        
        n_features = X.shape[1]
        self.categories_ = []
        
        for col_idx in range(n_features):
            col = X[:, col_idx]
            # Get unique values, sorted for consistency
            unique_vals = np.unique(col)
            self.categories_.append(unique_vals)
        
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, 'categories_')
        
        n_samples = X.shape[0]
        
        # Calculate output shape
        n_output_features = sum(len(cats) for cats in self.categories_)
        X_encoded = np.zeros((n_samples, n_output_features), dtype=int)
        
        # Fill in the 1s
        col_offset = 0
        for col_idx, categories in enumerate(self.categories_):
            for row_idx in range(n_samples):
                val = X[row_idx, col_idx]
                
                # Find position of this value in categories
                if val in categories:
                    pos = np.where(categories == val)[0][0]
                    X_encoded[row_idx, col_offset + pos] = 1
                # else: leave as 0 (encoding unseen category as all zeros)
            
            col_offset += len(categories)
        
        return X_encoded
```

### Test Cases for OneHotEncoder

```python
import numpy as np
import pytest
from glassbox.preprocessing import OneHotEncoder

def test_onehot_basic():
    """Test basic one-hot encoding"""
    X = np.array([
        ["red"],
        ["blue"],
        ["red"],
    ])
    
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    
    # Should have 2 columns (blue, red - sorted)
    assert X_encoded.shape == (3, 2)
    
    # Expected encoding (sorted order: "blue" then "red")
    expected = np.array([
        [0, 1],  # red
        [1, 0],  # blue
        [0, 1],  # red
    ])
    np.testing.assert_array_equal(X_encoded, expected)

def test_onehot_multicolumn():
    """Test one-hot with multiple columns"""
    X = np.array([
        ["red", "small"],
        ["blue", "large"],
    ])
    
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    
    # Column 0: ["blue", "red"] (2 categories)
    # Column 1: ["large", "small"] (2 categories)
    # Total: 4 output features
    assert X_encoded.shape == (2, 4)
    
    # Row 0: red=0 (blue), small=1 (small comes after large when sorted)
    # Sorting: large < small, so small has index 1
    # Actually: ["large", "small"] sorted → ["large", "small"]
    # Row 0: red (index 1 in ["blue", "red"]) + small (index 1 in ["large", "small"])
    # → [0, 1, 0, 1]

def test_onehot_unseen_category():
    """Test handling of unseen category during transform"""
    X_train = np.array([
        ["red"],
        ["blue"],
    ])
    
    encoder = OneHotEncoder()
    encoder.fit(X_train)
    
    # Transform data with unseen category
    X_test = np.array([
        ["red"],
        ["blue"],
        ["green"],  # ← Never seen during fit
    ])
    
    X_encoded = encoder.transform(X_test)
    
    # Unseen category should be encoded as all zeros
    assert X_encoded.shape == (3, 2)
    assert X_encoded[2, 0] == 0 and X_encoded[2, 1] == 0

def test_onehot_fit_transform():
    """Test fit_transform method"""
    X = np.array([["a"], ["b"], ["a"]])
    
    encoder = OneHotEncoder()
    result = encoder.fit_transform(X)
    
    # Should match fit().transform()
    encoder2 = OneHotEncoder()
    expected = encoder2.fit(X).transform(X)
    
    np.testing.assert_array_equal(result, expected)
```

---

## Part 5: Complete Pipeline Example

### Real-World Usage

```python
import numpy as np
from glassbox.preprocessing import (
    SimpleImputer,
    StandardScaler,
    OneHotEncoder,
)

# Simulate real dataset
X_raw = np.array([
    [25, "USA", 50000, np.nan],
    [30, "UK", np.nan, 100],
    [35, "USA", 60000, 150],
    [np.nan, "Canada", 55000, np.nan],
])

# Create pipeline
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
encoder = OneHotEncoder()

# Step 1: Impute missing values
X_imputed = imputer.fit_transform(X_raw)

# Step 2: Separate numerical and categorical
numerical_cols = [0, 2, 3]  # Age, salary, bonus
categorical_cols = [1]       # Country

X_numerical = X_imputed[:, numerical_cols]
X_categorical = X_imputed[:, categorical_cols]

# Step 3: Encode categorical
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Step 4: Scale numerical
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Step 5: Combine
X_processed = np.hstack([X_numerical_scaled, X_categorical_encoded])

print(f"Original shape: {X_raw.shape}")
print(f"Processed shape: {X_processed.shape}")
# Original shape: (4, 4)
# Processed shape: (4, 5)  # 3 numerical + 2 categorical (Canada, UK, USA)
```

---

## Summary of Key Test Patterns

```python
# Pattern 1: Test fit() stores correct values
def test_fits_stores_X():
    X = np.array([...])
    transformer.fit(X)
    # Assert transformer.attr_ == expected_value

# Pattern 2: Test transform() uses stored values
def test_transform_uses_fit_values():
    X_train = np.array([...])
    X_test = np.array([...])
    transformer.fit(X_train)
    result = transformer.transform(X_test)
    # Assert result contains transformed X_test

# Pattern 3: Test error handling
def test_raises_on_invalid_input():
    with pytest.raises(ValueError):
        transformer.fit(invalid_data)

# Pattern 4: Test fit_transform consistency
def test_fit_transform_equals_fit_then_transform():
    result1 = transformer.fit_transform(X)
    result2 = transformer.fit(X).transform(X)
    np.testing.assert_array_equal(result1, result2)

# Pattern 5: Test edge cases
def test_handles_edge_case():
    X = edge_case_data()
    result = transformer.fit_transform(X)
    # Assert result is reasonable
```

