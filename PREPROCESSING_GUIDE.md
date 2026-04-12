# Preprocessing Module - Complete Documentation

## Overview
The preprocessing module provides data cleaning, transformation, and encoding utilities. It follows a **scikit-learn inspired design** but built from scratch using NumPy.

**Key Philosophy:** All transformers inherit from `BaseTransformer`, follow the `fit() → transform()` pattern, and use validation functions to ensure data integrity.

---

## Architecture: The Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│  User Code (AutoFit Agent)                              │
│  pipeline = Pipeline([                                  │
│    ('imputer', SimpleImputer()),                        │
│    ('scaler', StandardScaler()),                        │
│    ('encoder', OneHotEncoder())                         │
│  ])                                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Transformer Layer (preprocessing/*.py)                 │
│  ├─ base.py (BaseTransformer)                           │
│  ├─ imputer.py (SimpleImputer)                          │
│  ├─ scalers.py (MinMaxScaler, StandardScaler)          │
│  └─ encoders.py (LabelEncoder, OneHotEncoder)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Validation Layer (utils/validation.py)                 │
│  ├─ check_array(X)                                      │
│  ├─ check_is_fitted(obj, attributes)                   │
│  └─ check_consistent_length(X, y)                      │
└─────────────────────────────────────────────────────────┘
```

Every transformer calls validation functions at the start of `fit()` and `transform()`.

---

## Part 1: Validation Functions (`utils/validation.py`)

### Why Validation?
Before transforming data, you must verify:
1. Input is actually a NumPy array (not a list, pandas DataFrame, etc.)
2. Transformer has been `fit()` before calling `transform()`
3. X and y have matching row counts

### Function Specifications

#### `check_array(X)`
**Purpose:** Ensure input is a NumPy array

```python
def check_array(X):
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
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(X).__name__}")
```

**Usage Example:**
```python
# Good
X = np.array([[1, 2], [3, 4]])
check_array(X)  # ✓ passes

# Bad
X = [[1, 2], [3, 4]]
check_array(X)  # ✗ raises TypeError: Expected np.ndarray, got list
```

---

#### `check_is_fitted(obj, attributes)`
**Purpose:** Verify a transformer has been fit before transform is called

**Logic:**
- A fitted transformer stores attributes (like `mean_`, `std_`, `categories_`)
- Before `transform()`, check these attributes exist
- If missing, it means `fit()` wasn't called first

```python
def check_is_fitted(obj, attributes):
    """
    Check that transformer has been fitted.
    
    Parameters
    ----------
    obj : object
        Transformer object to check
    attributes : str or list of str
        Attribute name(s) set during fit() (e.g., "mean_", "std_")
    
    Raises
    ------
    RuntimeError
        If any attribute doesn't exist on obj
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    
    for attr in attributes:
        if not hasattr(obj, attr):
            raise RuntimeError(
                f"Transformer not fitted. Missing attribute: {attr}"
            )
```

**Usage Example:**
```python
scaler = StandardScaler()
# scaler.transform(X)  # ✗ raises RuntimeError: Transformer not fitted

scaler.fit(X)
scaler.transform(X)  # ✓ now works because mean_ and std_ exist
```

**In Code:**
```python
# Inside StandardScaler.transform()
def transform(self, X):
    check_array(X)
    check_is_fitted(self, ['mean_', 'std_'])  # Validates both attributes exist
    # ... do transformation
```

---

#### `check_consistent_length(X, y)`
**Purpose:** Ensure X and y have the same number of rows

**Logic:**
- X is always a 2D array: shape (n_samples, n_features)
- y is a 1D array: shape (n_samples,)
- They must have matching `n_samples`

```python
def check_consistent_length(X, y):
    """
    Check that X and y have the same number of samples.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    
    Raises
    ------
    ValueError
        If n_samples don't match
    """
    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0]
    
    if n_samples_X != n_samples_y:
        raise ValueError(
            f"X and y have incompatible number of samples: "
            f"{n_samples_X} vs {n_samples_y}"
        )
```

**Usage Example:**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
y = np.array([0, 1])  # 2 samples

check_consistent_length(X, y)
# ✗ raises ValueError: X and y have incompatible number of samples: 3 vs 2
```

---

## Part 2: Base Transformer (`preprocessing/base.py`)

### The Contract
Every transformer **must** inherit from `BaseTransformer` and implement two methods:

| Method | Purpose | Returns | Stateful? |
|--------|---------|---------|-----------|
| `fit(X)` | Learn from data (store statistics) | `self` | ✓ Yes, stores fitted_ attributes |
| `transform(X)` | Apply learned transformation | Transformed array | ✗ No, reads fitted_ attributes |
| `fit_transform(X)` | fit() + transform() in one call | Transformed array | ✓ Yes (calls fit then transform) |

### Why This Pattern?
**Scenario:** You have 1000 training samples and 100 test samples
```python
X_train = np.array([...])  # 1000 samples
X_test = np.array([...])   # 100 samples

scaler = StandardScaler()
scaler.fit(X_train)           # Learn mean and std from training data
X_train_scaled = scaler.transform(X_train)   # Scale training data
X_test_scaled = scaler.transform(X_test)    # Scale test data using SAME statistics
```

**Key Point:** fit() learns from training data. transform() applies the learned transformation to ANY data. This prevents data leakage.

### Implementation

```python
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """
    Abstract base class for all preprocessing transformers.
    
    Every transformer must:
    1. Inherit from this class
    2. Implement fit(X) - learn and store statistics
    3. Implement transform(X) - apply learned transformation
    
    The fit_transform() method is provided automatically.
    """
    
    @abstractmethod
    def fit(self, X):
        """
        Learn transformation parameters from X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        
        Returns
        -------
        self : object
            Fitted transformer
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Apply learned transformation to X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform
        
        Returns
        -------
        X_transformed : np.ndarray
            Transformed data
        """
        pass
    
    def fit_transform(self, X):
        """
        Fit to X, then transform it.
        
        This is a convenience method. Equivalent to:
        return self.fit(X).transform(X)
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        
        Returns
        -------
        X_transformed : np.ndarray
            Fitted and transformed data
        """
        return self.fit(X).transform(X)
```

---

## Part 3: SimpleImputer (`preprocessing/imputer.py`)

### Problem: Missing Data
Real-world data has missing values (NaN).

```python
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, np.nan, 6.0],  # ← Missing value in column 1
    [7.0, 8.0, np.nan],  # ← Missing value in column 2
])
```

### Solution: SimpleImputer
Replace NaN with a computed statistic (mean, median, or mode).

### Strategies

#### 1. **"mean"** - Numerical columns
- Compute mean of non-NaN values in each column
- Replace all NaN with that mean
- Use case: standardize missing numbers

#### 2. **"median"** - Numerical columns (robust)
- Compute median of non-NaN values in each column
- Less affected by outliers than mean
- Use case: data with extreme values

#### 3. **"mode"** - Categorical columns
- Find most frequent value in each column
- Replace all NaN with that mode
- Use case: categorical data

### Implementation Outline

```python
class SimpleImputer(BaseTransformer):
    def __init__(self, strategy="mean"):
        """
        Parameters
        ----------
        strategy : {"mean", "median", "mode"}
            How to fill missing values
        """
        self.strategy = strategy
        self.fill_values_ = None  # Computed during fit()
    
    def fit(self, X):
        """
        Compute fill values for each column.
        
        Logic:
        1. For each column, compute the statistic (mean/median/mode)
        2. Ignore NaN values when computing
        3. Store fill values in self.fill_values_
        """
        # Example for "mean" strategy:
        # fill_values = [col[~np.isnan(col)].mean() for col in X.T]
        pass
    
    def transform(self, X):
        """
        Replace NaN with stored fill values.
        """
        # Example:
        # Copy X so we don't modify original
        # For each column: X_copy[np.isnan(X_copy[:, i]), i] = self.fill_values_[i]
        pass
```

### Edge Cases

#### Case 1: Entire Column is NaN
```python
X = np.array([
    [1.0, np.nan],
    [2.0, np.nan],
    [3.0, np.nan],
])
# Column 1 is ALL NaN → mean is NaN → can't impute properly
# ✗ Raise ValueError: "Column 1 is entirely NaN"
```

#### Case 2: NaN After Transform
If transform gets a column with all NaN values:
```python
scaler.fit(X_train)  # All NaN column → fill_value = NaN
X_test_imputed = scaler.transform(X_test)  # Result has NaN still
# This is bad. Prevent this in fit() by raising an error.
```

---

## Part 4: Scalers (`preprocessing/scalers.py`)

### Problem: Feature Scale Mismatch
Different features have different ranges:

```python
X = np.array([
    [1000, 0.001],  # Age in years, heart rate in Hz
    [2000, 0.002],
    [3000, 0.003],
])
# Column 0: range 1000-3000
# Column 1: range 0.001-0.003
# Machine learning models may be biased toward larger values
```

### Solution: Scaling
Transform each column to a standard range.

---

### 1. MinMaxScaler: Scale to [0, 1]

**Formula (per column):**
```
X_scaled = (X - min) / (max - min)
```

**Result:** Each column ranges from 0 to 1

**Implementation:**

```python
class MinMaxScaler(BaseTransformer):
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        """Store min and max per column."""
        # min_ = [col.min() for col in X.T]
        # max_ = [col.max() for col in X.T]
        pass
    
    def transform(self, X):
        """
        Apply scaling: (X - min) / (max - min)
        
        Guard: if max == min (constant column), denominator is 0
        Solution: set denominator to 1 (or return original column)
        """
        pass
```

**Edge Case: Constant Column**
```python
X_train = np.array([
    [5.0, 1.0],
    [5.0, 2.0],
    [5.0, 3.0],
])
# Column 0: all values are 5 → min=5, max=5 → (x - 5) / 0 → undefined!

# Fix: Detect when max == min
# Option 1: Set denominator to 1 → all values become 0
# Option 2: Return the column unchanged
```

---

### 2. StandardScaler: Standardize to mean=0, std=1

**Formula (per column):**
```
X_scaled = (X - mean) / std
```

**Result:** Each column has mean=0 and std=1

**Implementation:**

```python
class StandardScaler(BaseTransformer):
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """Store mean and std per column."""
        # mean_ = [col.mean() for col in X.T]
        # std_ = [col.std() for col in X.T]  # Use utils.column_std()
        pass
    
    def transform(self, X):
        """
        Apply scaling: (X - mean) / std
        
        Guard: if std == 0 (no variance), denominator is 0
        Solution: set denominator to 1
        """
        pass
```

**Edge Case: Zero Std**
```python
X_train = np.array([
    [5.0, 1.0],
    [5.0, 2.0],
    [5.0, 3.0],
])
# Column 0: all values are 5 → std=0 → (x - 5) / 0 → undefined!

# Fix: Detect when std == 0
# Set denominator to 1 → all values become 0
```

---

## Part 5: Encoders (`preprocessing/encoders.py`)

### Problem: Categorical Data
Machine learning models need numbers, but some features are categories:

```python
X = np.array([
    ["red", "small", 100],
    ["blue", "large", 200],
    ["red", "medium", 150],
])
# Column 0: "red", "blue" (strings, not numbers!)
# Column 1: "small", "medium", "large" (strings)
```

### Solution: Encoding
Convert categories to numbers.

---

### 1. LabelEncoder: Map Categories → Integers

**Mapping:**
```
"red" → 0
"blue" → 1
"green" → 2
```

**Use Case:** Binary or ordinal data

**Implementation:**

```python
class LabelEncoder(BaseTransformer):
    def __init__(self):
        self.classes_ = None  # Unique categories
        self.mapping_ = None  # Dict: category → integer
    
    def fit(self, X):
        """
        Learn unique categories and create mapping.
        
        For column ["red", "blue", "red", "green"]:
        1. Find unique values: ["red", "blue", "green"]
        2. Sort for consistency (optional but recommended)
        3. Create mapping: {"red": 0, "blue": 1, "green": 2}
        """
        pass
    
    def transform(self, X):
        """
        Replace each category with its integer.
        
        ["red", "blue", "red"] → [0, 1, 0]
        """
        pass
```

**Edge Case: Unseen Categories**
```python
encoder.fit(["red", "blue"])  # Only saw these two
encoder.transform(["red", "blue", "green"])  # "green" is NEW!

# Option: Raise warning, default to unknown value (e.g., -1)
```

---

### 2. OneHotEncoder: Create Binary Columns

**Transformation:**
```
Original:
[["red"],
 ["blue"],
 ["red"]]

One-Hot Encoded:
[[1, 0],    # red: is_red=1, is_blue=0
 [0, 1],    # blue: is_red=0, is_blue=1
 [1, 0]]    # red: is_red=1, is_blue=0

Column names: ["red", "blue"]
```

**Use Case:** Nominal (unordered) categorical data, input to algorithms

**Implementation:**

```python
class OneHotEncoder(BaseTransformer):
    def __init__(self):
        self.categories_ = None  # List of unique categories per column
        self.n_categories_ = None
    
    def fit(self, X):
        """
        Learn unique categories per column.
        
        For column ["red", "blue", "red", "green"]:
        1. Find unique values (sorted for consistency)
        2. Store: ["blue", "green", "red"]  (sorted alphabetically)
        """
        pass
    
    def transform(self, X):
        """
        Create binary columns for each category.
        
        Input shape: (n_samples, 1)
        Output shape: (n_samples, n_unique_categories)
        
        Example:
        ["red", "blue"] → [[1, 0, 0], [0, 1, 0]]
        """
        pass
```

**Edge Case: Unseen Categories**
```python
encoder.fit(["red", "blue"])  # Only saw these two
encoder.transform(["red", "blue", "green"])  # "green" is NEW!

# For unseen category "green": create all-zero row
# ["red", "blue", "green"] → [1, 0, 0]
# (or could raise warning)
```

---

## Summary: Integration Into AutoFit

```python
# Inside AutoFit.auto_fit():
from glassbox.preprocessing import (
    SimpleImputer,
    StandardScaler,
    OneHotEncoder,
)

X, y = load_data(csv_path)

# Step 1: Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Step 2: Encode categorical features
encoder = OneHotEncoder()
X_categorical = X[:, categorical_columns]
X_categorical_encoded = encoder.fit_transform(X_categorical)
X = np.hstack([X[:, numerical_columns], X_categorical_encoded])

# Step 3: Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# X is now clean, encoded, and scaled - ready for model training!
```

---

## Key Principles to Remember

| Principle | Why |
|-----------|-----|
| **fit() learns, transform() applies** | Prevents data leakage between train/test |
| **Validation at every step** | Catches bugs early with clear error messages |
| **Inherit from BaseTransformer** | Guarantees consistent API across all transformers |
| **Guard against edge cases** | Constant columns, all-NaN columns, unseen categories |
| **Use only NumPy** | Keeps the library dependency-free and understandable |
| **Store statistics with `_` suffix** | `mean_`, `std_`, `categories_` — convention signals fitted attributes |
| **Return self from fit()** | Enables method chaining: `scaler.fit(X).transform(X)` |

