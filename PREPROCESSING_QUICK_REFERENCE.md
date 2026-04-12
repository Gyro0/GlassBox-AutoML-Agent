# Preprocessing Implementation - Quick Reference

## File Implementation Checklist

### 1️⃣ `utils/validation.py`
**Status:** Day 1 anchor file  
**Dependencies:** None  
**Complexity:** Minimal

```python
# 3 functions, ~40 lines total

✓ check_array(X)
  - Input: anything
  - Check: isinstance(X, np.ndarray)
  - Raise: TypeError if not
  - Used by: Everything

✓ check_is_fitted(obj, attributes)
  - Input: transformer object, attribute names
  - Check: hasattr(obj, attr) for each attr
  - Raise: RuntimeError if missing
  - Used by: Every transformer's transform()

✓ check_consistent_length(X, y)
  - Input: X (2D array), y (1D array)
  - Check: X.shape[0] == y.shape[0]
  - Raise: ValueError if mismatch
  - Used by: Supervised learning transformers
```

---

### 2️⃣ `preprocessing/base.py`
**Status:** Day 1 merge point  
**Dependencies:** validation.py  
**Complexity:** Minimal

```python
# 1 abstract class, ~30 lines

✓ BaseTransformer (ABC)
  - fit(X) → self                    [abstract]
  - transform(X) → array             [abstract]
  - fit_transform(X) → array         [implemented: return self.fit(X).transform(X)]

Key: Every transformer inherits from this
```

---

### 3️⃣ `preprocessing/imputer.py`
**Status:** Building block  
**Dependencies:** base.py, validation.py  
**Complexity:** Medium

```python
class SimpleImputer(BaseTransformer):
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X):
        check_array(X)
        self.fill_values_ = []
        
        for col in X.T:
            non_nan = col[~np.isnan(col)]
            
            if len(non_nan) == 0:
                raise ValueError(f"Column is entirely NaN")
            
            if self.strategy == "mean":
                fill_value = np.mean(non_nan)
            elif self.strategy == "median":
                fill_value = np.median(non_nan)
            elif self.strategy == "mode":
                # Get most frequent value
                unique, counts = np.unique(non_nan, return_counts=True)
                fill_value = unique[np.argmax(counts)]
            
            self.fill_values_.append(fill_value)
        
        self.fill_values_ = np.array(self.fill_values_)
        return self

    def transform(self, X):
        check_array(X)
        check_is_fitted(self, 'fill_values_')
        
        X_filled = X.copy()
        for i, fill_value in enumerate(self.fill_values_):
            mask = np.isnan(X_filled[:, i])
            X_filled[mask, i] = fill_value
        
        return X_filled
```

---

### 4️⃣ `preprocessing/scalers.py`
**Status:** Building block  
**Dependencies:** base.py, validation.py, utils.matrix  
**Complexity:** Medium

```python
class MinMaxScaler(BaseTransformer):
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        check_array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, ['min_', 'max_'])
        
        # Guard: compute range, where range=0 set to 1
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1  # Avoid division by zero
        
        return (X - self.min_) / range_

class StandardScaler(BaseTransformer):
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        check_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)  # Use utils.column_std()
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, ['mean_', 'std_'])
        
        # Guard: where std=0 set to 1
        std_safe = self.std_.copy()
        std_safe[std_safe == 0] = 1
        
        return (X - self.mean_) / std_safe
```

---

### 5️⃣ `preprocessing/encoders.py`
**Status:** Building block  
**Dependencies:** base.py, validation.py  
**Complexity:** Medium-High

```python
class LabelEncoder(BaseTransformer):
    def __init__(self):
        self.classes_ = None
        self.mapping_ = None
    
    def fit(self, X):
        check_array(X)
        
        # Get unique categories (1D array)
        self.classes_ = np.unique(X.flatten())
        
        # Create mapping dict
        self.mapping_ = {cat: idx for idx, cat in enumerate(self.classes_)}
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, 'mapping_')
        
        X_encoded = np.zeros_like(X, dtype=int)
        for i, val in enumerate(X.flatten()):
            if val in self.mapping_:
                X_encoded.flat[i] = self.mapping_[val]
            else:
                # Warning: unseen category
                X_encoded.flat[i] = -1  # or raise warning
        
        return X_encoded.reshape(X.shape)

class OneHotEncoder(BaseTransformer):
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X):
        check_array(X)
        
        # For each column, get sorted unique values
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        check_array(X)
        check_is_fitted(self, 'categories_')
        
        n_samples, n_cols = X.shape
        total_categories = sum(len(cats) for cats in self.categories_)
        
        # Initialize output: all zeros
        X_encoded = np.zeros((n_samples, total_categories))
        
        # Fill in the 1s
        col_idx = 0
        for col in range(n_cols):
            for row in range(n_samples):
                val = X[row, col]
                # Find position of this value in categories list
                if val in self.categories_[col]:
                    pos = np.where(self.categories_[col] == val)[0][0]
                    X_encoded[row, col_idx + pos] = 1
                # else: leave as 0 (unseen category)
            col_idx += len(self.categories_[col])
        
        return X_encoded
```

---

## Testing Strategy

Each module should have tests in `tests/test_preprocessing.py`:

```python
# SimpleImputer tests
✓ test_imputer_mean_fills_nan()
✓ test_imputer_median_fills_nan()
✓ test_imputer_mode_fills_nan()
✓ test_imputer_raises_on_all_nan_column()
✓ test_imputer_consistent_between_fit_transform()

# MinMaxScaler tests
✓ test_minmax_scales_to_0_1()
✓ test_minmax_handles_constant_column()

# StandardScaler tests
✓ test_standard_scales_mean_0_std_1()
✓ test_standard_handles_zero_std()

# LabelEncoder tests
✓ test_label_encoder_maps_unique_categories()
✓ test_label_encoder_handles_unseen()

# OneHotEncoder tests
✓ test_onehot_creates_binary_columns()
✓ test_onehot_handles_unseen_categories()
```

---

## Critical Gotchas

| Issue | Fix |
|-------|-----|
| fit() doesn't return self | Add `return self` at end |
| transform() modifies input X | Use `X.copy()` first |
| Constant columns crash divisor | Check `if val == 0: set to 1` |
| fit() on train, transform() on test with unseen value | Handle gracefully (zero/warning) |
| Attributes not marked with `_` suffix | Always use `mean_`, `std_`, `classes_` |
| Forgetting to call validation functions | Every fit() and transform() needs check_array() |
| Using scikit-learn code | NumPy only — implement yourself |

---

## Build Order

**Day 1 (Foundation):**
1. ✅ `utils/validation.py` (3 functions)
2. ✅ `preprocessing/base.py` (1 class)
3. Merge to `dev` branch → Yasser & Abderrahim can now inherit

**Days 2-3 (Implementation):**
4. `preprocessing/imputer.py` (SimpleImputer)
5. `preprocessing/scalers.py` (MinMaxScaler, StandardScaler)
6. `preprocessing/encoders.py` (LabelEncoder, OneHotEncoder)
7. `tests/test_preprocessing.py` (All tests)
8. Merge to `dev` when all tests pass

---

## Integration with AutoFit

```python
# In glassbox/agent/autofit.py pseudo-code:

def auto_fit(X_raw, target_col, task="auto", ...):
    X, y = load_and_split(X_raw, target_col)
    
    # --- PREPROCESSING PIPELINE ---
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit(X).transform(X)  # fit on train data
    
    # Identify categorical columns
    cat_cols = [i for i in range(X.shape[1]) if is_categorical(X[:, i])]
    num_cols = [i for i in range(X.shape[1]) if not is_categorical(X[:, i])]
    
    # Encode categorical
    encoder = OneHotEncoder()
    X_cat_encoded = encoder.fit(X[:, cat_cols]).transform(X[:, cat_cols])
    
    # Scale numerical
    scaler = StandardScaler()
    X_num_scaled = scaler.fit(X[:, num_cols]).transform(X[:, num_cols])
    
    # Combine
    X_processed = np.hstack([X_num_scaled, X_cat_encoded])
    
    # --- MODEL TRAINING ---
    # (Yasser's models receive clean, scaled data)
    # ...
```

