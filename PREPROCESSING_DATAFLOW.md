# Preprocessing Pipeline - Data Flow Diagram

## End-to-End Data Transformation

```
┌─────────────────────────────────────────────────────────────────────┐
│ RAW DATA                                                            │
│ X = [[1.0,   "red",   np.nan,   100],                              │
│      [2.0,   "blue",  5.0,      200],                              │
│      [3.0,   "red",   6.0,      np.nan],                           │
│      [4.0,   np.nan,  7.0,      300]]                              │
│                                                                     │
│ Columns:  [0: numeric] [1: category] [2: numeric+NaN] [3: NaN]   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ STEP 1: SimpleImputer      │
                    │ Strategy: mean for numeric │
                    │          mode for category │
                    └──────────────┬──────────────┘
                                   │
│ X_imputed = [[1.0,   "red",   6.0,      200],
│              [2.0,   "blue",  5.0,      200],
│              [3.0,   "red",   6.0,      200],
│              [4.0,   "red",   7.0,      300]]
│
│ Processing:
│ - Column 0: no NaN → unchanged
│ - Column 1: NaN → replaced with mode "red"
│ - Column 2: NaN → replaced with mean 6.0
│ - Column 3: NaN → replaced with mean 200
└─────────────────────────────────────────────────────────────────────│
                                   │
                    ┌──────────────▼──────────────┐
                    │ STEP 2: OneHotEncoder      │
                    │ Encode categories → binary │
                    └──────────────┬──────────────┘
                                   │
│ X_categorical = X_imputed[:, [1]]  (only categorical column)
│ OneHotEncoder.fit() finds: classes_ = ["blue", "red"]  (sorted)
│
│ X_encoded = [[1, 0],    # "red" → [is_blue=0, is_red=1]
│              [0, 1],    # "blue" → [is_blue=1, is_red=0]
│              [1, 0],    # "red" → [is_blue=0, is_red=1]
│              [1, 0]]    # "red" → [is_blue=0, is_red=1]
│
│ Now we have numeric features only!
└─────────────────────────────────────────────────────────────────────│
                                   │
                    ┌──────────────▼──────────────┐
                    │ STEP 3: StandardScaler     │
                    │ (X - mean) / std           │
                    └──────────────┬──────────────┘
                                   │
│ X_combined = [[1.0, 6.0, 200, 1, 0],
│               [2.0, 5.0, 200, 0, 1],
│               [3.0, 6.0, 200, 1, 0],
│               [4.0, 7.0, 300, 1, 0]]
│
│ StandardScaler.fit() computes:
│ - mean_  = [2.5,  6.0,  225,  0.75, 0.25]
│ - std_   = [1.29, 0.82, 50,   0.43, 0.43]
│
│ X_scaled = [[-1.16, 0.00, -0.5,  0.58, -0.58],
│             [-0.39, -1.22, -0.5, -0.58,  1.74],
│             [ 0.39, 0.00, -0.5,  0.58, -0.58],
│             [ 1.16, 1.22,  1.5,  0.58, -0.58]]
│
│ Result: All columns have mean≈0, std≈1
└─────────────────────────────────────────────────────────────────────│
                                   │
                    ┌──────────────▼──────────────┐
                    │ READY FOR MODEL            │
                    │ (clean, normalized, typed) │
                    └──────────────────────────────┘
```

---

## fit() vs transform() - The Critical Distinction

### Training Pipeline (fit=True)

```
┌───────────────────────────────────────────────────┐
│ TRAINING DATA (1000 samples)                      │
│ X_train = np.array([...])                         │
└──────────────┬───────────────────────────────────┘
               │
               │ IMPUTER.fit(X_train)
               │ ├─ Compute: mean_col_2 = 5.3
               │ └─ Store: self.fill_values_ = [5.3, ...]
               │
               │ IMPUTER.transform(X_train)
               │ └─ Replace NaN in col 2 with 5.3
               │
│ X_train_clean = [[...], [...], ...]              │
└──────────────┬───────────────────────────────────┘
               │
               │ SCALER.fit(X_train_clean)
               │ ├─ Compute: mean = [2.5, 6.0, 225, ...]
               │ └─ Store: self.mean_ = [...]
               │
               │ SCALER.transform(X_train_clean)
               │ └─ Apply: (X - mean) / std
               │
│ X_train_scaled = [[−1.16, 0.00, ...], ...]       │
└──────────────┬───────────────────────────────────┘
               │
               │ TRAIN MODEL on X_train_scaled
               │ Model learns: "feature_1 strongly predicts target"
               │
└─────────────────────────────────────────────────┘
```

### Test Pipeline (fit=False, only transform)

```
┌─────────────────────────────────────────────────┐
│ TEST DATA (100 samples)                         │
│ X_test = np.array([...])                        │
└──────────────┬────────────────────────────────┘
               │
               │ IMPUTER.transform(X_test)  ← NO fit!
               │ └─ Use stored fill_values_
               │    (same mean from training!)
               │
│ X_test_clean = [[...], [...], ...]             │
└──────────────┬────────────────────────────────┘
               │
               │ SCALER.transform(X_test_clean)  ← NO fit!
               │ └─ Use stored mean_ and std_
               │    (from training data!)
               │
│ X_test_scaled = [[−1.16, 0.00, ...], ...]      │
└──────────────┬────────────────────────────────┘
               │
               │ PREDICT using same model
               │ ✓ Predictions are consistent!
               │
└─────────────────────────────────────────────────┘
```

### Why NOT fit() on test data?

```
❌ WRONG:
imputer.fit(X_test)      # Compute mean from test data!
X_test_scaled = imputer.transform(X_test)

Problem: Mean of test set ≠ mean of train set
         Model was trained on different scale
         Predictions will be WRONG

✓ CORRECT:
imputer.fit(X_train)
X_train_transformed = imputer.transform(X_train)

imputer.transform(X_test)  # Use SAME mean as training
# X_test is transformed using training statistics
```

---

## Memory Layout: What's Stored vs What's Computed

```
┌─────────────────────────────────────────────────────┐
│ SimpleImputer STATE (after fit)                    │
├─────────────────────────────────────────────────────┤
│ self.strategy = "mean"         (initialized)        │
│ self.fill_values_ = [5.3, 200] (COMPUTED, stored)   │
│                                                     │
│ Later at transform():                               │
│ ├─ Read: self.fill_values_                         │
│ └─ Apply: Replace NaN with these values             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ StandardScaler STATE (after fit)                    │
├─────────────────────────────────────────────────────┤
│ self.mean_ = [2.5, 6.0, 225, ...]  (COMPUTED)       │
│ self.std_  = [1.29, 0.82, 50, ...]  (COMPUTED)      │
│                                                     │
│ Later at transform():                               │
│ ├─ Read: self.mean_, self.std_                     │
│ └─ Apply: (X - mean) / std                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ OneHotEncoder STATE (after fit)                     │
├─────────────────────────────────────────────────────┤
│ self.categories_ = [                                │
│    ["blue", "red"],    # Column 1 categories       │
│ ]  (COMPUTED, sorted)                              │
│                                                     │
│ Later at transform():                               │
│ ├─ Read: self.categories_                          │
│ └─ Apply: Create binary columns, 1 if value        │
│           matches this category, 0 otherwise        │
└─────────────────────────────────────────────────────┘
```

---

## Error Handling: What Can Go Wrong?

```
┌─────────────────────────────────────────────────────┐
│ ERROR SCENARIO 1: Forgot to fit()                   │
├─────────────────────────────────────────────────────┤
│
│ scaler = StandardScaler()
│ scaler.transform(X)  # ✗ No fit first!
│
│ check_is_fitted(self, 'mean_')
│ ├─ hasattr(self, 'mean_') → False
│ └─ Raise RuntimeError: "Transformer not fitted"
│
│ Fix: scaler.fit(X_train).transform(X)
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ERROR SCENARIO 2: Entire column is NaN              │
├─────────────────────────────────────────────────────┤
│
│ X = np.array([
│   [1.0, np.nan],
│   [2.0, np.nan],
│   [3.0, np.nan],
│ ])
│
│ imputer = SimpleImputer(strategy="mean")
│ imputer.fit(X)
│
│ Column 1: non_nan = [] (empty!)
│ len(non_nan) == 0 → True
│ Raise ValueError: "Column 1 is entirely NaN"
│
│ Fix: Drop or remove that column before imputing
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ERROR SCENARIO 3: Constant column (zero division)   │
├─────────────────────────────────────────────────────┤
│
│ X = np.array([
│   [5.0, 100],
│   [5.0, 200],
│   [5.0, 300],
│ ])
│
│ scaler = StandardScaler()
│ scaler.fit(X)
│ ├─ mean_[0] = 5.0
│ └─ std_[0] = 0.0  (no variance!)
│
│ scaler.transform(X)
│ ├─ (X[0] - 5.0) / 0.0 → undefined!
│
│ Fix: Detect std_ == 0 in transform()
│ └─ std_safe = std_.copy()
│    std_safe[std_safe == 0] = 1  (avoid division)
│    result = (X - mean_) / std_safe
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ERROR SCENARIO 4: Unseen category in test           │
├─────────────────────────────────────────────────────┤
│
│ encoder = LabelEncoder()
│ encoder.fit(["red", "blue"])
│ ├─ mapping_ = {"red": 0, "blue": 1}
│
│ encoder.transform(["red", "blue", "green"])
│ ├─ "red" → 0 ✓
│ ├─ "blue" → 1 ✓
│ └─ "green" → not in mapping!
│
│ Options:
│ 1. Raise warning: "Unknown category: green"
│ 2. Map to -1 or 0
│ 3. Raise error
│
│ For OneHotEncoder: create all-zero row (safe)
└─────────────────────────────────────────────────────┘
```

---

## Validation Function Call Sequence

```
USER CODE:
│
├─ scaler = StandardScaler()
│
├─ scaler.fit(X_train)
│  │
│  └─ fit() {
│      check_array(X_train)  ← Validate input
│      ├─ if not isinstance(X_train, np.ndarray):
│      │  └─ raise TypeError
│      │
│      self.mean_ = X_train.mean()
│      self.std_ = X_train.std()
│      return self
│    }
│
├─ scaler.transform(X_test)
│  │
│  └─ transform() {
│      check_array(X_test)  ← Validate input
│      check_is_fitted(self, ['mean_', 'std_'])  ← Validate state
│      ├─ hasattr(self, 'mean_') → True ✓
│      ├─ hasattr(self, 'std_') → True ✓
│      │
│      return (X_test - self.mean_) / self.std_
│    }
│
└─ Done!
```

---

## Integration Checkpoint: What Yasser Receives

After preprocessing, Yasser's models in `glassbox/models/` receive:

```
Input to models:
├─ X shape: (n_samples, n_features)
│  ├─ All values are numbers (no categories)
│  ├─ No NaN values
│  ├─ All numeric columns are normalized (mean≈0, std≈1)
│  └─ OneHot columns are binary (0 or 1)
│
├─ y shape: (n_samples,)
│  ├─ Classification: integers (0, 1, 2, ...)
│  └─ Regression: floats
│
└─ Models can focus purely on learning,
   not data cleaning!
```

