"""Tests for preprocessing module."""

import csv
from pathlib import Path

import pytest
import numpy as np
from glassbox.preprocessing.base import BaseTransformer
from glassbox.preprocessing.imputer import SimpleImputer
from glassbox.preprocessing.scalers import MinMaxScaler, StandardScaler
from glassbox.preprocessing.encoders import LabelEncoder, OneHotEncoder
from glassbox.utils.validation import check_array, check_is_fitted, check_consistent_length


# ============================================================================
# Tests for Validation Functions
# ============================================================================

class TestCheckArray:
    """Tests for check_array validation function."""
    
    def test_check_array_accepts_numpy_array(self):
        """Valid numpy arrays should pass silently."""
        X = np.array([[1, 2], [3, 4]])
        check_array(X)  # Should not raise
    
    def test_check_array_rejects_list(self):
        """Lists should raise TypeError."""
        with pytest.raises(TypeError, match="Expected np.ndarray, got list"):
            check_array([[1, 2], [3, 4]])
    
    def test_check_array_rejects_dict(self):
        """Dicts should raise TypeError."""
        with pytest.raises(TypeError, match="Expected np.ndarray, got dict"):
            check_array({"a": 1})
    
    def test_check_array_rejects_tuple(self):
        """Tuples should raise TypeError."""
        with pytest.raises(TypeError, match="Expected np.ndarray, got tuple"):
            check_array((1, 2, 3))


class TestCheckIsFitted:
    """Tests for check_is_fitted validation function."""
    
    def test_check_is_fitted_raises_for_missing_attribute(self):
        """Should raise RuntimeError if attribute is missing."""
        class DummyTransformer:
            pass
        
        obj = DummyTransformer()
        with pytest.raises(RuntimeError, match="missing attribute: mean_"):
            check_is_fitted(obj, 'mean_')
    
    def test_check_is_fitted_with_multiple_attributes(self):
        """Should check all attributes in list."""
        class DummyTransformer:
            mean_ = np.array([1, 2])
            std_ = np.array([0.5, 0.5])
        
        obj = DummyTransformer()
        check_is_fitted(obj, ['mean_', 'std_'])  # Should pass
        
        with pytest.raises(RuntimeError, match="missing attribute"):
            check_is_fitted(obj, ['mean_', 'missing_'])


class TestCheckConsistentLength:
    """Tests for check_consistent_length validation function."""
    
    def test_check_consistent_length_passes_for_matching(self):
        """Should pass when X and y have same n_samples."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        check_consistent_length(X, y)  # Should not raise
    
    def test_check_consistent_length_raises_for_mismatch(self):
        """Should raise ValueError when lengths don't match."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])
        
        with pytest.raises(ValueError, match="incompatible number of samples: 3 vs 2"):
            check_consistent_length(X, y)


# ============================================================================
# Tests for SimpleImputer
# ============================================================================

class TestSimpleImputer:
    """Tests for SimpleImputer transformer."""
    
    def test_imputer_mean_strategy(self):
        """Test mean imputation strategy."""
        X_train = np.array([
            [1.0, 5.0],
            [2.0, np.nan],
            [3.0, 7.0],
        ])
        
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X_train)
        
        # mean of col 1: (5 + 7) / 2 = 6
        assert imputer.fill_values_[1] == 6.0
        
        X_transformed = imputer.transform(X_train)
        expected = np.array([
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
        ])
        np.testing.assert_array_equal(X_transformed, expected)
    
    def test_imputer_median_strategy(self):
        """Test median imputation strategy."""
        X_train = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
        ])
        
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)
        
        # median of col 1: (10, 30) → 20
        assert imputer.fill_values_[1] == 20.0
    
    def test_imputer_all_nan_column_raises(self):
        """Test that all-NaN column raises ValueError."""
        X_train = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
        ])
        
        imputer = SimpleImputer(strategy="mean")
        
        with pytest.raises(ValueError, match="entire column is NaN"):
            imputer.fit(X_train)
    
    def test_imputer_doesnt_modify_input(self):
        """Test that transform doesn't modify original X."""
        X_train = np.array([
            [1.0, np.nan],
            [2.0, 5.0],
        ])
        X_train_original = X_train.copy()
        
        imputer = SimpleImputer()
        imputer.fit(X_train)
        _ = imputer.transform(X_train)
        
        np.testing.assert_array_equal(X_train, X_train_original)
    
    def test_imputer_fit_transform(self):
        """Test fit_transform convenience method."""
        X = np.array([
            [1.0, np.nan],
            [2.0, 5.0],
        ])
        
        imputer = SimpleImputer()
        result = imputer.fit_transform(X)
        
        imputer2 = SimpleImputer()
        expected = imputer2.fit(X).transform(X)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_imputer_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        X = np.array([[1.0, np.nan]])
        imputer = SimpleImputer()
        
        with pytest.raises(RuntimeError, match="missing attribute"):
            imputer.transform(X)


# ============================================================================
# Tests for MinMaxScaler
# ============================================================================

class TestMinMaxScaler:
    """Tests for MinMaxScaler transformer."""
    
    def test_minmax_scales_to_0_1(self):
        """Test that scaled data is in [0, 1]."""
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ], dtype=float)
        
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # First sample: (1-1)/(3-1)=0, (10-10)/(30-10)=0
        # Last sample: (3-1)/(3-1)=1, (30-10)/(30-10)=1
        expected = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_minmax_handles_constant_column(self):
        """Test handling of constant column (min == max)."""
        X = np.array([
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
        ], dtype=float)
        
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # Column 0: min=5, max=5, range=0 → zeros
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0, 0, 0])


# ============================================================================
# Tests for StandardScaler
# ============================================================================

class TestStandardScaler:
    """Tests for StandardScaler transformer."""
    
    def test_standard_scaler_mean_std(self):
        """Test that scaled data has mean≈0, std≈1."""
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
    
    def test_standard_scaler_constant_column(self):
        """Test handling of constant column (std=0)."""
        X = np.array([
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
        ], dtype=float)
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        assert scaler.std_[0] == 0
        
        X_scaled = scaler.transform(X)
        
        # Constant column should become all 0s
        np.testing.assert_array_almost_equal(X_scaled[:, 0], [0, 0, 0])
    
    def test_standard_scaler_fit_transform_consistency(self):
        """Test that fit_transform equals fit().transform()."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        scaler1 = StandardScaler()
        result1 = scaler1.fit_transform(X)
        
        scaler2 = StandardScaler()
        result2 = scaler2.fit(X).transform(X)
        
        np.testing.assert_array_almost_equal(result1, result2)


# ============================================================================
# Tests for LabelEncoder
# ============================================================================

class TestLabelEncoder:
    """Tests for LabelEncoder transformer."""
    
    def test_label_encoder_basic(self):
        """Test basic label encoding."""
        X = np.array([["red"], ["blue"], ["red"]])
        
        encoder = LabelEncoder()
        encoder.fit(X)
        X_encoded = encoder.transform(X)
        
        # Should map to integers 0, 1 (sorted: "blue"=0, "red"=1)
        expected = np.array([[1], [0], [1]])
        np.testing.assert_array_equal(X_encoded, expected)
    
    def test_label_encoder_classes_sorted(self):
        """Test that classes are sorted for consistency."""
        X = np.array([["zebra"], ["apple"], ["zebra"]])
        
        encoder = LabelEncoder()
        encoder.fit(X)
        
        # Should be sorted: apple, zebra
        expected_classes = np.array(["apple", "zebra"])
        np.testing.assert_array_equal(encoder.classes_, expected_classes)
    
    def test_label_encoder_multiple_columns(self):
        """Test encoding with multiple columns."""
        X = np.array([
            ["red", "small"],
            ["blue", "large"],
        ])
        
        encoder = LabelEncoder()
        encoder.fit(X)
        X_encoded = encoder.transform(X)
        
        # All categories flattened and encoded
        assert X_encoded.shape == (2, 2)
        assert X_encoded.dtype == int

    def test_label_encoder_warns_without_printing_for_unseen_categories(self, capsys):
        """Unseen categories should warn and encode as -1 without print output."""
        encoder = LabelEncoder().fit(np.array([["red"], ["blue"]]))

        with pytest.warns(UserWarning, match="Unseen categories"):
            encoded = encoder.transform(np.array([["green"]]))

        captured = capsys.readouterr()
        assert captured.out == ""
        np.testing.assert_array_equal(encoded, np.array([[-1]]))


# ============================================================================
# Tests for OneHotEncoder
# ============================================================================

class TestOneHotEncoder:
    """Tests for OneHotEncoder transformer."""
    
    def test_onehot_basic(self):
        """Test basic one-hot encoding."""
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
    
    def test_onehot_unseen_category(self):
        """Test handling of unseen category during transform."""
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
            ["green"],  # Never seen during fit
        ])
        
        X_encoded = encoder.transform(X_test)
        
        # Unseen category should be encoded as all zeros
        assert X_encoded.shape == (3, 2)
        assert X_encoded[2, 0] == 0 and X_encoded[2, 1] == 0
    
    def test_onehot_fit_transform(self):
        """Test fit_transform method."""
        X = np.array([["a"], ["b"], ["a"]])
        
        encoder = OneHotEncoder()
        result = encoder.fit_transform(X)
        
        encoder2 = OneHotEncoder()
        expected = encoder2.fit(X).transform(X)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_onehot_invalid_strategy_raises(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            SimpleImputer(strategy="invalid")


# ============================================================================
# Integration Tests
# ============================================================================

class TestPreprocessingPipeline:
    """Integration tests for preprocessing pipeline."""
    
    def test_pipeline_combination(self):
        """Test combining multiple transformers."""
        # Create data with NaN and categorical features
        X_raw = np.array([
            [25.0, np.nan, "USA"],
            [30.0, 100.0, "UK"],
            [35.0, 150.0, "USA"],
        ])
        
        # Step 1: Impute numerical NaN
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X_raw[:, :2].astype(float))
        
        # Step 2: Scale numerical
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed.astype(float))
        
        # Step 3: Encode categorical
        encoder = OneHotEncoder()
        X_categorical = X_raw[:, [2]].astype(str)
        X_encoded = encoder.fit_transform(X_categorical)
        
        # Combine
        X_processed = np.hstack([X_scaled, X_encoded])
        
        # Check shape
        assert X_processed.shape == (3, 4)  # 2 numerical + 2 categorical

    def test_pipeline_with_real_csv_sample(self):
        """Load sample.csv and run numeric + categorical preprocessing end to end."""
        sample_path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        numeric = np.array(
            [
                [
                    float(row["age"]),
                    np.nan if row["income"] == "" else float(row["income"]),
                    float(row["visits"]),
                ]
                for row in rows
            ],
            dtype=float,
        )
        categorical = np.array(
            [
                [
                    row["plan"],
                    row["region"],
                    row["is_student"],
                ]
                for row in rows
            ],
            dtype=object,
        )

        numeric_imputed = SimpleImputer(strategy="mean").fit_transform(numeric)
        numeric_scaled = StandardScaler().fit_transform(numeric_imputed)
        categorical_imputed = SimpleImputer(strategy="mode").fit_transform(categorical)
        categorical_encoded = OneHotEncoder().fit_transform(categorical_imputed)
        processed = np.hstack([numeric_scaled, categorical_encoded.astype(float)])

        assert processed.shape[0] == len(rows)
        assert processed.shape[1] > numeric.shape[1]
        assert np.all(np.isfinite(processed))
