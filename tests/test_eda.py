import numpy as np

from glassbox.eda.auto_typer import infer_column_types
from glassbox.eda.correlation import build_pearson_correlation_matrix
from glassbox.eda.outlier import iqr_outlier_handler
from glassbox.eda.profiler import profile_numeric_columns


def test_profile_numeric_columns_returns_expected_keys_and_values():
	X = np.array(
		[
			[1.0, 2.0],
			[2.0, 2.0],
			[3.0, 4.0],
			[4.0, 4.0],
		]
	)

	profile = profile_numeric_columns(X)

	assert set(profile.keys()) == {0, 1}
	assert set(profile[0].keys()) == {
		"mean",
		"median",
		"mode",
		"std",
		"variance",
		"skewness",
		"kurtosis",
	}

	col0 = X[:, 0]
	col1 = X[:, 1]

	np.testing.assert_allclose(profile[0]["mean"], np.mean(col0))
	np.testing.assert_allclose(profile[0]["median"], np.median(col0))
	np.testing.assert_allclose(profile[0]["std"], np.std(col0))
	np.testing.assert_allclose(profile[0]["variance"], np.var(col0))

	np.testing.assert_allclose(profile[1]["mean"], np.mean(col1))
	np.testing.assert_allclose(profile[1]["median"], np.median(col1))
	np.testing.assert_allclose(profile[1]["std"], np.std(col1))
	np.testing.assert_allclose(profile[1]["variance"], np.var(col1))


def test_profile_numeric_columns_mode_tie_breaks_to_smallest_value():
	X = np.array(
		[
			[2.0],
			[1.0],
			[2.0],
			[1.0],
		]
	)

	profile = profile_numeric_columns(X)

	np.testing.assert_allclose(profile[0]["mode"], 1.0)


def test_profile_numeric_columns_handles_constant_column():
	X = np.array(
		[
			[5.0],
			[5.0],
			[5.0],
		]
	)

	profile = profile_numeric_columns(X)

	np.testing.assert_allclose(profile[0]["std"], 0.0)
	np.testing.assert_allclose(profile[0]["variance"], 0.0)
	np.testing.assert_allclose(profile[0]["skewness"], 0.0)
	np.testing.assert_allclose(profile[0]["kurtosis"], -3.0)


def test_profile_numeric_columns_requires_2d_input():
	with np.testing.assert_raises(ValueError):
		profile_numeric_columns(np.array([1.0, 2.0, 3.0]))


def test_profile_numeric_columns_supports_feature_names():
	X = np.array(
		[
			[1.0, 10.0],
			[2.0, 20.0],
			[3.0, 30.0],
		]
	)

	profile = profile_numeric_columns(X, feature_names=["a", "b"])

	assert set(profile.keys()) == {"a", "b"}
	np.testing.assert_allclose(profile["a"]["mean"], 2.0)
	np.testing.assert_allclose(profile["b"]["mean"], 20.0)


def test_infer_column_types_mixed_inputs():
	data = {
		"age": [21, 30, 40, 52],
		"zip_code": [10001, 10001, 10002, 10002],
		"city": ["A", "B", "A", "C"],
		"is_active": [0, 1, 1, 0],
	}

	inferred = infer_column_types(data, unique_ratio_threshold=0.5)

	assert inferred["age"] == "numerical"
	assert inferred["zip_code"] == "categorical"
	assert inferred["city"] == "categorical"
	assert inferred["is_active"] == "boolean"


def test_build_pearson_correlation_matrix_and_high_collinearity():
	X = np.array(
		[
			[1.0, 2.0, 7.0],
			[2.0, 4.0, 6.0],
			[3.0, 6.0, 5.0],
			[4.0, 8.0, 4.0],
		]
	)
	feature_names = ["x1", "x2", "x3"]

	corr, names, high = build_pearson_correlation_matrix(X, feature_names=feature_names, collinearity_threshold=0.85)

	assert names == feature_names
	np.testing.assert_allclose(corr[0, 0], 1.0)
	np.testing.assert_allclose(corr[1, 1], 1.0)
	np.testing.assert_allclose(corr[0, 1], 1.0)
	assert ("x1", "x2", 1.0) in high


def test_iqr_outlier_handler_flag_and_cap_modes():
	X = np.array(
		[
			[1.0, 10.0],
			[2.0, 11.0],
			[3.0, 12.0],
			[100.0, 13.0],
		]
	)

	mask = iqr_outlier_handler(X, mode="flag")
	capped = iqr_outlier_handler(X, mode="cap")

	assert mask.shape == X.shape
	assert mask[3, 0]
	assert capped[3, 0] < 100.0


def test_iqr_outlier_handler_can_return_outlier_row_indices():
	X = np.array(
		[
			[1.0, 10.0],
			[2.0, 11.0],
			[3.0, 12.0],
			[100.0, 13.0],
		]
	)

	indices = iqr_outlier_handler(X, mode="flag", return_row_indices=True)

	assert indices == [3]
