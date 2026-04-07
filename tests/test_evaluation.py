import numpy as np

from glassbox.evaluation.classification import build_confusion_matrix, classification_report
from glassbox.evaluation.regression import mean_absolute_error, mean_squared_error, r2_score


def test_build_confusion_matrix_binary_case():
	y_true = np.array([0, 1, 1, 0, 1])
	y_pred = np.array([0, 1, 0, 0, 1])

	matrix, labels = build_confusion_matrix(y_true, y_pred)

	assert labels == [0, 1]
	np.testing.assert_array_equal(matrix, np.array([[2, 0], [1, 2]]))


def test_classification_report_binary_average():
	y_true = np.array([0, 1, 1, 0, 1])
	y_pred = np.array([0, 1, 0, 0, 1])

	report = classification_report(y_true, y_pred, average="binary", positive_label=1)

	np.testing.assert_allclose(report["accuracy"], 0.8)
	np.testing.assert_allclose(report["precision"], 1.0)
	np.testing.assert_allclose(report["recall"], 2.0 / 3.0)
	np.testing.assert_allclose(report["f1"], 0.8)


def test_classification_report_macro_average_multiclass():
	y_true = np.array([0, 1, 2, 0, 1, 2])
	y_pred = np.array([0, 2, 2, 0, 1, 1])

	report = classification_report(y_true, y_pred, average="macro")

	np.testing.assert_allclose(report["accuracy"], 4.0 / 6.0)
	np.testing.assert_allclose(report["precision"], (1.0 + 0.5 + 0.5) / 3.0)
	np.testing.assert_allclose(report["recall"], (1.0 + 0.5 + 0.5) / 3.0)
	np.testing.assert_allclose(report["f1"], (1.0 + 0.5 + 0.5) / 3.0)


def test_regression_metrics_values():
	y_true = np.array([3.0, -0.5, 2.0, 7.0])
	y_pred = np.array([2.5, 0.0, 2.0, 8.0])

	np.testing.assert_allclose(mean_absolute_error(y_true, y_pred), 0.5)
	np.testing.assert_allclose(mean_squared_error(y_true, y_pred), 0.375)
	np.testing.assert_allclose(r2_score(y_true, y_pred), 0.9486081370449679)


def test_r2_score_constant_target_returns_zero():
	y_true = np.array([5.0, 5.0, 5.0])
	y_pred = np.array([4.0, 6.0, 5.0])

	np.testing.assert_allclose(r2_score(y_true, y_pred), 0.0)
