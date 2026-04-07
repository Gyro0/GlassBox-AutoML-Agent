import numpy as np

from glassbox.utils.matrix import column_mean, column_std, column_variance, dot, transpose


def test_dot_matrix_multiplication():
	A = np.array([[1.0, 2.0], [3.0, 4.0]])
	B = np.array([[2.0, 0.0], [1.0, 2.0]])

	result = dot(A, B)

	np.testing.assert_allclose(result, np.array([[4.0, 4.0], [10.0, 8.0]]))


def test_transpose_matrix():
	A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

	result = transpose(A)

	np.testing.assert_allclose(result, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))


def test_column_statistics():
	X = np.array(
		[
			[1.0, 2.0, 3.0],
			[3.0, 4.0, 5.0],
			[5.0, 6.0, 7.0],
		]
	)

	np.testing.assert_allclose(column_mean(X), np.array([3.0, 4.0, 5.0]))
	np.testing.assert_allclose(column_variance(X), np.array([8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0]))
	np.testing.assert_allclose(column_std(X), np.sqrt(np.array([8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0])))