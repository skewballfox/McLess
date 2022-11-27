import numpy as np
from numpy import floating
from numpy.typing import NDArray


def _set_weights(A: NDArray[floating], B: NDArray[floating], W: NDArray):
    normal_matrix = A.T @ A
    sign_det, log_det = np.linalg.slogdet(normal_matrix)
    # since $(A^{T}A)\hat{W}=A^{T}B$, then $\hat{W}=A^{+}B$
    pseudoinverse: NDArray[floating]
    if sign_det * log_det:  # Fails if det == 0
        # if $A^{T}A$ is nonsingular, $A^{+}=(A^{T}A)A^{T}$
        pseudoinverse = np.linalg.inv(normal_matrix) @ A.T
    else:
        # if $A^{T}A$ is singular, $A^{+}=V \Sigma^{-1} U^{T}$
        U, S, V = np.linalg.svd(normal_matrix, compute_uv=True)
        pseudoinverse = V @ np.linalg.inv(S) @ U.T
    W = pseudoinverse @ B


class McLess(object):
    def __init__():
        pass

    def fit(self, x_data: NDArray[floating], y_data: NDArray[floating]):
        data_count, self.__data_len = x_data.shape
        label_set = set(y_data)
        self.__label_count = len(label_set)
        # information_offset = label_count - data_len
        information_matrix = np.ones((data_count, self.__data_len + 1))
        information_matrix[:, 1:] = x_data

        # each column represents the bias towards a specific classification
        self.weight_matrix: NDArray[floating] = np.ones(
            (self.__data_len + 1, self.__label_count)
        )

        # originally B
        # each (i,j)^th entry represents a membership in the jth class for the ith data point
        source_matrix: NDArray[floating] = np.kron(x_data, y_data)

        _set_weights(information_matrix, source_matrix, self.weight_matrix)

    def score(self, x_test, y_test):
        data_count = x_test[0]
        # information_matrix = np.ones((data_count, self.__data_len + 1))
        # information_matrix[:, 1:] = x_test

        # originally B
        # each (i,j)^th entry represents a membership in the jth class for the ith data point
        # source_matrix: NDArray[floating] = np.kron(x_test, y_test)
        y_pred = self.predict(x_test)
        hit_count = np.sum(np.all((y_test - y_pred == 0)))
        return hit_count / data_count

    def predict(self, X):
        data_count = X[0]
        information_matrix = np.ones((data_count, self.__data_len + 1))
        information_matrix[:, 1:] = X
        pred_probs = X @ self.weight_matrix
        return np.argmax(pred_probs, axis=1)
