import numpy as np
from numpy import floating
from numpy.typing import NDArray
from typing import Optional, Callable


def _euclidian_distance(X: NDArray[floating]) -> NDArray[floating]:
    """feature expansion function for McLess, Computes a point p such that $\sigma(x) = ||x - p||

    Args:
        X (NDArray[floating]): the input matrix to the model with shape (number of points, number of features)

    Returns:
        NDArray[floating]: the 1D feature expansion  with shape (number of points, 1) to be added to the end of the information matrix
    """
    return X[:, 0]


def _create_information_matrix(
    X: NDArray[floating],
    feature_count: int,
    expansion_count: int,
    feature_expansions: list[Callable],
) -> NDArray[floating]:
    """helper function for McLess that creates the information matrix used to both set the weights and
    predict the classification of future data points

    Args:
        X (NDArray[floating]): the input data for the model
        feature_count (int): number of features the model is expecting, should cause an error if not equal to the number of columns in X
        expansion_count (int): number of feature expansions
        feature_expansions (list[Callable]): the list of feature expansions being performed on X,
            each function should take in a matrix(X) and output a 1D array

    Returns:
        NDArray[floating]: the information matrix, with shape (number of points, 1 + number of features + number of feature expansions)
    """
    data_count, cols = X.shape
    if cols != feature_count:
        print("NOW IS THE TIME TO PANIC")  # TODO: replace with error
    # modification of the input data, the matrix is set up col wise like [1;x;output of feature functions]
    information_matrix = np.ones((data_count, feature_count + 1 + expansion_count))
    information_matrix[:, 1 : feature_count + 1] = X
    for i, f in enumerate(feature_expansions):
        information_matrix[:, feature_count + i] = f(X)
    return information_matrix


def _set_weights(A: NDArray[floating], B: NDArray[floating]) -> NDArray[floating]:
    """helper function for McLess that sets the weight matrix. Note if using feature expansion, this should be performed on the input
    matrix A before passing it to this function to set the weights

    Args:
        A (NDArray[floating]): the information matrix with shape (number of points, 1 + number of features + number of feature expansions)
        B (NDArray[floating]): the one hot encoding of the labels with shape (number of points, number of labels)

    Returns:
        the weights for McLess equal to the pseudoinverse of A matrix multiplied with B,
        the weights should have shape (1 + number of features + number of feature expansions, number of labels).
        Note the first column represents the bias
    """
    normal_matrix: NDArray[floating] = A.T @ A
    sign_det, log_det = np.linalg.slogdet(normal_matrix)

    # since $(A^{T}A)\hat{W}=A^{T}B$, then $\hat{W}=A^{+}B$
    pseudoinverse: NDArray[floating]
    # print(f"A^T A shape {normal_matrix.shape}")
    if sign_det * log_det:  # Fails if det == 0
        # if $A^{T}A$ is nonsingular, $A^{+}=(A^{T}A)A^{T}$
        pseudoinverse = np.linalg.inv(normal_matrix) @ A.T
    else:
        # if $A^{T}A$ is singular, $A^{+}=V \Sigma^{-1} U^{T}$
        U, S, Vh = np.linalg.svd(normal_matrix, compute_uv=True)
        # unless I'm mistaken, Vh is returned transposed so...
        pseudoinverse = Vh.T @ np.linalg.inv(S) @ U.T

    return pseudoinverse @ B


class McLess(object):
    # NOTE: the prefixed underscores are because python lacks private variables, variables starting with underscores are considered internal implementaion
    # details and not meant to be directly interacted with by users of the class
    # see https://stackoverflow.com/a/1301409/11019565

    # this both list all the variables stored by the class and reduces the amount of memory used by class instances by
    # not storing it internally as a dict
    # see https://book.pythontips.com/en/latest/__slots__magic.html
    __slots__ = [
        "feature_names",  # name of the features each column of the input to the model represents
        "target_names",  # the name of the classification each data point belong to
        "_weight_matrix",  # the matrix used to predict the classification of future data points
        "_label_count",  # the number of labels/targets/classifications
        "_feature_count",  # the number of values associated with a single data point
        "_feature_expansion_functions",  # functions which operate on the features to extend the data associated with each point
        "_feature_expansion_count",  # the number of feature expansions
    ]

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
        feature_expansions: Optional[list[str | Callable]] = None,
    ):

        self._feature_expansion_count: int = 0
        self._feature_expansion_functions: list[Callable] = []

        # set feature names if provided
        self.feature_names: list[str] = (
            feature_names if feature_names is not None else []
        )

        # set label names if provided
        self.target_names: list[str] = target_names if target_names is not None else []
        # preemptively set label count
        self._label_count: int = len(self.target_names)  # 0
        # set the feature expansions for the information matrix
        if feature_expansions is not None:
            for f in feature_expansions:
                if type(f) == str:
                    if f == "euclidian":
                        self.__feature_expansion_functions.append(_euclidian_distance)
            self._feature_expansion_count = len(self.__feature_expansion_functions)

    def fit(self, x_data: NDArray[floating], y_data: NDArray[floating]):
        data_count, self._feature_count = x_data.shape

        if self._label_count == 0:
            label_set: set = set(y_data)
            self._label_count: int = len(label_set)

        information_matrix: NDArray[floating] = _create_information_matrix(
            x_data,
            self._feature_count,
            self._feature_expansion_count,
            self._feature_expansion_functions,
        )
        # originally B
        source_matrix: NDArray[floating] = np.zeros((data_count, self._label_count))
        # each (i,j)^th entry represents a membership in the jth class for the ith data point
        source_matrix[np.arange(data_count), y_data] = 1

        # each column represents the bias towards a specific classification
        self._weight_matrix: NDArray[floating] = _set_weights(
            information_matrix, source_matrix
        )

    def score(self, x_test, y_test) -> float:
        data_count: int = len(y_test)

        y_pred: NDArray[np.int32] = self.predict(x_test)
        # print(f"y\ntest:\n{y_test}\npred:\n{y_pred}")
        hit_count: int = np.count_nonzero(y_test == y_pred)
        # print(
        #     f"data count: {data_count} hit count: {hit_count} accuracy {hit_count/data_count}"
        # )
        return hit_count / data_count

    def predict(self, X: NDArray[floating]):

        information_matrix: NDArray[floating] = _create_information_matrix(
            X,
            self._feature_count,
            self._feature_expansion_count,
            self._feature_expansion_functions,
        )
        # print(X)
        # print(self.weight_matrix.shape)
        pred_probs: NDArray[floating] = information_matrix @ self._weight_matrix

        return np.argmax(pred_probs, axis=1)
