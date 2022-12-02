import sys
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from mcless.model import McLess
import numpy as np
from numpy import floating, float64
from numpy.typing import NDArray
import time

from tabulate import tabulate

# -----------------------------------------------
classifier_names = [
    "Logistic Regr",
    "KNeighbors-7 ",
    "Linear SVM ",
    "RBF SVM",
    "Random Forest",
    "Deep-NN",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "Gaussian Proc",
    "McLess",
    "McLess with feature expansion",
]
# -----------------------------------------------
classifiers = [
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(7),
    SVC(kernel="linear", C=0.5),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GaussianProcessClassifier(),
    McLess(),
    McLess(feature_expansions="euclidean"),
]


def bench_classifier(
    classifier,
    X: NDArray[float64],
    Y: NDArray[float64],
    training_size: float = 0.7e0,
    run_count=100,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    training_time: NDArray[float64] = np.zeros(run_count)
    inference_time: NDArray[float64] = np.zeros(run_count)
    Accuracy: NDArray[float64] = np.zeros(run_count)

    train_start: float = time.time()
    inference_start: float = time.time()

    test_data_percent: float = 1 - training_size
    for i in range(run_count):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            X, Y, test_size=test_data_percent, random_state=i, stratify=Y
        )
        train_start = time.time()
        classifier.fit(Xtrain, Ytrain)
        training_time[i] = time.time() - train_start
        # update average training time and accuracy
        # inference_start = time.time()
        # _ = classifier.predict(Xtest[0]) # trouble getting this to work
        # inference_time[i] = time.time() - inference_start
        Accuracy[i] = classifier.score(Xtest, Ytest)
    return training_time, inference_time, Accuracy


def bench_classifier_with_dataset(classifier, data_read):
    X = data_read.data
    Y = data_read.target
    data_file = data_read.filename
    return bench_classifier(classifier, X, Y)


def bench_all_with_dataset(data_read):
    global classifiers  # bad practice, but I'm short on time
    global classifier_names
    if len(classifier_names) != len(classifiers):
        return
    X = data_read.data
    Y = data_read.target
    data_file = data_read.filename
    results: Dict[str, (float, float, float, float)] = {}
    headers = [
        "training time avg",
        "training time std",
        # "inference time avg",
        # "inference time std",
        "accuracy avg",
        "accuracy std",
    ]
    for (cname, classifier) in zip(classifier_names, classifiers):
        print(f"testing classifer {cname}")
        (training, inference, accuracy) = bench_classifier_with_dataset(
            classifier, data_read
        )
        results[cname] = (
            training.mean(),
            training.std(),
            # inference.mean(),
            # inference.std(),
            accuracy.mean(),
            accuracy.std(),
        )
    return tabulate(results, headers=headers)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print(bench_all_with_dataset(load_iris()))
