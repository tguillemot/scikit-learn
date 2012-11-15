import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal

from sklearn.random_classifier import RandomClassifier


def _check_predict_proba(clf, X, y):
    out = clf.predict_proba(X)
    assert_equal(out.shape[0], len(X))
    assert_equal(out.shape[1], len(np.unique(y)))
    assert_array_equal(out.sum(axis=1), np.ones(len(X)))

def test_most_frequent_sampling():
    X = [[0], [0], [0], [0]]  # ignored
    y = [1, 2, 1, 1]

    clf = RandomClassifier(sampling="most_frequent", random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), np.ones(len(X)))
    _check_predict_proba(clf, X, y)


def test_stratified_sampling():
    X = [[0], [0], [0], [0], [0]]  # ignored
    y = [1, 2, 1, 1, 2]

    clf = RandomClassifier(sampling="stratified", random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), [1, 2, 2, 1, 1])
    _check_predict_proba(clf, X, y)


def test_uniform_sampling():
    X = [[0], [0], [0], [0]]  # ignored
    y = [1, 2, 1, 1]

    clf = RandomClassifier(sampling="uniform", random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.predict(X), [1, 2, 2, 1])
    _check_predict_proba(clf, X, y)
