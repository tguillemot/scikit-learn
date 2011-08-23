""" Test the cross_val module
"""

import numpy as np
from scipy.sparse import coo_matrix

from nose.tools import assert_true
from nose.tools import assert_raises

from ..base import BaseEstimator
from ..datasets import load_iris
from ..metrics import zero_one_score
from ..cross_val import StratifiedKFold
from ..svm import SVC
from ..svm.sparse import SVC as SparseSVC
from .. import cross_val
from ..cross_val import permutation_test_score


class MockClassifier(BaseEstimator):
    """Dummy classifier to test the cross-validation"""

    def __init__(self, a=0):
        self.a = a

    def fit(self, X, Y):
        return self

    def predict(self, T):
        return T.shape[0]

    def score(self, X=None, Y=None):
        return 1./(1+np.abs(self.a))


X = np.ones((10, 2))
X_sparse = coo_matrix(X)
y = np.arange(10) / 2

##############################################################################
# Tests

def test_kfold():
    # Check that errors are raise if there is not enough samples
    assert_raises(AssertionError, cross_val.KFold, 3, 4)
    y = [0, 0, 1, 1, 2]
    assert_raises(AssertionError, cross_val.StratifiedKFold, y, 3)


def test_cross_val_score():
    clf = MockClassifier()
    for a in range(-10, 10):
        clf.a = a
        # Smoke test
        score = cross_val.cross_val_score(clf, X, y)
        np.testing.assert_array_equal(score, clf.score(X, y))

        score = cross_val.cross_val_score(clf, X_sparse, y)
        np.testing.assert_array_equal(score, clf.score(X_sparse, y))


def test_permutation_score():
    iris = load_iris()
    X = iris.data
    X_sparse = coo_matrix(X)
    y = iris.target
    svm = SVC(kernel='linear')
    cv = StratifiedKFold(y, 2)

    score, scores, pvalue = permutation_test_score(
        svm, X, y, zero_one_score, cv)

    assert_true(score > 0.9)
    np.testing.assert_almost_equal(pvalue, 0.0, 1)

    score_label, _, pvalue_label = permutation_test_score(
        svm, X, y, zero_one_score, cv, labels=np.ones(y.size), random_state=0)

    assert_true(score_label == score)
    assert_true(pvalue_label == pvalue)

    # check that we obtain the same results with a sparse representation
    svm_sparse = SparseSVC(kernel='linear')
    cv_sparse = StratifiedKFold(y, 2, indices=True)
    score_label, _, pvalue_label = permutation_test_score(
        svm_sparse, X_sparse, y, zero_one_score, cv_sparse,
        labels=np.ones(y.size), random_state=0)

    assert_true(score_label == score)
    assert_true(pvalue_label == pvalue)

    # set random y
    y = np.mod(np.arange(len(y)), 3)

    score, scores, pvalue = permutation_test_score(
        svm, X, y, zero_one_score, cv)

    assert_true(score < 0.5)
    assert_true(pvalue > 0.4)


def test_cross_val_generator_with_indices():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 1, 2, 2])
    labels = np.array([1, 2, 3, 4])
    loo = cross_val.LeaveOneOut(4, indices=True)
    lpo = cross_val.LeavePOut(4, 2, indices=True)
    kf = cross_val.KFold(4, 2, indices=True)
    skf = cross_val.StratifiedKFold(y, 2, indices=True)
    lolo = cross_val.LeaveOneLabelOut(labels, indices=True)
    lopo = cross_val.LeavePLabelOut(labels, 2, indices=True)
    for cv in [loo, lpo, kf, skf, lolo, lopo]:
        for train, test in cv:
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]


def test_bootstrap_errors():
    assert_raises(ValueError, cross_val.Bootstrap, 10, n_train=100)
    assert_raises(ValueError, cross_val.Bootstrap, 10, n_test=100)
    assert_raises(ValueError, cross_val.Bootstrap, 10, n_train=1.1)
    assert_raises(ValueError, cross_val.Bootstrap, 10, n_test=1.1)
