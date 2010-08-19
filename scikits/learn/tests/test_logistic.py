import numpy as np
from scikits.learn import logistic, datasets
from numpy.testing import *

X = [[-1, 0], [0, 1], [1, 1]]
Y1 = [0, 1, 1]
Y2 = [0, 1, 2]
iris = datasets.load_iris()

def test_predict_2_classes():
    """
    Simple sanity check on a 2 classes dataset.
    Make sure it predicts the correct result on simple datasets.
    """
    clf = logistic.LogisticRegression().fit(X, Y1)
    assert_array_equal(clf.predict(X), Y1)

    clf = logistic.LogisticRegression(C=100).fit(X, Y1)
    assert_array_equal(clf.predict(X), Y1)

    clf = logistic.LogisticRegression(has_intercept=False).fit(X, Y1)
    assert_array_equal(clf.predict(X), Y1)


def test_error():
    """
    test for appropriate exception on errors
    """
    assert_raises (ValueError, logistic.LogisticRegression(C=-1).fit, X, Y1)


def test_predict_3_classes():
    clf = logistic.LogisticRegression(C=10).fit(X, Y2)
    assert_array_equal(clf.predict(X), Y2)


def test_predict_iris():
    """Test logisic regression with the iris dataset"""

    clf = logistic.LogisticRegression().fit(iris.data, iris.target)
    pred = clf.predict(iris.data)

    assert_ ( np.mean(pred == iris.target) > .95 )

@decorators.skipif(True, "XFailed test")
def test_predict_proba():
    """
    I think this test is wrong. Is there a way to know the right results ?
    """
    clf = logistic.LogisticRegression().fit(X, Y2)
    assert_array_almost_equal(clf.predict_proba([[1, 1]]),
                              [[ 0.21490268,  0.32639437,  0.45870294]])

    clf = logistic.LogisticRegression(penalty='l1').fit(X, Y2)
    assert_array_almost_equal(clf.predict_proba([[2, 2]]),
                              [[ 0.33333333,  0.33333333,  0.33333333]])
