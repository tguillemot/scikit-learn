"""
Testing for the base module (sklearn.ensemble.base).
"""

# Authors: Gilles Louppe
# License: BSD 3 clause

from numpy.testing import assert_equal
from nose.tools import assert_raises, assert_true

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron


def test_base():
    """Check BaseEnsemble methods."""
    ensemble = BaggingClassifier(base_estimator=Perceptron(), n_estimators=3)

    iris = load_iris()
    ensemble.fit(iris.data, iris.target)
    ensemble.estimators_ = [] # empty the list and create estimators manually

    ensemble._make_estimator()
    ensemble._make_estimator()
    ensemble._make_estimator()
    ensemble._make_estimator(append=False)

    assert_equal(3, len(ensemble))
    assert_equal(3, len(ensemble.estimators_))

    assert_true(isinstance(ensemble[0], Perceptron))
