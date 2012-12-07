"""
Testing for the boost module (sklearn.ensemble.boost).
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_classification_toy():
    """Check classification on a toy dataset."""
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)


def test_regression_toy():
    """Check classification on a toy dataset."""
    clf = AdaBoostRegressor(n_estimators=10)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)


def test_iris():
    """Check consistency on dataset iris."""
    for c in ("gini", "entropy"):
        # AdaBoost classification
        clf = AdaBoostClassifier(DecisionTreeClassifier(criterion=c),
                                 n_estimators=1)
        clf.fit(iris.data, iris.target)
        score = clf.score(iris.data, iris.target)
        assert score > 0.9, "Failed with criterion %s and score = %f" % (c,
                                                                         score)


def test_boston():
    """Check consistency on dataset boston house prices."""
    clf = AdaBoostRegressor(n_estimators=10)
    clf.fit(boston.data, boston.target)
    score = clf.score(boston.data, boston.target)
    assert score > 0.99


def test_probability():
    """Predict probabilities."""
    # AdaBoost classification
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(iris.data, iris.target)
    assert_array_almost_equal(np.sum(clf.predict_proba(iris.data), axis=1),
                              np.ones(iris.data.shape[0]))
    assert_array_almost_equal(clf.predict_proba(iris.data),
                              np.exp(clf.predict_log_proba(iris.data)))


def test_gridsearch():
    """Check that base trees can be grid-searched."""
    # AdaBoost classification
    boost = AdaBoostClassifier()
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)


def test_pickle():
    """Check pickability."""
    import pickle

    # Adaboost
    obj = AdaBoostClassifier()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(iris.data, iris.target)
    assert score == score2


if __name__ == "__main__":
    import nose
    nose.runmodule()
