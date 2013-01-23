"""
Testing for the boost module (sklearn.ensemble.boost).
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from nose.tools import assert_true
from nose.tools import assert_raises

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import datasets


# Common random state
rng = np.random.RandomState(0)

# Toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)

# Load the boston dataset and randomly permute it
boston = datasets.load_boston()
boston.data, boston.target = shuffle(boston.data, boston.target, random_state=rng)


def test_classification_toy():
    """Check classification on a toy dataset."""
    clf = AdaBoostClassifier()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)


def test_regression_toy():
    """Check classification on a toy dataset."""
    clf = AdaBoostRegressor()
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)


def test_iris():
    """Check consistency on dataset iris."""
    clf = AdaBoostClassifier()
    clf.fit(iris.data, iris.target)
    score = clf.score(iris.data, iris.target)
    assert score > 0.9, "Failed with criterion %s and score = %f" % (c,
                                                                     score)


def test_boston():
    """Check consistency on dataset boston house prices."""
    clf = AdaBoostRegressor()
    clf.fit(boston.data, boston.target)
    score = clf.score(boston.data, boston.target)
    assert score > 0.85


def test_staged_predict():
    """Check staged predictions."""
    # AdaBoost classification
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(iris.data, iris.target)

    predictions = clf.predict(iris.data)
    staged_predictions = [p for p in clf.staged_predict(iris.data)]
    proba = clf.predict_proba(iris.data)
    staged_probas = [p for p in clf.staged_predict_proba(iris.data)]
    score = clf.score(iris.data, iris.target)
    staged_scores = [s for s in clf.staged_score(iris.data, iris.target)]

    assert_equal(len(staged_predictions), 10)
    assert_array_almost_equal(predictions, staged_predictions[-1])
    assert_equal(len(staged_probas), 10)
    assert_array_almost_equal(proba, staged_probas[-1])
    assert_equal(len(staged_scores), 10)
    assert_array_almost_equal(score, staged_scores[-1])

    # AdaBoost regression
    clf = AdaBoostRegressor(n_estimators=10)
    clf.fit(boston.data, boston.target)

    predictions = clf.predict(boston.data)
    staged_predictions = [p for p in clf.staged_predict(boston.data)]
    score = clf.score(boston.data, boston.target)
    staged_scores = [s for s in clf.staged_score(boston.data, boston.target)]

    assert_equal(len(staged_predictions), 10)
    assert_array_almost_equal(predictions, staged_predictions[-1])
    assert_equal(len(staged_scores), 10)
    assert_array_almost_equal(score, staged_scores[-1])


def test_gridsearch():
    """Check that base trees can be grid-searched."""
    # AdaBoost classification
    boost = AdaBoostClassifier()
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)

    # AdaBoost regression
    boost = AdaBoostRegressor()
    parameters = {'n_estimators': (1, 2),
                  'base_estimator__max_depth': (1, 2)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(boston.data, boston.target)


def test_pickle():
    """Check pickability."""
    import pickle

    # Adaboost classifier
    obj = AdaBoostClassifier()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(iris.data, iris.target)
    assert score == score2

    # Adaboost regressor
    obj = AdaBoostRegressor()
    obj.fit(boston.data, boston.target)
    score = obj.score(boston.data, boston.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert_equal(type(obj2), obj.__class__)
    score2 = obj2.score(boston.data, boston.target)
    assert score == score2


def test_importances():
    """Check variable importances."""
    X, y = datasets.make_classification(n_samples=2000,
                                        n_features=10,
                                        n_informative=3,
                                        n_redundant=0,
                                        n_repeated=0,
                                        shuffle=False,
                                        random_state=1)

    clf = AdaBoostClassifier(compute_importances=True)

    clf.fit(X, y)
    importances = clf.feature_importances_
    n_important = sum(importances > 0.1)

    assert_equal(importances.shape[0], 10)
    assert_equal(n_important, 3)

    clf = AdaBoostClassifier()
    clf.fit(X, y)
    assert_true(clf.feature_importances_ is None)


def test_error():
    """Test that it gives proper exception on deficient input."""
    from sklearn.dummy import DummyClassifier
    from sklearn.dummy import DummyRegressor

    # Invalid values for parameters
    assert_raises(ValueError,
                  AdaBoostClassifier(learning_rate=-1).fit,
                  X, y)

    assert_raises(ValueError,
                  AdaBoostClassifier(algorithm="foo").fit,
                  X, y)

    assert_raises(TypeError,
                  AdaBoostClassifier(base_estimator=DummyRegressor()).fit,
                  X, y)

    assert_raises(TypeError,
                  AdaBoostRegressor(base_estimator=DummyClassifier()).fit,
                  X, y)


def test_base_estimator():
    """Test different base estimators."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    clf = AdaBoostClassifier(RandomForestClassifier())
    clf.fit(X, y)

    clf = AdaBoostClassifier(SVC(), algorithm="SAMME")
    clf.fit(X, y)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    clf = AdaBoostRegressor(RandomForestRegressor())
    clf.fit(X, y)

    clf = AdaBoostRegressor(SVR())
    clf.fit(X, y)


if __name__ == "__main__":
    import nose
    nose.runmodule()
