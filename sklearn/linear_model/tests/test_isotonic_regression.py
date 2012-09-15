import numpy as np
from numpy.testing import assert_array_equal

from sklearn.linear_model.isotonic_regression_ import isotonic_regression
from sklearn.linear_model import IsotonicRegression

from nose.tools import assert_raises


def test_isotonic_regression():
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    y_ = np.array([3, 6, 6, 8, 8, 8, 10])
    assert_array_equal(y_, isotonic_regression(y))

    x = np.arange(len(y))
    ir = IsotonicRegression(y_min=0., y_max=1.)
    ir.fit(x, y)
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))
    assert_array_equal(ir.transform(x), ir.predict(x))


def test_assert_raises_exceptions():
    ir = IsotonicRegression()
    rng = np.random.RandomState(42)
    assert_raises(ValueError, ir.fit, [0, 1, 2], [5, 7, 3], [0.1, 0.6])
    assert_raises(ValueError, ir.fit, [0, 1, 2], [5, 7])
    assert_raises(ValueError, ir.fit, rng.randn(3, 10), [0, 1, 2])
    assert_raises(ValueError, ir.transform, rng.randn(3, 10))
