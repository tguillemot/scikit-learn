from sklearn.utils.extmath import logsum

from numpy.testing import assert_almost_equal, assert_array_almost_equal
import numpy as np


def test_logsum():
    # Try to add some smallish numbers in logspace
    x = np.array([1e-40] * 1000000)
    logx = np.log(x)
    assert_almost_equal(np.exp(logsum(logx)), x.sum())

    X = np.vstack([x, x])
    logX = np.vstack([logx, logx])
    assert_array_almost_equal(np.exp(logsum(logX, axis=0)), X.sum(axis=0))
    assert_array_almost_equal(np.exp(logsum(logX, axis=1)), X.sum(axis=1))
