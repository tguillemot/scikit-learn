""" Test fast_dict. So far only smoke test
"""
import numpy as np
from nose.tools import assert_equal

from sklearn.utils.fast_dict import IntFloatDict, argmin

def test_int_float_dict():
    rng = np.random.RandomState(0)
    keys = np.unique(rng.randint(100, size=10).astype(np.intp))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value

    d.append(120, 3.)
    assert d[120] == 3.0
    [d.append(i+1000, 4.0) for i in xrange(2000)]
    assert d[1100] == 4.0


def test_int_float_dict_argmin():
    # Test the argmin implementation on the IntFloatDict
    keys = np.arange(100, dtype=np.intp)
    values = np.arange(100, dtype=np.float)
    d = IntFloatDict(keys, values)
    assert_equal(argmin(d), (0, 0))
