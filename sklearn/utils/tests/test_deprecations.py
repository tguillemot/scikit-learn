# This files is used to test the deprecation of sample_weights.
# It can be removed at 0.21.

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.deprecations import _deprecate_sample_weight


def test_deprecate_sample_weight():
    sample_weights = 1
    sample_props = dict(weights=2)

    assert_raise_message(
        ValueError, "Both `sample_weight` and `sample_props['weights']` "
        "provided to score. Please specify only one of the two.",
        _deprecate_sample_weight, sample_weights, sample_props, 'score')

    value = assert_warns_message(
        DeprecationWarning, "The `sample_weight` parameter is deprecated in "
        "0.19 and will be removed in 0.21. From now, all sample properties "
        "must be passed using the `sample_prop` attribute. The sample weights "
        "can be provided using the key `weights` of `sample_props` (ex: "
        "`sample_props = {'weights': weights}`).",
        _deprecate_sample_weight, sample_weights, None)
    assert_equal(value, sample_weights)

    value = assert_no_warnings(_deprecate_sample_weight, None, sample_props)
    assert_equal(value, sample_props['weights'])
