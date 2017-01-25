# This files introduces a common function to deprecate sample_weights.
# It canbe removed at 0.21.

import warnings


def _deprecate_sample_weight(sample_weight, sample_props, function_name='fit'):
    if sample_props is None:
        sample_props = {}

    if "weights" in sample_props.keys() and sample_weight is not None:
        raise ValueError(
            "Both `sample_weight` and `sample_props['weights']` provided to "
            "%s. Please specify only one of the two." % function_name)

    elif sample_weight is not None:
        warnings.warn("The `sample_weight` parameter is deprecated in 0.19 "
                      "and will be removed in 0.21. From now, all sample "
                      "properties must be passed using the `sample_prop` "
                      "attribute. The sample weights can be provided using "
                      "the key `weights` of `sample_props` (ex: `sample_props "
                      "= {'weights': weights}`).", DeprecationWarning)
        return sample_weight

    else:
        return sample_props.get('weights', None)


# raise warning when we provide something more
# ignore when used in meta-estimator

# put __only__ to send only on estimator.
# do to example :
# 1) One sample with cross_val_score generique
# 2) A second more complicated to block everything at the level estimator
#    We can use __only__ eventually.
#
# Use some example with dict(one=1)
