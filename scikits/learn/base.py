"""
Base class for all estimators.

"""
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>

# License: BSD Style
import inspect

import numpy as np

from .metrics import zero_one, mean_square_error

################################################################################
class BaseEstimator(object):
    """ Base class for all estimators in the scikit learn

        Note
        =====

        All estimators should specify all the parameters that can be set
        at the class level in their __init__ as explicit keyword
        arguments (no *args, **kwargs).

    """

    @classmethod
    def _get_param_names(cls):
        try:
            args, varargs, kw, default = inspect.getargspec(cls.__init__)
            assert varargs is None, (
                'scikit learn estimators should always specify their '
                'parameters in the signature of their init (no varargs).'
                )
            # Remove 'self'
            # XXX: This is going to fail if the init is a staticmethod, but
            # who would do this?
            args.pop(0)
        except TypeError:
            # No explicit __init__
            args = []
        return args


    def _get_params(self):
        out = dict()
        for key in self._get_param_names():
            out[key] = getattr(self, key)
        return out


    def _set_params(self, **params):
        """ Set the parameters of the estimator.
        """
        valid_params = self._get_param_names()
        for key, value in params.iteritems():
            assert key in valid_params, ('Invalid parameter %s '
                'for estimator %s' %
                (key, self.__class__.__name__))
            setattr(self, key, value)


    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=5, threshold=64, edgeitems=2)
        class_name = self.__class__.__name__

        # Do a multi-line justified repr:
        params_list = list()
        this_line_length = len(class_name)
        line_sep = ',\n' + (1+len(class_name)/2)*' '
        for i, (k, v) in enumerate(self._get_params().iteritems()):
            this_repr  = '%s=%s' % (k, repr(v))
            if i > 0: 
                if (this_line_length + len(this_repr) >= 75
                                            or '\n' in this_repr):
                    params_list.append(line_sep)
                    this_line_length += len(line_sep)
                else:
                    params_list.append(', ')
                    this_line_length += 2
            params_list.append(this_repr)
            this_line_length += len(this_repr)

        params_str = ''.join(params_list)
        np.set_printoptions(**options)
        return '%s(%s)' % (
                class_name,
                params_str
            )


################################################################################
class ClassifierMixin(object):
    """ Mixin class for all classifiers in the scikit learn
    """

    def score(self, X, y):
        return - zero_one(self.predict(X), y)


################################################################################
class RegressorMixin(object):
    """ Mixin class for all regression estimators in the scikit learn
    """

    def score(self, X, y):
        return - mean_square_error(self.predict(X), y)
    
