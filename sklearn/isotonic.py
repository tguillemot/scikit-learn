# Authors: Fabian Pedregosa <fabian@fseoane.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from .base import BaseEstimator, TransformerMixin, RegressorMixin
from .utils import as_float_array, check_arrays
from ._isotonic import _isotonic_regression
import warnings


def isotonic_regression(y, sample_weight=None, y_min=None, y_max=None,
                        weight=None, increasing=True):
    """Solve the isotonic regression model::

        min sum w[i] (y[i] - y_[i]) ** 2

        subject to y_min = y_[1] <= y_[2] ... <= y_[n] = y_max

    where:
        - y[i] are inputs (real numbers)
        - y_[i] are fitted
        - w[i] are optional strictly positive weights (default to 1.0)

    Parameters
    ----------
    y : iterable of floating-point values
        The data.

    sample_weight : iterable of floating-point values, optional, default: None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean, optional, default: True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False)

    Returns
    -------
    `y_` : list of floating-point values
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.
    """
    if weight is not None:
        warnings.warn("'weight' was renamed to 'sample_weight' and will "
                      "be removed in 0.16.",
                      DeprecationWarning)
        sample_weight = weight

    y = np.asarray(y, dtype=np.float)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=y.dtype)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float)
    if not increasing:
        y = y[::-1]
        sample_weight = sample_weight[::-1]

    if y_min is not None or y_max is not None:
        y = np.copy(y)
        sample_weight = np.copy(sample_weight)
        # upper bound on the cost function
        C = np.dot(sample_weight, y * y) * 10
        if y_min is not None:
            y[0] = y_min
            sample_weight[0] = C
        if y_max is not None:
            y[-1] = y_max
            sample_weight[-1] = C

    solution = np.empty(len(y))
    y_ = _isotonic_regression(y, sample_weight, solution)
    if increasing:
        return y_
    else:
        return y_[::-1]


class IsotonicRegression(BaseEstimator, TransformerMixin, RegressorMixin):
    """Isotonic regression model.

    The isotonic regression optimization problem is defined by::

        min sum w_i (y[i] - y_[i]) ** 2

        subject to y_[i] <= y_[j] whenever X[i] <= X[j]
        and min(y_) = y_min, max(y_) = y_max

    where:
        - ``y[i]`` are inputs (real numbers)
        - ``y_[i]`` are fitted
        - ``X`` specifies the order.
          If ``X`` is non-decreasing then ``y_`` is non-decreasing.
        - ``w[i]`` are optional strictly positive weights (default to 1.0)

    Parameters
    ----------
    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean or string, optional, default : 'auto'
        If boolean, whether or not to fit the isotonic regression with y
        increasing or decreasing.

        If string and set to "auto," determine whether y should
        increase or decrease based on the Spearman correlation estimate's
        sign.


    Attributes
    ----------
    `X_` : ndarray (n_samples, )
        A copy of the input X.

    `y_` : ndarray (n_samples, )
        Isotonic fit of y.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308
    """
    def __init__(self, y_min=None, y_max=None, increasing=True):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing

    def _check_fit_data(self, X, y, sample_weight=None):
        if len(X.shape) != 1:
            raise ValueError("X should be a vector")

    def _check_increasing(self, X, y):
        """
        Set the proper value of ``increasing`` based on the constructor
        parameter and the data.

        The Spearman correlation coefficent is estimated from the data,
        and the sign of the resulting estimate is used to set ``increasing``.

        In the event that the 95% confidence interval based on Fisher transform
        spans zero, a warning is raised.
        """
        # Determine increasing if Spearman requested
        increasing_bool = self.increasing

        if self.increasing == 'auto':
            # Calculate Spearman rho estimate and set accordingly
            rho, _ = spearmanr(X, y)
            if rho >= 0:
                increasing_bool = True
            else:
                increasing_bool = False

            # Run Fisher transform to get the rho CI
            F = 0.5 * np.log((1 + rho) / (1 - rho))
            F_se = 1 / np.sqrt(len(X) - 3)
            rho_0 = np.tanh(F - 2.0 * F_se)
            rho_1 = np.tanh(F + 2.0 * F_se)

            # Warn if the CI spans zero.
            if np.sign(rho_0) != np.sign(rho_1):
                warnings.warn("Confidence interval of the Spearman "
                              "correlation coefficient spans zero. "
                              "Determination of ``increasing`` may be "
                              "suspect.")

        return increasing_bool

    def fit(self, X, y, sample_weight=None, weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            Training data.

        y : array-like, shape=(n_samples,)
            Training target.

        sample_weight : array-like, shape=(n_samples,), optional, default: None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as `transform` needs X to interpolate
        new input data.
        """
        if weight is not None:
            warnings.warn("'weight' was renamed to 'sample_weight' and will "
                          "be removed in 0.16.",
                          DeprecationWarning)
            sample_weight = weight

        X, y, sample_weight = check_arrays(X, y, sample_weight,
                                           sparse_format='dense')
        y = as_float_array(y)
        self._check_fit_data(X, y, sample_weight)

        # Determine increasing if auto-determination requested
        increasing_bool = self._check_increasing(X, y)

        order = np.argsort(X)
        self.X_ = as_float_array(X[order], copy=False)
        self.y_ = isotonic_regression(y[order], sample_weight, self.y_min,
                                      self.y_max, increasing=increasing_bool)
        return self

    def transform(self, T):
        """Transform new data by linear interpolation

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.

        Returns
        -------
        `T_` : array, shape=(n_samples,)
            The transformed data
        """
        T = as_float_array(T)
        if len(T.shape) != 1:
            raise ValueError("X should be a vector")

        f = interpolate.interp1d(self.X_, self.y_, kind='linear',
                                 bounds_error=True)
        return f(T)

    def fit_transform(self, X, y, sample_weight=None, weight=None):
        """Fit model and transform y by linear interpolation.

        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            Training data.

        y : array-like, shape=(n_samples,)
            Training target.

        sample_weight : array-like, shape=(n_samples,), optional, default: None
            Weights. If set to None, all weights will be equal to 1 (equal
            weights).

        Returns
        -------
        `y_` : array, shape=(n_samples,)
            The transformed data.

        Notes
        -----
        X doesn't influence the result of `fit_transform`. It is however stored
        for future use, as `transform` needs X to interpolate new input
        data.
        """
        if weight is not None:
            warnings.warn("'weight' was renamed to 'sample_weight' and will "
                          "be removed in 0.16.",
                          DeprecationWarning)
            sample_weight = weight

        X, y, sample_weight = check_arrays(X, y, sample_weight,
                                           sparse_format='dense')
        y = as_float_array(y)
        self._check_fit_data(X, y, sample_weight)

        # Determine increasing if auto-determination requested
        increasing_bool = self._check_increasing(X, y)

        order = np.lexsort((y, X))
        order_inv = np.argsort(order)
        self.X_ = as_float_array(X[order], copy=False)
        self.y_ = isotonic_regression(y[order], sample_weight, self.y_min,
                                      self.y_max, increasing=increasing_bool)
        return self.y_[order_inv]

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape=(n_samples,)
            Data to transform.

        Returns
        -------
        `T_` : array, shape=(n_samples,)
            Transformed data.
        """
        return self.transform(T)
