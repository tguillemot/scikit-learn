"""
Generalized Linear models.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Vincent Michel <vincent.michel@inria.fr>
#
# License: BSD Style.

import numpy as np

from ..base import BaseEstimator, RegressorMixin
from ..metrics import explained_variance_score

###
### TODO: intercept for all models
### We should define a common function to center data instead of
### repeating the same code inside each fit method.
###
### Also, bayesian_ridge_regression and bayesian_regression_ard
### should be squashed into its respective objects.
###

class LinearModel(BaseEstimator, RegressorMixin):
    """Base class for Linear Models"""

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = np.asanyarray(X)
        return np.dot(X, self.coef_) + self.intercept_

    def _explained_variance(self, X, y):
        """Compute explained variance a.k.a. r^2"""
        return explained_variance_score(y, self.predict(X))

    @staticmethod
    def _center_data(X, y, fit_intercept):
        """
        Centers data to have mean zero along axis 0. This is here
        because nearly all Linear Models will want it's data to be
        centered.
        """
        if fit_intercept:
            Xmean = X.mean(axis=0)
            ymean = y.mean()
            X = X - Xmean
            y = y - ymean
        else:
            Xmean = np.zeros(X.shape[1])
            ymean = 0.
        return X, y, Xmean, ymean

    def _set_intercept(self, Xmean, ymean):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.intercept_ = ymean - np.dot(Xmean, self.coef_)
        else:
            self.intercept_ = 0

    def __str__(self):
        if self.coef_ is not None:
            return ("%s \n%s #... Fitted: explained variance=%s" %
                    (repr(self), ' '*len(self.__class__.__name__),
                     self.explained_variance_))
        else:
            return "%s \n#... Not fitted to data" % repr(self)


class LinearRegression(LinearModel):
    """
    Ordinary least squares Linear Regression.

    Attributes
    ----------
    `coef_` : array
        Estimated coefficients for the linear regression problem.

    `intercept_` : array
        Independent term in the linear model.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (numpy.linalg.lstsq) wrapped as a predictor object.

    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y, **params):
        """
        Fit linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values
        fit_intercept : boolean, optional
            wether to calculate the intercept for this model. If set
            to false, no intercept will be used in calculations
            (e.g. data is expected to be already centered).

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params(**params)
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        X, y, Xmean, ymean = LinearModel._center_data(X, y, self.fit_intercept)

        self.coef_, self.residues_, self.rank_, self.singular_ = \
                np.linalg.lstsq(X, y)

        self._set_intercept(Xmean, ymean)
        return self
