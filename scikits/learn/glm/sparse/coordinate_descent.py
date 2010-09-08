# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD Style.
"""Implementation of coordinate descent for the Elastic Net with sparse data."""

import numpy as np
from scipy import sparse

from ..base import LinearModel
from . import cd_fast


class ElasticNet(LinearModel):
    """Linear Model trained with L1 and L2 prior as regularizer

    This implementation works on scipy.sparse X and dense coef_.

    rho=1 is the lasso penalty. Currently, rho <= 0.01 is not
    reliable, unless you supply your own sequence of alpha.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term. Defaults to 1.0
    rho : float
        The ElasticNet mixing parameter, with 0 < rho <= 1.
    coef_ : ndarray of shape n_features
        The initial coeffients to warm-start the optimization
    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.
    """

    def __init__(self, alpha=1.0, rho=0.5, coef_=None,
                fit_intercept=True):
        self.alpha = alpha
        self.rho = rho
        self.fit_intercept = fit_intercept
        self.intercept_ = 0.0
        self._set_coef(coef_)

    def _set_coef(self, coef_):
        self.coef_ = coef_
        if coef_ is None:
            self.sparse_coef_ = None
        else:
            n_features = len(coef_)
            # sparse representation of the fitted coef for the predict method
            self.sparse_coef_ = sparse.csr_matrix(coef_)

    def fit(self, X, Y, maxit=1000, tol=1e-4, **params):
        """Fit Elastic Net model with coordinate descent

        X is expected to be a sparse matrix. For maximum effiency, use a
        sparse matrix in csr format (scipy.sparse.csc_matrix)
        """
        self._set_params(**params)
        X = sparse.csc_matrix(X)
        Y = np.asanyarray(Y, dtype=np.float64)
        # do not center data to avoid breaking the sparsity of X

        n_samples, n_features = X.shape[0], X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(n_features, dtype=np.float64)

        alpha = self.alpha * self.rho * n_samples
        beta = self.alpha * (1.0 - self.rho) * n_samples

        # TODO: add support for non centered data
        coef_, self.dual_gap_, self.eps_ = \
                cd_fast.enet_coordinate_descent(self.coef_, alpha, beta,
                                                X.data, X.indices, X.indptr,
                                                Y, maxit, tol)

        # update self.coef_ and self.sparse_coef_ consistently
        self._set_coef(coef_)

        if self.dual_gap_ > self.eps_:
            warnings.warn('Objective did not converge, you might want'
                                'to increase the number of interations')


        # Store explained variance for __str__
        # TODO: implement me!
        #self.explained_variance_ = 0
        # TODO: implement intercept_ fitting

        # return self for chaining fit and predict calls
        return self

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] with the predicted real values
        """
        # np.dot only works correctly if both arguments are sparse matrices
        assert sparse.issparse(X)
        return np.ravel(np.dot(self.sparse_coef_, X.T).todense()
                        + self.intercept_)

