"""Gaussian Mixture Model."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>

import numpy as np

from scipy import linalg

from .base import BaseMixture, _check_shape
from ..externals.six.moves import zip
from ..utils import check_array
from ..utils.validation import check_is_fitted


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0)) or
            any(np.greater(weights, 1))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1 - np.sum(weights)), 0.0):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), 'means')
    return means


def _check_precision_matrix(precision, precision_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if (not np.allclose(precision, precision.T) or
            np.any(np.less_equal(linalg.eigvalsh(precision), .0))):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % precision_type)


def _check_precision_positivity(precision, precision_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % precision_type)


def _check_precisions_full(precisions, precision_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for k, prec in enumerate(precisions):
        _check_precision_matrix(prec, precision_type)


def _check_precisions(precisions, precision_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    precision_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=precision_type is 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[precision_type],
                 '%s covariance' % precision_type)

    check_functions = {'full': _check_precisions_full,
                       'tied': _check_precision_matrix,
                       'diag': _check_precision_positivity,
                       'spherical': _check_precision_positivity}
    check_functions[precision_type](precisions, precision_type)

    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_precisions_cholesky_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    precisions_chol : array, shape (n_components, n_features, n_features)
        The cholesky decomposition of the precision matrix.
    """
    n_features = X.shape[1]
    n_components = means.shape[0]
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariance = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariance.flat[::n_features + 1] += reg_covar
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError("The algorithm has diverged because of too "
                             "few samples per components. "
                             "Try to decrease the number of components, or "
                             "increase reg_covar.")
        precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                     np.eye(n_features),
                                                     lower=True).T
    return precisions_chol


def _estimate_gaussian_precisions_cholesky_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    precisions_chol : array, shape (n_features, n_features)
        The cholesky decomposition of the precision matrix.
    """
    n_features = X.shape[1]
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariances = avg_X2 - avg_means2
    covariances /= X.shape[0]
    covariances.flat[::len(covariances) + 1] += reg_covar
    try:
        cov_chol = linalg.cholesky(covariances, lower=True)
    except linalg.LinAlgError:
        raise ValueError("The algorithm has diverged because of too "
                         "few samples per components. "
                         "Try to decrease the number of components, or "
                         "increase reg_covar.")
    precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                              lower=True).T
    return precisions_chol


def _estimate_gaussian_precisions_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance matrices.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    precisions : array, shape (n_components, n_features)
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    if np.any(np.less_equal(covariances, 0.0)):
        raise ValueError("The algorithm has diverged because of too "
                         "few samples per components. "
                         "Try to decrease the number of components, or "
                         "increase reg_covar.")
    return 1. / covariances


def _estimate_gaussian_precisions_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical covariance matrices.

    Parameters
    ----------
    responsibilities : array-like, shape (n_samples, n_components)

    X : array-like, shape (n_samples, n_features)

    nk : array-like, shape (n_components,)

    means : array-like, shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    precisions : array, shape (n_components,)
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = (avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar).mean(1)
    if np.any(np.less_equal(covariances, 0.0)):
        raise ValueError("The algorithm has diverged because of too "
                         "few samples per components. "
                         "Try to decrease the number of components, or "
                         "increase reg_covar.")
    return 1. / covariances


def _estimate_gaussian_parameters(X, resp, reg_covar, precision_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data array.

    resp : array-like, shape (n_samples, n_features)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to each covariance matrices.

    precision_type : {'full', 'tied', 'diag', 'spherical'}
        The type of covariance matrices.

    Returns
    -------
    nk : array, shape (n_components,)
        The numbers of data samples in the current components.

    means : array, shape (n_components, n_features)
        The centers of the current components.

    precisions : array
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the precision_type.
    """
    estimate_precisions = {
        "full": _estimate_gaussian_precisions_cholesky_full,
        "tied": _estimate_gaussian_precisions_cholesky_tied,
        "diag": _estimate_gaussian_precisions_diag,
        "spherical": _estimate_gaussian_precisions_spherical}

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    precisions = estimate_precisions[precision_type](resp, X, nk, means,
                                                     reg_covar)

    return nk, means, precisions


###############################################################################
# Gaussian mixture probability estimators

def _estimate_log_gaussian_prob_full(X, means, precisions_chol):
    """Estimate the log Gaussian probability for 'full' covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions_chol : array-like, shape (n_components, n_features, n_features)
        Cholesky decompositions of the precision matrices.

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        log_det = -2. * np.sum(np.log(np.diagonal(prec_chol)))
        y = np.dot(X - mu, prec_chol)
        log_prob[:, k] = - .5 * (n_features * np.log(2. * np.pi) + log_det +
                                 np.sum(np.square(y), axis=1))
    return log_prob


def _estimate_log_gaussian_prob_tied(X, means, precision_chol):
    """Estimate the log Gaussian probability for 'tied' covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precision_chol : array-like, shape (n_features, n_features)
        Cholesky decomposition of the precision matrix.

    Returns
    -------
    log_prob : array-like, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))
    log_det = -2. * np.sum(np.log(np.diagonal(precision_chol)))
    for k, mu in enumerate(means):
        y = np.dot(X - mu, precision_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    log_prob = - .5 * (n_features * np.log(2. * np.pi) + log_det + log_prob)
    return log_prob


def _estimate_log_gaussian_prob_diag(X, means, precisions):
    """Estimate the log Gaussian probability for 'diag' covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions : array-like, shape (n_components, n_features)

    Returns
    -------
    log_prob : array-like, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    log_prob = - .5 * (n_features * np.log(2. * np.pi) -
                       np.sum(np.log(precisions), 1) +
                       np.sum((means ** 2 * precisions), 1) -
                       2. * np.dot(X, (means * precisions).T) +
                       np.dot(X ** 2, precisions.T))
    return log_prob


def _estimate_log_gaussian_prob_spherical(X, means, precisions):
    """Estimate the log Gaussian probability for 'spherical' covariance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    means : array-like, shape (n_components, n_features)

    precisions : array-like, shape (n_components, )

    Returns
    -------
    log_prob : array-like, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    log_prob = - .5 * (n_features * np.log(2 * np.pi) -
                       n_features * np.log(precisions) +
                       np.sum(means ** 2, 1) * precisions -
                       2 * np.dot(X, means.T * precisions) +
                       np.outer(np.sum(X ** 2, axis=1), precisions))
    return log_prob


class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    precision_type : {'full', 'tied', 'diag', 'spherical'},
        defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
        'full' (each component has its own general covariance matrix).
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance),

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        log_likelihood average gain is below this threshold.

    reg_covar : float, defaults to 0.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::
        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init: array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init: array-like, optional.
        The user-provided initial precisions, defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'precision_type'::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state: RandomState or an int seed, defaults to None.
        A random number generator instance.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    Attributes
    ----------
    weights_ : array, shape (n_components,)
        The weights of each mixture components.
        `weights_` will not exist before a call to fit.

    means_ : array, shape (n_components, n_features)
        The mean of each mixture component.
        `means_` will not exist before a call to fit.

    precisions_cholesky_ : array
        The cholesky decomposition of each mixture component.
        The shape depends on `precision_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
        `precisions_cholesky_` will not exist before a call to fit.

    precisions_ : array
        The precision of each mixture component.
        The shape depends on `precision_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'
        `precisions_` will not exist before a call to fit.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
        `converged_` will not exist before a call to fit.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
        `n_iter_`  will not exist before a call to fit.
    """

    def __init__(self, n_components=1, precision_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.precision_type = precision_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        if self.precision_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'precision_type': %s "
                             "'precision_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.precision_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, X.shape[1])

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.precision_type,
                                                     self.n_components,
                                                     X.shape[1])

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        weights, means, precisions = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.precision_type)
        weights /= X.shape[0]

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init
        self._precisions = (precisions if self.precisions_init is None
                            else self.precisions_init)

    def _e_step(self, X):
        log_prob_norm, _, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), np.exp(log_resp)

    def _m_step(self, X, resp):
        self.weights_, self.means_, self._precisions = (
            _estimate_gaussian_parameters(X, resp, self.reg_covar,
                                          self.precision_type))
        self.weights_ /= X.shape[0]

    def _estimate_log_prob(self, X):
        estimate_log_prob_functions = {
            "full": _estimate_log_gaussian_prob_full,
            "tied": _estimate_log_gaussian_prob_tied,
            "diag": _estimate_log_gaussian_prob_diag,
            "spherical": _estimate_log_gaussian_prob_spherical
        }
        return estimate_log_prob_functions[self.precision_type](
            X, self.means_, self._precisions)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _check_is_fitted(self):
        check_is_fitted(self, ['weights_', 'means_', '_precisions'])

    def _get_parameters(self):
        return self.weights_, self.means_, self._precisions

    def _set_parameters(self, params):
        self.weights_, self.means_, self._precisions = params

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        if self.precision_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.precision_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.precision_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.precision_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic: float
            The greater the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float
            The greater the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    @property
    def precisions_(self):
        self._check_is_fitted()
        if self.precision_type is 'full':
            return np.array([np.dot(precs, precs.T)
                            for precs in self._precisions])
        elif self.precision_type is 'tied':
            return np.dot(self._precisions, self._precisions.T)
        else:
            return self._precisions

    @property
    def precisions_cholesky_(self):
        self._check_is_fitted()
        if self.precision_type in ['full', 'tied']:
            return self._precisions
        else:
            return np.sqrt(self._precisions)
