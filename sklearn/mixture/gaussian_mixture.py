"""Gaussian Mixture Model."""
import numpy as np

from scipy import linalg

from sklearn.externals.six.moves import zip

from .base import MixtureBase, check_shape, check_weights
from ..utils import check_array
from ..utils.validation import check_is_fitted


def _define_parameter_shape(n_components, n_features, covariance_type):
    """Define the shape of the parameters."""
    cov_shape = {'full': (n_components, n_features, n_features),
                 'tied': (n_features, n_features),
                 'diag': (n_components, n_features),
                 'spherical': (n_components,)}
    param_shape = {'weights': (n_components,),
                   'means': (n_components, n_features),
                   'covariances': cov_shape[covariance_type]}
    return param_shape


def _check_means(means, desired_shape):
    """Validate the user provided 'means'.

    'means' are the center of the components of the mixture in future space

    Parameters
    ----------
    means : array-like, (n_components, n_features)

    n_components : int

    n_features : int

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    check_shape(means, desired_shape, 'means')
    return means


def _check_covars(covars, desired_shape, covariance_type):
    """Validate user provided covariances.

    Parameters
    ----------
    covars : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    n_components : int

    n_features : int

    covariance_type : string

    Returns
    -------
    covars : array
    """
    covars = check_array(covars, dtype=[np.float64, np.float32],
                         ensure_2d=False, allow_nd=(covariance_type == 'full'))
    check_shape(covars, desired_shape, '%s covariance' % covariance_type)

    if covariance_type == 'full':
        for k, cov in enumerate(covars):
            if (not np.allclose(cov, cov.T) or
                    np.any(np.less_equal(linalg.eigvalsh(cov), 0.0))):
                raise ValueError("The component %d of 'full covariance' "
                                 "should be symmetric, positive-definite" % k)

    elif covariance_type == 'tied':
        if (not np.allclose(covars, covars.T) or
                np.any(np.less_equal(linalg.eigvalsh(covars), 0.0))):
            raise ValueError("'tied covariance' should be "
                             "symmetric, positive-definite")

    elif covariance_type in ('diag', 'spherical'):
        if np.any(np.less_equal(covars, 0.0)):
            raise ValueError("'%s covariance' should be positive"
                             % covariance_type)

    return covars


def _estimate_gaussian_suffstat_sk(resp, X, nk, xk, reg_covar, covar_type):
    """Estimate the covariance matrices.

    Parameters
    ----------
    resp : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk : array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    reg_covar : float

    covar_type : string

    Returns
    -------
    Sk : array,
        full : shape = (n_components, n_features, n_features)
        tied : shape = (n_components, n_features)
        diag : shape = (n_components, n_features)
        spherical : shape = (n_components,)
    """
    if covar_type == 'full':
        n_features = X.shape[1]
        n_components = xk.shape[0]
        covars = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - xk[k]
            covars[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covars[k].flat[::n_features + 1] += reg_covar

    elif covar_type == 'tied':
        avg_X2 = np.dot(X.T, X)
        avg_means2 = np.dot(nk * xk.T, xk)
        covars = avg_X2 - avg_means2
        covars /= X.shape[0]
        covars.flat[::len(covars) + 1] += reg_covar

    elif covar_type in ('diag', 'spherical'):
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = xk ** 2
        avg_X_means = xk * np.dot(resp.T, X) / nk[:, np.newaxis]
        covars = avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

        if covar_type == 'spherical':
            covars = covars.mean(axis=1)

    return covars


def _estimate_gaussian_suffstat(X, resp, reg_covar, cov_prec_type):
    """Estimate the sufficient statistics for Gaussian distribution.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The input data array.

    resp : array-like, shape = (n_samples, n_features)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization on covariance matrices.

    cov_prec_type : string
        The type of covariance or precision. It could be one of 'full', 'tied',
        'diag' and 'spherical'.


    Returns
    -------
    nk : array, shape = (n_components,)
        The numbers of data samples in the current components.

    xk : array, shape = (n_components, n_features)
        The centers of the current components.

    Sk : array
        The sample covariances of the current components.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    Sk = _estimate_gaussian_suffstat_sk(resp, X, nk, xk, reg_covar,
                                        cov_prec_type)
    return nk, xk, Sk


def _estimate_log_gaussian_prob(X, means, covars, covar_type):
    """Estimate the log probability of Gaussian distribution.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)

    means : array-like, shape = (n_components, n_features)

    covars : array-like

    covar_type : string

    Returns
    -------
    log_prob : array-like, shape = (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]

    if covar_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, cov) in enumerate(zip(means, covars)):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
            cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = - .5 * (n_features * np.log(2. * np.pi) +
                                     cv_log_det +
                                     np.sum(np.square(cv_sol), axis=1))

    elif covar_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        try:
            cov_chol = linalg.cholesky(covars, lower=True)
        except linalg.LinAlgError:
            raise ValueError("'covars' must be symmetric, positive-definite")
        cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
        for k, mu in enumerate(means):
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = np.sum(np.square(cv_sol), axis=1)
        log_prob = - .5 * (n_features * np.log(2. * np.pi) +
                           cv_log_det +
                           log_prob)

    elif covar_type == 'diag':
        if np.any(np.less_equal(covars, 0.0)):
            raise ValueError("'diag covariance' should be positive")
        log_prob = - .5 * (n_features * np.log(2. * np.pi) +
                           np.sum(np.log(covars), 1) +
                           np.sum((means ** 2 / covars), 1) -
                           2. * np.dot(X, (means / covars).T) +
                           np.dot(X ** 2, (1. / covars).T))

    elif covar_type == 'spherical':
        if np.any(np.less_equal(covars, 0.0)):
            raise ValueError("'spherical covariance' should be positive")
        log_prob = - .5 * (n_features * np.log(2 * np.pi) +
                           n_features * np.log(covars) +
                           np.sum(means ** 2, 1) / covars -
                           2 * np.dot(X, means.T / covars) +
                           np.outer(np.sum(X ** 2, axis=1), 1. / covars))

    return log_prob


class GaussianMixture(MixtureBase):
    """Gaussian Mixture Model.

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, and maximum-likelihood
    estimation of the parameters of a GMM distribution.

    Parameters
    ----------
    n_components : int, defaults to 1.
        Number of mixture components.

    covariance_type : string, defaults to 'full'.
        String describing the type of covariance parameters to
        use.  Must be one of
        'spherical' (each component has its own single variance),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'full' (each component has its own general covariance matrix).

    reg_covar : float, defaults to 0.
        Non-negative regularization to the diagonal of covariance.

    tol : float, defaults to 1e-6.
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold.

    max_iter : int, defaults to 100.
        Number of EM iterations to perform.

    n_init : int, defaults to 1.
        Number of initializations to perform. The best results is kept.

    params : string, defaults to None.
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.

    init_params : string, defaults to 'kmeans'.
        Controls how parameters are initialized unless the parameters are
        provided by users. It should be one of "kmeans", "random", None.
        Defaults to None. If it is not None, the variable responsibilities are
        initialized by the chosen method, which are used to further initialize
        weights, means, and covariances.

    weights : array-like, shape (`n_components`, ), defaults to None.
        User-provided initial weights. If it None, weights
        are initialized by `init_params`.

    means: array-like, shape (`n_components`, `n_features`),
        defaults to None.
        User-provided initial means. If it None, means
        are initialized by `init_params`.

    covars: array-like, defaults to None.
        User-provided initial covariances. Defaults to None. If it None, covars
        are initialized by `init_params`. The shape
        depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state: RandomState or an int seed, defaults to None.
        A random number generator instance.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the log probability and the time needed
        for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    """

    def __init__(self, n_components=1, covariance_type='full',
                 random_state=None, tol=1e-6, reg_covar=0.,
                 max_iter=100, n_init=1, init_params='kmeans',
                 weights=None, means=None, covars=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GaussianMixture, self).__init__(
            n_components=n_components, random_state=random_state, tol=tol,
            reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.covariance_type = covariance_type

        self.weights_init = weights
        self.weights_ = None
        self.means_init = means
        self.means_ = None
        self.covars_init = covars
        self.covars_ = None

    def _check_parameters_values(self):
        super(GaussianMixture, self)._check_parameters_values()

        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
                raise ValueError("Invalid value for 'covariance_type': %s "
                                 "'covariance_type' should be in "
                                 "['spherical', 'tied', 'diag', 'full']"
                                 % self.covariance_type)

    def _estimate_suffstat(self, X, resp):
        return _estimate_gaussian_suffstat(X, resp, self.reg_covar,
                                           self.covariance_type)

    def _check_initial_parameters(self):
        param_shape = _define_parameter_shape(
            self.n_components, self.n_features, self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = check_weights(
                self.weights_init, param_shape['weights'])

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, param_shape['means'])

        if self.covars_init is not None:
            self.covars_init = _check_covars(
                self.covars_init, param_shape['covariances'],
                self.covariance_type)

    def _initialize_parameters(self, X, resp):
        nk, xk, Sk = self._estimate_suffstat(X, resp)
        self.weights_ = (self._estimate_weights(X, nk)
                         if self.weights_init is None
                         else self.weights_init)
        self.means_ = (self._estimate_means(xk) if self.means_init is None
                       else self.means_init)
        self.covars_ = (self._estimate_covariances(Sk)
                        if self.covars_init is None
                        else self.covars_init)

    def _estimate_weights(self, X, nk):
        return nk / X.shape[0]

    def _estimate_means(self, xk):
        return xk

    def _estimate_covariances(self, Sk):
        return Sk

    def _m_step(self, X, resp):
        nk, xk, Sk = self._estimate_suffstat(X, resp)
        self.weights_ = self._estimate_weights(X, nk)
        self.means_ = self._estimate_means(xk)
        self.covars_ = self._estimate_covariances(Sk)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(X, self.means_, self.covars_,
                                           self.covariance_type)

    def _e_step(self, X):
        log_prob_norm, _, resp, _ = self._estimate_log_prob_resp(X)
        self._log_likelihood = np.sum(log_prob_norm)
        return self._log_likelihood, resp

    def _check_is_fitted(self):
        check_is_fitted(self, 'weights_')
        check_is_fitted(self, 'means_')
        check_is_fitted(self, 'covars_')

    def _get_parameters(self):
        return self.weights_, self.means_, self.covars_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covars_ = params

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.n_features
        if self.covariance_type == 'full':
            cov_params = self.n_components * ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * ndim
        elif self.covariance_type == 'tied':
            cov_params = ndim * (ndim + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the greater the better)
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
        aic: float (the greater the better)
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def _snapshot(self, X):
        """For debug."""
        self._log_snapshot.append((self.weights_, self.means_, self.covars_,
                                   self._log_likelihood))
