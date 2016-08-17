"""Bayesian Gaussian Mixture Model."""
# Author: Wei Xue <xuewei4d@gmail.com>
#         Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.special import digamma, gammaln

from .base import BaseMixture, _check_shape
from .gaussian_mixture import _check_precision_matrix
from .gaussian_mixture import _check_precision_positivity
from .gaussian_mixture import _compute_log_det_cholesky
from .gaussian_mixture import _compute_precision_cholesky
from .gaussian_mixture import _estimate_gaussian_parameters
from .gaussian_mixture import _estimate_log_gaussian_prob
from ..utils import check_array
from ..utils.validation import check_is_fitted


def _log_dirichlet_norm(dirichlet_concentration):
    """Compute the log of the Dirichlet distribution normalization term.

    Parameters
    ----------
    dirichlet_concentration : array-like, shape (n_samples,)
        The parameters values of the Dirichlet distribution.

    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return (gammaln(np.sum(dirichlet_concentration)) -
            np.sum(gammaln(dirichlet_concentration)))


def _log_wishart_norm(freedom_degrees, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    freedom_degrees : array-like, shape (n_components,)
        The parameters values of the Whishart distribution.

    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the cholesky decomposition of the precision matrix.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like, shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(freedom_degrees * log_det_precisions_chol +
             freedom_degrees * n_features * .5 * np.log(2.) +
             np.sum(gammaln(.5 * (freedom_degrees -
                                  np.arange(0, n_features)[:,
                                                           np.newaxis])), 0))


class BayesianGaussianMixture(BaseMixture):
    """Variational estimation of a Gaussian mixture.

    Representation of a variational inference for a Bayesian Gaussian mixture
    model probability distribution. This class allows to do inference of an
    approximate posterior distribution over the parameters of a Gaussian
    mixture distribution. The number of components can be inferred from the
    data.

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components: int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'},
        defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::
        'full' (each component has its own general covariance matrix).
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance),

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    max_iter : int, default to 100.
        The number of EM iterations to perform.

    n_init : int, default to 1.
        The number of initializations to perform. The best results is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::
        'kmeans' : responsibilities are initialized using kmeans.
        'random' : responsibilities are initialized randomly.

    dirichlet_concentration_prior : float, optional.
        The user-provided dirichlet concentration prior parameter of the
        Dirichlet distribution. The higher concentration puts more mass in the
        center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex. The value of the parameter must be greater than 0. If is None,
        the dirichlet concentration prior is set to `1. / n_components`.

    mean_precision_prior : float, optional.
        The user-provided mean precision prior parameter of the Gaussian
        distribution. Controls the extend to where means can be placed. Smaller
        values concentrates the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, mean precision prior is set to 1.

    mean_prior : array-like, shape (`n_features`,), optional
        The user-provided mean prior of the Gaussian distribution.
        If it is None, the mean prior is set to the mean of X.

    freedom_degrees_prior : float, optional.
        The user-provided number of degrees of freedom prior parameter of the
        covariance distribution. If it is None, the prior of the number of
        degrees of freedom is set to `n_features`.

    covariance_prior : float or array-like, optional
        The user-provided covariance prior of the covariance distribution.
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::
            (`n_features`, `n_features`) if 'full',
            (`n_features`, `n_features`) if 'tied',
            (`n_features`)               if 'diag',
            float                        if 'spherical'

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
    weights_ : array-like, shape (`n_components`,)
        The weights of each mixture components.

    means_ : array-like, shape (`n_components`, `n_features`)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of EM.

    dirichlet_concentration_prior_ : float
        The dirichlet concentration prior parameter of the Dirichlet
        distribution used during the fit process. The higher
        concentration puts more mass in the center and will lead to more
        components being active, while a lower concentration parameter will
        lead to more mass at the edge of the simplex.

    dirichlet_concentration_ : array-like, shape (`n_components`, )
        The dirichlet concentration parameters of the Dirichlet distribution of
        each component.

    mean_precision_prior : float
        The mean precision prior parameters of the Gaussian distributions of
        the means used during the fit process. Controls the extend to where
        means can be placed. Smaller values concentrates the means of each
        clusters around `mean_prior`.

    mean_precision_ : array-like, shape (`n_components`, )
        The mean precision parameters of the Gaussian distributions of
        the means.

    means_prior_ : array-like, shape (`n_features`,)
        The mean prior of each mixture component.

    freedom_degrees_prior_ : float
        The prior of the number of degrees of freedom parameters of the
        covariance distribution.

    freedom_degrees_ : array-like, shape (`n_components`,)
        The number of degrees of freedom parameters of the covariance
        distribution.

    covariance_prior_ : float or array-like
        The covariance prior of the covariance distribution.
        The shape depends on `covariance_type`::
            (`n_features`, `n_features`) if 'full',
            (`n_features`, `n_features`) if 'tied',
            (`n_features`)               if 'diag',
            float                        if 'spherical'

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.
    """

    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 max_iter=100, n_init=1, init_params='kmeans',
                 dirichlet_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 freedom_degrees_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=20):
        super(BayesianGaussianMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=0,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.dirichlet_concentration_prior = dirichlet_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.freedom_degrees_prior = freedom_degrees_prior
        self.covariance_prior = covariance_prior

    def _check_parameters(self, X):
        """Check the parameters are well defined.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)
        self._check_weights_parameters()
        self._check_means_parameters(X)
        self._check_precision_parameters(X)
        self._checkcovariance_prior_parameter(X)

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""
        if self.dirichlet_concentration_prior is None:
            self.dirichlet_concentration_prior_ = 1. / self.n_components
        elif self.dirichlet_concentration_prior > 0.:
            self.dirichlet_concentration_prior_ = (
                self.dirichlet_concentration_prior)
        else:
            raise ValueError("The parameter 'dirichlet_concentration_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.dirichlet_concentration_prior)

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.
        elif self.mean_precision_prior > 0.:
            self.mean_precision_prior_ = self.mean_precision_prior
        else:
            raise ValueError("The parameter 'mean_precision_prior' should be "
                             "greater than 0., but got %.3f."
                             % self.mean_precision_prior)

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(self.mean_prior,
                                           dtype=[np.float64, np.float32],
                                           ensure_2d=False)
            _check_shape(self.mean_prior_, (n_features, ), 'means')

    def _check_precision_parameters(self, X):
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.freedom_degrees_prior is None:
            self.freedom_degrees_prior_ = n_features
        elif self.freedom_degrees_prior > n_features - 1.:
            self.freedom_degrees_prior_ = self.freedom_degrees_prior
        else:
            raise ValueError("The parameter 'freedom_degrees_prior' "
                             "should be greater than %d, but got %.3f."
                             % (n_features - 1, self.freedom_degrees_prior))

    def _checkcovariance_prior_parameter(self, X):
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                'full': np.atleast_2d(np.cov(X.T)),
                'tied': np.atleast_2d(np.cov(X.T)),
                'diag': np.var(X, axis=0, ddof=1),
                'spherical': np.var(X, axis=0, ddof=1).mean()
            }[self.covariance_type]

        elif self.covariance_type in ['full', 'tied']:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features, n_features),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_matrix(self.covariance_prior_,
                                    self.covariance_type)
        elif self.covariance_type == 'diag':
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            _check_shape(self.covariance_prior_, (n_features,),
                         '%s covariance_prior' % self.covariance_type)
            _check_precision_positivity(self.covariance_prior_,
                                        self.covariance_type)
        # spherical case
        elif self.covariance_prior > 0.:
            self.covariance_prior_ = self.covariance_prior
        else:
            raise ValueError("The parameter 'spherical covariance_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.covariance_prior)

    def _initialize(self, X, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        nk, xk, sk = _estimate_gaussian_parameters(X, resp, self.reg_covar,
                                                   self.covariance_type)

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_weights(self, nk):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)
        """
        self.dirichlet_concentration_ = (
            self.dirichlet_concentration_prior_ + nk)

    def _estimate_means(self, nk, xk):
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)
        """
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = ((self.mean_precision_prior_ * self.mean_prior_ +
                        nk[:, np.newaxis] * xk) /
                       self.mean_precision_[:, np.newaxis])

    def _estimate_precisions(self, nk, xk, sk):
        """Estimate the precisions parameters of the precision distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)

        sk : array-like
            The shape depends of `covariance_type`:
            'full' : (n_components, n_features, n_features)
            'tied' : (n_features, n_features)
            'diag' : (n_components, n_features)
            'spherical' : (n_components,)
        """
        {"full": self._estimate_wishart_full,
         "tied": self._estimate_wishart_tied,
         "diag": self._estimate_wishart_diag,
         "spherical": self._estimate_wishart_spherical
         }[self.covariance_type](nk, xk, sk)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

    def _estimate_wishart_full(self, nk, xk, sk):
        """Estimate the full Wishart distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)

        sk : array-like, shape (n_components, n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `freedom_degrees_k = freedom_degrees_0 + Nk` is the correct formula
        self.freedom_degrees_ = self.freedom_degrees_prior_ + nk

        self.covariances_ = np.empty((self.n_components, n_features,
                                      n_features))

        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covariances_[k] = (self.covariance_prior_ + nk[k] * sk[k] +
                                    nk[k] * self.mean_precision_prior_ /
                                    self.mean_precision_[k] * np.outer(diff,
                                                                       diff))

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.freedom_degrees_[:, np.newaxis, np.newaxis]

    def _estimate_wishart_tied(self, nk, xk, sk):
        """Estimate the tied Wishart distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)

        sk : array-like, shape (n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `freedom_degrees_k = freedom_degrees_0 + Nk` is the correct formula
        self.freedom_degrees_ = (
            self.freedom_degrees_prior_ + nk.sum() / self.n_components)

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_ + sk * nk.sum() / self.n_components +
            self.mean_precision_prior_ / self.n_components * np.dot(
                (nk / self.mean_precision_) * diff.T, diff))

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.freedom_degrees_

    def _estimate_wishart_diag(self, nk, xk, sk):
        """Estimate the diag Wishart distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)

        sk : array-like, shape (n_components, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `freedom_degrees_k = freedom_degrees_0 + Nk` is the correct formula
        self.freedom_degrees_ = self.freedom_degrees_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_ + nk[:, np.newaxis] * (
                sk + (self.mean_precision_prior_ /
                      self.mean_precision_)[:, np.newaxis] * np.square(diff)))

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.freedom_degrees_[:, np.newaxis]

    def _estimate_wishart_spherical(self, nk, xk, sk):
        """Estimate the spherical Wishart distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        nk : array-like, shape (n_components,)

        xk : array-like, shape (n_components, n_features)

        sk : array-like, shape (n_components,)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `freedom_degrees_k = freedom_degrees_0 + Nk` is the correct formula
        self.freedom_degrees_ = self.freedom_degrees_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_ + nk * (
                sk + self.mean_precision_prior_ / self.mean_precision_ *
                np.mean(np.square(diff), 1)))

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.freedom_degrees_

    def _check_is_fitted(self):
        check_is_fitted(self, ['dirichlet_concentration_', 'mean_precision_',
                               'means_', 'freedom_degrees_',
                               'covariances_', 'precisions_',
                               'precisions_cholesky_'])

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log-responsibility : array, shape (n_samples, n_components)
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm, log_resp

    def _estimate_log_weights(self):
        return (digamma(self.dirichlet_concentration_) -
                digamma(np.sum(self.dirichlet_concentration_)))

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.freedom_degrees_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type) -
            .5 * n_features * np.log(self.freedom_degrees_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.freedom_degrees_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to decrease at
        each iteration.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log-resp : array, shape (n_samples, n_components)

        log_prob_norm : float

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        n_features, = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.freedom_degrees_)` because
        # the precision matrix is normalized.
        log_det_precisions_chol = (_compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features) -
            .5 * n_features * np.log(self.freedom_degrees_))

        if self.covariance_type == 'tied':
            log_wishart = self.n_components * np.float64(_log_wishart_norm(
                self.freedom_degrees_, log_det_precisions_chol, n_features))
        else:
            log_wishart = np.sum(_log_wishart_norm(
                self.freedom_degrees_, log_det_precisions_chol, n_features))

        return (-np.sum(np.exp(log_resp) * log_resp) - log_wishart -
                _log_dirichlet_norm(self.dirichlet_concentration_) -
                0.5 * n_features * np.sum(np.log(self.mean_precision_)))

    def _get_parameters(self):
        return (self.dirichlet_concentration_,
                self.mean_precision_, self.means_,
                self.freedom_degrees_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.dirichlet_concentration_, self.mean_precision_, self.means_,
         self.freedom_degrees_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        self. weights_ = (self.dirichlet_concentration_ /
                          np.sum(self.dirichlet_concentration_))

        if self.covariance_type == 'full':
            self.precisions_ = np.array([
                np.dot(prec_chol, prec_chol.T)
                for prec_chol in self.precisions_cholesky_])

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2
