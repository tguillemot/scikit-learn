"""Base class for mixture models."""
from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np

from .. import cluster
from ..base import BaseEstimator
from ..base import DensityMixin
from ..externals import six
from ..utils import ConvergenceWarning
from ..utils import check_array, check_random_state
from ..utils.extmath import logsumexp


def check_shape(param, param_shape, name):
    """Validate the input data X.

    Raise informative messages otherwise.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components=None, n_features=None):
    """Check the input data X.

    Parameters
    ----------
    X : array-like, (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32])
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def check_weights(weights, desired_shape):
    """Check the user provided 'weights'.

    'weights' are the proportions of components of the mixture

    Parameters
    ----------
    weights : array-like, shape = (n_components,)

    n_components : int

    Returns
    -------
    weights : array, shape = (n_components,)
    """
    # check value
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)

    # check shape
    check_shape(weights, desired_shape, 'weights')

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


class MixtureBase(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    """Base class for mixture models.

    This abstract class specifies the interface by abstract methods and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components, tol, reg_covar,
                 max_iter, n_init, init_params, random_state, warm_start,
                 verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.n_iter_ = 0
        self.n_features = None
        self.converged_ = False

        self._started = False

    def _check_parameters_values(self):
        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

    # e-step functions
    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log weights in EM algorithm, E[ ln pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape = (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X):
        """Estimate the log probabilities ln P(X | Z).

        Log probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape = (n_samples, n_component)
        """
        pass

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log probabilities, ln P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape = (n_features, n_component)
        """
        return self._estimate_log_weights() + self._estimate_log_prob(X)

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape = (n_samples,)
            ln p(X)

        log_prob : array, shape = (n_samples, n_components)
            ln p(X|Z) + log weights

        responsibilities : array, shape = (n_samples, n_components)
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
            resp = np.exp(log_resp)
        return log_prob_norm, weighted_log_prob, resp, log_resp

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound less than `tol`,
        otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        X = _check_X(X, self.n_components)
        self.n_features = X.shape[1]
        self._check_parameters_values()
        self._check_initial_parameters()

        max_log_likelihood = -np.infty
        best_params = self._get_parameters()

        if self.verbose > 10:
            self._log_snapshot = []

        do_init = not(self.warm_start and self._started)
        n_init = self.n_init if do_init else 1

        for init in range(n_init):
            # print verbose msg
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize(X)
                self._started = True

            current_log_likelihood = -np.infty
            self.converged_ = False

            for self.n_iter_ in range(self.max_iter):
                prev_log_likelihood = current_log_likelihood

                # e step
                current_log_likelihood, resp = self._e_step(X)
                change = current_log_likelihood - prev_log_likelihood

                # log the most verbose msg, debug only
                if self.verbose > 10:
                    self._snapshot(X)
                # print verbose msg
                self._print_verbose_msg_iter_end(self.n_iter_, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

                # m step
                self._m_step(X, resp)

            if not self.converged_:
                # compute the log-likelihood of the last m-step
                warnings.warn('Initialization %d is not converged. '
                              'Try different init parameters, '
                              'or increase n_init, tol '
                              'or check for degenerate data.'
                              % (init + 1), ConvergenceWarning)
                current_log_likelihood, _ = self._e_step(X)

            # print verbose msg
            self._print_verbose_msg_init_end(current_log_likelihood)

            if current_log_likelihood > max_log_likelihood:
                # max_log_likelihood is always updated,
                # since we compute the log-likelihood of the initialization
                max_log_likelihood = current_log_likelihood
                best_params = self._get_parameters()
                best_iter_ = self.n_iter_

        self.max_likelihood_ = max_log_likelihood
        self._set_parameters(best_params)
        self.best_iter_ = best_iter_
        return self

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape = (n_samples,)
            log probabilities
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)

        log_prob = self._estimate_weighted_log_prob(X)
        return logsumexp(log_prob, axis=1)

    def score(self, X, y=None):
        """Compute the log likelihood of the model on the given data X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_dimensions)

        Returns
        -------
        log likelihood : float
        """
        X = _check_X(X, None, self.n_features)
        p = self.score_samples(X)
        return p.mean()

    def _initialize(self, X):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        """
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[range(n_samples), label] = 1

        elif self.init_params == 'random':
            resp = random_state.rand(X.shape[0], self.n_components)
            resp = resp / resp.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize_parameters(X, resp)

    @abstractmethod
    def _initialize_parameters(self, X, resp):
        pass

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose == 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose == 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t ll change %.5f" % (
                    n_iter, cur_time - self._iter_prev_time, diff_ll))
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose == 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" %
                  (self.converged_, time() - self._init_prev_time, ll))

    def predict(self, X, y=None):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,) component labels
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data per each component.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        resp : array, shape = (n_samples, n_components)
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)
        _, _, resp, _ = self._estimate_log_prob_resp(X)
        return resp

    @abstractmethod
    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log-likelihood : scalar

        responsibility : array, shape = (n_samples, n_components)
        """
        pass

    @abstractmethod
    def _m_step(self, X, resp):
        pass

    @abstractmethod
    def _check_initial_parameters(self):
        pass

    @abstractmethod
    def _estimate_suffstat(self, X, resp):
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass

    def _snapshot(self, X):
        """For debug."""
        pass
