# Author: Wei Xue <xuewei4d@gmail.com>
#         Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import warnings

import numpy as np
from scipy.special import gammaln

from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_almost_equal

from sklearn.mixture.bayesian_mixture import _log_dirichlet_norm
from sklearn.mixture.bayesian_mixture import _log_wishart_norm

from sklearn.mixture import BayesianGaussianMixture

from sklearn.mixture.tests.test_gaussian_mixture import RandomData
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import assert_greater_equal

COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']


def test_log_dirichlet_norm():
    rng = np.random.RandomState(0)

    dirichlet_concentration = rng.rand(2)
    expected_norm = (gammaln(np.sum(dirichlet_concentration)) -
                     np.sum(gammaln(dirichlet_concentration)))
    predected_norm = _log_dirichlet_norm(dirichlet_concentration)

    assert_almost_equal(expected_norm, predected_norm)


def test_log_wishart_norm():
    rng = np.random.RandomState(0)

    n_components, n_features = 5, 2
    freedom_degrees = np.abs(rng.rand(n_components)) + 1.
    log_det_precisions_chol = n_features * np.log(range(2, 2 + n_components))

    expected_norm = np.empty(5)
    for k, (freedom_degrees_k, log_det_k) in enumerate(
            zip(freedom_degrees, log_det_precisions_chol)):
        expected_norm[k] = -(
            freedom_degrees_k * (log_det_k + .5 * n_features * np.log(2.)) +
            np.sum(gammaln(.5 * (freedom_degrees_k -
                                 np.arange(0, n_features)[:, np.newaxis])), 0))
    predected_norm = _log_wishart_norm(freedom_degrees,
                                       log_det_precisions_chol, n_features)

    assert_almost_equal(expected_norm, predected_norm)


def test_bayesian_mixture_covariance_type():
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 2
    X = rng.rand(n_samples, n_features)

    covariance_type = 'bad_covariance_type'
    bgmm = BayesianGaussianMixture(covariance_type=covariance_type)
    assert_raise_message(ValueError,
                         "Invalid value for 'covariance_type': %s "
                         "'covariance_type' should be in "
                         "['spherical', 'tied', 'diag', 'full']"
                         % covariance_type,
                         bgmm.fit, X)


def test_bayesian_mixture_weights_prior_initialisation():
    rng = np.random.RandomState(0)
    n_samples, n_components, n_features = 10, 5, 2
    X = rng.rand(n_samples, n_features)

    # Check raise message for a bad value of dirichlet_concentration_prior
    bad_dirichlet_concentration_prior_ = 0.
    bgmm = BayesianGaussianMixture(
        dirichlet_concentration_prior=bad_dirichlet_concentration_prior_)
    assert_raise_message(ValueError,
                         "The parameter 'dirichlet_concentration_prior' "
                         "should be greater than 0., but got %.3f."
                         % bad_dirichlet_concentration_prior_,
                         bgmm.fit, X)

    # Check correct init for a given value of dirichlet_concentration_prior
    dirichlet_concentration_prior = rng.rand()
    bgmm = BayesianGaussianMixture(
        dirichlet_concentration_prior=dirichlet_concentration_prior).fit(X)
    assert_almost_equal(dirichlet_concentration_prior,
                        bgmm.dirichlet_concentration_prior_)

    # Check correct init for the default value of dirichlet_concentration_prior
    bgmm = BayesianGaussianMixture(n_components=n_components).fit(X)
    assert_almost_equal(1. / n_components, bgmm.dirichlet_concentration_prior_)


