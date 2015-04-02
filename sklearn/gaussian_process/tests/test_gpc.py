"""Testing for Gaussian process classification """

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Licence: BSD 3 clause

import numpy as np

from scipy.optimize import approx_fprime

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn.utils.testing import (assert_true, assert_greater,
                                   assert_almost_equal, assert_array_equal)


def f(x):
    return x * np.sin(x)
X = np.atleast_2d(np.linspace(0, 10, 30)).T
X2 = np.atleast_2d([2., 4., 5.5, 6.5, 7.5]).T
y = np.array(f(X).ravel() > 0, dtype=int)


kernels = [RBF(l=0.1), RBF(l=1.0, l_bounds=(1e-3, 1e3)),
           C(1.0, (1e-2, 1e2)) *  RBF(l=1.0, l_bounds=(1e-3, 1e3))]


def test_predict_consistent():
    """ Check binary predict decision has also predicted probability above 0.5.
    """
    for kernel in kernels:
        gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
        assert_array_equal(gpc.predict(X),
                           gpc.predict_proba(X)[:, 1] >=0.5)


def test_lml_improving():
    """ Test that hyperparameter-tuning improves log-marginal likelihood. """
    for kernel in kernels:
        gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
        assert_greater(gpc.log_marginal_likelihood(gpc.kernel_.theta),
                       gpc.log_marginal_likelihood(kernel.theta))


def test_converged_to_local_maximum():
    """ Test that we are in local maximum after hyperparameter-optimization. """
    for kernel in kernels:
        gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

        lml, lml_gradient = gpc.log_marginal_likelihood(gpc.kernel_.theta, True)

        assert_almost_equal(lml_gradient, 0, 2)


def test_lml_gradient():
    """ Compare analytic and numeric gradient of log marginal likelihood. """
    for kernel in kernels:
        gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

        lml, lml_gradient = gpc.log_marginal_likelihood(kernel.theta, True)
        lml_gradient_approx = \
            approx_fprime(kernel.theta,
                          lambda theta: gpc.log_marginal_likelihood(theta,
                                                                    False),
                          1e-10)

        assert_almost_equal(lml_gradient, lml_gradient_approx, 3)

def test_random_starts():
    """
    Test that an increasing number of random-starts of GP fitting only
    increases the log marginal likelihood of the chosen theta.
    """
    n_samples, n_features = 25, 3
    np.random.seed(0)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features) * 2 - 1
    y = (np.sin(X).sum(axis=1) + np.sin(3 * X).sum(axis=1)) > 0

    kernel = C(1.0, (1e-2, 1e2)) \
        * RBF(l=[1e-3] * n_features, l_bounds=[(1e-4, 1e+2)] * n_features)
    last_lml = -np.inf
    for n_restarts_optimizer in range(1, 10):
        gp = GaussianProcessClassifier(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,
            random_state=0,).fit(X, y)
        lml = gp.log_marginal_likelihood(gp.theta_)
        assert_greater(lml, last_lml - np.finfo(np.float32).eps)
        last_lml = lml
