# Author: Christian Osendorfer <osendorf@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Licence: BSD3

import warnings
import numpy as np

from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils import ConvergenceWarning
from sklearn.decomposition import FactorAnalysis


def test_factor_analysis():
    """Test FactorAnalysis ability to recover the data covariance structure
    """
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 20, 5, 3

    # Some random settings for the generative model
    W = rng.randn(n_components, n_features)
    # latent variable of dim 3, 20 of it
    h = rng.randn(n_samples, n_components)
    # using gamma to model different noise variance
    # per component
    noise = rng.gamma(1, size=n_features) * rng.randn(n_samples, n_features)

    # generate observations
    # wlog, mean is 0
    X = np.dot(h, W) + noise
    assert_raises(ValueError, FactorAnalysis, svd_method='foo')
    fa_fail = FactorAnalysis()
    fa_fail.svd_method = 'foo'
    assert_raises(ValueError, fa_fail.fit, X)
    fas = []
    for method in ['randomized', 'lapack']:
        fa = FactorAnalysis(n_components=n_components, svd_method=method)
        fa.fit(X)
        fas.append(fa)

        X_t = fa.transform(X)
        assert_equal(X_t.shape, (n_samples, n_components))

        assert_almost_equal(fa.loglike_[-1], fa.score(X).sum())

        diff = np.all(np.diff(fa.loglike_))
        assert_greater(diff, 0., 'Log likelihood dif not increase')

        # Sample Covariance
        scov = np.cov(X, rowvar=0., bias=1.)

        # Model Covariance
        mcov = fa.get_covariance()
        diff = np.sum(np.abs(scov - mcov)) / W.size
        assert_less(diff, 0.1, "Mean absolute difference is %f" % diff)
        fa = FactorAnalysis(n_components=n_components,
                            noise_variance_init=np.ones(n_features))
        assert_raises(ValueError, fa.fit, X[:, :2])


    f = lambda x, y: np.abs(getattr(x, y))  # sign will not be equal
    fa1, fa2 = fas
    for attr in ['loglike_', 'components_', 'noise_variance_']:
        assert_almost_equal(f(fa1, attr), f(fa2, attr))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        fa1.max_iter = 1
        fa1.verbose = True
        fa1.fit(X)
        assert_true(w[-1].category == ConvergenceWarning)

        warnings.simplefilter('always', DeprecationWarning)
        FactorAnalysis(verbose=1)
        assert_true(w[-1].category == DeprecationWarning)

    fa2 = FactorAnalysis(n_components=n_components,
                         noise_variance_init=np.ones(n_features))
    assert_raises(ValueError, fa2.fit, X[:, :2])

    # Test get_covariance and get_precision with n_components < n_features
    cov = fa.get_covariance()
    precision = fa.get_precision()
    assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    # Test get_covariance and get_precision with n_components == n_features
    fa.n_components = n_features
    fa.fit(X)
    cov = fa.get_covariance()
    precision = fa.get_precision()
    assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    # Test get_covariance and get_precision with n_components == 0
    fa.n_components = 0
    fa.fit(X)
    cov = fa.get_covariance()
    precision = fa.get_precision()
    assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)
