import numpy as np

from ...datasets.samples_generator import make_spd_matrix
from ...utils.testing import assert_array_equal
from ...utils.testing import assert_raise_message

rng = np.random.RandomState(0)

COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']


def generate_data(n_samples, n_features, weights, means, covariances,
                  covariance_type):
    X = []
    if covariance_type == 'spherical':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['spherical'])):
            X.append(rng.multivariate_normal(m, c * np.eye(n_features),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'diag':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'tied':
        for k, (w, m) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, covariances['tied'],
                                             int(np.round(w * n_samples))))
    if covariance_type == 'full':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['full'])):
            X.append(rng.multivariate_normal(m, c,
                                             int(np.round(w * n_samples))))

    X = np.vstack(X)
    return X


class RandomData(object):
    def __init__(self, rng):
        n_samples = 500
        n_components = 2
        n_features = 2
        weights = rng.rand(n_components)
        weights = weights / weights.sum()
        means = rng.rand(n_components, n_features) * 50
        covariances = {'spherical': .5 + rng.rand(n_components),
                       'diag': (.5 + rng.rand(n_components, n_features)) ** 2,
                       'tied': make_spd_matrix(n_features, random_state=rng),
                       'full': np.array([make_spd_matrix(
                           n_features, random_state=rng) * .5
                           for _ in range(n_components)])}
        X = dict(zip(COVARIANCE_TYPE, [generate_data(
            n_samples, n_features, weights, means, covariances, cov_type)
            for cov_type in COVARIANCE_TYPE]))
        Y = np.hstack([k * np.ones(np.round(w * n_samples)) for k, w in
                      enumerate(weights)])
        (self.n_samples, self.n_components, self.n_features, self.weights,
         self.means, self.covariances, self.X, self.Y) = (
            n_samples, n_components, n_features, weights, means, covariances,
            X, Y)


RandData = RandomData(rng)


def test_check_X():
    from sklearn.mixture.gmm import _check_X
    n_samples = RandData.n_samples
    n_components = RandData.n_components
    n_features = RandData.n_features

    X_bad_dim = rng.rand(n_samples)
    assert_raise_message(ValueError,
                         'Expected the input data X have 2 dimensions, '
                         'but got %d dimension(s)' % X_bad_dim.ndim,
                         _check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components - 1, n_features)
    assert_raise_message(ValueError,
                         'Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X_bad_dim.shape[0]),
                         _check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components, n_features + 1)
    assert_raise_message(ValueError,
                         'Expected the input data X have %d features, '
                         'but got %d features'
                         % (n_features, X_bad_dim.shape[1]),
                         _check_X, X_bad_dim, n_components, n_features)

    X = rng.rand(n_samples, n_features)
    assert_array_equal(X, _check_X(X, n_components, n_features))
