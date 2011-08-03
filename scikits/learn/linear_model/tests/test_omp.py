# Author: Vlad Niculae
# License: BSD style

import warnings

import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_equal, assert_array_almost_equal
from nose.plugins.skip import SkipTest

from .. import orthogonal_mp, orthogonal_mp_gram, OrthogonalMatchingPursuit
from ...utils.fixes import count_nonzero
from ...utils import check_random_state
from ...datasets import generate_sparse_coded_signal

n_samples, n_features, n_nonzero_coefs = 20, 30, 5
y, X, gamma = generate_sparse_coded_signal(3, n_features, n_samples,
                                           n_nonzero_coefs, random_state=0)
G, Xy = np.dot(X.T, X), np.dot(X.T, y)
# this makes X (n_samples, n_features)
# and y (n_samples, 3)


def test_correct_shapes():
    assert_equal(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5).shape,
                 (n_features,))
    assert_equal(orthogonal_mp(X, y, n_nonzero_coefs=5).shape,
                 (n_features, 3))


def test_correct_shapes_gram():
    assert_equal(orthogonal_mp_gram(G, Xy[:, 0], n_nonzero_coefs=5).shape, 
                 (n_features,))
    assert_equal(orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5).shape,
                 (n_features, 3))


def test_n_nonzero_coefs():
    assert count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5)) <= 5
    assert count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5,
                                       precompute_gram=True)) <= 5


def test_eps():
    eps = 0.5
    gamma = orthogonal_mp(X, y[:, 0], eps=eps)
    gamma_gram = orthogonal_mp(X, y[:, 0], eps=eps, precompute_gram=True)
    assert np.sum((y[:, 0] - np.dot(X, gamma)) ** 2) <= eps
    assert np.sum((y[:, 0] - np.dot(X, gamma_gram)) ** 2) <= eps


def test_with_without_gram():
    assert_array_almost_equal(orthogonal_mp(X, y, n_nonzero_coefs=5),
                              orthogonal_mp(X, y, n_nonzero_coefs=5,
                                            precompute_gram=True))


def test_with_without_gram_eps():
    assert_array_almost_equal(orthogonal_mp(X, y, eps=1.),
                              orthogonal_mp(X, y, eps=1., precompute_gram=True)
                              )


def test_unreachable_accuracy():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert_array_almost_equal(orthogonal_mp(X, y, eps=0),
                                  orthogonal_mp(X, y, 
                                              n_nonzero_coefs=n_features))                                                
        assert_array_almost_equal(orthogonal_mp(X, y, eps=0, 
                                                precompute_gram=True),
                                  orthogonal_mp(X, y, precompute_gram=True,
                                                n_nonzero_coefs=n_features))
        assert len(w) > 0  # warnings should be raised


def test_bad_input():
    assert_raises(ValueError, orthogonal_mp, X, y, eps=-1)
    assert_raises(ValueError, orthogonal_mp, X, y, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp, X, y,
                  n_nonzero_coefs=n_features + 1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, eps=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy,
                  n_nonzero_coefs=n_features + 1)


def test_perfect_signal_recovery():
    # XXX: use signal generator
    idx, = gamma[:, 0].nonzero()
    gamma_rec = orthogonal_mp(X, y[:, 0], 5)
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], 5)
    assert_equal(idx, np.flatnonzero(gamma_rec))
    assert_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_estimator_shapes():
    OMP = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    OMP.fit(X, y[:, 0])
    assert_equal(OMP.coef_.shape, (n_features, ))
    assert_equal(OMP.intercept_.shape, ())
    assert count_nonzero(OMP.coef_) <= 5

    OMP.fit(X, y)
    assert_equal(OMP.coef_.shape, (3, n_features))
    assert_equal(OMP.intercept_.shape, (3, ))
    assert count_nonzero(OMP.coef_) <= 3 * 5


def test_identical_regressors():
  newX = X.copy()
  newX[:, 1] = newX[:, 0]
  gamma = np.zeros(n_features)
  gamma[0] = gamma[1] = 1.
  newy = np.dot(newX, gamma)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    orthogonal_mp(newX, newy, 2)
    assert len(w) == 1
