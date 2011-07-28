# Author: Vlad Niculae
# License: BSD style

import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_equal, assert_array_almost_equal
from nose.plugins.skip import SkipTest

from .. import orthogonal_mp, orthogonal_mp_gram


def generate_data(n_samples, n_features):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    X /= np.sqrt(np.sum((X ** 2), axis=0))
    gamma = np.random.randn(n_features)
    return X, np.dot(X, gamma)


def test_correct_shapes():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    assert_equal(orthogonal_mp(X, y, n_nonzero_coefs=6).shape, (n_features,))
    y = np.random.randn(len(y), 3)
    assert_equal(orthogonal_mp(X, y, n_nonzero_coefs=6).shape, (n_features, 3))


def test_correct_shapes_gram():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    G, Xy = np.dot(X.T, X), np.dot(X.T, y)
    assert_equal(orthogonal_mp_gram(G, Xy, n_nonzero_coefs=6).shape,
                 (n_features,))
    y = np.random.randn(n_samples, 3)
    Xy = np.dot(X.T, y)
    assert_equal(orthogonal_mp_gram(G, Xy, n_nonzero_coefs=6).shape,
                 (n_features, 3))


def test_n_features():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    assert_equal(np.sum(orthogonal_mp(X, y, n_nonzero_coefs=6) != 0), 6)
    assert_equal(np.sum(orthogonal_mp(X, y, n_nonzero_coefs=6,
                                      compute_gram=True) != 0), 6)


def test_eps():
    n_samples, n_features = 10, 15
    eps = 0.5
    X, y = generate_data(n_samples, n_features)
    gamma = orthogonal_mp(X, y, eps=eps)
    gamma_gram = orthogonal_mp(X, y, eps=eps, compute_gram=True)
    assert(np.sum((y - np.dot(X, gamma)) ** 2) <= eps)
    assert(np.sum((y - np.dot(X, gamma_gram)) ** 2) <= eps)


def test_with_without_gram():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    assert_array_almost_equal(orthogonal_mp(X, y, n_nonzero_coefs=6),
                              orthogonal_mp(X, y, n_nonzero_coefs=6,
                                            compute_gram=True))


def test_with_without_gram_eps():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    assert_array_almost_equal(orthogonal_mp(X, y, eps=1.),
                              orthogonal_mp(X, y, eps=1., compute_gram=True))


def test_unreachable_accuracy():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    assert_array_almost_equal(orthogonal_mp(X, y, eps=0),
                              orthogonal_mp(X, y, n_nonzero_coefs=n_features))


def test_bad_input():
    n_samples, n_features = 10, 15
    X, y = generate_data(n_samples, n_features)
    G, Xy = np.dot(X.T, X), np.dot(X.T, y)
    assert_raises(ValueError, orthogonal_mp, X, y, eps=-1)
    assert_raises(ValueError, orthogonal_mp, X, y, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp, X, y,
                  n_nonzero_coefs=n_features + 1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, eps=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy,
                  n_nonzero_coefs=n_features + 1)


def test_perfect_signal_recovery():
    n_samples, n_features = 10, 15
    n_atoms = 4
    X, _ = generate_data(n_samples, n_features)
    gamma = np.zeros(n_features)
    idx = np.arange(n_features)
    np.random.shuffle(idx)
    idx = idx[:n_atoms]
    gamma[idx] = np.random.normal(0, 5, n_atoms)
    y = np.dot(X, gamma)
    G, Xy = np.dot(X.T, X), np.dot(X.T, y)
    assert_array_almost_equal(orthogonal_mp(X, y, n_atoms), gamma, decimal=1)
    assert_array_almost_equal(orthogonal_mp_gram(G, Xy, n_atoms), gamma,
                              decimal=1)
