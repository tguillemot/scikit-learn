# Author: Vlad Niculae
# Licence: BSD 3 clause

import warnings

import numpy as np

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater

from sklearn.linear_model import (orthogonal_mp, orthogonal_mp_gram,
                                  OrthogonalMatchingPursuit,
                                  OrthogonalMatchingPursuitCV)
from sklearn.utils.fixes import count_nonzero
from sklearn.datasets import make_sparse_coded_signal

n_samples, n_features, n_nonzero_coefs, n_targets = 20, 30, 5, 3
y, X, gamma = make_sparse_coded_signal(n_targets, n_features, n_samples,
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
    assert_true(count_nonzero(orthogonal_mp(X, y[:, 0],
                              n_nonzero_coefs=5)) <= 5)
    assert_true(count_nonzero(orthogonal_mp(X, y[:, 0], n_nonzero_coefs=5,
                                            precompute=True)) <= 5)


def test_tol():
    tol = 0.5
    gamma = orthogonal_mp(X, y[:, 0], tol=tol)
    gamma_gram = orthogonal_mp(X, y[:, 0], tol=tol, precompute=True)
    assert_true(np.sum((y[:, 0] - np.dot(X, gamma)) ** 2) <= tol)
    assert_true(np.sum((y[:, 0] - np.dot(X, gamma_gram)) ** 2) <= tol)


def test_with_without_gram():
    assert_array_almost_equal(
        orthogonal_mp(X, y, n_nonzero_coefs=5),
        orthogonal_mp(X, y, n_nonzero_coefs=5, precompute=True))


def test_with_without_gram_tol():
    assert_array_almost_equal(
        orthogonal_mp(X, y, tol=1.),
        orthogonal_mp(X, y, tol=1., precompute=True))


def test_unreachable_accuracy():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert_array_almost_equal(
            orthogonal_mp(X, y, tol=0),
            orthogonal_mp(X, y, n_nonzero_coefs=n_features))

        assert_array_almost_equal(
            orthogonal_mp(X, y, tol=0, precompute=True),
            orthogonal_mp(X, y, precompute=True,
                          n_nonzero_coefs=n_features))
        assert_greater(len(w), 0)  # warnings should be raised


def test_bad_input():
    assert_raises(ValueError, orthogonal_mp, X, y, tol=-1)
    assert_raises(ValueError, orthogonal_mp, X, y, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp, X, y,
                  n_nonzero_coefs=n_features + 1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, tol=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy, n_nonzero_coefs=-1)
    assert_raises(ValueError, orthogonal_mp_gram, G, Xy,
                  n_nonzero_coefs=n_features + 1)


def test_perfect_signal_recovery():
    # XXX: use signal generator
    idx, = gamma[:, 0].nonzero()
    gamma_rec = orthogonal_mp(X, y[:, 0], 5)
    gamma_gram = orthogonal_mp_gram(G, Xy[:, 0], 5)
    assert_array_equal(idx, np.flatnonzero(gamma_rec))
    assert_array_equal(idx, np.flatnonzero(gamma_gram))
    assert_array_almost_equal(gamma[:, 0], gamma_rec, decimal=2)
    assert_array_almost_equal(gamma[:, 0], gamma_gram, decimal=2)


def test_estimator():
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y[:, 0])
    assert_equal(omp.coef_.shape, (n_features,))
    assert_equal(omp.intercept_.shape, ())
    assert_true(count_nonzero(omp.coef_) <= n_nonzero_coefs)

    omp.fit(X, y)
    assert_equal(omp.coef_.shape, (n_targets, n_features))
    assert_equal(omp.intercept_.shape, (n_targets,))
    assert_true(count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs)

    omp.set_params(fit_intercept=False, normalize=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        omp.fit(X, y[:, 0], Gram=G, Xy=Xy[:, 0])
        assert_equal(omp.coef_.shape, (n_features,))
        assert_equal(omp.intercept_, 0)
        assert_true(count_nonzero(omp.coef_) <= n_nonzero_coefs)
        assert_true(len(w) == 2)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        omp.fit(X, y, Gram=G, Xy=Xy)
        assert_equal(omp.coef_.shape, (n_targets, n_features))
        assert_equal(omp.intercept_, 0)
        assert_true(count_nonzero(omp.coef_) <= n_targets * n_nonzero_coefs)
        assert_true(len(w) == 2)


def test_scaling_with_gram():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Use only 1 nonzero coef to be faster and to avoid warnings
        omp1 = OrthogonalMatchingPursuit(n_nonzero_coefs=1,
                                         fit_intercept=False, normalize=False)
        omp2 = OrthogonalMatchingPursuit(n_nonzero_coefs=1,
                                         fit_intercept=True, normalize=False)
        omp3 = OrthogonalMatchingPursuit(n_nonzero_coefs=1,
                                         fit_intercept=False, normalize=True)
        omp1.fit(X, y, Gram=G)
        omp1.fit(X, y, Gram=G, Xy=Xy)
        assert_true(len(w) == 3)
        omp2.fit(X, y, Gram=G)
        assert_true(len(w) == 5)
        omp2.fit(X, y, Gram=G, Xy=Xy)
        assert_true(len(w) == 8)
        omp3.fit(X, y, Gram=G)
        assert_true(len(w) == 10)
        omp3.fit(X, y, Gram=G, Xy=Xy)
        assert_true(len(w) == 13)


def test_identical_regressors():
    newX = X.copy()
    newX[:, 1] = newX[:, 0]
    gamma = np.zeros(n_features)
    gamma[0] = gamma[1] = 1.
    newy = np.dot(newX, gamma)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        orthogonal_mp(newX, newy, 2)
        assert_true(len(w) == 1)


def test_swapped_regressors():
    gamma = np.zeros(n_features)
    # X[:, 21] should be selected first, then X[:, 0] selected second,
    # which will take X[:, 21]'s place in case the algorithm does
    # column swapping for optimization (which is the case at the moment)
    gamma[21] = 1.0
    gamma[0] = 0.5
    new_y = np.dot(X, gamma)
    new_Xy = np.dot(X.T, new_y)
    gamma_hat = orthogonal_mp(X, new_y, 2)
    gamma_hat_gram = orthogonal_mp_gram(G, new_Xy, 2)
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])
    assert_array_equal(np.flatnonzero(gamma_hat_gram), [0, 21])


def test_no_atoms():
    y_empty = np.zeros_like(y)
    Xy_empty = np.dot(X.T, y_empty)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gamma_empty = orthogonal_mp(X, y_empty, 1)
        gamma_empty_gram = orthogonal_mp_gram(G, Xy_empty, 1)
    assert_equal(np.all(gamma_empty == 0), True)
    assert_equal(np.all(gamma_empty_gram == 0), True)


def test_omp_path():
    path = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp(X, y, n_nonzero_coefs=5, return_path=False)
    assert_equal(path.shape, (n_features, n_targets, 5))
    assert_array_almost_equal(path[:, :, -1], last)
    path = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=True)
    last = orthogonal_mp_gram(G, Xy, n_nonzero_coefs=5, return_path=False)
    assert_equal(path.shape, (n_features, n_targets, 5))
    assert_array_almost_equal(path[:, :, -1], last)


def test_omp_cv():
    y_ = y[:, 0]
    gamma_ = gamma[:, 0]
    ompcv = OrthogonalMatchingPursuitCV(normalize=True, fit_intercept=False,
                                        max_iter=10, cv=5)
    ompcv.fit(X, y_)
    assert_equal(ompcv.n_nonzero_coefs_, n_nonzero_coefs)
    assert_array_almost_equal(ompcv.coef_, gamma_)
    omp = OrthogonalMatchingPursuit(normalize=True, fit_intercept=False,
                                    n_nonzero_coefs=ompcv.n_nonzero_coefs_)
    omp.fit(X, y_)
    assert_array_almost_equal(ompcv.coef_, omp.coef_)
