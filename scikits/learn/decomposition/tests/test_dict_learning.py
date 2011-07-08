import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.plugins.skip import SkipTest

from .. import DictionaryLearning, DictionaryLearningOnline


def test_dict_learning_shapes():
    n_samples, n_features = 10, 8
    n_atoms = 5
    X = np.random.randn(n_samples, n_features)
    dico = DictionaryLearning(n_atoms).fit(X)
    assert dico.components_.shape == (n_atoms, n_features)


def test_dict_learning_overcomplete():
    n_samples, n_features = 10, 8
    n_atoms = 12
    X = np.random.randn(n_samples, n_features)
    dico = DictionaryLearning(n_atoms).fit(X)
    assert dico.components_.shape == (n_atoms, n_features)


def test_dict_learning_reconstruction():
    np.random.seed(0)
    n_samples, n_features = 10, 8
    n_atoms = 5
    U = np.zeros((n_samples, n_atoms)).ravel()
    U[np.random.randint(len(U), size=len(U) / n_samples)] = 1.0
    U = U.reshape((n_samples, n_atoms))
    V = np.random.randn(n_atoms, n_features)

    X = np.dot(U, V)
    dico = DictionaryLearning(n_atoms, transform_method='omp')
    code = dico.fit(X).transform(X, eps=0.01)

    assert_array_almost_equal(np.dot(code, dico.components_), X)

    dico.transform_method = 'lasso_lars'
    code = dico.transform(X, alpha=0.01)
    # decimal=1 because lars is sensitive to roundup errors
    assert_array_almost_equal(np.dot(code, dico.components_), X, decimal=1)


def test_dict_learning_split():
    np.random.seed(0)
    n_samples, n_features = 10, 8
    n_atoms = 5
    U = np.zeros((n_samples, n_atoms)).ravel()
    U[np.random.randint(len(U), size=len(U) / n_samples)] = 1.0
    U = U.reshape((n_samples, n_atoms))
    V = np.random.randn(n_atoms, n_features)

    X = np.dot(U, V)
    dico = DictionaryLearning(n_atoms, transform_method='threshold')
    code = dico.fit(X).transform(X, alpha=1)
    dico.split_sign = True
    split_code = dico.transform(X, alpha=1)

    assert_array_equal(split_code[:, :n_atoms] - split_code[:, n_atoms:], code)


def test_dict_learning_online_shapes():
    n_samples, n_features = 10, 8
    n_atoms = 5
    X = np.random.randn(n_samples, n_features)
    dico = DictionaryLearningOnline(n_atoms, n_iter=20).fit(X)
    assert dico.components_.shape == (n_atoms, n_features)


def test_dict_learning_online_overcomplete():
    n_samples, n_features = 10, 8
    n_atoms = 12
    X = np.random.randn(n_samples, n_features)
    dico = DictionaryLearningOnline(n_atoms, n_iter=20).fit(X)
    assert dico.components_.shape == (n_atoms, n_features)


def test_dict_learning_online_partial_fit():
    raise SkipTest
    n_samples, n_features = 10, 8
    n_atoms = 12
    X = np.random.randn(n_samples, n_features)
    V = np.random.randn(n_atoms, n_features)  # random init
    dico1 = DictionaryLearningOnline(n_atoms, n_iter=10, chunk_size=1,
                                     shuffle=False, dict_init=V,
                                     transform_method='threshold').fit(X)
    dico2 = DictionaryLearningOnline(n_atoms, n_iter=1, dict_init=V,
                                     transform_method='threshold')
    for ii, sample in enumerate(X):
        dico2.partial_fit(sample, iter_offset=ii * dico2.n_iter)

    code1 = dico1.transform(X, alpha=1)
    code2 = dico2.transform(X, alpha=1)
    X1 = np.dot(code1, dico1.components_)
    X2 = np.dot(code2, dico2.components_)
    assert_array_equal(X1, X2)
