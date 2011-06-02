import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from scikits.learn.preprocessing import Binarizer
from scikits.learn.preprocessing import KernelCenterer
from scikits.learn.preprocessing import LabelBinarizer
from scikits.learn.preprocessing import SampleNormalizer
from scikits.learn.preprocessing import Scaler
from scikits.learn.preprocessing import scale

from scikits.learn import datasets
from scikits.learn.linear_model.stochastic_gradient import SGDClassifier

np.random.seed(0)

iris = datasets.load_iris()


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def test_scaler():
    """Test scaling of dataset along all axis"""
    # First test with 1D data
    X = np.random.randn(5)

    scaler = Scaler()
    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert_array_almost_equal(X_scaled.mean(axis=0), 0.0)
    assert_array_almost_equal(X_scaled.std(axis=0), 1.0)

    X = np.random.randn(4, 5)

    scaler = Scaler()
    X_scaled = scaler.fit(X).transform(X, copy=False)
    assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), 5 * [1.0])
    # Check that X has not been copied
    assert X_scaled is X

    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_array_almost_equal(X_scaled.mean(axis=0), 5 * [0.0])
    assert_array_almost_equal(X_scaled.std(axis=0), 5 * [1.0])
    # Check that X has not been copied
    assert X_scaled is not X

    X_scaled = scale(X, axis=1, with_std=False)
    assert_array_almost_equal(X_scaled.mean(axis=1), 4 * [0.0])
    X_scaled = scale(X, axis=1, with_std=True)
    assert_array_almost_equal(X_scaled.std(axis=1), 4 * [1.0])
    # Check that the data hasn't been modified
    assert X_scaled is not X


def test_normalizer_l1():
    np.random.seed(0)
    X_orig = np.random.randn(4, 5)
    X_orig[3, :] = 0.0

    for init in (np.array, sp.csr_matrix):

        X = init(X_orig.copy())

        normalizer = SampleNormalizer(norm='l1', copy=True)
        X_norm = normalizer.transform(X)
        assert X_norm is not X
        X_norm1 = toarray(X_norm)

        normalizer = SampleNormalizer(norm='l1', copy=False)
        X_norm = normalizer.transform(X)
        assert X_norm is X
        X_norm2 = toarray(X_norm)

        for X_norm in (X_norm1, X_norm2):
            row_sums = np.abs(X_norm).sum(axis=1)
            for i in range(3):
                assert_almost_equal(row_sums[i], 1.0)
            assert_almost_equal(row_sums[3], 0.0)


def test_normalizer_l2():
    np.random.seed(0)
    X_orig = np.random.randn(4, 5)
    X_orig[3, :] = 0.0

    for init in (np.array, sp.csr_matrix):

        X = init(X_orig.copy())

        normalizer = SampleNormalizer(norm='l2', copy=True)
        X_norm1 = normalizer.transform(X)
        assert X_norm1 is not X
        X_norm1 = toarray(X_norm1)

        normalizer = SampleNormalizer(norm='l2', copy=False)
        X_norm2 = normalizer.transform(X)
        assert X_norm2 is X
        X_norm2 = toarray(X_norm2)

        for X_norm in (X_norm1, X_norm2):
            for i in xrange(3):
                assert_almost_equal(la.norm(X_norm[i]), 1.0)
            assert_almost_equal(la.norm(X_norm[3]), 0.0)


def test_binarizer():
    X_ = np.array([[1, 0, 5], [2, 3, 0]])

    for init in (np.array, sp.csr_matrix):

        X = init(X_.copy())

        binarizer = Binarizer(threshold=2.0, copy=True)
        X_bin = toarray(binarizer.transform(X))
        assert_equal(np.sum(X_bin == 0), 4)
        assert_equal(np.sum(X_bin == 1), 2)

        binarizer = Binarizer(copy=True)
        X_bin = binarizer.transform(X)
        assert X_bin is not X
        X_bin = toarray(X_bin)
        assert_equal(np.sum(X_bin == 0), 2)
        assert_equal(np.sum(X_bin == 1), 4)

        binarizer = Binarizer(copy=False)
        X_bin = binarizer.transform(X)
        assert X_bin is X
        X_bin = toarray(X_bin)
        assert_equal(np.sum(X_bin == 0), 2)
        assert_equal(np.sum(X_bin == 1), 4)


def test_label_binarizer():
    lb = LabelBinarizer()

    # two-class case
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # multi-class case
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array([[0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [1, 0, 0, 0]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_multilabel():
    lb = LabelBinarizer()

    inp = [(2, 3), (1,), (1, 2)]
    expected = np.array([[0, 1, 1],
                         [1, 0, 0],
                         [1, 1, 0]])
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_iris():
    lb = LabelBinarizer()
    Y = lb.fit_transform(iris.target)
    clfs = [SGDClassifier().fit(iris.data, Y[:, k])
            for k in range(len(lb.classes_))]
    Y_pred = np.array([clf.decision_function(iris.data) for clf in clfs]).T
    y_pred = lb.inverse_transform(Y_pred)
    accuracy = np.mean(iris.target == y_pred)
    y_pred2 = SGDClassifier().fit(iris.data, iris.target).predict(iris.data)
    accuracy2 = np.mean(iris.target == y_pred2)
    assert_almost_equal(accuracy, accuracy2)


def test_center_kernel():
    """Test that KernelCenterer is equivalent to Scaler in feature space"""
    X_fit = np.random.random((5, 4))
    scaler = Scaler(with_std=False)
    scaler.fit(X_fit)
    X_fit_centered = scaler.transform(X_fit)
    K_fit = np.dot(X_fit, X_fit.T)

    # center fit time matrix
    centerer = KernelCenterer()
    K_fit_centered = np.dot(X_fit_centered, X_fit_centered.T)
    K_fit_centered2 = centerer.fit_transform(K_fit)
    assert_array_almost_equal(K_fit_centered, K_fit_centered2)

    # center predict time matrix
    X_pred = np.random.random((2, 4))
    K_pred = np.dot(X_pred, X_fit.T)
    X_pred_centered = scaler.transform(X_pred)
    K_pred_centered = np.dot(X_pred_centered, X_fit_centered.T)
    K_pred_centered2 = centerer.transform(K_pred)
    assert_array_almost_equal(K_pred_centered, K_pred_centered2)
