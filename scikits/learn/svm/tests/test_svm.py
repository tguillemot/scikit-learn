"""
Testing for Support Vector Machine module (scikits.learn.svm)

TODO: remove hard coded numerical results when possible
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises

from scikits.learn import svm, linear_model, datasets, metrics
from scikits.learn.datasets.samples_generator import test_dataset_classif

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
Y = [1, 1, 1, 2, 2, 2]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [1, 2, 2]

# also load the iris dataset
iris = datasets.load_iris()
perm = np.random.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_libsvm_parameters():
    """
    Test parameters on classes that make use of libsvm.
    """

    clf = svm.SVC(kernel='linear').fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[0.25, -.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.])
    assert_array_equal(clf.predict(X), Y)


def test_libsvm_iris():
    """
    Check consistency on dataset iris.
    """

    # shuffle the dataset so that labels are not ordered

    for k in ('linear', 'rbf'):
        clf = svm.SVC(kernel=k).fit(iris.data, iris.target)
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.9

    assert_array_equal(clf.label_, np.sort(clf.label_))


def test_precomputed():
    """
    SVC with a precomputed kernel.

    We test it with a toy dataset and with iris.
    """
    clf = svm.SVC(kernel='precomputed')
    # Gram matrix for train data (square matrix)
    # (we use just a linear kernel)
    K = np.dot(X, np.array(X).T)
    clf.fit(K, Y)
    # Gram matrix for test data (rectangular matrix)
    KT = np.dot(T, np.array(X).T)
    pred = clf.predict(KT)

    assert_array_equal(clf.dual_coef_, [[0.25, -.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)

    # Gram matrix for test data but compute KT[i,j]
    # for support vectors j only.
    KT = np.zeros_like(KT)
    for i in range(len(T)):
        for j in clf.support_:
            KT[i, j] = np.dot(T[i], X[j])

    pred = clf.predict(KT)
    assert_array_equal(pred, true_result)

    # same as before, but using a callable function instead of the kernel
    # matrix. kernel is just a linear kernel

    kfunc = lambda x, y: np.dot(x, y.T)
    clf = svm.SVC(kernel=kfunc)
    clf.fit(X, Y)
    pred = clf.predict(T)

    assert_array_equal(clf.dual_coef_, [[0.25, -.25]])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)

    # test a precomputed kernel with the iris dataset
    # and check parameters against a linear SVC
    clf = svm.SVC(kernel='precomputed')
    clf2 = svm.SVC(kernel='linear')
    K = np.dot(iris.data, iris.data.T)
    clf.fit(K, iris.target)
    clf2.fit(iris.data, iris.target)
    pred = clf.predict(K)
    assert_array_almost_equal(clf.support_, clf2.support_)
    assert_array_almost_equal(clf.dual_coef_, clf2.dual_coef_)
    assert_array_almost_equal(clf.intercept_, clf2.intercept_)
    assert_almost_equal(np.mean(pred == iris.target), .99, decimal=2)

    # Gram matrix for test data but compute KT[i,j]
    # for support vectors j only.
    K = np.zeros_like(K)
    for i in range(len(iris.data)):
        for j in clf.support_:
            K[i, j] = np.dot(iris.data[i], iris.data[j])

    pred = clf.predict(K)
    assert_almost_equal(np.mean(pred == iris.target), .99, decimal=2)

    clf = svm.SVC(kernel=kfunc)
    clf.fit(iris.data, iris.target)
    assert_almost_equal(np.mean(pred == iris.target), .99, decimal=2)


def test_SVR():
    """
    Test Support Vector Regression
    """

    clf = svm.SVR(kernel='linear')
    clf.fit(X, Y)
    pred = clf.predict(T)

    assert_array_almost_equal(clf.dual_coef_, [[-0.1, 0.1]])
    assert_array_almost_equal(clf.coef_, [[0.2, 0.2]])
    assert_array_almost_equal(clf.support_vectors_, [[-1, -1], [1, 1]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_almost_equal(clf.intercept_, [1.5])
    assert_array_almost_equal(pred, [1.1, 2.3, 2.5])

    # the same with kernel='rbf'
    clf = svm.SVR(kernel='rbf')
    clf.fit(X, Y)
    pred = clf.predict(T)

    assert_array_almost_equal(clf.dual_coef_,
                              [[-0.014, -0.515, -0.013, 0.515, 0.013, 0.013]],
                              decimal=3)
    assert_raises(NotImplementedError, lambda: clf.coef_)
    assert_array_almost_equal(clf.support_vectors_, X)
    assert_array_almost_equal(clf.intercept_, [1.49997261])
    assert_array_almost_equal(pred, [1.10001274, 1.86682485, 1.73300377])


def test_oneclass():
    """
    Test OneClassSVM
    """
    clf = svm.OneClassSVM()
    clf.fit(X)
    pred = clf.predict(T)

    assert_array_almost_equal(pred, [1, -1, -1])
    assert_array_almost_equal(clf.intercept_, [-1.351], decimal=3)
    assert_array_almost_equal(clf.dual_coef_, [[0.750, 0.749, 0.749, 0.750]],
                              decimal=3)
    assert_raises(NotImplementedError, lambda: clf.coef_)


def test_tweak_params():
    """
    Make sure some tweaking of parameters works.

    We change clf.dual_coef_ at run time and expect .predict() to change
    accordingly. Notice that this is not trivial since it involves a lot
    of C/Python copying in the libsvm bindings.

    The success of this test ensures that the mapping between libsvm and
    the python classifier is complete.
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[.25, -.25]])
    assert_array_equal(clf.predict([[-.1, -.1]]), [1])
    clf.dual_coef_ = np.array([[.0, 1.]])
    assert_array_equal(clf.predict([[-.1, -.1]]), [2])


def test_probability():
    """
    Predict probabilities using SVC

    This uses cross validation, so we use a slightly bigger testing set.
    """

    clf = svm.SVC(probability=True)
    clf.fit(iris.data, iris.target)

    # predict on a simple dataset
    T = [[0, 0, 0, 0],
         [2, 2, 2, 2]]
    assert_array_almost_equal(clf.predict_proba(T),
                [[0.993, 0.003, 0.002],
                 [0.740, 0.223, 0.035]],
                 decimal=2)

    assert_almost_equal(clf.predict_proba(T),
                        np.exp(clf.predict_log_proba(T)), 8)

    # make sure probabilities sum to one
    pprob = clf.predict_proba(X)
    assert_array_almost_equal(pprob.sum(axis=1),
                               np.ones(len(X)))


def test_decision_function():
    """
    Test decision_function

    Sanity check, test that decision_function implemented in python
    returns the same as the one in libsvm

    TODO: proabably could be simplified
    """
    clf = svm.SVC(kernel='linear').fit(iris.data, iris.target)

    data = iris.data[0]

    sv_start = np.r_[0, np.cumsum(clf.n_support_)]
    n_features = iris.data.shape[1]
    n_class = 3

    kvalue = np.dot(data, clf.support_vectors_.T)

    dec = np.empty(n_class * (n_class - 1) / 2)
    p = 0
    for i in range(n_class):
        for j in range(i + 1, n_class):
            coef1 = clf.dual_coef_[j - 1]
            coef2 = clf.dual_coef_[i]
            idx1 = slice(sv_start[i], sv_start[i + 1])
            idx2 = slice(sv_start[j], sv_start[j + 1])
            s = np.dot(coef1[idx1],  kvalue[idx1]) + \
                np.dot(coef2[idx2], kvalue[idx2]) + \
                clf.intercept_[p]
            dec[p] = s
            p += 1

    assert_array_almost_equal(-dec, np.ravel(clf.decision_function(data)))


def test_weight():
    """
    Test class weights
    """
    clf = svm.SVC()
    # we give a small weights to class 1
    clf.fit(X, Y, {1: 0.1})
    # so all predicted values belong to class 2
    assert_array_almost_equal(clf.predict(X), [2] * 6)

    X_, y_ = test_dataset_classif(n_samples=200, n_features=100, param=[5, 1],
                                  seed=0)
    for clf in (linear_model.LogisticRegression(), svm.LinearSVC(), svm.SVC()):
        clf.fit(X_[: 180], y_[: 180], class_weight={0: 5})
        y_pred = clf.predict(X_[180:])
        assert np.sum(y_pred == y_[180:]) >= 11


def test_sample_weights():
    """
    Test weights on individual samples
    """
    clf = svm.SVC()
    clf.fit(X, Y)
    assert_array_equal(clf.predict(X[2]), [1.])

    sample_weight = [.1] * 3 + [10] * 3
    clf.fit(X, Y, sample_weight=sample_weight)
    assert_array_equal(clf.predict(X[2]), [2.])


def test_auto_weight():
    """Test class weights for imbalanced data"""
    # compute reference metrics on iris dataset that is quite balanced by
    # default
    X, y = iris.data, iris.target
    clf = svm.SVC(kernel="linear").fit(X, y)
    assert_almost_equal(metrics.f1_score(y, clf.predict(X)), 0.99, 2)

    # make the same prediction using automated class_weight
    clf_auto = svm.SVC(kernel="linear").fit(X, y, class_weight="auto")
    assert_almost_equal(metrics.f1_score(y, clf_auto.predict(X)), 0.99, 2)

    # Make sure that in the balanced case it does not change anything
    # to use "auto"
    assert_array_almost_equal(clf.coef_, clf_auto.coef_, 6)

    # build an very very imbalanced dataset out of iris data
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]

    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)

    # fit a model on the imbalanced data without class weight info
    y_pred = svm.SVC().fit(X_imbalanced, y_imbalanced).predict(X)
    assert_almost_equal(metrics.f1_score(y, y_pred), 0.88, 2)

    # fit a model with auto class_weight enabled
    clf = svm.SVC().fit(X_imbalanced, y_imbalanced, class_weight="auto")
    y_pred = clf.predict(X)
    assert_almost_equal(metrics.f1_score(y, y_pred), 0.92, 2)


def test_error():
    """
    Test that it gives proper exception on deficient input
    """
    # impossible value of C
    assert_raises(ValueError, svm.SVC(C=-1).fit, X, Y)

    # impossible value of nu
    clf = svm.NuSVC(nu=0.0)
    assert_raises(ValueError, clf.fit, X, Y)

    Y2 = Y[:-1]  # wrong dimensions for labels
    assert_raises(ValueError, clf.fit, X, Y2)
    assert_raises(AssertionError, svm.SVC, X, Y2)

    # Test with arrays that are non-contiguous.
    Xf = np.asfortranarray(X)
    clf = svm.SVC()
    clf.fit(Xf, Y)
    assert_array_equal(clf.predict(T), true_result)


def test_LinearSVC():
    """
    Test basic routines using LinearSVC
    """
    clf = svm.LinearSVC().fit(X, Y)

    # by default should have intercept
    assert clf.fit_intercept

    assert_array_equal(clf.predict(T), true_result)
    assert_array_almost_equal(clf.intercept_, [0], decimal=3)

    # the same with l1 penalty
    clf = svm.LinearSVC(penalty='l1', dual=False).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # l2 penalty with dual formulation
    clf = svm.LinearSVC(penalty='l2', dual=True).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # l2 penalty, l1 loss
    clf = svm.LinearSVC(penalty='l2', loss='l1', dual=True).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    dec = clf.decision_function(X)
    assert_array_equal(np.argmax(dec, 1), true_result)

def test_LinearSVC_iris():
    """
    Test that LinearSVC gives plausible predictions on the iris dataset
    """
    clf = svm.LinearSVC().fit(iris.data, iris.target)
    assert np.mean(clf.predict(iris.data) == iris.target) > 0.95


def test_dense_liblinear_intercept_handling(classifier=svm.LinearSVC):
    """
    Test that dense liblinear honours intercept_scaling param
    """
    X = [[2, 1],
         [3, 1],
         [1, 3],
         [2, 3]]
    y = [0, 0, 1, 1]
    clf = classifier(fit_intercept=True, penalty='l1', loss='l2',
                     dual=False, C=1, eps=1e-7)
    assert clf.intercept_scaling == 1, clf.intercept_scaling
    assert clf.fit_intercept

    # when intercept_scaling is low the intercept value is highly "penalized"
    # by regularization
    clf.intercept_scaling = 1
    clf.fit(X, y)
    assert_almost_equal(clf.intercept_, 0, decimal=5)

    # when intercept_scaling is sufficiently high, the intercept value
    # is not affected by regularization
    clf.intercept_scaling = 100
    clf.fit(X, y)
    intercept1 = clf.intercept_
    assert intercept1 > 1

    # when intercept_scaling is sufficiently high, the intercept value
    # doesn't depend on intercept_scaling value
    clf.intercept_scaling = 1000
    clf.fit(X, y)
    intercept2 = clf.intercept_
    assert_array_almost_equal(intercept1, intercept2, decimal=2)


def test_liblinear_predict():
    """
    Test liblinear predict

    Sanity check, test that predict implemented in python
    returns the same as the one in libliblinear

    """
    clf = svm.LinearSVC().fit(iris.data, iris.target)
    prediction = liblinear_prediction_function(iris.data, clf,
                                               np.unique(iris.target))
    assert_array_equal(clf.predict(iris.data), prediction)


def liblinear_prediction_function(farray, clf, labels):
    """
       Note this is only intended for multiclass prediction.   For
       binary, the usual "argmax" procedure works.

       TODO: 1) Add case for binary prediction?
             2) redefine weights in the actual liblinear API?
    """

    weights = clf.raw_coef_
    a, b = weights.shape
    weights = weights.reshape((b, a))

    D = np.column_stack([farray, np.ones(farray.shape[0])])

    H = np.dot(D, weights)

    return H.argmax(1)


if __name__ == '__main__':
    import nose
    nose.runmodule()
