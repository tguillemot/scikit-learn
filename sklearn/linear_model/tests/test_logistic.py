import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
import scipy

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import raises
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_raise_message
from sklearn.utils import ConvergenceWarning

from sklearn.linear_model.logistic import (
    LogisticRegression,
    logistic_regression_path, LogisticRegressionCV,
    _logistic_loss_and_grad, _logistic_grad_hess,
    _multinomial_grad_hess, _logistic_loss,
    )
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import log_loss


X = [[-1, 0], [0, 1], [1, 1]]
X_sp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
iris = load_iris()

sp_version = tuple([int(s) for s in scipy.__version__.split('.')])


def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]

    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)

    assert_equal(predicted.shape, (n_samples,))
    assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    assert_equal(probabilities.shape, (n_samples, n_classes))
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)


def test_predict_2_classes():
    # Simple sanity check on a 2 classes dataset
    # Make sure it predicts the correct result on simple datasets.
    check_predictions(LogisticRegression(random_state=0), X, Y1)
    check_predictions(LogisticRegression(random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(C=100, random_state=0), X, Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(fit_intercept=False,
                                         random_state=0), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False,
                                         random_state=0), X_sp, Y1)


def test_error():
    # Test for appropriate exception on errors
    msg = "Penalty term must be positive"
    assert_raise_message(ValueError, msg,
                         LogisticRegression(C=-1).fit, X, Y1)
    assert_raise_message(ValueError, msg,
                         LogisticRegression(C="test").fit, X, Y1)

    for LR in [LogisticRegression, LogisticRegressionCV]:
        msg = "Tolerance for stopping criteria must be positive"
        assert_raise_message(ValueError, msg, LR(tol=-1).fit, X, Y1)
        assert_raise_message(ValueError, msg, LR(tol="test").fit, X, Y1)

        msg = "Maximum number of iteration must be positive"
        assert_raise_message(ValueError, msg, LR(max_iter=-1).fit, X, Y1)
        assert_raise_message(ValueError, msg, LR(max_iter="test").fit, X, Y1)


def test_predict_3_classes():
    check_predictions(LogisticRegression(C=10), X, Y2)
    check_predictions(LogisticRegression(C=10), X_sp, Y2)


def test_predict_iris():
    # Test logistic regression with the iris dataset
    n_samples, n_features = iris.data.shape

    target = iris.target_names[iris.target]

    # Test that both multinomial and OvR solvers handle
    # multiclass data correctly and give good accuracy
    # score (>0.95) for the training data.
    for clf in [LogisticRegression(C=len(iris.data)),
                LogisticRegression(C=len(iris.data), solver='lbfgs',
                                   multi_class='multinomial'),
                LogisticRegression(C=len(iris.data), solver='newton-cg',
                                   multi_class='multinomial'),
                LogisticRegression(C=len(iris.data), solver='sag', tol=1e-2,
                                   multi_class='ovr', random_state=42)]:
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        assert_greater(np.mean(pred == target), .95)

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1),
                                  np.ones(n_samples))

        pred = iris.target_names[probabilities.argmax(axis=1)]
        assert_greater(np.mean(pred == target), .95)


def test_multinomial_validation():
    for solver in ['lbfgs', 'newton-cg']:
        lr = LogisticRegression(C=-1, solver=solver, multi_class='multinomial')
        assert_raises(ValueError, lr.fit, [[0, 1], [1, 0]], [0, 1])


def test_check_solver_option():
    X, y = iris.data, iris.target
    for LR in [LogisticRegression, LogisticRegressionCV]:

        msg = ("Logistic Regression supports only liblinear, newton-cg, lbfgs"
               " and sag solvers, got wrong_name")
        lr = LR(solver="wrong_name")
        assert_raise_message(ValueError, msg, lr.fit, X, y)

        msg = "multi_class should be either multinomial or ovr, got wrong_name"
        lr = LR(solver='newton-cg', multi_class="wrong_name")
        assert_raise_message(ValueError, msg, lr.fit, X, y)

        # all solver except 'newton-cg' and 'lfbgs'
        for solver in ['liblinear', 'sag']:
            msg = ("Solver %s does not support a multinomial backend." %
                   solver)
            lr = LR(solver=solver, multi_class='multinomial')
            assert_raise_message(ValueError, msg, lr.fit, X, y)

        # all solvers except 'liblinear'
        for solver in ['newton-cg', 'lbfgs', 'sag']:
            msg = ("Solver %s supports only l2 penalties, got l1 penalty." %
                   solver)
            lr = LR(solver=solver, penalty='l1')
            assert_raise_message(ValueError, msg, lr.fit, X, y)

            msg = ("Solver %s supports only dual=False, got dual=True" %
                   solver)
            lr = LR(solver=solver, dual=True)
            assert_raise_message(ValueError, msg, lr.fit, X, y)


def test_multinomial_binary():
    # Test multinomial LR on a binary problem.
    target = (iris.target > 0).astype(np.intp)
    target = np.array(["setosa", "not-setosa"])[target]

    for solver in ['lbfgs', 'newton-cg']:
        clf = LogisticRegression(solver=solver, multi_class='multinomial')
        clf.fit(iris.data, target)

        assert_equal(clf.coef_.shape, (1, iris.data.shape[1]))
        assert_equal(clf.intercept_.shape, (1,))
        assert_array_equal(clf.predict(iris.data), target)

        mlr = LogisticRegression(solver=solver, multi_class='multinomial',
                                 fit_intercept=False)
        mlr.fit(iris.data, target)
        pred = clf.classes_[np.argmax(clf.predict_log_proba(iris.data),
                                      axis=1)]
        assert_greater(np.mean(pred == target), .9)


def test_sparsify():
    # Test sparsify and densify members.
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    clf = LogisticRegression(random_state=0).fit(iris.data, target)

    pred_d_d = clf.decision_function(iris.data)

    clf.sparsify()
    assert_true(sp.issparse(clf.coef_))
    pred_s_d = clf.decision_function(iris.data)

    sp_data = sp.coo_matrix(iris.data)
    pred_s_s = clf.decision_function(sp_data)

    clf.densify()
    pred_d_s = clf.decision_function(sp_data)

    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)


def test_inconsistent_input():
    # Test that an exception is raised on inconsistent input
    rng = np.random.RandomState(0)
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0

    clf = LogisticRegression(random_state=0)

    # Wrong dimensions for training data
    y_wrong = y_[:-1]
    assert_raises(ValueError, clf.fit, X, y_wrong)

    # Wrong dimensions for test data
    assert_raises(ValueError, clf.fit(X_, y_).predict,
                  rng.random_sample((3, 12)))


def test_write_parameters():
    # Test that we can write to coef_ and intercept_
    clf = LogisticRegression(random_state=0)
    clf.fit(X, Y1)
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    assert_array_almost_equal(clf.decision_function(X), 0)


@raises(ValueError)
def test_nan():
    # Test proper NaN handling.
    # Regression test for Issue #252: fit used to go into an infinite loop.
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    LogisticRegression(random_state=0).fit(Xnan, Y1)


def test_consistency_path():
    # Test that the path algorithm is consistent
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = np.logspace(0, 4, 10)

    f = ignore_warnings
    # can't test with fit_intercept=True since LIBLINEAR
    # penalizes the intercept
    for solver in ('lbfgs', 'newton-cg', 'liblinear', 'sag'):
        coefs, Cs, _ = f(logistic_regression_path)(
            X, y, Cs=Cs, fit_intercept=False, tol=1e-5, solver=solver,
            random_state=0)
        for i, C in enumerate(Cs):
            lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-5,
                                    random_state=0)
            lr.fit(X, y)
            lr_coef = lr.coef_.ravel()
            assert_array_almost_equal(lr_coef, coefs[i], decimal=4,
                                      err_msg="with solver = %s" % solver)

    # test for fit_intercept=True
    for solver in ('lbfgs', 'newton-cg', 'liblinear', 'sag'):
        Cs = [1e3]
        coefs, Cs, _ = f(logistic_regression_path)(
            X, y, Cs=Cs, fit_intercept=True, tol=1e-6, solver=solver,
            intercept_scaling=10000., random_state=0)
        lr = LogisticRegression(C=Cs[0], fit_intercept=True, tol=1e-4,
                                intercept_scaling=10000., random_state=0)
        lr.fit(X, y)
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        assert_array_almost_equal(lr_coef, coefs[0], decimal=4,
                                  err_msg="with solver = %s" % solver)


def test_liblinear_dual_random_state():
    # random_state is relevant for liblinear solver only if dual=True
    X, y = make_classification(n_samples=20)
    lr1 = LogisticRegression(random_state=0, dual=True, max_iter=1, tol=1e-15)
    lr1.fit(X, y)
    lr2 = LogisticRegression(random_state=0, dual=True, max_iter=1, tol=1e-15)
    lr2.fit(X, y)
    lr3 = LogisticRegression(random_state=8, dual=True, max_iter=1, tol=1e-15)
    lr3.fit(X, y)

    # same result for same random state
    assert_array_almost_equal(lr1.coef_, lr2.coef_)
    # different results for different random states
    msg = "Arrays are not almost equal to 6 decimals"
    assert_raise_message(AssertionError, msg,
                         assert_array_almost_equal, lr1.coef_, lr3.coef_)


def test_logistic_loss_and_grad():
    X_ref, y = make_classification(n_samples=20)
    n_features = X_ref.shape[1]

    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.zeros(n_features)

        # First check that our derivation of the grad is correct
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.)
        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.)[0], 1e-3
            )
        assert_array_almost_equal(grad, approx_grad, decimal=2)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(
            w, X, y, alpha=1.
            )
        assert_array_almost_equal(loss, loss_interp)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.)[0], 1e-3
            )
        assert_array_almost_equal(grad_interp, approx_grad, decimal=2)


def test_logistic_grad_hess():
    rng = np.random.RandomState(0)
    n_samples, n_features = 50, 5
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = .1 * np.ones(n_features)

        # First check that _logistic_grad_hess is consistent
        # with _logistic_loss_and_grad
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.)
        grad_2, hess = _logistic_grad_hess(w, X, y, alpha=1.)
        assert_array_almost_equal(grad, grad_2)

        # Now check our hessian along the second direction of the grad
        vector = np.zeros_like(grad)
        vector[1] = 1
        hess_col = hess(vector)

        # Computation of the Hessian is particularly fragile to numerical
        # errors when doing simple finite differences. Here we compute the
        # grad along a path in the direction of the vector and then use a
        # least-square regression to estimate the slope
        e = 1e-3
        d_x = np.linspace(-e, e, 30)
        d_grad = np.array([
            _logistic_loss_and_grad(w + t * vector, X, y, alpha=1.)[1]
            for t in d_x
            ])

        d_grad -= d_grad.mean(axis=0)
        approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()

        assert_array_almost_equal(approx_hess_col, hess_col, decimal=3)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(w, X, y, alpha=1.)
        loss_interp_2 = _logistic_loss(w, X, y, alpha=1.)
        grad_interp_2, hess = _logistic_grad_hess(w, X, y, alpha=1.)
        assert_array_almost_equal(loss_interp, loss_interp_2)
        assert_array_almost_equal(grad_interp, grad_interp_2)


def test_logistic_cv():
    # test for LogisticRegressionCV object
    n_samples, n_features = 50, 5
    rng = np.random.RandomState(0)
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    lr_cv = LogisticRegressionCV(Cs=[1.], fit_intercept=False,
                                 solver='liblinear')
    lr_cv.fit(X_ref, y)
    lr = LogisticRegression(C=1., fit_intercept=False)
    lr.fit(X_ref, y)
    assert_array_almost_equal(lr.coef_, lr_cv.coef_)

    assert_array_equal(lr_cv.coef_.shape, (1, n_features))
    assert_array_equal(lr_cv.classes_, [-1, 1])
    assert_equal(len(lr_cv.classes_), 2)

    coefs_paths = np.asarray(list(lr_cv.coefs_paths_.values()))
    assert_array_equal(coefs_paths.shape, (1, 3, 1, n_features))
    assert_array_equal(lr_cv.Cs_.shape, (1, ))
    scores = np.asarray(list(lr_cv.scores_.values()))
    assert_array_equal(scores.shape, (1, 3, 1))


def test_logistic_cv_sparse():
    X, y = make_classification(n_samples=50, n_features=5,
                               random_state=0)
    X[X < 1.0] = 0.0
    csr = sp.csr_matrix(X)

    clf = LogisticRegressionCV(fit_intercept=True)
    clf.fit(X, y)
    clfs = LogisticRegressionCV(fit_intercept=True)
    clfs.fit(csr, y)
    assert_array_almost_equal(clfs.coef_, clf.coef_)
    assert_array_almost_equal(clfs.intercept_, clf.intercept_)
    assert_equal(clfs.C_, clf.C_)


def test_intercept_logistic_helper():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    # Fit intercept case.
    alpha = 1.
    w = np.ones(n_features + 1)
    grad_interp, hess_interp = _logistic_grad_hess(w, X, y, alpha)
    loss_interp = _logistic_loss(w, X, y, alpha)

    # Do not fit intercept. This can be considered equivalent to adding
    # a feature vector of ones, i.e column of one vectors.
    X_ = np.hstack((X, np.ones(10)[:, np.newaxis]))
    grad, hess = _logistic_grad_hess(w, X_, y, alpha)
    loss = _logistic_loss(w, X_, y, alpha)

    # In the fit_intercept=False case, the feature vector of ones is
    # penalized. This should be taken care of.
    assert_almost_equal(loss_interp + 0.5 * (w[-1] ** 2), loss)

    # Check gradient.
    assert_array_almost_equal(grad_interp[:n_features], grad[:n_features])
    assert_almost_equal(grad_interp[-1] + alpha * w[-1], grad[-1])

    rng = np.random.RandomState(0)
    grad = rng.rand(n_features + 1)
    hess_interp = hess_interp(grad)
    hess = hess(grad)
    assert_array_almost_equal(hess_interp[:n_features], hess[:n_features])
    assert_almost_equal(hess_interp[-1] + alpha * grad[-1], hess[-1])


def test_ovr_multinomial_iris():
    # Test that OvR and multinomial are correct using the iris dataset.
    train, target = iris.data, iris.target
    n_samples, n_features = train.shape

    # Use pre-defined fold as folds generated for different y
    cv = StratifiedKFold(target, 3)
    clf = LogisticRegressionCV(cv=cv)
    clf.fit(train, target)

    clf1 = LogisticRegressionCV(cv=cv)
    target_copy = target.copy()
    target_copy[target_copy == 0] = 1
    clf1.fit(train, target_copy)

    assert_array_almost_equal(clf.scores_[2], clf1.scores_[2])
    assert_array_almost_equal(clf.intercept_[2:], clf1.intercept_)
    assert_array_almost_equal(clf.coef_[2][np.newaxis, :], clf1.coef_)

    # Test the shape of various attributes.
    assert_equal(clf.coef_.shape, (3, n_features))
    assert_array_equal(clf.classes_, [0, 1, 2])
    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert_array_almost_equal(coefs_paths.shape, (3, 3, 10, n_features + 1))
    assert_equal(clf.Cs_.shape, (10, ))
    scores = np.asarray(list(clf.scores_.values()))
    assert_equal(scores.shape, (3, 3, 10))

    # Test that for the iris data multinomial gives a better accuracy than OvR
    for solver in ['lbfgs', 'newton-cg']:
        clf_multi = LogisticRegressionCV(
            solver=solver, multi_class='multinomial', max_iter=15
            )
        clf_multi.fit(train, target)
        multi_score = clf_multi.score(train, target)
        ovr_score = clf.score(train, target)
        assert_greater(multi_score, ovr_score)

        # Test attributes of LogisticRegressionCV
        assert_equal(clf.coef_.shape, clf_multi.coef_.shape)
        assert_array_equal(clf_multi.classes_, [0, 1, 2])
        coefs_paths = np.asarray(list(clf_multi.coefs_paths_.values()))
        assert_array_almost_equal(coefs_paths.shape, (3, 3, 10,
                                                      n_features + 1))
        assert_equal(clf_multi.Cs_.shape, (10, ))
        scores = np.asarray(list(clf_multi.scores_.values()))
        assert_equal(scores.shape, (3, 3, 10))


def test_logistic_regression_solvers():
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)

    ncg = LogisticRegression(solver='newton-cg', fit_intercept=False)
    lbf = LogisticRegression(solver='lbfgs', fit_intercept=False)
    lib = LogisticRegression(fit_intercept=False)
    sag = LogisticRegression(solver='sag', fit_intercept=False,
                             random_state=42)
    ncg.fit(X, y)
    lbf.fit(X, y)
    sag.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=3)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, lib.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, ncg.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, lbf.coef_, decimal=3)


def test_logistic_regression_solvers_multiclass():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)
    tol = 1e-6
    ncg = LogisticRegression(solver='newton-cg', fit_intercept=False, tol=tol)
    lbf = LogisticRegression(solver='lbfgs', fit_intercept=False, tol=tol)
    lib = LogisticRegression(fit_intercept=False, tol=tol)
    sag = LogisticRegression(solver='sag', fit_intercept=False, tol=tol,
                             max_iter=1000, random_state=42)
    ncg.fit(X, y)
    lbf.fit(X, y)
    sag.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=4)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, lib.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, ncg.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, lbf.coef_, decimal=4)

def test_logistic_regressioncv_class_weights():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)

    # Test the liblinear fails when class_weight of type dict is
    # provided, when it is multiclass. However it can handle
    # binary problems.
    clf_lib = LogisticRegressionCV(class_weight={0: 0.1, 1: 0.2},
                                   solver='liblinear')
    assert_raises(ValueError, clf_lib.fit, X, y)
    y_ = y.copy()
    y_[y == 2] = 1
    clf_lib.fit(X, y_)
    assert_array_equal(clf_lib.classes_, [0, 1])

    # Test for class_weight=balanced
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               random_state=0)
    clf_lbf = LogisticRegressionCV(solver='lbfgs', fit_intercept=False,
                                   class_weight='balanced')
    clf_lbf.fit(X, y)
    clf_lib = LogisticRegressionCV(solver='liblinear', fit_intercept=False,
                                   class_weight='balanced')
    clf_lib.fit(X, y)
    clf_sag = LogisticRegressionCV(solver='sag', fit_intercept=False,
                                   class_weight='balanced', max_iter=2000)
    clf_sag.fit(X, y)
    assert_array_almost_equal(clf_lib.coef_, clf_lbf.coef_, decimal=4)
    assert_array_almost_equal(clf_sag.coef_, clf_lbf.coef_, decimal=4)
    assert_array_almost_equal(clf_lib.coef_, clf_sag.coef_, decimal=4)

def test_logistic_regression_sample_weights():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)

    # Test that liblinear fails when sample weights are provided
    clf_lib = LogisticRegression(solver='liblinear')
    assert_raises(ValueError, clf_lib.fit, X, y, 
                  sample_weight=np.ones(y.shape[0]))

    # Test that passing sample_weight as ones is the same as
    # not passing them at all (default None)
    clf_sw_none = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_sw_none.fit(X, y)
    clf_sw_ones = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_sw_ones.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(clf_sw_none.coef_, clf_sw_ones.coef_, decimal=4)

    # Test that sample weights work the same with the lbfgs
    # and newton-cg solvers
    clf_sw_lbfgs = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_sw_lbfgs.fit(X, y, sample_weight=y+1)
    clf_sw_n = LogisticRegression(solver='newton-cg', fit_intercept=False)
    clf_sw_n.fit(X, y, sample_weight=y+1)
    assert_array_almost_equal(clf_sw_lbfgs.coef_, clf_sw_n.coef_, decimal=4)

    # Test that passing class_weight as [1,2] is the same as 
    # passing class weight = [1,1] but adjusting sample weights
    # to be 2 for all instances of class 2
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=2, random_state=0)
    clf_cw_12 = LogisticRegression(solver='lbfgs', fit_intercept=False,
                                     class_weight = {0: 1, 1: 2})
    clf_cw_12.fit(X, y)
    sample_weight = np.ones(y.shape[0])
    sample_weight[y == 1] = 2
    clf_sw_12 = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_sw_12.fit(X, y, sample_weight=sample_weight)
    assert_array_almost_equal(clf_cw_12.coef_, clf_sw_12.coef_, decimal=4)

def test_logistic_regressioncv_sample_weights():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)

    # Test that liblinear fails when sample weights are provided
    clf_lib = LogisticRegressionCV(solver='liblinear')
    assert_raises(ValueError, clf_lib.fit, X, y, 
                  sample_weight=np.ones(y.shape[0]))

    # Test that passing sample_weight as ones is the same as
    # not passing them at all (default None)
    clf_sw_none = LogisticRegressionCV(solver='lbfgs', fit_intercept=False)
    clf_sw_none.fit(X, y)
    clf_sw_ones = LogisticRegressionCV(solver='lbfgs', fit_intercept=False)
    clf_sw_ones.fit(X, y, sample_weight=np.ones(y.shape[0]))
    assert_array_almost_equal(clf_sw_none.coef_, clf_sw_ones.coef_, decimal=4)

    # Test that sample weights work the same with the lbfgs
    # and newton-cg solvers
    clf_sw_lbfgs = LogisticRegressionCV(solver='lbfgs', fit_intercept=False)
    clf_sw_lbfgs.fit(X, y, sample_weight=y+1)
    clf_sw_n = LogisticRegressionCV(solver='newton-cg', fit_intercept=False)
    clf_sw_n.fit(X, y, sample_weight=y+1)
    assert_array_almost_equal(clf_sw_lbfgs.coef_, clf_sw_n.coef_, decimal=4)

    # Test that passing class_weight as [1,2] is the same as 
    # passing class weight = [1,1] but adjusting sample weights
    # to be 2 for all instances of class 2
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=2, random_state=0)
    clf_cw_12 = LogisticRegressionCV(solver='lbfgs', fit_intercept=False,
                                     class_weight = {0: 1, 1: 2})
    clf_cw_12.fit(X, y)
    sample_weight = np.ones(y.shape[0])
    sample_weight[y == 1] = 2
    clf_sw_12 = LogisticRegressionCV(solver='lbfgs', fit_intercept=False)
    clf_sw_12.fit(X, y, sample_weight=sample_weight)
    assert_array_almost_equal(clf_cw_12.coef_, clf_sw_12.coef_, decimal=4)


def test_logistic_regression_convergence_warnings():
    # Test that warnings are raised if model does not converge

    X, y = make_classification(n_samples=20, n_features=20)
    clf_lib = LogisticRegression(solver='liblinear', max_iter=2, verbose=1)
    assert_warns(ConvergenceWarning, clf_lib.fit, X, y)
    assert_equal(clf_lib.n_iter_, 2)


def test_logistic_regression_multinomial():
    # Tests for the multinomial option in logistic regression

    # Some basic attributes of Logistic Regression
    n_samples, n_features, n_classes = 50, 20, 3
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=10,
                               n_classes=n_classes, random_state=0)
    clf_int = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf_int.fit(X, y)
    assert_array_equal(clf_int.coef_.shape, (n_classes, n_features))

    clf_wint = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                  fit_intercept=False)
    clf_wint.fit(X, y)
    assert_array_equal(clf_wint.coef_.shape, (n_classes, n_features))

    # Similar tests for newton-cg solver option
    clf_ncg_int = LogisticRegression(solver='newton-cg',
                                     multi_class='multinomial')
    clf_ncg_int.fit(X, y)
    assert_array_equal(clf_ncg_int.coef_.shape, (n_classes, n_features))

    clf_ncg_wint = LogisticRegression(solver='newton-cg', fit_intercept=False,
                                      multi_class='multinomial')
    clf_ncg_wint.fit(X, y)
    assert_array_equal(clf_ncg_wint.coef_.shape, (n_classes, n_features))

    # Compare solutions between lbfgs and newton-cg
    assert_almost_equal(clf_int.coef_, clf_ncg_int.coef_, decimal=3)
    assert_almost_equal(clf_wint.coef_, clf_ncg_wint.coef_, decimal=3)
    assert_almost_equal(clf_int.intercept_, clf_ncg_int.intercept_, decimal=3)

    # Test that the path give almost the same results. However since in this
    # case we take the average of the coefs after fitting across all the
    # folds, it need not be exactly the same.
    for solver in ['lbfgs', 'newton-cg']:
        clf_path = LogisticRegressionCV(solver=solver,
                                        multi_class='multinomial', Cs=[1.])
        clf_path.fit(X, y)
        assert_array_almost_equal(clf_path.coef_, clf_int.coef_, decimal=3)
        assert_almost_equal(clf_path.intercept_, clf_int.intercept_, decimal=3)


def test_multinomial_grad_hess():
    rng = np.random.RandomState(0)
    n_samples, n_features, n_classes = 100, 5, 3
    X = rng.randn(n_samples, n_features)
    w = rng.rand(n_classes, n_features)
    Y = np.zeros((n_samples, n_classes))
    ind = np.argmax(np.dot(X, w.T), axis=1)
    Y[range(0, n_samples), ind] = 1
    w = w.ravel()
    sample_weights = np.ones(X.shape[0])
    grad, hessp = _multinomial_grad_hess(w, X, Y, alpha=1.,
                                         sample_weight=sample_weights)
    # extract first column of hessian matrix
    vec = np.zeros(n_features * n_classes)
    vec[0] = 1
    hess_col = hessp(vec)

    # Estimate hessian using least squares as done in
    # test_logistic_grad_hess
    e = 1e-3
    d_x = np.linspace(-e, e, 30)
    d_grad = np.array([
        _multinomial_grad_hess(w + t * vec, X, Y, alpha=1.,
                               sample_weight=sample_weights)[0]
        for t in d_x
        ])
    d_grad -= d_grad.mean(axis=0)
    approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()
    assert_array_almost_equal(hess_col, approx_hess_col)


def test_liblinear_decision_function_zero():
    # Test negative prediction when decision_function values are zero.
    # Liblinear predicts the positive class when decision_function values
    # are zero. This is a test to verify that we do not do the same.
    # See Issue: https://github.com/scikit-learn/scikit-learn/issues/3600
    # and the PR https://github.com/scikit-learn/scikit-learn/pull/3623
    X, y = make_classification(n_samples=5, n_features=5)
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X, y)

    # Dummy data such that the decision function becomes zero.
    X = np.zeros((5, 5))
    assert_array_equal(clf.predict(X), np.zeros(5))


def test_liblinear_logregcv_sparse():
    # Test LogRegCV with solver='liblinear' works for sparse matrices

    X, y = make_classification(n_samples=10, n_features=5)
    clf = LogisticRegressionCV(solver='liblinear')
    clf.fit(sparse.csr_matrix(X), y)


def test_logreg_intercept_scaling():
    # Test that the right error message is thrown when intercept_scaling <= 0

    for i in [-1, 0]:
        clf = LogisticRegression(intercept_scaling=i)
        msg = ('Intercept scaling is %r but needs to be greater than 0.'
               ' To disable fitting an intercept,'
               ' set fit_intercept=False.' % clf.intercept_scaling)
        assert_raise_message(ValueError, msg, clf.fit, X, Y1)


def test_logreg_intercept_scaling_zero():
    # Test that intercept_scaling is ignored when fit_intercept is False

    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X, Y1)
    assert_equal(clf.intercept_, 0.)


def test_logreg_cv_penalty():
    # Test that the correct penalty is passed to the final fit.
    X, y = make_classification(n_samples=50, n_features=20, random_state=0)
    lr_cv = LogisticRegressionCV(penalty="l1", Cs=[1.0], solver='liblinear')
    lr_cv.fit(X, y)
    lr = LogisticRegression(penalty="l1", C=1.0, solver='liblinear')
    lr.fit(X, y)
    assert_equal(np.count_nonzero(lr_cv.coef_), np.count_nonzero(lr.coef_))


def test_logreg_predict_proba_multinomial():
    X, y = make_classification(
        n_samples=10, n_features=20, random_state=0, n_classes=3, n_informative=10)

    # Predicted probabilites using the true-entropy loss should give a smaller loss
    # than those using the ovr method.
    clf_multi = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf_multi.fit(X, y)
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_ovr = LogisticRegression(multi_class="ovr", solver="lbfgs")
    clf_ovr.fit(X, y)
    clf_ovr_loss = log_loss(y, clf_ovr.predict_proba(X))
    assert_greater(clf_ovr_loss, clf_multi_loss)

    # Predicted probabilites using the soft-max function should give a smaller loss
    # than those using the logistic function.
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_wrong_loss = log_loss(y, clf_multi._predict_proba_lr(X))
    assert_greater(clf_wrong_loss, clf_multi_loss)


@ignore_warnings
def test_max_iter():
    # Test that the maximum number of iteration is reached
    X, y_bin = iris.data, iris.target.copy()
    y_bin[y_bin == 2] = 0

    solvers = ['newton-cg', 'liblinear', 'sag']
    # old scipy doesn't have maxiter
    if sp_version >= (0, 12):
        solvers.append('lbfgs')

    for max_iter in range(1, 5):
        for solver in solvers:
            lr = LogisticRegression(max_iter=max_iter, tol=1e-15,
                                    random_state=0, solver=solver)
            lr.fit(X, y_bin)
            assert_equal(lr.n_iter_[0], max_iter)


def test_n_iter():
    # Test that self.n_iter_ has the correct format.
    X, y = iris.data, iris.target
    y_bin = y.copy()
    y_bin[y_bin == 2] = 0

    n_Cs = 4
    n_cv_fold = 2

    for solver in ['newton-cg', 'liblinear', 'sag', 'lbfgs']:
        # OvR case
        n_classes = 1 if solver == 'liblinear' else np.unique(y).shape[0]
        clf = LogisticRegression(tol=1e-2, multi_class='ovr',
                                 solver=solver, C=1.,
                                 random_state=42, max_iter=100)
        clf.fit(X, y)
        assert_equal(clf.n_iter_.shape, (n_classes,))

        n_classes = np.unique(y).shape[0]
        clf = LogisticRegressionCV(tol=1e-2, multi_class='ovr',
                                   solver=solver, Cs=n_Cs, cv=n_cv_fold,
                                   random_state=42, max_iter=100)
        clf.fit(X, y)
        assert_equal(clf.n_iter_.shape, (n_classes, n_cv_fold, n_Cs))
        clf.fit(X, y_bin)
        assert_equal(clf.n_iter_.shape, (1, n_cv_fold, n_Cs))

        # multinomial case
        n_classes = 1
        if solver in ('liblinear', 'sag'):
            break

        clf = LogisticRegression(tol=1e-2, multi_class='multinomial',
                                 solver=solver, C=1.,
                                 random_state=42, max_iter=100)
        clf.fit(X, y)
        assert_equal(clf.n_iter_.shape, (n_classes,))

        clf = LogisticRegressionCV(tol=1e-2, multi_class='multinomial',
                                   solver=solver, Cs=n_Cs, cv=n_cv_fold,
                                   random_state=42, max_iter=100)
        clf.fit(X, y)
        assert_equal(clf.n_iter_.shape, (n_classes, n_cv_fold, n_Cs))
        clf.fit(X, y_bin)
        assert_equal(clf.n_iter_.shape, (1, n_cv_fold, n_Cs))


@ignore_warnings
def test_warm_start():
    # A 1-iteration second fit on same data should give almost same result
    # with warm starting, and quite different result without warm starting.
    # Warm starting does not work with liblinear solver.
    X, y = iris.data, iris.target

    solvers = ['newton-cg', 'sag']
    # old scipy doesn't have maxiter
    if sp_version >= (0, 12):
        solvers.append('lbfgs')

    for warm_start in [True, False]:
        for fit_intercept in [True, False]:
            for solver in solvers:
                for multi_class in ['ovr', 'multinomial']:
                    if solver == 'sag' and multi_class == 'multinomial':
                        break
                    clf = LogisticRegression(tol=1e-4, multi_class=multi_class,
                                             warm_start=warm_start,
                                             solver=solver,
                                             random_state=42, max_iter=100,
                                             fit_intercept=fit_intercept)
                    clf.fit(X, y)
                    coef_1 = clf.coef_

                    clf.max_iter = 1
                    with ignore_warnings():
                        clf.fit(X, y)
                    cum_diff = np.sum(np.abs(coef_1 - clf.coef_))
                    msg = ("Warm starting issue with %s solver in %s mode "
                           "with fit_intercept=%s and warm_start=%s"
                           % (solver, multi_class, str(fit_intercept),
                              str(warm_start)))
                    if warm_start:
                        assert_greater(2.0, cum_diff, msg)
                    else:
                        assert_greater(cum_diff, 2.0, msg)
