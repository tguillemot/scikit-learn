
from ...base import ClassifierMixin, RegressorMixin
from .base import SparseBaseLibSVM, SparseBaseLibLinear
from ...linear_model.sparse.base import CoefSelectTransformerMixin


class SVC(SparseBaseLibSVM, ClassifierMixin):
    """SVC for sparse matrices (csr).

    See :class:`sklearn.svm.SVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm.sparse import SVC
    >>> clf = SVC()
    >>> clf.fit(X, y)
    SVC(C=1.0, coef0=0.0, degree=3, gamma=0.25, kernel='rbf', probability=False,
      shrinking=True, tol=0.001)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'c_svc', kernel, degree, gamma, coef0,
                         tol, C, 0., 0.,
                         shrinking, probability)



class NuSVC (SparseBaseLibSVM, ClassifierMixin):
    """NuSVC for sparse matrices (csr).

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm.sparse import NuSVC
    >>> clf = NuSVC()
    >>> clf.fit(X, y)
    NuSVC(coef0=0.0, degree=3, gamma=0.25, kernel='rbf', nu=0.5,
       probability=False, shrinking=True, tol=0.001)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]
    """


    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'nu_svc', kernel, degree,
                         gamma, coef0, tol, 0., nu, 0.,
                         shrinking, probability)




class SVR (SparseBaseLibSVM, RegressorMixin):
    """SVR for sparse matrices (csr)

    See :class:`sklearn.svm.SVR` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> from sklearn.svm.sparse import SVR
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = SVR(C=1.0, epsilon=0.2)
    >>> clf.fit(X, y)
    SVR(C=1.0, coef0=0.0, degree=3, epsilon=0.2, gamma=0.1, kernel='rbf', nu=0.5,
      probability=False, shrinking=True, tol=0.001)
    """

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=1e-3, C=1.0, nu=0.5, epsilon=0.1,
                 shrinking=True, probability=False):

        SparseBaseLibSVM.__init__(self, 'epsilon_svr', kernel,
                         degree, gamma, coef0, tol, C, nu,
                         epsilon, shrinking, probability)





class NuSVR (SparseBaseLibSVM, RegressorMixin):
    """NuSVR for sparse matrices (csr)

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> from sklearn.svm.sparse import NuSVR
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = NuSVR(nu=0.1, C=1.0)
    >>> clf.fit(X, y)
    NuSVR(C=1.0, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1, kernel='rbf',
       nu=0.1, probability=False, shrinking=True, tol=0.001)
    """

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3,
                 gamma=0.0, coef0=0.0, shrinking=True, epsilon=0.1,
                 probability=False, tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'nu_svr', kernel,
                         degree, gamma, coef0, tol, C, nu,
                         epsilon, shrinking, probability)



class OneClassSVM (SparseBaseLibSVM):
    """NuSVR for sparse matrices (csr)

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).
    """

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=1e-3, nu=0.5, shrinking=True,
                 probability=False):

        SparseBaseLibSVM.__init__(self, 'one_class', kernel, degree,
                         gamma, coef0, tol, 0.0, nu, 0.0,
                         shrinking, probability)

    def fit(self, X, class_weight={}, sample_weight=[]):
        super(OneClassSVM, self).fit(
            X, [], class_weight=class_weight, ample_weight=sample_weight)


class LinearSVC(SparseBaseLibLinear, ClassifierMixin,
                CoefSelectTransformerMixin):
    """
    Linear Support Vector Classification, Sparse Version

    Similar to SVC with parameter kernel='linear', but uses internally
    liblinear rather than libsvm, so it has more flexibility in the
    choice of penalties and loss functions and should be faster for
    huge datasets.

    Parameters
    ----------
    loss : string, 'l1' or 'l2' (default 'l2')
        Specifies the loss function. With 'l1' it is the standard SVM
        loss (a.k.a. hinge Loss) while with 'l2' it is the squared loss.
        (a.k.a. squared hinge Loss)

    penalty : string, 'l1' or 'l2' (default 'l2')
        Specifies the norm used in the penalization. The 'l2' penalty
        is the standard used in SVC. The 'l1' leads to ``coef_``
        vectors that are sparse.

    C : float, optional (default=1.0)
        penalty parameter C of the error term.

    dual : bool, (default True)
        Select the algorithm to either solve the dual or primal
        optimization problem.

    intercept_scaling : float, default: 1
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased

    Attributes
    ----------
    `coef_` : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
        Wiehgiths asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

    `intercept_` : array, shape = [1] if n_classes == 2 else [n_classes]
        constants in decision function

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller eps parameter.

    See also
    --------
    SVC

    References
    ----------
    LIBLINEAR -- A Library for Large Linear Classification
    http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    """
    pass
