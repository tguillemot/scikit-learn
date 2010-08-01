"""
Support Vector Machine algorithms for sparse matrices.

Warning: this module is a work in progress. It is not tested and surely
contains bugs.

Notes
-----

Some fields, like dual_coef_ are not sparse matrices strictly speaking.
However, they are converted to a sparse matrix for consistency and
efficiency when multiplying to other sparse matrices.

Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
License: New BSD
"""

import numpy as np
from scipy import sparse

from ..base import BaseEstimator
from .. import svm, _libsvm

class SparseBaseLibsvm(BaseEstimator):

    _kernel_types = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    _svm_types = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

    def __init__(self, impl, kernel, degree, gamma, coef0, cache_size,
                 eps, C, nu, p, shrinking, probability):
        assert impl in self._svm_types, \
            "impl should be one of %s, %s was given" % (
                self._svm_types, impl)
        assert kernel in self._kernel_types or callable(kernel), \
            "kernel should be one of %s or a callable, %s was given." % (
                self._kernel_types, kernel)
        self.kernel = kernel
        self.impl = impl
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.eps = eps
        self.C = C
        self.nu = nu
        self.p = p
        self.shrinking = int(shrinking)
        self.probability = int(probability)

        # container for when we call fit
        self._support_data    = np.empty (0, dtype=np.float64, order='C')
        self._support_indices = np.empty (0, dtype=np.int32, order='C')
        self._support_indptr  = np.empty (0, dtype=np.int32, order='C')

        # strictly speaking, dual_coef is not sparse (see Notes above)
        self._dual_coef_data    = np.empty (0, dtype=np.float64, order='C')
        self._dual_coef_indices = np.empty (0, dtype=np.int32,   order='C')
        self._dual_coef_indptr  = np.empty (0, dtype=np.int32,   order='C')
        self.intercept_         = np.empty (0, dtype=np.float64, order='C')

        # only used in classification
        self.nSV_ = np.empty(0, dtype=np.int32, order='C')


    def fit(self, X, Y, class_weight={}):
        """
        X is expected to be a sparse matrix. For maximum effiency, use a
        sparse matrix in csr format (scipy.sparse.csr_matrix)
        """

        X = sparse.csr_matrix(X)
        X.data = np.asanyarray(X.data, dtype=np.float64, order='C')
        Y      = np.asanyarray(Y,      dtype=np.float64, order='C')

        solver_type = self._svm_types.index(self.impl)
        kernel_type = self._kernel_types.index(self.kernel)

        self.weight       = np.asarray(class_weight.values(),
                                      dtype=np.float64, order='C')
        self.weight_label = np.asarray(class_weight.keys(),
                                       dtype=np.int32, order='C')

        self.label_, self.probA_, self.probB_ = _libsvm.csr_train_wrap(
                 X.shape[1], X.data, X.indices, X.indptr, Y,
                 solver_type, kernel_type, self.degree,
                 self.gamma, self.coef0, self.eps, self.C,
                 self._support_data, self._support_indices,
                 self._support_indptr, self._dual_coef_data,
                 self.intercept_, self.weight_label, self.weight,
                 self.nSV_, self.nu, self.cache_size, self.p,
                 self.shrinking,
                 int(self.probability))

        # TODO: explicitly specify size
        self.support_ = sparse.csr_matrix((self._support_data,
                                           self._support_indices,
                                           self._support_indptr))

        # TODO: is this always a 1-d array ?
        n_classes = len(self.label_) - 1
        dual_coef_indices =  np.tile(np.arange(self.support_.shape[0]),
                                     n_classes)
        dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                     dual_coef_indices.size / n_classes)

        self.dual_coef_ = sparse.csr_matrix((self._dual_coef_data,
                                             dual_coef_indices,
                                             dual_coef_indptr))

        return self


    def predict(self, T):
        """
        This function does classification or regression on an array of
        test vectors T.

        For a classification model, the predicted class for each
        sample in T is returned.  For a regression model, the function
        value of T calculated is returned.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        T : scipy.sparse.csr, shape = [nsamples, nfeatures]

        Returns
        -------
        C : array, shape = [nsample]
        """
        T = sparse.csr_matrix(T)
        T.data = np.asanyarray(T.data, dtype=np.float64, order='C')
        kernel_type = self._kernel_types.index(self.kernel)
        return _libsvm.csr_predict_from_model_wrap(T.data,
                      T.indices, T.indptr, self.support_.data,
                      self.support_.indices, self.support_.indptr,
                      self.dual_coef_.data, self.intercept_,
                      self._svm_types.index(self.impl),
                      kernel_type, self.degree,
                      self.gamma, self.coef0, self.eps, self.C,
                      self.weight_label, self.weight,
                      self.nu, self.cache_size, self.p,
                      self.shrinking, self.probability,
                      self.nSV_, self.label_, self.probA_,
                      self.probB_)

    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise NotImplementedError(
                'coef_ is only available when using a linear kernel')
        return np.dot(self.dual_coef_, self.support_)


class SVC(SparseBaseLibsvm):
    """SVC for sparse matrices (csr)

    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).
    """
    def __init__(self, impl='c_svc', kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, cache_size=100.0, eps=1e-3, C=1.0, nu=0.5, p=0.1,
                 shrinking=True, probability=False):

        SparseBaseLibsvm.__init__(self, impl, kernel, degree, gamma, coef0,
                         cache_size, eps, C, nu, p,
                         shrinking, probability)

