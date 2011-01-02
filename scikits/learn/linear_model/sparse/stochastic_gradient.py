# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""Implementation of Stochastic Gradient Descent (SGD) with sparse data."""

import numpy as np
from scipy import sparse

from ...externals.joblib import Parallel, delayed
from ..base import BaseSGDClassifier, BaseSGDRegressor
from ..sgd_fast_sparse import plain_sgd

## TODO add flag for intercept learning rate heuristic
##

class SGDClassifier(BaseSGDClassifier):
    """Linear model fitted by minimizing a regularized empirical loss with SGD

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works on scipy.sparse X and dense coef_.

    Parameters
    ----------
    loss : str, 'hinge' or 'log' or 'modified_huber'
        The loss function to be used. Defaults to 'hinge'. The hinge loss is a
        margin loss used by standard linear SVM models. The 'log' loss is the
        loss of logistic regression models and can be used for probability
        estimation in binary classifiers. 'modified_huber' is another smooth
        loss that brings tolerance to outliers.

    penalty : str, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to 'l2' which
        is the standard regularizer for linear SVM models. 'l1' and 'elasticnet'
        migh bring sparsity to the model (feature selection) not achievable with
        'l2'.

    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001

    rho : float
        The Elastic Net mixing parameter, with 0 < rho <= 1.
        Defaults to 0.85.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter: int
        The number of passes over the training data (aka epochs).
        Defaults to 5.

    shuffle: bool
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.

    verbose: integer, optional
        The verbosity level

    n_jobs: integer, optional
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation. -1 means 'all CPUs'. Defaults
        to 1.

    Attributes
    ----------
    `coef_` : array, shape = [n_features] if n_classes == 2 else [n_classes,
    n_features]
        Weights asigned to the features.

    `intercept_` : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from scikits.learn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.sparse.SGDClassifier()
    >>> clf.fit(X, y)
    SGDClassifier(loss='hinge', n_jobs=1, shuffle=False, verbose=0, n_iter=5,
           fit_intercept=True, penalty='l2', rho=1.0, alpha=0.0001)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]

    See also
    --------
    LinearSVC, LogisticRegression

    """

    def _set_coef(self, coef_):
        self.coef_ = coef_
        if coef_ is None:
            self.sparse_coef_ = None
        else:
            # sparse representation of the fitted coef for the predict method
            self.sparse_coef_ = sparse.csr_matrix(coef_)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            class_weight={}, **params):
        """Fit linear model with Stochastic Gradient Descent

        X is expected to be a sparse matrix. For maximum efficiency, use a
        sparse matrix in CSR format (scipy.sparse.csr_matrix)

        Parameters
        ----------
        X : scipy sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values
        coef_init : array, shape = [n_features] if n_classes == 2 else
        [n_classes,n_features]
            The initial coeffients to warm-start the optimization.
        intercept_init : array, shape = [1] if n_classes == 2 else [n_classes]
            The initial intercept to warm-start the optimization.
        class_weight : dict, {class_label : weight} or "auto"
            Weights associated with classes. If not given, all classes
            are supposed to have weight one.

            The "auto" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params(**params)
        X = sparse.csr_matrix(X)
        y = np.asanyarray(y, dtype=np.float64)

        # largest class id is positive class
        self.classes = np.unique(y)

        self.weight = self._get_class_weight(class_weight, self.classes, y)

        if self.classes.shape[0] > 2:
            self._fit_multiclass(X, y, coef_init, intercept_init)
        elif self.classes.shape[0] == 2:
            self._fit_binary(X, y, coef_init, intercept_init)
        else:
            raise ValueError("The number of class labels must be " \
                             "greater than one. ")
        # return self for chaining fit and predict calls
        return self

    def _fit_binary(self, X, y, coef_init, intercept_init):
        """Fit a binary classifier.
        """
        # encode original class labels as 1 (classes[1]) or -1 (classes[0]).
        y_new = np.ones(y.shape, dtype=np.float64) * -1.0
        y_new[y == self.classes[1]] = 1.0
        y = y_new

        n_samples, n_features = X.shape[0], X.shape[1]
        self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")
        self.intercept_ = np.zeros(1, dtype=np.float64, order="C")
        if coef_init is not None:
            coef_init = np.asanyarray(coef_init)
            if coef_init.shape != (n_features,):
                raise ValueError("Provided coef_init does not match dataset.")
            self.coef_ = coef_init
        if intercept_init is not None:
            intercept_init = np.asanyarray(intercept_init)
            if intercept_init.shape != (1,):
                raise ValueError("Provided intercept_init " \
                                 "does not match dataset.")
            else:
                self.intercept_ = intercept_init

        X_data = np.array(X.data, dtype=np.float64, order="C")
        X_indices = X.indices
        X_indptr = X.indptr
        coef_, intercept_ = plain_sgd(self.coef_,
                                      self.intercept_,
                                      self.loss_function,
                                      self.penalty_type,
                                      self.alpha, self.rho,
                                      X_data,
                                      X_indices, X_indptr, y,
                                      self.n_iter,
                                      int(self.fit_intercept),
                                      int(self.verbose),
                                      int(self.shuffle),
                                      self.weight[1], self.weight[0])

        # update self.coef_ and self.sparse_coef_ consistently
        self._set_coef(self.coef_)
        self.intercept_ = np.asarray(intercept_)

    def _fit_multiclass(self, X, y, coef_init, intercept_init):
        """Fit a multi-class classifier with a combination
        of binary classifiers, each predicts one class versus
        all others (OVA: One Versus All).
        """
        n_classes = self.classes.shape[0]
        n_samples, n_features = X.shape[0], X.shape[1]
        self.coef_ = np.zeros((n_classes, n_features),
                         dtype=np.float64, order="C")
        self.intercept_ = np.zeros(n_classes, dtype=np.float64, order="C")

        if coef_init is not None:
            coef_init = np.asanyarray(coef_init)
            if coef_init.shape != (n_classes, n_features):
                raise ValueError("Provided coef_ does not match dataset. ")
            else:
                self.coef_ = coef_init

        if intercept_init is not None:
            intercept_init = np.asanyarray(intercept_init)
            if intercept_init.shape != (n_classes, ):
                raise ValueError("Provided intercept_init " \
                                 "does not match dataset.")
            else:
                self.intercept_ = intercept_init

        X_data = np.array(X.data, dtype=np.float64, order="C")
        X_indices = X.indices
        X_indptr = X.indptr

        res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_train_ova_classifier)(i, c, X_data, X_indices,
                                               X_indptr, y, self.coef_[i],
                                               self.intercept_[i],
                                               self.loss_function,
                                               self.penalty_type, self.alpha,
                                               self.rho, self.n_iter,
                                               self.fit_intercept,
                                               self.verbose, self.shuffle,
                                               self.weight[i])
            for i, c in enumerate(self.classes))

        for i, coef, intercept in res:
            self.coef_[i] = coef
            self.intercept_[i] = intercept

        self._set_coef(self.coef_)
        self.intercept_ = self.intercept_

    def decision_function(self, X):
        """Predict signed 'distance' to the hyperplane (aka confidence score).

        Parameters
        ----------
        X : scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples, n_classes]
          The signed 'distances' to the hyperplane(s).
        """
        # np.dot only works correctly if both arguments are sparse matrices
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        scores = np.asarray(np.dot(X, self.sparse_coef_.T).todense()
                            + self.intercept_)
        if self.classes.shape[0] == 2:
            return np.ravel(scores)
        else:
            return scores


def _train_ova_classifier(i, c, X_data, X_indices, X_indptr, y, coef_,
                          intercept_, loss_function, penalty_type, alpha,
                          rho, n_iter, fit_intercept, verbose, shuffle,
                          weight_pos):
    """Inner loop for One-vs.-All scheme"""
    y_i = np.ones(y.shape, dtype=np.float64) * -1.0
    y_i[y == c] = 1.0
    coef, intercept = plain_sgd(coef_, intercept_,
                                loss_function, penalty_type,
                                alpha, rho, X_data, X_indices,
                                X_indptr, y_i, n_iter,
                                fit_intercept, verbose,
                                shuffle, weight_pos, 1.0)
    return (i, coef, intercept)


class SGDRegressor(BaseSGDRegressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and
    achieve online feature selection.

    This implementation works with data represented as dense numpy arrays
    of floating point values for the features.

    Parameters
    ----------
    loss : str, 'squared_loss' or 'huber'
        The loss function to be used. Defaults to 'squared_loss' which
        refers to the ordinary least squares fit. 'huber' is an epsilon
        insensitive loss function for robust regression.

    penalty : str, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' migh bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001

    rho : float
        The Elastic Net mixing parameter, with 0 < rho <= 1.
        Defaults to 0.85.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter: int
        The number of passes over the training data (aka epochs).
        Defaults to 5.

    shuffle: bool
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.

    verbose: integer, optional
        The verbosity level

    p : float
        Epsilon in the epsilon insensitive huber loss function;
        only if `loss=='huber'`.

    Attributes
    ----------
    `coef_` : array, shape = [n_features]
        Weights asigned to the features.

    `intercept_` : array, shape = [1]
        The intercept term.

    Examples
    --------
    >>> import numpy as np
    >>> from scikits.learn import linear_model    
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = linear_model.sparse.SGDRegressor()
    >>> clf.fit(X, y)
    SGDRegressor(loss='squared_loss', shuffle=False, verbose=0, n_iter=5,
           fit_intercept=True, penalty='l2', p=0.1, rho=1.0, alpha=0.0001)

    See also
    --------
    RidgeRegression, ElasticNet, Lasso, SVR

    """

    def _set_coef(self, coef_):
        self.coef_ = coef_
        if coef_ is None:
            self.sparse_coef_ = None
        else:
            # sparse representation of the fitted coef for the predict method
            self.sparse_coef_ = sparse.csr_matrix(coef_)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            **params):
        """Fit linear model with Stochastic Gradient Descent

        X is expected to be a sparse matrix. For maximum efficiency, use a
        sparse matrix in CSR format (scipy.sparse.csr_matrix)

        Parameters
        ----------
        X : scipy sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values
        coef_init : array, shape = [n_features]
            The initial coeffients to warm-start the optimization.
        intercept_init : array, shape = [1]

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params(**params)
        X = sparse.csr_matrix(X)
        y = np.asanyarray(y, dtype=np.float64)

        n_samples, n_features = X.shape[0], X.shape[1]
        self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")
        self.intercept_ = np.zeros(1, dtype=np.float64, order="C")
        if coef_init is not None:
            coef_init = np.asanyarray(coef_init)
            if coef_init.shape != (n_features,):
                raise ValueError("Provided coef_init does not match dataset.")
            self.coef_ = coef_init
        if intercept_init is not None:
            intercept_init = np.asanyarray(intercept_init)
            if intercept_init.shape != (1,):
                raise ValueError("Provided intercept_init " \
                                 "does not match dataset.")
            else:
                self.intercept_ = intercept_init

        X_data = np.array(X.data, dtype=np.float64, order="C")
        X_indices = X.indices
        X_indptr = X.indptr
        coef_, intercept_ = plain_sgd(self.coef_,
                                      self.intercept_,
                                      self.loss_function,
                                      self.penalty_type,
                                      self.alpha, self.rho,
                                      X_data,
                                      X_indices, X_indptr, y,
                                      self.n_iter,
                                      int(self.fit_intercept),
                                      int(self.verbose),
                                      int(self.shuffle),
                                      1.0, 1.0)

        # update self.coef_ and self.sparse_coef_ consistently
        self._set_coef(self.coef_)
        self.intercept_ = np.asarray(intercept_)
        return self

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]
           Whether the numpy.array or scipy.sparse matrix is accepted dependes
           on the actual implementation

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        # np.dot only works correctly if both arguments are sparse matrices
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        scores = np.asarray(np.dot(X, self.sparse_coef_.T).todense()
                            + self.intercept_).ravel()
        return scores
