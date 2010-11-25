# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""Stochastic Gradient Descent (SGD) abstract base class"""

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from .sgd_fast import Hinge, Log, ModifiedHuber, SquaredLoss, Huber


class BaseSGD(BaseEstimator):
    """Base class for dense and sparse SGD"""

    def __init__(self, loss, penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0):
        self.loss = str(loss)
        self.penalty = str(penalty)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.fit_intercept = bool(fit_intercept)
        self.n_iter = int(n_iter)
        if self.n_iter <= 0:
            raise ValueError("n_iter must be greater than zero.")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        self.shuffle = bool(shuffle)
        self.verbose = int(verbose)
        self._get_loss_function()
        self._get_penalty_type()

    def _get_loss_function(self):
        """Get concrete LossFunction"""
        raise NotImplementedError("BaseSGD is an abstract class.")

    def _get_penalty_type(self):
        penalty_types = {"l2": 2, "l1": 1, "elasticnet": 3}
        try:
            self.penalty_type = penalty_types[self.penalty]
            if self.penalty_type == 2:
                self.rho = 1.0
            elif self.penalty_type == 1:
                self.rho = 0.0
        except KeyError:
            raise ValueError("Penalty %s is not supported. " % self.penalty)


class ClassifierBaseSGD(BaseSGD, ClassifierMixin):
    """Base class for dense and sparse classification using SGD.
    """

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, n_jobs=1):
        super(ClassifierBaseSGD, self).__init__(loss=loss, penalty=penalty,
                                                alpha=alpha, rho=rho,
                                                fit_intercept=fit_intercept,
                                                n_iter=n_iter, shuffle=shuffle,
                                                verbose=verbose)
        self.n_jobs = int(n_jobs)

    def _get_loss_function(self):
        """Get concrete LossFunction"""
        loss_functions = {
            "hinge": Hinge(),
            "log": Log(),
            "modified_huber": ModifiedHuber(),
        }
        try:
            self.loss_function = loss_functions[self.loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % self.loss)

    def _get_class_weight(self, class_weight, classes, y):
        """
        Estimate class weights for unbalanced datasets.
        """
        if class_weight == {}:
            weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        elif class_weight == 'auto':
            weight = np.array([1.0 / np.sum(y==i) for i in classes],
                              dtype=np.float64, order='C')
            weight *= classes.shape[0] / np.sum(weight)
        else:
            weight = np.zeros(classes.shape[0], dtype=np.float64, order='C')
            for i, c in enumerate(classes):
                weight[i] = class_weight.get(i, 1.0)

        return weight

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
        scores = self.decision_function(X)
        if self.classes.shape[0] == 2:
            indices = np.array(scores > 0, dtype=np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes[np.ravel(indices)]

    def predict_proba(self, X):
        """Predict class membership probability

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples,
        n_classes]
            Contains the membership probabilities of the positive class.

        """
        if (isinstance(self.loss_function, Log) and
            self.classes.shape[0] == 2):
            return 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        else:
            raise NotImplementedError("%s loss does not provide "
                                      "this functionality" % self.loss)


class RegressorBaseSGD(BaseSGD, RegressorMixin):
    """Base class for dense and sparse regression using SGD.
    """
    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1):
        self.epsilon=float(epsilon)
        super(RegressorBaseSGD, self).__init__(loss=loss, penalty=penalty,
                                               alpha=alpha, rho=rho,
                                               fit_intercept=fit_intercept,
                                               n_iter=n_iter, shuffle=shuffle,
                                               verbose=verbose)

    def _get_loss_function(self):
        """Get concrete LossFunction"""
        loss_functions = {
            "squared_loss": SquaredLoss(),
            "huber": Huber(self.epsilon),
        }
        try:
            self.loss_function = loss_functions[self.loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % self.loss)

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
        X = np.asanyarray(X)
        return np.dot(X, self.coef_) + self.intercept_
