"""Gradient Boosting methods

This module contains methods for fitting gradient boosted
regression trees for both classification and regression

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ the the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  classification problems.
"""

# Authors: Peter Prettenhofer, Scott White
# License: BSD Style.

from __future__ import division
from abc import ABCMeta, abstractmethod

import numpy as np

from .base import BaseEnsemble
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..utils import check_random_state

from ..tree.tree import Tree
from ..tree._tree import _find_best_split
from ..tree._tree import _predict_regression_tree_inplace as _tree_predict
from ..tree._tree import _apply_tree
from ..tree._tree import MSE
from ..tree._tree import DTYPE


__all__ = ["GradientBoostingClassifier",
           "GradientBoostingRegressor"]


# ignore overflows due to exp(-pred) in BinomailDeviance
#np.seterr(invalid='raise', under='raise', divide='raise', over='ignore')


class MedianPredictor(object):
    """A simple initial estimator that predicts the median
    of the training targets.
    """

    def fit(self, X, y):
        self.median = np.median(y)

    def predict(self, X):
        y = np.empty((X.shape[0], 1), dtype=DTYPE)
        y.fill(self.median)
        return y


class MeanPredictor(object):
    """A simple initial estimator that predicts the mean
    of the training targets.
    """

    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        y = np.empty((X.shape[0], 1), dtype=DTYPE)
        y.fill(self.mean)
        return y


class ClassPriorPredictor(object):
    """A simple initial estimator that predicts the class prior
    of the training targets.
    """

    def fit(self, X, y):
        self.prior = np.log(np.sum(y) / np.sum(1.0 - y))

    def predict(self, X):
        y = np.empty((X.shape[0], 1), dtype=DTYPE)
        y.fill(self.prior)
        return y


class MultiClassPriorPredictor(object):
    """A simple initial estimator that predicts the multi class priors
    of the training targets.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.priors = np.empty((self.n_classes,), dtype=DTYPE)
        for k in range(0, self.n_classes):
            self.priors[k] = y[y == self.classes_[k]].shape[0] \
                             / float(y.shape[0])

    def predict(self, X):
        y = np.empty((X.shape[0], self.n_classes), dtype=DTYPE)
        y[:] = self.priors
        return y


class LossFunction(object):
    """Abstract base class for various loss functions.

    Attributes
    ----------
    K : int
        The number of classes; 1 for regression.
    """
    __metaclass__ = ABCMeta

    def __init__(self, n_classes):
        self.K = n_classes

    def init_estimator(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y, pred):
        """Compute the loss of prediction ``pred`` and ``y``. """

    def is_multi_class(self):
        return False

    @abstractmethod
    def negative_gradient(self, y, y_pred, **kargs):
        """Compute the negative gradient.

        Paramters
        ---------
        y : np.ndarray, shape=(n,)
            The target labels.
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learn_rate=1.0, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : np.ndarray, shape=(n, m)
            The data array.
        y : np.ndarray, shape=(n,)
            The target labels.
        residual : np.ndarray, shape=(n,)
            The residuals (usually the negative gradient).
        y_pred : np.ndarray, shape=(n,):
            The predictions.
        """
        # compute leaf for each sample in ``X``.
        terminal_regions = np.empty((X.shape[0], ), dtype=np.int32)
        _apply_tree(X, tree.children, tree.feature, tree.threshold,
                    terminal_regions)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        for leaf in np.where(tree.children[:, 0] == Tree.LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions,
                                         leaf, X, y, residual,
                                         y_pred[:, k])

        # update predictions (both in-bag and out-of-bag)
        y_pred[:, k] += learn_rate * tree.value[:, 0].take(terminal_regions,
                                                           axis=0)

    @abstractmethod
    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Template method for updating terminal regions (=leafs). """


class RegressionLossFunction(LossFunction):
    """Base class for regression loss functions. """
    __metaclass__ = ABCMeta

    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression")
        super(RegressionLossFunction, self).__init__(n_classes)


class LeastSquaresError(RegressionLossFunction):
    """Loss function for least squares (LS) estimation.
    Terminal regions need not to be updated for least squares. """

    def init_estimator(self):
        return MeanPredictor()

    def __call__(self, y, pred):
        return np.mean((y - pred.ravel()) ** 2.0)

    def negative_gradient(self, y, pred, **kargs):
        return y - pred.ravel()

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learn_rate=1.0, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
        # update predictions
        y_pred[:, k] += learn_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        pass


class LeastAbsoluteError(RegressionLossFunction):
    """Loss function for least absolute deviation (LAD) regression. """

    def init_estimator(self):
        return MedianPredictor()

    def __call__(self, y, pred):
        return np.abs(y - pred.ravel()).mean()

    def negative_gradient(self, y, pred, **kargs):
        return np.sign(y - pred.ravel())

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """LAD updates terminal regions to median estimates. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        tree.value[leaf, 0] = np.median(y.take(terminal_region, axis=0) - \
                                        pred.take(terminal_region, axis=0))


class BinomialDeviance(LossFunction):
    """Binomial deviance loss function for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.
    """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("%s requires 2 classes." %
                             self.__class__.__name__)
        # we only need to fit one tree for binary clf.
        super(BinomialDeviance, self).__init__(1)

    def init_estimator(self):
        return ClassPriorPredictor()

    def __call__(self, y, pred):
        """Compute the deviance (= negative log-likelihood). """
        # logaddexp(0, v) == log(1.0 + exp(v))
        pred = pred.ravel()
        return np.sum(np.logaddexp(0.0, -2.0 * y * pred)) / y.shape[0]

    def negative_gradient(self, y, pred, **kargs):
        return y - 1.0 / (1.0 + np.exp(-pred.ravel()))

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)

        numerator = residual.sum()
        denominator = np.sum((y - residual) * (1.0 - y + residual))

        if denominator == 0.0:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator


class MultinomialDeviance(LossFunction):

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError("%s requires more than 2 classes."
                             % self.__class__.__name__)
        super(MultinomialDeviance, self).__init__(n_classes)

    def init_estimator(self):
        return MultiClassPriorPredictor()

    def __call__(self, y, pred):
        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.K), dtype=DTYPE)
        for k in range(self.K):
            Y[:, k] = y == k

        return np.sum(-1 * (Y * pred).sum(axis=1) +
                      np.log(np.exp(pred).sum(axis=1)))

    def is_multi_class(self):
        return True

    def negative_gradient(self, y, pred, k=0):
        """Compute negative gradient for the ``k``-th class. """
        return y - np.exp(pred[:, k]) / np.sum(np.exp(pred), axis=1)

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        """Make a single Newton-Raphson step. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)

        y = y.take(terminal_region, axis=0)

        numerator = residual.sum()
        numerator *= (self.K - 1) / self.K

        denominator = np.sum((y - residual) * (1.0 - y + residual))
        #denominator = np.sum(abs(residual) * (1 - abs(residual)))

        if denominator == 0.0:
            tree.value[leaf, 0] = 0.0
        else:
            tree.value[leaf, 0] = numerator / denominator


LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'lad': LeastAbsoluteError,
                  'bdeviance': BinomialDeviance,
                  'mdeviance': MultinomialDeviance,
                  'deviance': None}  # for both, multinomial and binomial


class BaseGradientBoosting(BaseEnsemble):
    """Abstract base class for Gradient Boosting. """

    def __init__(self, loss, learn_rate, n_estimators, min_samples_split,
                 min_samples_leaf, max_depth, init, subsample, random_state):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        self.n_estimators = n_estimators

        if learn_rate <= 0.0:
            raise ValueError("learn_rate must be greater than 0")
        self.learn_rate = learn_rate

        if loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss '%s' not supported. " % loss)
        self.loss = loss

        if min_samples_split <= 0:
            raise ValueError("min_samples_split must be larger than 0.")
        self.min_samples_split = min_samples_split

        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be larger than 0.")
        self.min_samples_leaf = min_samples_leaf

        if subsample <= 0.0 or subsample > 1:
            raise ValueError("subsample must be in (0,1]")
        self.subsample = subsample

        if max_depth <= 0:
            raise ValueError("max_depth must be larger than 0.")
        self.max_depth = max_depth

        if init is not None:
            if not hasattr(init, 'fit') or not hasattr(init, 'predict'):
                raise ValueError("init must be valid estimator")
        self.init = init

        self.random_state = check_random_state(random_state)

        self.estimators_ = []

    def fit_stage(self, X, X_argsorted, y, y_pred, sample_mask):
        """Fit another stage of ``n_classes`` trees to the boosting model. """
        loss = self.loss_
        self.estimators_.append([])
        original_y = y
        for k in range(loss.K):
            if loss.is_multi_class():
                y = np.array(original_y == k, dtype=DTYPE)
            residual = loss.negative_gradient(y, y_pred, k=k)

            # induce regression tree on residuals
            tree = Tree(1, self.n_features)
            tree.build(X, residual, MSE(), self.max_depth,
                               self.min_samples_split, self.min_samples_leaf,
                               0.0, self.n_features, self.random_state,
                               _find_best_split, sample_mask,
                               X_argsorted)

            # update tree leafs
            self.loss_.update_terminal_regions(tree, X, y, residual, y_pred,
                                               sample_mask, self.learn_rate,
                                               k=k)
            # add tree to ensemble
            self.estimators_[-1].append(tree)

        return y_pred

    def fit(self, X, y, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Use fortran-style
            to avoid memory copies.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes
            0, 1, ..., n_classes-1

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asfortranarray(X, dtype=DTYPE)
        y = np.ascontiguousarray(y)

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Number of labels does not match " \
                             "number of samples.")
        self.n_features = n_features

        loss = LOSS_FUNCTIONS[self.loss](self.n_classes)

        # store loss object for future use
        self.loss_ = loss

        if self.init is None:
            self.init = loss.init_estimator()

        # create argsorted X for fast tree induction
        X_argsorted = np.asfortranarray(
            np.argsort(X.T, axis=1).astype(np.int32).T)

        # fit initial model
        self.init.fit(X, y)

        # init predictions
        y_pred = self.init.predict(X)

        self.estimators_ = []

        self.train_deviance = np.zeros((self.n_estimators,), dtype=DTYPE)
        self.oob_deviance = np.zeros((self.n_estimators), dtype=DTYPE)

        sample_mask = np.ones((n_samples,), dtype=np.bool)

        # perform boosting iterations
        for i in range(self.n_estimators):

            # subsampling
            if self.subsample < 1.0:
                sample_mask = self.random_state.rand(n_samples) \
                              >= (1.0 - self.subsample)

            # fit next stage of trees
            y_pred = self.fit_stage(X, X_argsorted, y, y_pred, sample_mask)

            # track deviance (= loss)
            if self.subsample < 1.0:
                self.train_deviance[i] = loss(y[sample_mask],
                                              y_pred[sample_mask])
                self.oob_deviance[i] = loss(y[~sample_mask],
                                            y_pred[~sample_mask])
            else:
                # no need to fancy index w/ no subsampling
                self.train_deviance[i] = loss(y, y_pred)

            if monitor:
                stop = monitor(self, i)
                if stop:
                    break

        return self

    def _predict(self, X, old_pred=None, stage_index=-1):
        """Predict targets with current model. Re-uses predictions
        from previous iteration if available.

        Returns
        -------
        y_pred : np.ndarray, shape=(n, K)
            The predictions of the current model, where ``n == X.shape[0]``
            and ``K == self._loss.K``.
        """
        learn_rate = self.learn_rate
        if old_pred is not None:
            y = old_pred
            stage = self.estimators_[stage_index]
            for k, tree in enumerate(stage):
                _tree_predict(X, tree.children, tree.feature,
                              tree.threshold, tree.value, learn_rate,
                              k, y)
        else:
            y = self.init.predict(X)
            for stage in self.estimators_:
                for k, tree in enumerate(stage):
                    _tree_predict(X, tree.children, tree.feature,
                                  tree.threshold, tree.value, learn_rate,
                                  k, y)

        return y

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    @property
    def feature_importances_(self):
        if not self.estimators_ or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, " \
                             "call `fit` before `feature_importances_`.")
        total_sum = np.zeros((self.n_features, ), dtype=DTYPE)
        for stage in self.estimators_:
            stage_sum = sum(tree.compute_feature_importances(method='squared')
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        importances = 100.0 * (importances / importances.max())
        return importances


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification. GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage a regression
    tree is fit on the negative gradient of binomial or multinomial
    deviance.

    Parameters
    ----------
    loss : {'deviance', 'ls'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. 'ls' refers to least squares
        regression.

    learn_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learn_rate`.
        There is a trade-off between learn_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.

    Examples
    --------
    >>> samples = [[0, 0, 2], [1, 0, 0]]
    >>> labels = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier().fit(samples, labels)
    >>> print gb.predict([[0.5, 0, 0]])
    [0]

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    def __init__(self, loss='deviance', learn_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=1, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None):

        super(GradientBoostingClassifier, self).__init__(
            loss, learn_rate, n_estimators, min_samples_split,
            min_samples_leaf, max_depth, init, subsample, random_state)

    def fit(self, X, y, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Use fortran-style
            to avoid memory copies.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes
            0, 1, ..., n_classes-1

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        y = np.searchsorted(self.classes_, y)
        if self.loss == 'deviance':
            self.loss = 'mdeviance' if len(self.classes_) > 2 else 'bdeviance'

        return super(GradientBoostingClassifier, self).fit(X, y,
                                                           monitor=monitor)

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_.take(np.argmax(P, axis=1), axis=0)

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        X = X.astype(DTYPE)
        if len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, " \
                             "call `fit` before `predict_proba`.")

        P = np.ones((X.shape[0], self.n_classes), dtype=DTYPE)
        f = self._predict(X)

        if not self.loss_.is_multi_class():
            P[:, 1] = 1.0 / (1.0 + np.exp(-f.ravel()))
            P[:, 0] -= P[:, 1]
        else:
            P = np.exp(f) / np.sum(np.exp(f), axis=1)[:, np.newaxis]
        return P


class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression. GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage a regression
    tree is fit on the negative gradient of the given loss function.

    Parameters
    ----------
    loss : {'ls', 'lad'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function soley based on order information of the input
        variables.

    learn_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learn_rate`.
        There is a trade-off between learn_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.

    Examples
    --------
    >>> samples = [[0, 0, 2], [1, 0, 0]]
    >>> labels = [0, 1]
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> gb = GradientBoostingRegressor().fit(samples, labels)
    >>> print gb.predict([[0, 0, 0]])    # doctest: +ELLIPSIS
    [  1.32806997e-05]

    See also
    --------
    sklearn.tree.DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    def __init__(self, loss='ls', learn_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=1, min_samples_leaf=1,
                 max_depth=3, init=None, random_state=None):

        super(GradientBoostingRegressor, self).__init__(
            loss, learn_rate, n_estimators, min_samples_split,
            min_samples_leaf, max_depth, init, subsample, random_state)

    def fit(self, X, y, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Use fortran-style
            to avoid memory copies.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes
            0, 1, ..., n_classes-1

        Returns
        -------
        self : object
            Returns self.
        """
        self.n_classes = 1
        return super(GradientBoostingRegressor, self).fit(X, y,
                                                          monitor=monitor)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = X.astype(DTYPE)
        if len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, " \
                             "call `fit` before `predict`.")
        y = self._predict(X).ravel()
        return y
