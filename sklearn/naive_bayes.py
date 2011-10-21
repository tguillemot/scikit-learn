# -*- coding: utf-8 -*-

"""
Naive Bayes models
==================

Naive Bayes algorithms are a set of supervised learning methods based on
applying Bayes' theorem with strong (naive) feature independence assumptions.

See http://scikit-learn.sourceforge.net/modules/naive_bayes.html for
complete documentation.
"""

# Author: Vincent Michel <vincent.michel@inria.fr>
#         Minor fixes by Fabian Pedregosa
#         Amit Aides <amitibo@tx.technion.ac.il>
#         Yehuda Finkelstein <yehudaf@tx.technion.ac.il>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         (parts based on earlier work by Mathieu Blondel)
#
# License: BSD Style.

import numpy as np
from scipy.sparse import issparse

from .base import BaseEstimator, ClassifierMixin
from .preprocessing import binarize, LabelBinarizer
from .utils import safe_asanyarray, atleast2d_or_csr
from .utils.extmath import safe_sparse_dot, logsum
from .utils.fixes import unique



class BaseNB(BaseEstimator, ClassifierMixin):
    """Abstract base class for naive Bayes estimators

    Any estimator based on this class should provide:

    __init__
    fit(X, y)
    _joint_log_likelihood(X)
        Compute the unnormalized posterior log probability of X,
        i.e. log P(c) + log P(x|c) for all rows x of X,
        as an array-like of shape [n_classes, n_samples].

    Input is passed to _joint_log_likelihood as is by predict, predict_proba
    and predict_log_proba.
    """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        y_pred = self.unique_y[np.argmax(jll, axis=1)]
        return y_pred

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class
            in the model, where classes are ordered arithmetically.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsum(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically.
        """
        return np.exp(self.predict_log_proba(X))


class GaussianNB(BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : array, shape = [n_samples]
        Target vector relative to X

    Attributes
    ----------
    class_prior : array, shape = [n_classes]
        probability of each class.

    theta : array, shape = [n_classes, n_features]
        mean of each feature per class

    sigma : array, shape = [n_classes, n_features]
        variance of each feature per class

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log-probability of each class using the model.


    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print clf.predict([[-0.8, -1]])
    [1]
    """

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self.unique_y = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]
        _, n_features = X.shape

        self.theta = np.empty((n_classes, n_features))
        self.sigma = np.empty((n_classes, n_features))
        self.class_prior = np.empty(n_classes)
        for i, y_i in enumerate(unique_y):
            self.theta[i, :] = np.mean(X[y == y_i, :], axis=0)
            self.sigma[i, :] = np.var(X[y == y_i, :], axis=0)
            self.class_prior[i] = np.float(np.sum(y == y_i)) / n_classes
        return self

    def _joint_log_likelihood(self, X):
        X = np.atleast_2d(X)
        joint_log_likelihood = []
        for i in xrange(np.size(self.unique_y)):
            jointi = np.log(self.class_prior[i])
            n_ij = - 0.5 * np.sum(np.log(np.pi * self.sigma[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta[i, :]) ** 2) / \
                                    (self.sigma[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

    def fit(self, X, y, class_prior=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        class_prior : array, shape [n_classes]
            Custom prior probability per class.
            Overrides the fit_prior parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = atleast2d_or_csr(X)
        y = safe_asanyarray(y)

        if X.shape[0] != y.shape[0]:
            msg = "X and y have incompatible shapes."
            if issparse(X):
                msg += "\nNote: Sparse matrices cannot be indexed w/ boolean \
                masks (use `indices=True` in CV)."
            raise ValueError(msg)

        self.unique_y, inv_y_ind = unique(y, return_inverse=True)
        n_classes = self.unique_y.size

        if class_prior:
            assert len(class_prior) == n_classes, \
                   'Number of priors must match number of classs'
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            y_count = np.bincount(inv_y_ind)
            self.class_log_prior_ = np.log(y_count) - np.log(len(y))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

        Y = LabelBinarizer().fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        N_c, N_c_i = self._count(X, Y)

        self.feature_log_prob_ = (np.log(N_c_i + self.alpha)
                    - np.log(N_c.reshape(-1, 1)
                           + self.alpha * X.shape[1]))

        return self

    @staticmethod
    def _count(X, Y):
        """Count feature occurrences.

        Returns (N_c, N_c_i), where
            N_c is the count of all features in all samples of class c;
            N_c_i is the count of feature i in all samples of class c.
        """
        N_c_i = safe_sparse_dot(Y.T, X)
        N_c = np.sum(N_c_i, axis=1)

        return N_c, N_c_i

    intercept_ = property(lambda self: self.class_log_prior_)
    coef_ = property(lambda self: self.feature_log_prob_)


class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------
    alpha: float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior: boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log probability of each class using the model.

    Attributes
    ----------
    `intercept_`, `class_log_prior_` : array, shape = [n_classes]
        Log probability of each class (smoothed).

    `feature_log_prob_`, `coef_` : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

        (`intercept_` and `coef_` are properties referring to
        `class_log_prior_` and `feature_log_prob_`, respectively.)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, Y)
    MultinomialNB(alpha=1.0, fit_prior=True)
    >>> print clf.predict(X[2])
    [3]

    References
    ----------
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    """

    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        X = atleast2d_or_csr(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_


class BernoulliNB(BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Note: this class does not check whether features are actually boolean.

    Parameters
    ----------
    alpha: float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    binarize: float or None, optional
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior: boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log probability of each class using the model.

    Attributes
    ----------
    `class_log_prior_` : array, shape = [n_classes]
        Log probability of each class (smoothed).

    `feature_log_prob_` : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
    >>> print clf.predict(X[2])
    [3]

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234–265.

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41–48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior

    def _count(self, X, Y):
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return super(BernoulliNB, self)._count(X, Y)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""

        X = atleast2d_or_csr(X)

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        X_neg_prob = (neg_prob.sum(axis=1)
                    - safe_sparse_dot(X, neg_prob.T))
        jll = safe_sparse_dot(X, self.coef_.T) + X_neg_prob

        return jll + self.intercept_
