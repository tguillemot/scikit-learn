"""
Linear Discriminant Analysis (LDA)
"""

# Authors: Clemens Brunner
#          Martin Billinger
#          Matthieu Perrot
#          Mathieu Blondel

# License: BSD 3-Clause

from __future__ import print_function
import warnings

import numpy as np
from scipy import linalg

from .base import BaseEstimator, ClassifierMixin, TransformerMixin
from .covariance import ledoit_wolf, empirical_covariance
from .utils.extmath import logsumexp
from .utils.multiclass import unique_labels
from .utils import check_array, check_X_y
from .preprocessing import StandardScaler, LabelEncoder


def _cov(X, estimator='empirical'):
    """
    Estimate the covariance matrix using a specified estimator.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Data vector

    estimator : string
        Covariance estimator, possible values:
          - 'empirical': empirical covariance matrix (default)
          - 'ledoit_wolf': shrunk covariance matrix using the Ledoit-Wolf lemma

    Returns
    -------
    s : array-like, shape = [n_features, n_features]
        Estimated covariance matrix
    """
    if estimator == 'ledoit_wolf':
        # standardize features
        sc = StandardScaler()
        X = sc.fit_transform(X)
        std = np.diag(sc.std_)
        s = std.dot(ledoit_wolf(X)[0]).dot(std)  # rescale covariance matrix
    elif estimator == 'empirical':
        s = empirical_covariance(X)
    else:
        raise ValueError('unknown covariance estimation method')
    return s


class LDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Linear Discriminant Analysis (LDA)

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions.

    Parameters
    ----------
    solver : string, optional
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default). Does not need to
                   compute the covariance matrix, therefore this solver is
                   recommended for very large feature dimensions.
          - 'lsqr': Least squares solution, can be combined with shrinkage (see
                    below)
          - 'eigen': Eigenvalue decomposition, can be combined with shrinkage
                    (see below)

    alpha : string or float, optional
        Shrinkage parameter, possible values:
          - None: No shrinkage (default)
          - 'ledoit_wolf': Ledoit-Wolf shrinkage (determines optimal shrinkage
                           parameter analytically)
          - 'empirical': Equivalent with no shrinkage (or a value of 0)
          - 0..1: If set to a number between 0 and 1, the shrinkage parameter is
                  set to this value (0 means no shrinkage)

    priors : array, optional, shape = [n_classes]
        Priors on classes

    n_comps : int
        Number of components (< n_classes - 1) for dimensionality reduction

    Attributes
    ----------
    coef_ : array, shape = [n_features] or [n_classes, n_features]
        Weight vector(s)

    intercept_ : array, shape = [n_features]
        Intercept term

    cov_ : array-like, shape = [n_features, n_features]
        Covariance matrix (shared by all classes)

    means_ : array-like, shape = [n_classes, n_features]
        Class means

    priors_ : array-like, shape = [n_classes]
        Class priors (sum to 1)

    scalings_ : array-like, shape = [rank, n_classes - 1]
        Scaling of the features in the space spanned by the class centroids

    xbar_ : array-like, shape = [n_features]
        Overall mean

    classes_ : array-like, shape = [n_classes]
        Unique class labels

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.lda import LDA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LDA()
    >>> clf.fit(X, y)
    >>> print(clf.predict([[-0.8, -1]]))
    [0]

    See also
    --------
    sklearn.qda.QDA: Quadratic discriminant analysis
    """

    def __init__(self, solver='svd', alpha=None, priors=None, n_comps=None):
        self.solver = solver
        self.alpha = alpha
        self.priors = priors
        self.n_comps = n_comps
        
    def _solve_lsqr(self, X, y, cov_estimator):
        """Least squares solver

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        cov_estimator : string
            Covariance estimator, possible values:
              - 'empirical': Empirical covariance matrix
              - 'ledoit_wolf': Shrunk covariance matrix using Ledoit-Wolf
        """
        classes = unique_labels(y)
        means = []
        covs = []
        for group in classes:
            Xg = X[y == group, :]
            meang = Xg.mean(0)
            means.append(meang)
            covg = _cov(Xg, cov_estimator)
            covg = np.atleast_2d(covg)
            covs.append(covg)
    
        self.cov_ = np.mean(covs, 0)
        self.means_ = np.asarray(means)
        self.xbar_ = np.dot(self.priors_, self.means_)
    
        # TODO: weight covariances with priors?
        Sw = np.mean(covs, 0)  # Within-class scatter
        means = self.means_ - self.xbar_
    
        self.coef_ = np.linalg.lstsq(Sw, means.T, rcond=1e-11)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(means, self.coef_.T))
                           + np.log(self.priors_))

    def _solve_eigen(self, X, y, cov_estimator):
        """Eigenvalue decomposition solver

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        cov_estimator : string
            Covariance estimator, possible values:
              - 'empirical': Empirical covariance matrix
              - 'ledoit_wolf': Shrunk covariance matrix using Ledoit-Wolf
        """
        classes = unique_labels(y)
        means = []
        covs = []
        for group in classes:
            Xg = X[y == group, :]
            meang = Xg.mean(0)
            means.append(meang)
            covg = _cov(Xg, cov_estimator)
            covg = np.atleast_2d(covg)
            covs.append(covg)

        self.cov_ = np.mean(covs, 0)
        self.means_ = np.asarray(means)
        self.xbar_ = np.dot(self.priors_, means)

        # TODO: weight covariances with priors?
        Sw = np.mean(covs, 0)  # Within-class scatter
        St = _cov(X, cov_estimator)
        Sb = St - Sw
        means = self.means_ - self.xbar_

        _, self.scalings_ = linalg.eigh(Sb, Sw)
        
        coef = np.dot(means, self.scalings_)
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(means, coef.T))
                           + np.log(self.priors_))
    
    def _solve_svd(self, X, y, tol=1.0e-4):
        """SVD solver

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        tol : float
            Threshold used for rank estimation
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
    
        # compute class means and covariance matrix
        means = []
        # covs = []
        Xc = []
        for group in self.classes_:
            Xg = X[y == group, :]
            meang = Xg.mean(0)
            means.append(meang)
            Xgc = Xg - meang
            Xc.append(Xgc)
            # covg = _cov(Xg, self.estimator)
            # covg = np.atleast_2d(covg)
            # covs.append(covg)
        self.means_ = np.asarray(means)
        self.xbar_ = np.dot(self.priors_, self.means_)
        # cov_ = np.mean(covs, 0)
    
        Xc = np.concatenate(Xc, axis=0)
    
        # 1) within (univariate) scaling by with classes std-dev
        std = Xc.std(axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.
        fac = 1. / (n_samples - n_classes)
    
        # 2) Within variance scaling
        X = np.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        U, S, V = linalg.svd(X, full_matrices=False)
    
        rank = np.sum(S > tol)
        if rank < n_features:
            warnings.warn("variables are collinear")
        # Scaling of within covariance is: V' 1/S
        scalings = (V[:rank] / std).T / S[:rank]
    
        # 3) Between variance scaling
        # Scale weighted centers
        X = np.dot(((np.sqrt((n_samples * self.priors_) * fac)) *
                    (self.means_ - self.xbar_).T).T, scalings)
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, V = linalg.svd(X, full_matrices=0)
    
        rank = np.sum(S > tol * S[0])
        self.scalings_ = np.dot(scalings, V.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = (-0.5 * np.sum(coef**2, axis=1)
                           + np.log(self.priors_))
        self.coef_ = np.dot(coef, self.scalings_.T)

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array, shape = [n_samples]
            Target values
        """
        X, y = check_X_y(X, y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape

        # TODO: support equal priors (should probably be the default)
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        if self.alpha is None:
            self.alpha = 'empirical'

        if self.solver == 'svd':
            self._solve_svd(X, y, self.alpha)
        elif self.solver == 'lsqr':
            self._solve_lsqr(X, y, self.alpha)
        elif self.solver == 'eigen':
            self._solve_eigen(X, y, self.alpha)

    def _decision_function(self, X):
        X = check_array(X)
        return np.dot(X - self.xbar_, self.coef_.T) + self.intercept_

    def decision_function(self, X):
        """
        This function returns the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,], giving the
            log likelihood ratio of the positive class.
        """
        dec_func = self._decision_function(X)
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def transform(self, X):
        """
        Project the data so as to maximize class separation (large separation
        between projected class means and small variance within each class).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components]
        """
        X = check_array(X)
        # center and scale data
        X_new = np.dot(X - self.xbar_, self.scalings_)
        n_components = X.shape[1] if self.n_components is None \
            else self.n_components
        return X_new[:, :n_components]

    def predict(self, X):
        """
        This function does classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
        """
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X):
        """
        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
        """
        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        """
        This function returns posterior log-probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
        """
        values = self._decision_function(X)
        loglikelihood = (values - values.max(axis=1)[:, np.newaxis])
        normalization = logsumexp(loglikelihood, axis=1)
        return loglikelihood - normalization[:, np.newaxis]
