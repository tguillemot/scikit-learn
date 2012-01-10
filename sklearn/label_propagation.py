#coding=utf8
"""
Label propagation in the context of this module refers to a set of
semisupervised classification algorithms. In the high level, these algorithms
work by forming a fully-connected graph between all points given and solving
for the steady-state distribution of labels at each point.

These algorithms perform very well in practice. The cost of running can be very
expensive, at approximately O(N^3) where N is the number of (labeled and
unlabeled) points. The theory (why they perform so well) is motivated by
intuitions from random walk algorithms and geometric relationships in the data.
For more information see the references below.

Model Features
--------------
Label clamping:
  The algorithm tries to learn distributions of labels over the dataset. In the
  "Hard Clamp" mode, the true ground labels are never allowed to change. They
  are clamped into position. In the "Soft Clamp" mode, they are allowed some
  wiggle room, but some alpha of their original value will always be retained.
  Hard clamp is the same as soft clamping with alpha set to 1.

Kernel:
  A function which projects a vector into some higher dimensional space. See
  the documentation for SVMs for more info on kernels.

Example
-------
>>> from sklearn import datasets
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> random_unlabeled_points = np.where(np.random.random_integers(0, 1,
...        size=len(iris.target)))
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
LabelPropagation(alpha=1, gamma=20, kernel='rbf', max_iters=30, n_neighbors=7,
         tol=0.001, unlabeled_identifier=-1)

=======
>>> label_prop_model.fit(iris.data, labels, unlabeled_identifier=-1)
... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
LabelPropagation(...)

Notes
-----
References:
[1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
Learning (2006), pp. 193-216
"""

import numpy as np
from scipy import sparse

from .base import BaseEstimator, ClassifierMixin
from .metrics.pairwise import rbf_kernel
from .neighbors.graph import kneighbors_graph
from .utils.graph import graph_laplacian
from .utils.fixes import divide_out

from neighbors.unsupervised import NearestNeighbors

# Authors: Clay Woolam <clay@woolam.org>

EPSILON = 1e-9


class BaseLabelPropagation(BaseEstimator, ClassifierMixin):
    """
    Base class for label propagation module.

    Parameters
    ----------
    kernel : string
      string identifier for kernel function to use
      only 'rbf' kernel is currently supported
    gamma : float
      parameter for rbf kernel
    alpha : float
      clamping factor

    unlabeled_identifier : any object, same class as label objects
      a special identifier label that represents unlabeled examples
      in the training set

    max_iters : float
      change maximum number of iterations allowed
    tol : float
      threshold to consider the system at steady state
    """

    def __init__(self, kernel='rbf', gamma=20, alpha=1,
            unlabeled_identifier=-1, max_iters=30, n_neighbors=2,
            tol=1e-3):
        self.max_iters = max_iters
        self.tol = tol

        # object referring to a point that is unlabeled
        self.unlabeled_identifier = unlabeled_identifier

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if y is None:
                return kneighbors_graph(X, self.n_neighbors)
            else:
                return NearestNeighbors(self.n_neighbors).fit(X)\
                        .kneighbors(y, return_distance=False)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf \
                             supported at this time" % self.kernel)

    def _build_graph(self):
        raise NotImplementedError("Graph construction must be implemented \
                to fit a label propagation model.")

    def predict(self, X):
        """Performs inductive inference across the model.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input data
        """
        probas = self.predict_proba(X)
        return self.unique_labels_[np.argmax(probas, axis=1)].flatten()

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array_like, shape = (n_features, n_features)

        Return
        ------
        inference : array of normalized probability distributions across class
        labels
        """
        X_2d = np.atleast_2d(X)
        weight_matrices = self._get_kernel(self.X_, X_2d)
        if self.kernel == 'knn':
            inference = []
            for weight_matrix in weight_matrices:
                ine = np.sum(self.label_distributions_[weight_matrix], axis=0)
                inference.append(ine)
            inference = np.array(inference)
        else:
            weight_matrices = weight_matrices.T
            inference = np.dot(weight_matrices, self.label_distributions_)
        normalizer = np.atleast_2d(np.sum(inference, axis=1)).T
        divide_out(inference, normalizer, out=inference)
        return inference

    def fit(self, X, y):
        """
        Fit a semi-supervised label propagation model based on input data
        matrix X and corresponding label matrix Y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_freatures]
          A {n_samples by n_samples} size matrix will be created from this
          (keep dataset fewer than 2000 points)

        y : array_like, shape = [n_labeled_samples]
          n_labeled_samples (unlabeled points marked with a special identifier)
          All unlabeled samples will be transductively assigned labels

        Returns
        -------
        updated LabelPropagation object with a new transduction results
        """
        self.X_ = np.asanyarray(X)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        unique_labels = np.unique(y)
        unique_labels = (unique_labels[unique_labels !=
                                                    self.unlabeled_identifier])
        self.unique_labels_ = unique_labels

        n_samples, n_classes = len(y), len(unique_labels)

        y = np.asanyarray(y)
        unlabeled = y == self.unlabeled_identifier
        clamp_weights = np.ones((n_samples, 1))
        clamp_weights[unlabeled, 0] = self.alpha

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in unique_labels:
            self.label_distributions_[y == label, unique_labels == label] = 1

        y_static = np.copy(self.label_distributions_)
        if self.alpha > 0.:
            y_static = y_static * (1 - self.alpha)
        y_static[unlabeled] = 0

        l_previous = np.zeros((self.X_.shape[0], n_classes))
        self.label_distributions_.resize((self.X_.shape[0], n_classes))

        remaining_iter = self.max_iters
        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()
        while (_not_converged(self.label_distributions_, l_previous, self.tol)
                and remaining_iter > 1):
            l_previous = self.label_distributions_
            if sparse.isspmatrix(graph_matrix):
                self.label_distributions_ = graph_matrix *\
                        self.label_distributions_
            else:
                self.label_distributions_ = np.dot(graph_matrix,
                        self.label_distributions_)
            # clamp
            self.label_distributions_ = np.multiply(clamp_weights,
                    self.label_distributions_) + y_static
            remaining_iter -= 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        divide_out(self.label_distributions_, normalizer,
                  out=self.label_distributions_)
        # set the transduction item
        transduction = self.unique_labels_[np.argmax(self.label_distributions_,
                axis=1)]
        self.transduction_ = transduction.flatten()
        return self


class LabelPropagation(BaseLabelPropagation):
    """
    Computes a stochastic affinity matrix and uses hard clamping.

    Parameters
    ----------
    kernel : string
      string identifier for kernel function to use
      only 'rbf' kernel is currently supported
    gamma : float
      parameter for rbf kernel
    n_neighbors : integer > 0
      parameter for knn kernel
    alpha : float
      clamping factor
    unlabeled_identifier : any object, same class as label objects
      a special identifier label that represents unlabeled examples
      in the training set
    max_iters : float
      change maximum number of iterations allowed
    tol : float
      threshold to consider the system at steady state

    Examples
    --------
    >>> from sklearn import datasets
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> random_unlabeled_points = np.where(np.random.random_integers(0, 1,
    ...    size=len(iris.target)))
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LabelPropagation(...)

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    See Also
    --------
    LabelSpreading : Alternate label proagation strategy more robust to noise
    """
    def _build_graph(self):
        """
        Builds a matrix representing a fully connected graph between each point
        in the dataset.

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired)
        """
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        if sparse.isspmatrix(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            divide_out(affinity_matrix, normalizer[:, np.newaxis],
                    out=affinity_matrix)
        return affinity_matrix


class LabelSpreading(BaseLabelPropagation):
    """
    Similar to the basic Label Propgation algorithm, but uses affinity matrix
    based on the graph laplacian and soft clamping accross the labels.

    Parameters
    ----------
    kernel : string
      string identifier for kernel function to use
      only 'rbf' kernel is currently supported
    gamma : float
      parameter for rbf kernel
    n_neighbors : integer > 0
      parameter for knn kernel
    alpha : float
      clamping factor
    unlabeled_identifier : any object, same class as label objects
      a special identifier label that represents unlabeled examples
      in the training set
    max_iters : float
      change maximum number of iterations allowed
    tol : float
      threshold to consider the system at steady state

    Examples
    --------
    >>> from sklearn import datasets
    >>> label_prop_model = LabelSpreading()
    >>> iris = datasets.load_iris()
    >>> random_unlabeled_points = np.where(np.random.random_integers(0, 1,
    ...    size=len(iris.target)))
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels, unlabeled_identifier=-1)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LabelSpreading(...)

    References
    ----------
    Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
    Bernhard Schölkopf. Learning with local and global consistency (2004)
    http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.115.3219

    See Also
    --------
    Label Propagation : Unregularized graph based semi-supervised learning
    """

    def __init__(self, kernel='rbf', gamma=20, alpha=0.2,
            unlabeled_identifier=-1, max_iters=30, tol=1e-3):
        # this one has different base parameters
        super(LabelSpreading, self).__init__(kernel=kernel, gamma=gamma,
                alpha=alpha, unlabeled_identifier=unlabeled_identifier,
                max_iters=max_iters, tol=tol)

    def _build_graph(self):
        """Graph matrix for Label Spreading computes the graph laplacian"""
        # compute affinity matrix (or gram matrix)
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        laplacian = graph_laplacian(affinity_matrix, normed=True)
        laplacian = -laplacian
        if sparse.isspmatrix(laplacian):
            diag_mask = (laplacian.row == laplacian.col)
            laplacian.data[diag_mask] = 0.0
        else:
            laplacian.flat[::n_samples + 1] = 0.0  # set diag to 0.0
        return laplacian


### Helper functions

def _not_converged(y_truth, y_prediction, tol=1e-3):
    """basic convergence check"""
    return np.sum(np.abs(np.asarray(y_truth - y_prediction))) > tol
