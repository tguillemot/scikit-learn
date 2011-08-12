"""
Label propagation in the context of this module refers to a set of
semisupervised classification algorithms. In the high level, these algorithms
work by forming a fully-connected graph between all points given and solving
for the steady-state distribution of labels at each point.

These algorithms perform very well in practice. The cost of running can be very
expensive, at approximately O(N^3) where N is the number of (labeled and
unlabeled) points. The theory (why they perform so well) is motivated by
intuitions from random walk algorithms and geometric relationships in the data.
For more information see [1].

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
>>> from scikits.learn import datasets
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> random_unlabeled_points = np.where(np.random.random_integers(0, 1,
        size=len(iris.target)))
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels, unlabeled_identifier=-1)
... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
LabelPropagation(...)


References
----------
[1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
    Learning (2006), pp. 193-216
"""
import numpy as np
from .base import BaseEstimator, ClassifierMixin
from .externals.joblib.logger import Logger
from .metrics.pairwise import rbf_kernel

logger = Logger()

# Authors: Clay Woolam <clay@woolam.org>

# really low epsilon (we don't really want machine eps)
EPSILON = 1e-9
# Main classes


class BaseLabelPropagation(BaseEstimator, ClassifierMixin):
    """
    Base class for label propagation module.

    Parameters
    ----------
    kernel : function (array_1, array_2) -> float
      kernel function to use
    gamma : float
      parameter to initialize the kernel function
    alpha : float
      clamping factor

    max_iters : float
      change maximum number of iterations allowed
    conv_threshold : float
      threshold to consider the system at steady state
    """

    def __init__(self, kernel='rbf', gamma=20, alpha=1,
            unlabeled_identifier=-1, max_iters=100,
            conv_threshold=1e-3, suppress_warning=False):
        self.max_iters = max_iters
        self.conv_threshold = conv_threshold
        self.gamma = gamma
        self.suppress_warning = suppress_warning

        # object referring to a point that is unlabeled
        self.unlabeled_identifier = unlabeled_identifier

        self.kernel = kernel

        # clamping factor
        self.alpha = alpha

    def _get_kernel(self, X, Y):
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf \
                             supported at this time" % self.kernel)

    def _build_graph(self):
        """
        Builds a matrix representing a fully connected graph between each point
        in the dataset

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired)
        """
        return self._get_kernel(self._X, self._X)

    def predict(self, X):
        """
        Performs inductive inference across the model

        Parameters
        ----------
        X : array_like, shape = [n_points, n_features]

        Returns
        -------
        y : array_like, shape = [n_points]
            Predictions for input data
        """
        ym = self.predict_proba(X)
        preds = np.argmax(ym, axis=1)
        return [self.num_to_label[pred] for pred in preds.flat]

    def predict_proba(self, X):
        """
        Returns a probability distribution (categorical distribution)
        over labels for a single input point.

        Parameters
        ----------
        X : array_like, shape = (n_features, n_features)
        """
        ary = np.atleast_2d(X)
        return self._get_kernel(self._X, ary).T * self._y

    def fit(self, X, y, **params):
        """
        Fit a semi-supervised label propagation model based on input data
        matrix X and corresponding label matrix Y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_freatures]
          A {n_samples by n_samples} size matrix will be created from this
          (keep dataset fewer than 2000 points)
        y : array, shape = [n_labeled_samples]
          n_labeled_samples (unlabeled points marked with a special identifier)
          All unlabeled samples will be transductively assigned labels

        Examples
        --------
        >>> from scikits.learn import datasets
        >>> label_prop_model = BaseLabelPropagation()
        >>> iris = datasets.load_iris()
        >>> random_unlabeled_points = np.where(np.random.random_integers(0, 1,
            size=len(iris.target)))
        >>> labels = np.copy(iris.target)
        >>> labels[random_unlabeled_points] = -1
        >>> label_prop_model.fit(iris.data, labels, unlabeled_identifier=-1)
        ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        BaseLabelPropagation(...)

        Warning
        -------
        Base class works, but intended for instruction. It will generate a n
        onstochastic affinity matrix.

        Returns
        -------
        updated LabelPropagation object with a new variable called transduction
        """
        self._set_params(**params)
        self._X = np.asanyarray(X)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        unq_labels = np.unique(y)
        unq_labels = unq_labels[unq_labels != self.unlabeled_identifier]
        
        num_labels, num_classes = len(y), len(unq_labels)
        self.label_map = dict(zip(unq_labels, range(num_classes)))

        y_st = np.asanyarray(y)
        self.unlabeled_points = np.where(y_st == self.unlabeled_identifier)
        alpha_ary = np.ones((num_labels, 1))
        alpha_ary[self.unlabeled_points, 0] = self.alpha

        self._y = np.zeros((num_labels, num_classes))
        for label in unq_labels:
            self._y[np.where(y_st == label), self.label_map[label]] = 1

        # this should do elementwise multiplication between the two arrays
        # Y_alpha = self._y * (1 - alpha_ary)
        Y_alpha = np.copy(self._y)
        Y_alpha = Y_alpha * (1 - self.alpha)
        Y_alpha[self.unlabeled_points] = 0

        y_p = np.zeros((self._X.shape[0], num_classes))
        self._y.resize((self._X.shape[0], num_classes))

        max_iters = self.max_iters
        ct = self.conv_threshold
        while not_converged(self._y, y_p, ct) and max_iters > 1:
            y_p = self._y
            self._y = np.dot(graph_matrix, self._y)
            # clamp
            self._y = np.multiply(alpha_ary, self._y) + Y_alpha
            max_iters -= 1
        # set the transduction item

        num_to_label = dict([reversed(itm) for itm in self.label_map.items()])
        transduction = map(lambda x: num_to_label[np.argmax(x)], self._y)
        self.num_to_label, self.transduction = num_to_label, transduction
        return self


class LabelPropagation(BaseLabelPropagation):
    """
    Original label propagation algorithm. Computes a basic affinity matrix and
    uses hard clamping.
    """
    def _build_graph(self):
        affinity_matrix = self._get_kernel(self._X, self._X)
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=0))
        deg_inv = np.linalg.inv(degree_matrix)
        aff_ideg = deg_inv * np.matrix(affinity_matrix)
        return aff_ideg


class LabelSpreading(BaseLabelPropagation):
    """
    Similar to the basic Label Propgation algorithm, but uses affinity matrix
    based on the graph laplacian and uses soft clamping for labels.

    Parameters
    ----------
    alpha : float
      "clamping factor" or how much of the original labels you want to keep
    """
    
    def __init__(self, kernel='rbf', gamma=20, alpha=0.2,
            unlabeled_identifier=-1, max_iters=100,
            conv_threshold=1e-3, suppress_warning=False):
        # this one has different base parameters
        super(LabelSpreading, self).__init__()

    def _build_graph(self):
        """
        Graph matrix for Label Spreading uses the Graph Laplacian
        """
        # compute affinity matrix (or gram matrix)
        n_samples = self._X.shape[0]
        affinity_matrix = self._get_kernel(self._X, self._X)
        affinity_matrix[np.diag_indices(n_samples)] = 0
        degree_matrix = np.diag(np.sum(affinity_matrix, axis=0))
        deg_invsq = np.sqrt(np.linalg.inv(degree_matrix))
        laplacian = deg_invsq * np.matrix(affinity_matrix) * deg_invsq
        return laplacian

### Helper functions


def not_converged(y, y_hat, threshold=1e-3):
    """basic convergence check"""
    return np.sum(np.abs(np.asarray(y - y_hat))) > threshold
