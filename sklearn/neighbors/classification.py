"""Nearest Neighbor Classification"""

# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck <L.J.Buitinck@uva.nl>
#
# License: BSD, (C) INRIA, University of Amsterdam

import numpy as np
from scipy import stats
from ..utils.extmath import weighted_mode

from .base import \
    _check_weights, _get_weights, \
    NeighborsBase, KNeighborsMixin,\
    RadiusNeighborsMixin, SupervisedIntegerMixin
from ..base import ClassifierMixin
from ..utils import atleast2d_or_csr


class KNeighborsClassifier(NeighborsBase, KNeighborsMixin,
                           SupervisedIntegerMixin, ClassifierMixin):
    """Classifier implementing the k-nearest neighbors vote.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    weights : str or callable
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`scipy.spatial.cKDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    warn_on_equidistant : boolean, optional.  Defaults to True.
        Generate a warning if equidistant neighbors are discarded.
        For classification or regression based on k-neighbors, if
        neighbor k and neighbor k+1 have identical distances but
        different labels, then the result will be dependent on the
        ordering of the training data.
        If the fit method is ``'kd_tree'``, no warnings will be generated.

    p: integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> neigh = KNeighborsClassifier(n_neighbors=2)
    >>> neigh.fit(X, y) # doctest: +ELLIPSIS
    KNeighborsClassifier(...)
    >>> print neigh.predict([[1.5]])
    [0]

    See also
    --------
    RadiusNeighborsClassifier
    KNeighborsRegressor
    RadiusNeighborsRegressor
    NearestNeighbors

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 algorithm='auto', leaf_size=30,
                 warn_on_equidistant=True, p=2):
        self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size,
                          warn_on_equidistant=warn_on_equidistant,
                          p=p)
        self.weights = _check_weights(weights)

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X: array
            A 2-D array representing the test points.

        Returns
        -------
        labels: array
            List of class labels (one for each data sample).
        """
        X = atleast2d_or_csr(X)

        neigh_dist, neigh_ind = self.kneighbors(X)
        pred_labels = self._y[neigh_ind]

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = stats.mode(pred_labels, axis=1)
        else:
            mode, _ = weighted_mode(pred_labels, weights, axis=1)

        return mode.flatten().astype(np.int)


class RadiusNeighborsClassifier(NeighborsBase, RadiusNeighborsMixin,
                                SupervisedIntegerMixin, ClassifierMixin):
    """Classifier implementing a vote among neighbors within a given radius

    Parameters
    ----------
    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth`radius_neighbors`
        queries.

    weights : str or callable
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Uniform weights are used by default.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`scipy.spatial.cKDtree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p: integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.


    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import RadiusNeighborsClassifier
    >>> neigh = RadiusNeighborsClassifier(radius=1.0)
    >>> neigh.fit(X, y) # doctest: +ELLIPSIS
    RadiusNeighborsClassifier(...)
    >>> print neigh.predict([[1.5]])
    [0]

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsRegressor
    KNeighborsRegressor
    NearestNeighbors

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, radius=1.0, weights='uniform',
                 algorithm='auto', leaf_size=30, p=2):
        self._init_params(radius=radius,
                          algorithm=algorithm,
                          leaf_size=leaf_size,
                          p=p)
        self.weights = _check_weights(weights)

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X: array
            A 2-D array representing the test points.

        Returns
        -------
        labels: array
            List of class labels (one for each data sample).
        """
        X = atleast2d_or_csr(X)

        neigh_dist, neigh_ind = self.radius_neighbors(X)
        pred_labels = [self._y[ind] for ind in neigh_ind]

        for pl in pred_labels:
            # Check that all have at least 1 neighbor
            if len(pl) < 1:
                raise ValueError('no neighbors found for a test sample, '
                                 'try using larger radius or removing '
                                 'outliers')
        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode = np.asarray([stats.mode(pl)[0] for pl in pred_labels],
                              dtype=np.int)
        else:
            mode = np.asarray([weighted_mode(pl, w)[0]
                               for (pl, w) in zip(pred_labels, weights)],
                              dtype=np.int)

        return mode.flatten().astype(np.int)
