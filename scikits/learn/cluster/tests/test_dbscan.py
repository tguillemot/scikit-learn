"""
Tests for DBSCAN clustering algorithm
"""

import numpy as np
from numpy.testing import assert_equal
from scipy.spatial import distance

from ..dbscan_ import DBSCAN, dbscan
from .common import generate_clustered_data


n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters)


def test_dbscan_similarity():
    """Tests the DBSCAN algorithm with a similarity array."""
    # Parameters chosen specifically for this task.
    eps = 0.15
    min_points = 10
    # Compute similarities
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    # Compute DBSCAN
    core_points, labels = dbscan(D, metric="precomputed",
                                 eps=eps, min_points=min_points)
    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - (1 if -1 in labels else 0)

    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN(metric="precomputed")
    labels = db.fit(D, eps=eps, min_points=min_points).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)


def test_dbscan_feature():
    """Tests the DBSCAN algorithm with a feature vector array."""
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_points = 10
    metric = 'euclidean'
    # Compute DBSCAN
    # parameters chosen for task
    core_points, labels = dbscan(X, metric=metric,
                                 eps=eps, min_points=min_points)

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN()
    labels = db.fit(X, metric=metric,
                    eps=eps, min_points=min_points).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)


def test_dbscan_callable():
    """Tests the DBSCAN algorithm with a callable metric."""
    # Parameters chosen specifically for this task.
    # Different eps to other test, because distance is not normalised.
    eps = 0.8
    min_points = 10
    # metric is the function reference, not the string key.
    metric = distance.euclidean
    # Compute DBSCAN
    # parameters chosen for task
    core_points, labels = dbscan(X, metric=metric,
                                 eps=eps, min_points=min_points)

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_1, n_clusters)

    db = DBSCAN()
    labels = db.fit(X, metric=metric,
                    eps=eps, min_points=min_points).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
    assert_equal(n_clusters_2, n_clusters)
