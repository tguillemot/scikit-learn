"""
Testing for Clustering methods

"""

import numpy as np
from numpy.testing import assert_equal
from scipy.spatial import distance

from ..dbscan_ import DBSCAN, dbscan
from .common import generate_clustered_data


n_clusters = 3
X = generate_clustered_data(n_clusters=n_clusters)


def test_dbscan():
    """
    Tests the DBSCAN algorithm

    """
    # Compute similarities
    D = distance.squareform(distance.pdist(X))
    S = 1 - (D / np.max(D))

    # Compute DBSCAN
    # parameters chosen for task
    core_points, labels = dbscan(S, eps=0.85, min_points=10)

    # number of clusters, ignoring noise if present
    n_clusters_1 = len(set(labels)) - (1 if -1 in labels else 0)

    assert_equal(n_clusters, n_clusters_1)

    db = DBSCAN()
    labels = db.fit(S, eps=0.85, min_points=10).labels_
    core_points = db.core_points_

    n_clusters_2 = len(set(labels)) - (1 if -1 in labels else 0)
    assert_equal(n_clusters, n_clusters_2)
