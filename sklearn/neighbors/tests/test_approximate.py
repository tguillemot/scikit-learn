"""
Testing for the approximate neighbor search using
Locality Sensitive Hashing Forest module
(sklearn.neighbors.LSHForest).
"""

# Author: Maheshakya Wijewardena

import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_warns

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors


def test_neighbors_accuracy_with_n_candidates():
    """Checks whether accuracy increases as `n_candidates` increases."""
    n_candidates_values = np.array([.1, 50, 500])
    n_samples = 100
    n_features = 10
    n_iter = 10
    n_points = 5
    rng = np.random.RandomState(42)
    accuracies = np.zeros(n_candidates_values.shape[0], dtype=float)
    X = rng.rand(n_samples, n_features)

    for i, n_candidates in enumerate(n_candidates_values):
        lshf = LSHForest(n_candidates=n_candidates)
        lshf.fit(X)
        for j in range(n_iter):
            query = X[rng.randint(0, n_samples)]
            neighbors = lshf.kneighbors(query, n_neighbors=n_points,
                                        return_distance=False)
            distances = pairwise_distances(query, X, metric='cosine')
            ranks = np.argsort(distances)[0, :n_points]

            intersection = np.intersect1d(ranks, neighbors).shape[0]
            ratio = intersection / float(n_points)
            accuracies[i] = accuracies[i] + ratio

        accuracies[i] = accuracies[i] / float(n_iter)
    # Sorted accuracies should be equal to original accuracies
    assert_true(np.all(np.diff(accuracies) >= 0),
                msg="Accuracies are not non-decreasing.")


def test_neighbors_accuracy_with_n_estimators():
    """Checks whether accuracy increases as `n_estimators` increases."""
    n_estimators = np.array([1, 10, 100])
    n_samples = 100
    n_features = 10
    n_iter = 10
    n_points = 5
    rng = np.random.RandomState(42)
    accuracies = np.zeros(n_estimators.shape[0], dtype=float)
    X = rng.rand(n_samples, n_features)

    for i, t in enumerate(n_estimators):
        lshf = LSHForest(n_candidates=500, n_estimators=t)
        lshf.fit(X)
        for j in range(n_iter):
            query = X[rng.randint(0, n_samples)]
            neighbors = lshf.kneighbors(query, n_neighbors=n_points,
                                        return_distance=False)
            distances = pairwise_distances(query, X, metric='cosine')
            ranks = np.argsort(distances)[0, :n_points]

            intersection = np.intersect1d(ranks, neighbors).shape[0]
            ratio = intersection / float(n_points)
            accuracies[i] = accuracies[i] + ratio

        accuracies[i] = accuracies[i] / float(n_iter)
    # Sorted accuracies should be equal to original accuracies
    assert_true(np.all(np.diff(accuracies) >= 0),
                msg="Accuracies are not non-decreasing.")


def test_kneighbors():
    """Checks whether desired number of neighbors are returned.

    It is guaranteed to return the requested number of neighbors
    if `min_hash_match` is set to 0. Returned distances should be
    in ascending order.
    """
    n_samples = 12
    n_features = 2
    n_iter = 10
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)

    lshf = LSHForest(min_hash_match=0)
    # Test unfitted estimator
    assert_raises(ValueError, lshf.kneighbors, X[0])

    lshf.fit(X)

    for i in range(n_iter):
        n_neighbors = rng.randint(0, n_samples)
        query = X[rng.randint(0, n_samples)]
        neighbors = lshf.kneighbors(query, n_neighbors=n_neighbors,
                                    return_distance=False)
        # Desired number of neighbors should be returned.
        assert_equal(neighbors.shape[1], n_neighbors)

    # Multiple points
    n_queries = 5
    queries = X[rng.randint(0, n_samples, n_queries)]
    distances, neighbors = lshf.kneighbors(queries,
                                           n_neighbors=1,
                                           return_distance=True)
    assert_equal(neighbors.shape[0], n_queries)
    assert_equal(distances.shape[0], n_queries)
    # Test only neighbors
    neighbors = lshf.kneighbors(queries, n_neighbors=1,
                                return_distance=False)
    assert_equal(neighbors.shape[0], n_queries)
    # Test random point(not in the data set)
    query = rng.randn(n_features)
    lshf.kneighbors(query, n_neighbors=1,
                    return_distance=False)
    # Test n_neighbors at initialization
    neighbors = lshf.kneighbors(query, return_distance=False)
    assert_equal(neighbors.shape[1], 5)


def test_radius_neighbors():
    """Checks whether Returned distances are less than `radius`

    At least one point should be returned when the `radius` is set
    to mean distance from the considering point to other points in
    the database.
    Moreover, this test compares the radius neighbors of LSHForest
    with the `sklearn.neighbors.NearestNeighbors`.
    """
    n_samples = 12
    n_features = 2
    n_iter = 10
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)

    lshf = LSHForest()
    # Test unfitted estimator
    assert_raises(ValueError, lshf.radius_neighbors, X[0])

    lshf.fit(X)

    for i in range(n_iter):
        query = X[rng.randint(0, n_samples)]
        mean_dist = np.mean(pairwise_distances(query, X, metric='cosine'))
        neighbors = lshf.radius_neighbors(query, radius=mean_dist,
                                          return_distance=False)
        # At least one neighbor should be returned.
        assert_greater(neighbors.shape[0], 0)
        # All distances should be less than mean_dist
        distances, neighbors = lshf.radius_neighbors(query,
                                                     radius=mean_dist,
                                                     return_distance=True)
        assert_array_less(distances[0], mean_dist)

    # Multiple points
    n_queries = 5
    queries = X[rng.randint(0, n_samples, n_queries)]
    distances, neighbors = lshf.radius_neighbors(queries,
                                                 return_distance=True)
    assert_equal(neighbors.shape[0], n_queries)
    assert_equal(distances.shape[0], n_queries)
    # dists and inds should not be 2D arrays
    assert_equal(distances.ndim, 1)
    assert_equal(neighbors.ndim, 1)

    # Compare with exact neighbor search
    query = X[rng.randint(0, n_samples)]
    mean_dist = np.mean(pairwise_distances(query, X, metric='cosine'))
    nbrs = NearestNeighbors(algorithm='brute', metric='cosine')
    nbrs.fit(X)

    distances_approx, _ = lshf.radius_neighbors(query, radius=mean_dist)
    distances_exact, _ = nbrs.radius_neighbors(query, radius=mean_dist)
    # Distances of exact neighbors is less than or equal to approximate
    assert_true(np.all(np.less_equal(np.sort(distances_exact[0]),
                                     np.sort(distances_approx[0]))))


def test_distances():
    """Checks whether returned neighbors are from closest to farthest."""
    n_samples = 12
    n_features = 2
    n_iter = 10
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)

    lshf = LSHForest()
    lshf.fit(X)

    for i in range(n_iter):
        n_neighbors = rng.randint(0, n_samples)
        query = X[rng.randint(0, n_samples)]
        distances, neighbors = lshf.kneighbors(query,
                                               n_neighbors=n_neighbors,
                                               return_distance=True)
        # Returned neighbors should be from closest to farthest.
        assert_true(np.all(np.diff(distances[0]) >= 0))

        mean_dist = np.mean(pairwise_distances(query, X, metric='cosine'))
        distances, neighbors = lshf.radius_neighbors(query,
                                                     radius=mean_dist,
                                                     return_distance=True)
        assert_true(np.all(np.diff(distances[0]) >= 0))


def test_fit():
    """Checks whether `fit` method sets all attribute values correctly."""
    n_samples = 12
    n_features = 2
    n_estimators = 5
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)

    lshf = LSHForest(n_estimators=n_estimators)
    lshf.fit(X)

    # _input_array = X
    assert_array_equal(X, lshf._fit_X)
    # A hash function g(p) for each tree
    assert_equal(n_estimators, len(lshf.hash_functions_))
    # Hash length = 32
    assert_equal(32, lshf.hash_functions_[0].components_.shape[0])
    # Number of trees_ in the forest
    assert_equal(n_estimators, len(lshf.trees_))
    # Each tree has entries for every data point
    assert_equal(n_samples, len(lshf.trees_[0]))
    # Original indices after sorting the hashes
    assert_equal(n_estimators, len(lshf.original_indices_))
    # Each set of original indices in a tree has entries for every data point
    assert_equal(n_samples, len(lshf.original_indices_[0]))


def test_partial_fit():
    """Checks whether inserting array is consitent with fitted data.

    `partial_fit` method should set all attribute values correctly.
    """
    n_samples = 12
    n_samples_partial_fit = 3
    n_features = 2
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)
    X_partial_fit = rng.rand(n_samples_partial_fit, n_features)

    lshf = LSHForest()

    # Test unfitted estimator
    lshf.partial_fit(X)
    assert_array_equal(X, lshf._fit_X)

    lshf.fit(X)

    # Insert wrong dimension
    assert_raises(ValueError, lshf.partial_fit,
                  np.random.randn(n_samples_partial_fit, n_features - 1))

    lshf.partial_fit(X_partial_fit)

    # size of _input_array = samples + 1 after insertion
    assert_equal(lshf._fit_X.shape[0],
                 n_samples + n_samples_partial_fit)
    # size of original_indices_[1] = samples + 1
    assert_equal(len(lshf.original_indices_[0]),
                 n_samples + n_samples_partial_fit)
    # size of trees_[1] = samples + 1
    assert_equal(len(lshf.trees_[1]),
                 n_samples + n_samples_partial_fit)


def test_hash_functions():
    """Checks randomness of hash functions.

    Variance and mean of each hash function (projection vector)
    should be different from flattened array of hash functions.
    If hash functions are not randomly built (seeded with
    same value), variances and means of all functions are equal.
    """
    n_samples = 12
    n_features = 2
    n_estimators = 5
    rng = np.random.RandomState(42)
    X = rng.rand(n_samples, n_features)

    lshf = LSHForest(n_estimators=n_estimators,
                     random_state=rng.randint(0, np.iinfo(np.int32).max))
    lshf.fit(X)

    hash_functions = []
    for i in range(n_estimators):
        hash_functions.append(lshf.hash_functions_[i].components_)

    for i in range(n_estimators):
        assert_not_equal(np.var(hash_functions),
                         np.var(lshf.hash_functions_[i].components_))

    for i in range(n_estimators):
        assert_not_equal(np.mean(hash_functions),
                         np.mean(lshf.hash_functions_[i].components_))


def test_candidates():
    """Checks whether candidates are sufficient.

    This should handle the cases when number of candidates is 0.
    User should be warned when number of candidates is less than
    requested number of neighbors.
    """
    X_train = [[5, 5, 2], [21, 5, 5], [1, 1, 1], [8, 9, 1], [6, 10, 2]]
    X_test = [7, 10, 3]

    # For zero candidates
    lshf = LSHForest(min_hash_match=32)
    lshf.fit(X_train)

    assert_warns(UserWarning, lshf.kneighbors, X_test, n_neighbors=3)
    distances, neighbors = lshf.kneighbors(X_test, n_neighbors=3)
    assert_equal(distances.shape[1], 3)

    # For candidates less than n_neighbors
    lshf = LSHForest(min_hash_match=31)
    lshf.fit(X_train)

    assert_warns(UserWarning, lshf.kneighbors, X_test, n_neighbors=5)
    distances, neighbors = lshf.kneighbors(X_test, n_neighbors=5)
    assert_equal(distances.shape[1], 5)


def test_graphs():
    """Smoke tests for graph methods."""
    n_samples_sizes = [5, 10, 20]
    n_features = 3
    rng = np.random.RandomState(42)

    for n_samples in n_samples_sizes:
        X = rng.rand(n_samples, n_features)
        lshf = LSHForest(min_hash_match=0)
        lshf.fit(X)

        kneighbors_graph = lshf.kneighbors_graph(X)
        radius_neighbors_graph = lshf.radius_neighbors_graph(X)

        assert_equal(kneighbors_graph.shape[0], n_samples)
        assert_equal(kneighbors_graph.shape[1], n_samples)
        assert_equal(radius_neighbors_graph.shape[0], n_samples)
        assert_equal(radius_neighbors_graph.shape[1], n_samples)


if __name__ == "__main__":
    import nose
    nose.runmodule()
