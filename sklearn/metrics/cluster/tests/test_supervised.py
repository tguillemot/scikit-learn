import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import expected_mutual_information
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import entropy

from sklearn.utils.testing import assert_raise_message
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal


score_funcs = [
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
]


def test_error_messages_on_wrong_input():
    for score_func in score_funcs:
        expected = ('labels_true and labels_pred must have same size,'
                    ' got 2 and 3')
        assert_raise_message(ValueError, expected, score_func,
                             [0, 1], [1, 1, 1])

        expected = "labels_true must be 1D: shape is (2"
        assert_raise_message(ValueError, expected, score_func,
                             [[0, 1], [1, 0]], [1, 1, 1])

        expected = "labels_pred must be 1D: shape is (2"
        assert_raise_message(ValueError, expected, score_func,
                             [0, 1, 0], [[1, 1], [0, 0]])


def test_perfect_matches():
    for score_func in score_funcs:
        assert_equal(score_func([], []), 1.0)
        assert_equal(score_func([0], [1]), 1.0)
        assert_equal(score_func([0, 0, 0], [0, 0, 0]), 1.0)
        assert_equal(score_func([0, 1, 0], [42, 7, 42]), 1.0)
        assert_equal(score_func([0., 1., 0.], [42., 7., 42.]), 1.0)
        assert_equal(score_func([0., 1., 2.], [42., 7., 2.]), 1.0)
        assert_equal(score_func([0, 1, 2], [42, 7, 2]), 1.0)


def test_homogeneous_but_not_complete_labeling():
    # homogeneous but not complete clustering
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 2, 2])
    assert_almost_equal(h, 1.00, 2)
    assert_almost_equal(c, 0.69, 2)
    assert_almost_equal(v, 0.81, 2)


def test_complete_but_not_homogeneous_labeling():
    # complete but not homogeneous clustering
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 1, 1])
    assert_almost_equal(h, 0.58, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 0.73, 2)


def test_not_complete_and_not_homogeneous_labeling():
    # neither complete nor homogeneous but not so bad either
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)


def test_non_consicutive_labels():
    # regression tests for labels with gaps
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 2, 2, 2],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)

    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 4, 0, 4, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)

    ari_1 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    ari_2 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    assert_almost_equal(ari_1, 0.24, 2)
    assert_almost_equal(ari_2, 0.24, 2)


def uniform_labelings_scores(score_func, n_samples, k_range, n_runs=10,
                             seed=42):
    # Compute score for random uniform cluster labelings
    random_labels = np.random.RandomState(seed).random_integers
    scores = np.zeros((len(k_range), n_runs))
    for i, k in enumerate(k_range):
        for j in range(n_runs):
            labels_a = random_labels(low=0, high=k - 1, size=n_samples)
            labels_b = random_labels(low=0, high=k - 1, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def test_adjustment_for_chance():
    # Check that adjusted scores are almost zero on random labels
    n_clusters_range = [2, 10, 50, 90]
    n_samples = 100
    n_runs = 10

    scores = uniform_labelings_scores(
        adjusted_rand_score, n_samples, n_clusters_range, n_runs)

    max_abs_scores = np.abs(scores).max(axis=1)
    assert_array_almost_equal(max_abs_scores, [0.02, 0.03, 0.03, 0.02], 2)


def test_adjusted_mutual_info_score():
    # Compute the Adjusted Mutual Information and test against known values
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    # Mutual information
    mi = mutual_info_score(labels_a, labels_b)
    assert_almost_equal(mi, 0.41022, 5)
    # Expected mutual information
    C = contingency_matrix(labels_a, labels_b)
    n_samples = np.sum(C)
    emi = expected_mutual_information(C, n_samples)
    assert_almost_equal(emi, 0.15042, 5)
    # Adjusted mutual information
    ami = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami, 0.27502, 5)
    ami = adjusted_mutual_info_score([1, 1, 2, 2], [2, 2, 3, 3])
    assert_equal(ami, 1.0)
    # Test with a very large array
    a110 = np.array([list(labels_a) * 110]).flatten()
    b110 = np.array([list(labels_b) * 110]).flatten()
    ami = adjusted_mutual_info_score(a110, b110)
    # This is not accurate to more than 2 places
    assert_almost_equal(ami, 0.37, 2)


def test_entropy():
    ent = entropy([0, 0, 42.])
    assert_almost_equal(ent, 0.6365141, 5)
    assert_almost_equal(entropy([]), 1)


def test_contingency_matrix():
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])
    C = contingency_matrix(labels_a, labels_b)
    C2 = np.histogram2d(labels_a, labels_b,
                        bins=(np.arange(1, 5),
                              np.arange(1, 5)))[0]
    assert_array_almost_equal(C, C2)
    C = contingency_matrix(labels_a, labels_b, eps=.1)
    assert_array_almost_equal(C, C2 + .1)


def test_exactly_zero_info_score():
    # Check numerical stability when information is exactly zero
    for i in np.logspace(1, 4, 4).astype(np.int):
        labels_a, labels_b = np.ones(i, dtype=np.int),\
            np.arange(i, dtype=np.int)
        assert_equal(normalized_mutual_info_score(labels_a, labels_b,
                                                  max_n_classes=1e4), 0.0)
        assert_equal(v_measure_score(labels_a, labels_b,
                                     max_n_classes=1e4), 0.0)
        assert_equal(adjusted_mutual_info_score(labels_a, labels_b,
                                                max_n_classes=1e4), 0.0)
        assert_equal(normalized_mutual_info_score(labels_a, labels_b,
                                                  max_n_classes=1e4), 0.0)


def test_v_measure_and_mutual_information(seed=36):
    # Check relation between v_measure, entropy and mutual information
    for i in np.logspace(1, 4, 4).astype(np.int):
        random_state = np.random.RandomState(seed)
        labels_a, labels_b = random_state.random_integers(0, 10, i),\
            random_state.random_integers(0, 10, i)
        assert_almost_equal(v_measure_score(labels_a, labels_b),
                            2.0 * mutual_info_score(labels_a, labels_b) /
                            (entropy(labels_a) + entropy(labels_b)), 0)


def test_max_n_classes():
    labels_true = np.random.random(5003)
    labels_pred = np.random.random(5003)
    labels_zero = np.zeros(5003)
    labels_true[:2] = 0
    labels_zero[:3] = 1
    labels_pred[:2] = 0
    for score_func in score_funcs:
        expected = ("Too many classes for a clustering metric. If you "
                    "want to increase the limit, pass parameter "
                    "max_n_classes to the scoring function")
        assert_raise_message(ValueError, expected, score_func,
                             labels_true, labels_pred,
                             max_n_classes=500)
        expected = ("Too many clusters for a clustering metric. If you "
                    "want to increase the limit, pass parameter "
                    "max_n_classes to the scoring function")
        assert_raise_message(ValueError, expected, score_func,
                             labels_zero, labels_pred,
                             max_n_classes=500)
