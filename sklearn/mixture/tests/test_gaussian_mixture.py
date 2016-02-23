import sys
import numpy as np
from scipy import stats

from ...datasets.samples_generator import make_spd_matrix
from ...utils.testing import assert_array_almost_equal
from ...utils.testing import assert_array_equal
from ...utils.testing import assert_equal
from ...utils.testing import assert_true
from ...utils.testing import assert_raise_message
from ...utils.testing import assert_almost_equal
from ...utils.testing import assert_allclose
from ...utils.testing import assert_greater
from ...utils.testing import assert_warns_message
from ...utils import ConvergenceWarning

from ...mixture.gaussian_mixture import GaussianMixture
from ...mixture.gaussian_mixture import estimate_Gaussian_suffstat_Sk
from ...covariance import EmpiricalCovariance
from ...metrics.cluster import adjusted_rand_score
from ...externals.six.moves import cStringIO as StringIO


rng = np.random.RandomState(0)
COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']


def generate_data(n_samples, n_features, weights, means, covariances,
                  covariance_type):
    X = []
    if covariance_type == 'spherical':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['spherical'])):
            X.append(rng.multivariate_normal(m, c * np.eye(n_features),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'diag':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'tied':
        for k, (w, m) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, covariances['tied'],
                                             int(np.round(w * n_samples))))
    if covariance_type == 'full':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['full'])):
            X.append(rng.multivariate_normal(m, c,
                                             int(np.round(w * n_samples))))

    X = np.vstack(X)
    return X


class RandomData(object):
    def __init__(self, rng):
        n_samples = 500
        n_components = 2
        n_features = 2
        weights = rng.rand(n_components)
        weights = weights / weights.sum()
        means = rng.rand(n_components, n_features) * 50
        covariances = {'spherical': .5 + rng.rand(n_components),
                       'diag': (.5 + rng.rand(n_components, n_features)) ** 2,
                       'tied': make_spd_matrix(n_features, random_state=rng),
                       'full': np.array([make_spd_matrix(
                           n_features, random_state=rng) * .5
                           for _ in range(n_components)])}
        X = dict(zip(COVARIANCE_TYPE, [generate_data(
            n_samples, n_features, weights, means, covariances, cov_type)
            for cov_type in COVARIANCE_TYPE]))
        Y = np.hstack([k * np.ones(np.round(w * n_samples)) for k, w in
                      enumerate(weights)])
        (self.n_samples, self.n_components, self.n_features, self.weights,
         self.means, self.covariances, self.X, self.Y) = (
            n_samples, n_components, n_features, weights, means, covariances,
            X, Y)


RandData = RandomData(rng)


def test_GaussianMixture_parameters():
    # test bad parameters

    # n_init should be greater than 0
    n_init = rng.randint(-10, 1)
    assert_raise_message(ValueError,
                         "Invalid value for 'n_init': %d "
                         "Estimation requires at least one run"
                         % n_init,
                         GaussianMixture, n_init=n_init)

    max_iter = 0
    assert_raise_message(ValueError,
                         "Invalid value for 'max_iter': %d "
                         "Estimation requires at least one iteration"
                         % max_iter,
                         GaussianMixture, max_iter=max_iter)

    reg_covar = -1
    assert_raise_message(ValueError,
                         "Invalid value for 'reg_covar': %.5f "
                         "regularization on covariance must be "
                         "non-negative" % reg_covar,
                         GaussianMixture, reg_covar=reg_covar)

    # covariance_type should be in [spherical, diag, tied, full]
    covariance_type = 'bad_covariance_type'
    assert_raise_message(ValueError,
                         "Invalid value for 'covariance_type': %s "
                         "'covariance_type' should be in "
                         "['spherical', 'tied', 'diag', 'full']"
                         % covariance_type,
                         GaussianMixture,
                         covariance_type=covariance_type)

    init_params = 'bad_method'
    g = GaussianMixture(init_params=init_params)
    assert_raise_message(ValueError,
                         "Unimplemented initialization method '%s'"
                         % init_params, g._initialize, X=rng.rand(10, 2))


def test_check_X():
    from sklearn.mixture.base import check_X
    n_samples = RandData.n_samples
    n_components = RandData.n_components
    n_features = RandData.n_features

    X_bad_dim = rng.rand(n_samples)
    assert_raise_message(ValueError,
                         'Expected the input data X have 2 dimensions, '
                         'but got %d dimension(s)' % X_bad_dim.ndim,
                         check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components - 1, n_features)
    assert_raise_message(ValueError,
                         'Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X_bad_dim.shape[0]),
                         check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components, n_features + 1)
    assert_raise_message(ValueError,
                         'Expected the input data X have %d features, '
                         'but got %d features'
                         % (n_features, X_bad_dim.shape[1]),
                         check_X, X_bad_dim, n_components, n_features)

    X = rng.rand(n_samples, n_features)
    assert_array_equal(X, check_X(X, n_components, n_features))


def test_check_weights():
    n_components = RandData.n_components

    weights_bad = rng.rand(n_components, 1)
    g = GaussianMixture(weights=weights_bad, n_components=n_components)
    assert_raise_message(ValueError,
                         "The parameter 'weights' should have the shape of "
                         "(%d,), "
                         "but got %s" % (n_components, str(weights_bad.shape)),
                         g._check_initial_parameters)

    weights_bad = rng.rand(n_components) + 1
    g = GaussianMixture(weights=weights_bad, n_components=n_components)
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights_bad), np.max(weights_bad)),
                         g._check_initial_parameters)

    weights_bad = rng.rand(n_components)
    weights_bad = weights_bad/(weights_bad.sum() + 1)
    g = GaussianMixture(weights=weights_bad, n_components=n_components)
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights_bad),
                         g._check_initial_parameters)

    weights = RandData.weights
    g = GaussianMixture(weights=weights, n_components=n_components)
    g._check_initial_parameters()
    assert_array_equal(weights, g.weights_init)


def test_check_means():
    n_components = RandData.n_components
    n_features = RandData.n_features

    means_bad = rng.rand(n_components + 1, n_features)
    g = GaussianMixture(means=means_bad, n_components=n_components)
    g.n_features = n_features
    assert_raise_message(ValueError,
                         "The parameter 'means' should have the shape of "
                         "(%d, %d), but got %s"
                         % (n_components, n_features, str(means_bad.shape)),
                         g._check_initial_parameters)

    means = RandData.means
    g = GaussianMixture(means=means, n_components=n_components)
    g.n_features = n_features
    g._check_initial_parameters()
    assert_array_equal(means, g.means_init)


def test_check_covars():
    n_components = RandData.n_components
    n_features = RandData.n_features

    # full
    covars_bad = rng.rand(n_components + 1, n_features, n_features)
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='full')
    g.n_features = n_features
    assert_raise_message(
        ValueError,
        "The parameter 'full covariance' should have the "
        "shape of (%d, %d, %d), but got %s"
        % (n_components, n_features, n_features, str(covars_bad.shape)),
        g._check_initial_parameters)

    covars_bad = rng.rand(n_components, n_features, n_features)
    covars_bad[0] = np.eye(n_features)
    covars_bad[0, 0, 0] = -1
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='full')
    g.n_features = n_features
    assert_raise_message(
        ValueError,
        "The component %d of 'full covariance' "
        "should be symmetric, positive-definite"
        % 0, g._check_initial_parameters)

    covars = RandData.covariances['full']
    g = GaussianMixture(n_components=n_components, covars=covars,
                        covariance_type='full')
    g.n_features = n_features
    g._check_initial_parameters()
    assert_array_equal(covars, g.covars_init)

    # tied
    covars_bad = rng.rand(n_features + 1, n_features + 1)
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='tied')
    g.n_features = n_features
    assert_raise_message(
        ValueError,
        "The parameter 'tied covariance' should have the shape of "
        "(%d, %d), but got %s"
        % (n_features, n_features, str(covars_bad.shape)),
        g._check_initial_parameters)
    covars_bad = np.eye(n_features)
    covars_bad[0, 0] = -1
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='tied')
    g.n_features = n_features
    assert_raise_message(ValueError, "'tied covariance' should be "
                         "symmetric, positive-definite",
                         g._check_initial_parameters)
    covars = RandData.covariances['tied']
    g = GaussianMixture(n_components=n_components, covars=covars,
                        covariance_type='tied')
    g.n_features = n_features
    g._check_initial_parameters()
    assert_array_equal(covars, g.covars_init)

    # diag
    covars_bad = rng.rand(n_components + 1, n_features)
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='diag')
    g.n_features = n_features
    assert_raise_message(
        ValueError,
        "The parameter 'diag covariance' should have the shape of "
        "(%d, %d), but got %s"
        % (n_components, n_features, str(covars_bad.shape)),
        g._check_initial_parameters)

    covars_bad = np.ones((n_components, n_features)) * -1
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='diag')
    g.n_features = n_features
    assert_raise_message(ValueError, "'diag covariance' should be positive",
                         g._check_initial_parameters)
    covars = RandData.covariances['diag']
    g = GaussianMixture(n_components=n_components, covars=covars,
                        covariance_type='diag')
    g.n_features = n_features
    g._check_initial_parameters()
    assert_array_equal(covars, g.covars_init)

    # spherical
    covars_bad = rng.rand(n_components + 1)
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='spherical')
    g.n_features = n_features
    assert_raise_message(
        ValueError,
        "The parameter 'spherical covariance' should have the "
        "shape of (%d,), but got %s"
        % (n_components, str(covars_bad.shape)),
        g._check_initial_parameters)
    covars_bad = np.ones(n_components)
    covars_bad[0] = -1
    g = GaussianMixture(n_components=n_components, covars=covars_bad,
                        covariance_type='spherical')
    g.n_features = n_features
    assert_raise_message(ValueError, "'spherical covariance' should be "
                                     "positive",
                         g._check_initial_parameters)
    covars = RandData.covariances['spherical']
    g = GaussianMixture(n_components=n_components, covars=covars,
                        covariance_type='spherical')
    g.n_features = n_features
    g._check_initial_parameters()
    assert_array_equal(covars, g.covars_init)


def test_suffstat_Sk_full():
    # compare the EmpiricalCovariance.covariance fitted on X*sqrt(resp)
    # with _sufficient_sk_full, n_components=1
    n_samples = RandData.n_samples
    n_features = RandData.n_features

    # special case 1, assuming data is "centered"
    X = rng.rand(n_samples, n_features)
    resp = rng.rand(n_samples, 1)
    X_resp = np.sqrt(resp) * X
    nk = np.array([n_samples])
    xk = np.zeros((1, n_features))
    covars_pred = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0, 'full')
    ecov = EmpiricalCovariance(assume_centered=True)
    ecov.fit(X_resp)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)

    # special case 2, assuming resp are all ones
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean().reshape((1, -1))
    covars_pred = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0, 'full')
    ecov = EmpiricalCovariance(assume_centered=False)
    ecov.fit(X)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)


def test_suffstat_Sk_tied():
    # use equation Nk * Sk / N = S_tied
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0,
                                                     'full')
    covars_pred_full = np.sum(nk[:, np.newaxis, np.newaxis] * covars_pred_full,
                              0) / n_samples

    covars_pred_tied = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0,
                                                     'tied')
    ecov = EmpiricalCovariance()
    ecov.covariance_ = covars_pred_full
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='spectral'), 0)


def test_suffstat_Sk_diag():
    # test against 'full' case
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0,
                                                     'full')
    covars_pred_full = np.array([np.diag(np.diag(d)) for d in
                                 covars_pred_full])
    covars_pred_diag = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0,
                                                     'diag')
    covars_pred_diag = np.array([np.diag(d) for d in covars_pred_diag])
    ecov = EmpiricalCovariance()
    for (cov_full, cov_diag) in zip(covars_pred_full, covars_pred_diag):
        ecov.covariance_ = cov_full
        assert_almost_equal(ecov.error_norm(cov_diag, norm='frobenius'), 0)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='spectral'), 0)


def test_Gaussian_suffstat_Sk_spherical():
    # computing spherical covariance equals to the variance of one-dimension
    # data after flattening, n_components=1
    n_samples = RandData.n_samples
    n_features = RandData.n_features

    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, 0,
                                                          'spherical')
    covars_pred_spherical2 = (np.dot(X.flatten().T, X.flatten()) /
                              (n_features * n_samples))
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)


def _naive_lmvnpdf_diag(X, means, covars):
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars)
    for i, (mean, std) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp


def test_GaussianMixture_log_probabilities():
    # test aginst with _naive_lmvnpdf_diag
    from ..gaussian_mixture import (_estimate_log_Gaussian_prob_full,
                                    _estimate_log_Gaussian_prob_diag,
                                    _estimate_log_Gaussian_prob_tied,
                                    _estimate_log_Gaussian_prob_spherical)
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    means = RandData.means
    covars_diag = rng.rand(n_components, n_features)
    X = rng.rand(n_samples, n_features)
    log_prob_naive = _naive_lmvnpdf_diag(X, means, covars_diag)

    # full covariances
    covars_full = np.array([np.diag(x) for x in covars_diag])

    log_prob = _estimate_log_Gaussian_prob_full(X, means, covars_full)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # diag covariances
    log_prob = _estimate_log_Gaussian_prob_diag(X, means, covars_diag)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # tied
    covars_tied = covars_full.mean(axis=0)
    log_prob_naive = _naive_lmvnpdf_diag(X, means,
                                         [np.diag(covars_tied)] * n_components)
    log_prob = _estimate_log_Gaussian_prob_tied(X, means, covars_tied)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # spherical
    covars_spherical = covars_diag.mean(axis=1)
    log_prob_naive = _naive_lmvnpdf_diag(X, means,
                                         [[k] * n_features for k in
                                          covars_spherical])
    log_prob = _estimate_log_Gaussian_prob_spherical(X, means,
                                                     covars_spherical)
    assert_array_almost_equal(log_prob, log_prob_naive)

# skip tests on weighted_log_probabilities, log_weights


def test_GaussianMixture_estimate_log_prob_resp():
    # test whether responsibilities are normalized
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    X = rng.rand(n_samples, n_features)
    for cov_type in COVARIANCE_TYPE:
        weights = RandData.weights
        means = RandData.means
        covariances = RandData.covariances[cov_type]
        g = GaussianMixture(n_components=n_components, random_state=rng,
                            weights=weights, means=means, covars=covariances,
                            covariance_type=cov_type)
        g._initialize(X)
        _, _, resp, _ = g._estimate_log_prob_resp(X)
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))


def test_GaussianMixture_predict_predict_proba():
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        Y = RandData.Y
        g = GaussianMixture(n_components=RandData.n_components,
                            random_state=rng, weights=RandData.weights,
                            means=RandData.means,
                            covars=RandData.covariances[cov_type],
                            covariance_type=cov_type)
        g._initialize(X)
        Y_pred = g.predict(X)
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        assert_array_equal(Y_pred, Y_pred_proba)
        assert_greater(adjusted_rand_score(Y, Y_pred), .95)


def test_GaussianMixture_fit():
    # recover the ground truth
    n_features = RandData.n_features
    n_components = RandData.n_components

    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = GaussianMixture(n_components=n_components, n_init=20, max_iter=100,
                            reg_covar=0, random_state=rng,
                            covariance_type=cov_type)
        g.fit(X)
        # needs more data to pass the test with rtol=1e-7
        assert_allclose(np.sort(g.weights_), np.sort(RandData.weights),
                        rtol=0.1, atol=1e-2)

        arg_idx1 = g.means_[:, 0].argsort()
        arg_idx2 = RandData.means[:, 0].argsort()
        assert_allclose(g.means_[arg_idx1], RandData.means[arg_idx2],
                        rtol=0.1, atol=1e-2)

        if cov_type == 'spherical':
            cov_pred = np.array([np.eye(n_features) * c for c in g.covars_])
            cov_test = np.array([np.eye(n_features) * c for c in
                                 RandData.covariances['spherical']])
        elif cov_type == 'diag':
            cov_pred = np.array([np.diag(d) for d in g.covars_])
            cov_test = np.array([np.diag(d) for d in
                                 RandData.covariances['diag']])
        elif cov_type == 'tied':
            cov_pred = np.array([g.covars_] * n_components)
            cov_test = np.array([RandData.covariances['tied']] * n_components)
        elif cov_type == 'full':
            cov_pred = g.covars_
            cov_test = RandData.covariances['full']
        arg_idx1 = np.trace(cov_pred, axis1=1, axis2=2).argsort()
        arg_idx2 = np.trace(cov_test, axis1=1, axis2=2).argsort()
        for k, h in zip(arg_idx1, arg_idx2):
            ecov = EmpiricalCovariance()
            ecov.covariance_ = cov_test[h]
            # the accuracy depends on the number of data and randomness, rng
            assert_allclose(ecov.error_norm(cov_pred[k]), 0, atol=0.1)


def test_GaussianMixture_fit_best_params():
    n_components = RandData.n_components
    n_init = 10
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = GaussianMixture(n_components=n_components, n_init=1,
                            max_iter=100, reg_covar=0, random_state=rng,
                            covariance_type=cov_type)
        ll = []
        for _ in range(n_init):
            g.fit(X)
            ll.append(g.score(X))
        ll = np.array(ll)
        g_best = GaussianMixture(n_components=n_components,
                                 n_init=n_init, max_iter=100, reg_covar=0,
                                 random_state=rng, covariance_type=cov_type)
        g_best.fit(X)
        assert_almost_equal(ll.min(), g_best.score(X))


def test_GaussianMixture_fit_convergence_warning():
    n_components = RandData.n_components
    max_iter = 1
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = GaussianMixture(n_components=n_components, n_init=1,
                            max_iter=max_iter, reg_covar=0, random_state=rng,
                            covariance_type=cov_type)
        assert_warns_message(ConvergenceWarning,
                             'Initialization %d is not converged. '
                             'Try different init parameters, '
                             'or increase n_init, tol '
                             'or check for degenerate data.'
                             % max_iter, g.fit, X)


def test_GaussianMixture_n_parameters():
    # Test that the right number of parameters is estimated
    n_samples, n_features, n_components = 7, 5, 2
    n_params = {'spherical': 13, 'diag': 21, 'tied': 26, 'full': 41}
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(
            n_components=n_components, covariance_type=cv_type,
            random_state=rng)
        g.n_features = n_features
        assert_equal(g._n_parameters(), n_params[cv_type])


def test_GaussianMixture_aic_bic():
    # Test the aic and bic criteria
    n_samples, n_features, n_components = 50, 3, 2
    X = rng.randn(n_samples, n_features)
    # standard gaussian entropy
    SGH = 0.5 * (np.log(np.linalg.det(np.cov(X.T, bias=1))) +
                 n_features * (1 + np.log(2 * np.pi)))
    for cv_type in COVARIANCE_TYPE:
        g = GaussianMixture(
            n_components=n_components, covariance_type=cv_type,
            random_state=rng)
        g.fit(X)
        aic = 2 * n_samples * SGH + 2 * g._n_parameters()
        bic = (2 * n_samples * SGH +
               np.log(n_samples) * g._n_parameters())
        bound = n_features * 3. / np.sqrt(n_samples)
        assert_true(np.abs(g.aic(X) - aic) / n_samples < bound)
        assert_true(np.abs(g.bic(X) - bic) / n_samples < bound)


def test_GaussianMixture_verbose():
    n_components = RandData.n_components
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = GaussianMixture(n_components=n_components, n_init=1,
                            max_iter=100, reg_covar=0, random_state=rng,
                            covariance_type=cov_type, verbose=1)
        h = GaussianMixture(n_components=n_components, n_init=1,
                            max_iter=100, reg_covar=0, random_state=rng,
                            covariance_type=cov_type, verbose=2)
        k = GaussianMixture(n_components=n_components, n_init=1,
                            max_iter=100, reg_covar=0, random_state=rng,
                            covariance_type=cov_type, verbose=20)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            g.fit(X)
            h.fit(X)
            k.fit(X)
        finally:
            sys.stdout = old_stdout
