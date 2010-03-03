# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD Style.

# $Id$

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises

from ..cd import lasso_coordinate_descent_slow
from ..cd import lasso_coordinate_descent_fast
from ..cd import Lasso

from ..cd import enet_coordinate_descent_slow
from ..cd import enet_coordinate_descent_fast
from ..cd import ElasticNet
from ..cd import lasso_path
from ..cd import enet_path


def test_lasso_cd_python_cython_sanity():
    n_samples, n_features, maxit = 100, 50, 150
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)

    model_slow = Lasso(alpha=1)
    assert_array_almost_equal(model_slow.density(), 0)
    model_slow.learner = lasso_coordinate_descent_slow
    model_slow.fit(X, y, maxit=maxit)

    # check the convergence using the KKT condition
    assert_array_almost_equal(model_slow.gap, 0, 4)

    model_fast = Lasso(alpha=1)
    model_fast.learner = lasso_coordinate_descent_fast
    model_fast.fit(X, y, maxit=maxit)

    # check t convergence using the KKT condition
    assert_array_almost_equal(model_fast.gap, 0, 4)

    # check that python and cython implementations behave exactly the same
    assert_array_almost_equal(model_slow.w, model_fast.w)
    assert_array_almost_equal(model_slow.E, model_fast.E)

    # check that the priori induces sparsity in the weights (feature selection)
    assert_array_almost_equal(model_fast.density(), 0.88, 2)

def test_enet_cd_python_cython_sanity():
    n_samples, n_features, maxit = 100, 50, 150
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)

    model_slow = ElasticNet(alpha=1, beta=10)
    model_slow.learner = enet_coordinate_descent_slow
    model_slow.fit(X, y, maxit=maxit)

    # check the convergence using the KKT condition
    assert_array_almost_equal(model_slow.gap, 0, 4)

    model_fast = ElasticNet(alpha=1, beta=10)
    model_fast.learner = enet_coordinate_descent_fast
    model_fast.fit(X, y, maxit=maxit)

    # check t convergence using the KKT condition
    assert_array_almost_equal(model_fast.gap, 0, 4)

    assert_array_almost_equal(model_slow.w, model_fast.w)
    assert_array_almost_equal(model_slow.E, model_fast.E)

    # check that the priori induces sparsity in the weights
    # (feature selection) but not 
    assert_array_almost_equal(model_slow.density(), 0.90, 2)


def test_lasso_enet_cd_paths():
    """Test Lasso and Elastic-Net path functions
    """
    n_samples, n_features, maxit = 5, 10, 30
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)

    alphas_lasso, weights_lasso = lasso_path(X, y, factor=0.97, n_alphas = 50)
    alphas_enet, weights_enet = enet_path(X, y, factor=0.97, n_alphas = 50,
                                          beta=0.1)
