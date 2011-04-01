import numpy as np
from .. import nmf
from nose.tools import ok_, assert_true, assert_false, raises

rng = np.random.mtrand.RandomState(0)
    
@raises(ValueError)
def test_initialize_nn_input():
    """
    Test _initialize_nmf_ behaviour on negative input
    """
    nmf._initialize_nmf_(-np.ones((2,2)), 2)

def test_initialize_nn_output():
    """
    Test that _initialize_nmf_ does not suggest negative values anywhere.
    """

    data = np.abs(rng.randn(10,10))
    for var in (None, 'a', 'ar'):
        W, H = nmf._initialize_nmf_(data, 10)
        assert_false((W < 0).any() or (H < 0).any())

def test_initialize_close():
    """
    Test that _initialize_nmf_ error is
    less than the standard deviation 
    of the entries in the matrix
    """
    A = np.abs(rng.randn(10,10))
    W, H = nmf._initialize_nmf_(A, 10)
    error = np.linalg.norm(np.dot(W, H) - A)
    sdev = np.linalg.norm(A - A.mean())
    assert_true(error <= sdev)

def test_initialize_variants():
    """
    Test that the variants 'a' and 'ar'
    differ from basic NNDSVD only where
    the basic version has zeros
    """
    data = np.abs(rng.randn(10,10))
    W0, H0 = nmf._initialize_nmf_(data, 10, variant=None)
    Wa, Ha = nmf._initialize_nmf_(data, 10, variant='a')
    War, Har = nmf._initialize_nmf_(data, 10, variant='ar')

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_true(np.allclose(evl[ref <> 0], ref[ref <> 0]))
        
@raises(ValueError)
def test_fit_nn_input():
    """
    Test model fit behaviour on negative input
    """
    A = -np.ones((2,2))
    m = nmf.NMF(2, init=None)
    m.fit(A)

def test_fit_nn_output():
    """
    Test that the model does not use negative values anywhere
    """
    A = np.c_[5 * np.ones(5) - xrange(1, 6),
              5 * np.ones(5) + xrange(1, 6)]
    for init in (None, 'nndsvd', 'cro'):
        model = nmf.NMF(2, init=init)
        transf = model.fit_transform(A)
        assert_false((model.components_ < 0).any() or
                     (transf < 0).any())

def test_fit_nn_close():
    """
    Test that the fit is "close enough"
    """
    assert nmf.NMF(5).fit(np.abs(
      rng.randn(6, 5))).reconstruction_err_ < 0.05

@raises(ValueError)
def test_nls_nn_input():
    """
    Test NLS behaviour on negative input
    """
    A = np.ones((2,2))
    nmf._nls_subproblem_(A, A, -A, 0.001, 20)

def test_nls_nn_output():
    """
    Test NLS doesn't return negative input.
    """
    A = np.atleast_2d(range(1,5))
    Ap, _, _ = nmf._nls_subproblem_(np.dot(A.T, -A), A.T, A, 0.001, 100)
    assert_false((Ap < 0).any())

def test_nls_close():
    """
    Test that the NLS results should be close
    """
    A = np.atleast_2d(range(1,5))
    Ap, _, _ = nmf._nls_subproblem_(np.dot(A.T, A), A.T, np.zeros_like(A), 
                                    0.001, 100)
    assert_true((np.abs(Ap - A) < 0.01).all())

def test_nmf_transform():
    """
    Test that NMF.transform returns close values
    (transform uses scipy.optimize.nnls for now)
    """
    A = np.abs(rng.randn(6, 5))
    m = nmf.NMF(5)
    transf = m.fit_transform(A)
    assert_true(np.allclose(transf, m.transform(A), atol=1e-2, rtol=0))

def test_nmf_sparseness():
    """
    Test that sparsity contraints actually increase
    sparseness where appropriate
    """
    
    A = np.abs(rng.randn(10, 10))
    m = nmf.NMF(5).fit(A)
    data_sp = nmf.NMF(5, sparseness='data').fit(A).data_sparseness_
    comp_sp = nmf.NMF(5, sparseness='components').fit(A).comp_sparseness_
    assert_true(data_sp > m.data_sparseness_ and comp_sp > m.comp_sparseness_)
    
if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
