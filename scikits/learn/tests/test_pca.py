import numpy as np
from numpy.random import randn
from nose.tools import assert_true


#from scikits.learn import datasets
#from scikits.learn.pca import PCA, ProbabilisticPCA, _assess_dimension_, _infer_dimension_
from .. import datasets
from ..pca import PCA, ProbabilisticPCA, _assess_dimension_, _infer_dimension_

iris = datasets.load_iris()

X = iris.data

def test_pca():
    """
    PCA
    """
    pca = PCA(n_comp=2)
    X_r = pca.fit(X).transform(X)
    np.testing.assert_equal(X_r.shape[1], 2)

    pca = PCA()
    pca.fit(X)
    np.testing.assert_almost_equal(pca.explained_variance_ratio_.sum(),
                                   1.0, 3)

def test_pca_check_projection():
    """test that the projection of data is correct
    """
    n, p = 100, 3
    X = randn(n, p)*.1
    X[:10] += np.array([3, 4, 5])
    pca = PCA(n_comp=2)
    pca.fit(X)
    Xt = 0.1* randn(1, p) + np.array([3, 4, 5])
    Yt = pca.transform(Xt)
    Yt /= np.sqrt((Yt**2).sum())
    np.testing.assert_almost_equal(Yt[0][0], 1., 1)

def test_pca_dim():
    """
    """
    n, p = 100, 5
    X = randn(n, p)*.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    pca = PCA(n_comp='mle')
    pca.fit(X)
    assert_true(pca.n_comp==1)

def test_infer_dim_1():
    """
    """
    n, p = 1000, 5
    X = randn(n, p)*0.1 + randn(n, 1)*np.array([3, 4, 5, 1, 2])+ np.array(
        [1, 0, 7, 4, 6])
    pca = PCA(n_comp=p)
    pca.fit(X)
    spect = pca.explained_variance_
    ll = []
    for k in range(p):
         ll.append(_assess_dimension_(spect, k, n, p))
    ll = np.array(ll)
    assert_true(ll[1]>ll.max()-.01*n)

def test_infer_dim_2():
    """
    """
    n, p = 1000, 5
    X = randn(n, p)*.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    pca = PCA(n_comp=p)
    pca.fit(X)
    spect = pca.explained_variance_
    assert_true(_infer_dimension_(spect, n, p)>1)

def test_infer_dim_3():
    """
    """
    n, p = 100, 5
    X = randn(n, p)*.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    X[30:40] += 2*np.array([-1, 1, -1, 1, -1])
    pca = PCA(n_comp=p)
    pca.fit(X)
    spect = pca.explained_variance_
    print _infer_dimension_(spect, n, p)
    assert_true(_infer_dimension_(spect, n, p)>2)


def test_probabilistic_pca_1():
    """test that probabilistic PCA yields a readonable score
    """
    n, p = 1000, 3
    X = randn(n, p)*.1 + np.array([3, 4, 5])
    ppca = ProbabilisticPCA(n_comp=2)
    ppca.fit(X)
    ll1 = ppca.score(X)
    h = 0.5*np.log(2*np.pi*np.exp(1)/0.1**2)*p
    np.testing.assert_almost_equal(ll1.mean()/h, 1, 0)

def test_probabilistic_pca_2():
    """test that probabilistic PCA correctly separated different datasets
    """
    n, p = 100, 3
    X = randn(n, p)*.1 + np.array([3, 4, 5])
    ppca = ProbabilisticPCA(n_comp=2)
    ppca.fit(X)
    ll1 = ppca.score(X)
    ll2 = ppca.score(randn(n, p)*.2 + np.array([3, 4, 5]))
    assert_true(ll1.mean()>ll2.mean())

def test_probabilistic_pca_3():
    """The homoscedastic model should work slightly worth
    than the heteroscedastic one in over-fitting condition
    """
    n, p = 100, 3
    X = randn(n, p)*.1 + np.array([3, 4, 5])
    ppca = ProbabilisticPCA(n_comp=2)
    ppca.fit(X)
    ll1 = ppca.score(X)
    ppca.fit(X, False)
    ll2 = ppca.score(X)
    assert_true(ll1.mean()<ll2.mean())

def test_probabilistic_pca_4():
    """Check that ppca select the right model
    """
    n, p = 200, 3
    Xl = randn(n, p) + randn(n, 1)*np.array([3, 4, 5]) + np.array([1, 0, 7])
    Xt = randn(n, p) + randn(n, 1)*np.array([3, 4, 5]) + np.array([1, 0, 7])
    ll = np.zeros(p)
    for k in range(p):
        ppca = ProbabilisticPCA(n_comp=k)
        ppca.fit(Xl)
        ll[k] = ppca.score(Xt).mean()
        
    assert_true(ll.argmax()==1)


    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
