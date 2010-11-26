"""
Benchmarks of Lasso vs LassoLARS

First, we fix a training set and increase the number of
samples. Then we plot the computation time as function of
the number of samples.

In the second benchmark, we increase the number of dimensions of the
training set. Then we plot the computation time as function of
the number of dimensions.
 
In both cases, only 10% of the features are informative.
"""
import gc
from time import time
import numpy as np

from bench_glmnet import make_data



def compute_bench(alpha, n_samples, n_features):

    lasso_results = []
    larslasso_results = []
    larslasso_gram_results = []

    n_tests = 1000
    it = 0

    for ns in n_samples:
        for nf in n_features:
            it += 1
            print '=================='
            print 'Iteration %s of %s' % (it, max(len(n_samples), len(n_features)))
            print '=================='
            k = nf // 10
            X, Y, X_test, Y_test, coef_ = make_data(
                n_samples=ns, n_tests=n_tests, n_features=nf,
                noise=0.1, k=k)

            X /= np.sqrt(np.sum(X**2, axis=0)) # Normalize data

            gc.collect()
            print "benching Lasso: "
            clf = Lasso(alpha=alpha, fit_intercept=False)
            tstart = time()
            clf.fit(X, Y)
            lasso_results.append(time() - tstart)

            gc.collect()
            print "benching LassoLARS: "
            clf = LassoLARS(alpha=alpha, fit_intercept=False)
            tstart = time()
            clf.fit(X, Y, normalize=False, precompute=False)
            larslasso_results.append(time() - tstart)

            gc.collect()
            print "benching LassoLARS (precomp. Gram): "
            clf = LassoLARS(alpha=alpha, fit_intercept=False)
            tstart = time()
            clf.fit(X, Y, normalize=False, precompute=True)
            larslasso_gram_results.append(time() - tstart)

    return lasso_results, larslasso_results, larslasso_gram_results


if __name__ == '__main__':
    from scikits.learn.linear_model import Lasso, LassoLARS
    import pylab as pl

    alpha = 0.01 # regularization parameter

    n_features = 500
    list_n_samples = range(500, 20501, 1000);
    lasso_results, larslasso_results, larslasso_gram_results = \
                    compute_bench(alpha, list_n_samples, [n_features])

    pl.subplot(211)
    pl.title('Lasso benchmark (%d features - alpha=%s)' % (n_features, alpha))
    pl.plot(list_n_samples, lasso_results, 'b-', label='Lasso')
    pl.plot(list_n_samples, larslasso_results,'r-', label='LassoLARS')
    pl.plot(list_n_samples, larslasso_gram_results,'g-', label='LassoLARS (with precomputed Gram matrix)')
    pl.legend(loc='upper left')
    pl.xlabel('number of samples')
    pl.ylabel('time (in seconds)')
    pl.axis('tight')

    n_samples = 2000
    list_n_features = range(500, 3001, 500);
    lasso_results, larslasso_results, larslasso_gram_results = \
                        compute_bench(alpha, [n_samples], list_n_features)

    pl.subplot(212)
    pl.title('Lasso benchmark (%d samples - alpha=%s)' % (n_samples, alpha))
    pl.plot(list_n_features, lasso_results, 'b-', label='Lasso')
    pl.plot(list_n_features, larslasso_results,'r-', label='LassoLARS')
    pl.plot(list_n_features, larslasso_gram_results,'g-', label='LassoLARS (with precomputed Gram matrix)')
    pl.legend(loc='upper left')
    pl.xlabel('number of features')
    pl.ylabel('time (in seconds)')
    pl.axis('tight')
    pl.show()

