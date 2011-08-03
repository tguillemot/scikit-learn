"""Benchmarks of orthogonal matching pursuit (:ref:`OMP`) versus least angle
regression (:ref:`least_angle_regression`)

The input data is mostly low rank but is a fat infinite tail.
"""
import gc
from time import time
import sys

import numpy as np
from collections import defaultdict

from scikits.learn.linear_model import lars_path, orthogonal_mp, \
                                       orthogonal_mp_gram
from scikits.learn.datasets.samples_generator import make_regression_dataset, \
                                                   generate_sparse_coded_signal


def compute_bench(samples_range, features_range):

    it = 0

    results = defaultdict(lambda: [])
    lars, lars_gram, omp, omp_gram = [], [], [], []

    max_it = len(samples_range) * len(features_range)
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1
            n_informative = n_features / 10
            print '===================='
            print 'Iteration %03d of %03d' % (it, max_it)
            print '===================='
            # dataset_kwargs = {
            #     'n_train_samples': n_samples,
            #     'n_test_samples': 2,
            #     'n_features': n_features,
            #     'n_informative': n_informative,
            #     'effective_rank': min(n_samples, n_features) / 10,
            #     #'effective_rank': None,
            #     'bias': 0.0,
            # }
            dataset_kwargs = {
                'n_samples': 1,
                'n_components': n_features,
                'n_features': n_samples,
                'n_nonzero_coefs': n_informative,
                'random_state': 0
            }
            print "n_samples: %d" % n_samples
            print "n_features: %d" % n_features
            #X, y, _, _, _ = make_regression_dataset(**dataset_kwargs)
            y, X, _ = generate_sparse_coded_signal(**dataset_kwargs)
            X = np.asfortranarray(X)
            #X /= np.sqrt((X**2).sum(axis=0))

            gc.collect()
            print "benching lars_path (with Gram):",
            sys.stdout.flush()
            tstart = time()
            G = np.dot(X.T, X)  # precomputed Gram matrix
            Xy = np.dot(X.T, y)
            lars_path(X, y, Xy=Xy, Gram=G, max_iter=n_informative)
            delta = time() - tstart
            print "%0.3fs" % delta
            lars_gram.append(delta)

            gc.collect()
            print "benching lars_path (without Gram):",
            sys.stdout.flush()
            tstart = time()
            lars_path(X, y, Gram=None, max_iter=n_informative)
            delta = time() - tstart
            print "%0.3fs" % delta
            lars.append(delta)

            gc.collect()
            print "benching orthogonal_mp (with Gram):",
            sys.stdout.flush()
            tstart = time()
            orthogonal_mp(X, y, precompute_gram=True,
                          n_nonzero_coefs=n_informative)
            delta = time() - tstart
            print "%0.3fs" % delta
            omp_gram.append(delta)

            gc.collect()
            print "benching orthogonal_mp (without Gram):",
            sys.stdout.flush()
            tstart = time()

            orthogonal_mp(X, y, precompute_gram=False,
                          n_nonzero_coefs=n_informative)
            delta = time() - tstart
            print "%0.3fs" % delta
            omp.append(delta)
    results['ratio of OMP versus LARS \n timing improvement \n (w/ Gram)'] = (
        np.array(lars_gram) / 
        np.array(omp_gram))
    results['ratio of OMP versus LARS \n timing improvement \n (w/o Gram)'] = (
        np.array(lars) / 
        np.array(omp))
    return results


if __name__ == '__main__':
    from mpl_toolkits.mplot3d import axes3d  # register the 3d projection
    import matplotlib.pyplot as plt

    samples_range = np.linspace(1000, 3000, 5).astype(np.int)
    features_range = np.linspace(1000, 3000, 5).astype(np.int)
    results = compute_bench(samples_range, features_range)

    max_time = max(max(t) for t in results.itervalues())

    fig = plt.figure()
    i = 1
    for c, (label, timings) in zip('bcry', sorted(results.iteritems())):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        X, Y = np.meshgrid(samples_range, features_range)
        Z = np.asarray(timings).reshape(samples_range.shape[0],
                                        features_range.shape[0])

        # plot the actual surface
        ax.plot_surface(X, Y, Z.T, cstride=1, rstride=1, color=c, alpha=0.8)

        # dummy point plot to stick the legend to since surface plot do not
        # support legends (yet?)
        #ax.plot([1], [1], [1], color=c, label=label)

        ax.set_xlabel('n_samples')
        ax.set_ylabel('n_features')
        ax.set_zlabel('time (s)')
        ax.set_zlim3d(0.0, max_time * 1.1)
        ax.set_title(label)
        #ax.legend()
        i += 1
    plt.show()
