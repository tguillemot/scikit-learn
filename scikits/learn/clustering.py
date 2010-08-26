""" Algorithms for clustering : Meanshift,  Affinity propagation and spectral 
clustering.

Author: Alexandre Gramfort alexandre.gramfort@inria.fr
        Gael Varoquaux gael.varoquaux@normalesup.org
"""
import warnings

from math import floor
import numpy as np

from scipy import cluster, sparse
from scipy.sparse.linalg.eigen.arpack import eigen_symmetric
from scipy.sparse.linalg import lobpcg
try:
    from pyamg import smoothed_aggregation_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False 

from .base import BaseEstimator
from .utils.graph import graph_laplacian

################################################################################
# MeanShift
################################################################################

def euclidian_distances(X, Y=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vector

    Parameters
    ----------
    X, array of shape (n_samples_1, n_features)

    Y, array of shape (n_samples_2, n_features), default None
            if Y is None, then Y=X is used instead

    Returns
    -------
    distances, array of shape (n_samples_1, n_samples_2)
    """
    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError, "incompatible dimension for X and Y matrices"

    XX = np.sum(X * X, axis=1)[:,np.newaxis]
    if Y is None:
        YY = XX.T
    else:
        YY = np.sum(Y * Y, axis=1)[np.newaxis,:]
    distances = XX + YY # Using broadcasting
    distances -= 2 * np.dot(X, Y.T)
    distances = np.maximum(distances, 0)
    distances = np.sqrt(distances)
    return distances


def estimate_bandwidth(X, quantile=0.3):
    """Estimate the bandwith ie the radius to use with an RBF kernel
    in the MeanShift algorithm

    X: array [n_samples, n_features]
        Input points

    quantile: float, default 0.3
        should be between [0, 1]
        0.5 means that the median is all pairwise distances is used
    """
    distances = euclidian_distances(X)
    distances = np.triu(distances, 1)
    distances_sorted = np.sort(distances[distances > 0])
    bandwidth = distances_sorted[floor(quantile * len(distances_sorted))]
    return bandwidth


def mean_shift(X, bandwidth=None):
    """Perform MeanShift Clustering of data using a flat kernel

    Parameters
    ----------

    X : array [n_samples, n_features]
        Input points

    bandwidth : float, optional
        kernel bandwidth
        If bandwidth is not defined, it is set using
        a heuristic given by the median of all pairwise distances

    Returns
    -------

    cluster_centers : array [n_clusters, n_features]
        Coordinates of cluster centers

    labels : array [n_samples]
        cluster labels for each point

    Notes
    -----
    See examples/plot_meanshift.py for an example.

    K. Funkunaga and L.D. Hosteler, "The Estimation of the Gradient of a
    Density Function, with Applications in Pattern Recognition"

    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X)

    n_points, n_features = X.shape

    n_clusters = 0
    bandwidth_squared = bandwidth**2
    points_idx_init = np.arange(n_points)
    stop_thresh     = 1e-3*bandwidth # when mean has converged
    cluster_centers = [] # center of clusters
    # track if a points been seen already
    been_visited_flag = np.zeros(n_points, dtype=np.bool)
    # number of points to possibly use as initilization points
    n_points_init = n_points
    # used to resolve conflicts on cluster membership
    cluster_votes = []

    random_state = np.random.RandomState(0)

    while n_points_init:
        # pick a random seed point
        tmp_index   = random_state.randint(n_points_init)
        # use this point as start of mean
        start_idx   = points_idx_init[tmp_index]
        my_mean     = X[start_idx, :] # intilize mean to this points location
        # points that will get added to this cluster
        my_members  = np.zeros(n_points, dtype=np.bool)
        # used to resolve conflicts on cluster membership
        this_cluster_votes = np.zeros(n_points, dtype=np.uint16)

        while True: # loop until convergence

            # dist squared from mean to all points still active
            sqrt_dist_to_all = np.sum((my_mean - X)**2, axis=1)

            # points within bandwidth
            in_idx = sqrt_dist_to_all < bandwidth_squared
            # add a vote for all the in points belonging to this cluster
            this_cluster_votes[in_idx] += 1

            my_old_mean  = my_mean # save the old mean
            my_mean      = np.mean(X[in_idx,:], axis=0) # compute the new mean
            # add any point within bandwidth to the cluster
            my_members   = np.logical_or(my_members, in_idx)
            # mark that these points have been visited
            been_visited_flag[my_members] = True

            if np.linalg.norm(my_mean-my_old_mean) < stop_thresh:

                # check for merge possibilities
                merge_with = -1
                for c in range(n_clusters):
                    # distance from possible new clust max to old clust max
                    dist_to_other = np.linalg.norm(my_mean -
                                                        cluster_centers[c])
                    # if its within bandwidth/2 merge new and old
                    if dist_to_other < bandwidth/2:
                        merge_with = c
                        break

                if merge_with >= 0: # something to merge
                    # record the max as the mean of the two merged
                    # (I know biased twoards new ones)
                    cluster_centers[merge_with] = 0.5 * (my_mean+
                                                cluster_centers[merge_with])
                    # add these votes to the merged cluster
                    cluster_votes[merge_with] += this_cluster_votes
                else: # its a new cluster
                    n_clusters += 1 # increment clusters
                    cluster_centers.append(my_mean) # record the mean
                    cluster_votes.append(this_cluster_votes)

                break

        # we can initialize with any of the points not yet visited
        points_idx_init = np.where(been_visited_flag == False)[0]
        n_points_init   = points_idx_init.size # number of active points in set

    # a point belongs to the cluster with the most votes
    labels = np.argmax(cluster_votes, axis=0)

    return cluster_centers, labels


################################################################################
class MeanShift(BaseEstimator):
    """MeanShift clustering


    Parameters
    ----------

    bandwidth: float, optional
        Bandwith used in the RBF kernel
        If not set, the bandwidth is estimated.
        See clustering.estimate_bandwidth

    Methods
    -------

    fit(X):
        Compute MeanShift clustering

    Attributes
    ----------

    cluster_centers_: array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_:
        Labels of each point

    Notes
    -----

    Reference:

    K. Funkunaga and L.D. Hosteler, "The Estimation of the Gradient of a
    Density Function, with Applications in Pattern Recognition"


    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def fit(self, X, **params):
        """compute MeanShift"""
        self._set_params(**params)
        self.cluster_centers_, self.labels_ = mean_shift(X, self.bandwidth)
        return self


################################################################################
# Affinity Propagation
################################################################################

def affinity_propagation(S, p=None, convit=30, maxit=200, damping=0.5,
            copy=True):
    """Perform Affinity Propagation Clustering of data

    Parameters
    ----------

    S: array [n_points, n_points]
        Matrix of similarities between points
    p: array [n_points,] or float, optional
        Preferences for each point
    damping : float, optional
        Damping factor
    copy: boolean, optional
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    Returns
    -------

    cluster_centers_indices: array [n_clusters]
        index of clusters centers

    labels : array [n_points]
        cluster labels for each point

    Notes
    -----
    See examples/plot_affinity_propagation.py for an example.

    Reference:
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    """
    if copy:
        # Copy the affinity matrix to avoid modifying it inplace
        S = S.copy()

    n_points = S.shape[0]

    assert S.shape[0] == S.shape[1]

    if p is None:
        p = np.median(S)

    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    random_state = np.random.RandomState(0)

    # Place preferences on the diagonal of S
    S.flat[::(n_points+1)] = p

    A = np.zeros((n_points, n_points))
    R = np.zeros((n_points, n_points)) # Initialize messages

    # Remove degeneracies
    S += (  np.finfo(np.double).eps*S
          + np.finfo(np.double).tiny*100
         )*random_state.randn(n_points, n_points)

    # Execute parallel affinity propagation updates
    e = np.zeros((n_points, convit))

    ind = np.arange(n_points)

    for it in range(maxit):
        # Compute responsibilities
        Rold = R.copy()
        AS = A + S

        I = np.argmax(AS, axis=1)
        Y = AS[np.arange(n_points), I]#np.max(AS, axis=1)

        AS[ind, I[ind]] = - np.finfo(np.double).max

        Y2 = np.max(AS, axis=1)
        R = S - Y[:, np.newaxis]

        R[ind, I[ind]] = S[ind, I[ind]] - Y2[ind]

        R = (1-damping)*R + damping*Rold # Damping

        # Compute availabilities
        Aold = A
        Rp = np.maximum(R, 0)
        Rp.flat[::n_points+1] = R.flat[::n_points+1]

        A = np.sum(Rp, axis=0)[np.newaxis, :] - Rp

        dA = np.diag(A)
        A = np.minimum(A, 0)

        A.flat[::n_points+1] = dA

        A = (1-damping)*A + damping*Aold # Damping

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convit] = E
        K = np.sum(E, axis=0)

        if it >= convit:
            se = np.sum(e, axis=1);
            unconverged = np.sum((se == convit) + (se == 0)) != n_points
            if (not unconverged and (K>0)) or (it==maxit):
                print "Converged after %d iterations." % it
                break
    else:
        print "Did not converged"

    I = np.where(np.diag(A+R) > 0)[0]
    K = I.size # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K) # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c==k)[0]
            j = np.argmax(np.sum(S[ii, ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        labels = np.empty((n_points, 1))
        cluster_centers_indices = None
        labels.fill(np.nan)

    return cluster_centers_indices, labels

################################################################################
class AffinityPropagation(BaseEstimator):
    """Perform Affinity Propagation Clustering of data

    Parameters
    ----------

    damping : float, optional
        Damping factor

    maxit : int, optional
        Maximum number of iterations

    convit : int, optional
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy: boolean, optional
        Make a copy of input data. True by default.

    Methods
    -------

    fit:
        Compute the clustering

    Attributes
    ----------

    cluster_centers_indices_ : array, [n_clusters]
        Indices of cluster centers

    labels_ : array, [n_samples]
        Labels of each point

    Notes
    -----
    See examples/plot_affinity_propagation.py for an example.

    Reference:

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """

    def __init__(self, damping=.5, maxit=200, convit=30, copy=True):
        self.damping = damping
        self.maxit = maxit
        self.convit = convit
        self.copy = copy

    def fit(self, S, p=None, **params):
        """compute MeanShift

        Parameters
        ----------

        S: array [n_points, n_points]
            Matrix of similarities between points
        p: array [n_points,] or float, optional
            Preferences for each point
        damping : float, optional
            Damping factor
        copy: boolean, optional
            If copy is False, the affinity matrix is modified inplace by the
            algorithm, for memory efficiency

        """
        self._set_params(**params)
        self.cluster_centers_indices_, self.labels_ = affinity_propagation(S, p,
                maxit=self.maxit, convit=self.convit, damping=self.damping,
                copy=self.copy)
        return self

################################################################################
# Spectral clustering 
################################################################################

def spectral_embedding(adjacency, k=8, mode=None):
    """ Spectral embedding: project the sample on the k first
        eigen vectors of the graph laplacian. 

        Parameters
        -----------
        adjacency: array-like or sparse matrix, shape: (p, p)
            The adjacency matrix of the graph to embed.
        k: integer, optional
            The dimension of the projection subspace.
        mode: {None, 'arpack' or 'amg'}
            The eigenvalue decomposition strategy to use. AMG (Algebraic
            MultiGrid) is much faster, but requires pyamg to be
            installed.

        Returns
        --------
        embedding: array, shape: (p, k)
            The reduced samples

        Notes
        ------
        The graph should contain only one connect component,
        elsewhere the results make little sens.
    """
    if mode == 'amg' and not amg_loaded:
        warnings.warn('pyamg not available, using scipy.sparse')
    if mode is None:
        mode = ('amg' if amg_loaded else 'arpack')
    laplacian, dd = graph_laplacian(adjacency,
                                    normed=True, return_diag=True)
    if mode == 'arpack' or not sparse.isspmatrix(laplacian):
        # We need to put the diagonal at zero
        if not sparse.isspmatrix(laplacian):
            n_nodes = laplacian.shape[0]
            laplacian[::n_nodes+1] = 0
        else:
            laplacian = laplacian.tocoo()
            diag_idx = (laplacian.row == laplacian.col)
            laplacian.data[diag_idx] = 0
            # If the matrix has a small number of diagonals (as in the
            # case of structured matrices comming from images), the
            # dia format might be best suited for matvec products:
            n_diags = np.unique(laplacian.row - laplacian.col).size
            if n_diags <= 7:
                # 3 or less outer diagonals on each side
                laplacian = laplacian.todia()
            else:
                # csr has the fastest matvec and is thus best suited to
                # arpack
                laplacian = laplacian.tocsr()
        lambdas, diffusion_map = eigen_symmetric(-laplacian, k=k, which='LA')
        embedding = diffusion_map.T[-2::-1]*dd
    elif mode == 'amg':
        # Use AMG to get a preconditionner and speed up the eigen value
        # problem.
        ml = smoothed_aggregation_solver(laplacian.tocsr())
        X = np.random.rand(laplacian.shape[0], k) 
        X[:, 0] = 1. / dd.ravel()
        M = ml.aspreconditioner()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12, 
                                        largest=False)
        embedding = diffusion_map.T[1:] * dd
    else:
        raise ValueError("Unknown value for mode: '%s'." % mode)
    return embedding


def spectral_clustering(adjacency, k=8, mode=None):
    """ Spectral clustering: apply k-means to a projection of the 
        graph laplacian, finds normalized graph cuts.

        Parameters
        -----------
        adjacency: array-like or sparse matrix, shape: (p, p)
            The adjacency matrix of the graph to embed.
        k: integer, optional
            The dimension of the projection subspace.
        mode: {None, 'arpack' or 'amg'}
            The eigenvalue decomposition strategy to use. AMG (Algebraic
            MultiGrid) is much faster, but requires pyamg to be
            installed.

        Returns
        --------
        labels: array of integers, shape: p
            The labels of the clusters.

        Notes
        ------
        The graph should contain only one connect component,
        elsewhere the results make little sens.
    """
    maps = spectral_embedding(adjacency, k=k+1, mode=mode)
    this_maps = maps[:k - 1]
    _, labels = cluster.vq.kmeans2(this_maps.T, k)
    return labels




