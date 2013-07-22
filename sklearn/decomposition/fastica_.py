"""
Python implementation of the fast ICA algorithms.

Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.
"""

# Author: Pierre Lafaye de Micheaux, Stefan van der Walt, Gael Varoquaux,
#         Bertrand Thirion, Alexandre Gramfort
# License: BSD 3 clause
import warnings
import numpy as np
from scipy import linalg

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six import moves
from ..utils import array2d, as_float_array, check_random_state, deprecated


__all__ = ['fastica', 'FastICA']


def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W

    Parameters
    ----------
    w: array of shape(n), to be orthogonalized
    W: array of shape(p, n), null space definition
    j: int < p

    caveats
    -------
    assumes that W is orthogonal
    w changed in place
    """
    w -= np.dot(np.dot(w, W[:j].T), W[:j])
    return w


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    K = np.dot(W, W.T)
    s, u = linalg.eigh(K)
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """Deflationary FastICA using fun approx to neg-entropy function

    Used internally by FastICA.
    """

    n_components = w_init.shape[0]
    W = np.zeros((n_components, n_components), dtype=float)

    # j is the index of the extracted component
    for j in range(n_components):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())

        for _ in moves.xrange(max_iter):
            wtx = np.dot(w.T, X)
            gwtx, g_wtx = g(wtx, fun_args)

            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w

            _gs_decorrelation(w1, W, j)

            w1 /= np.sqrt((w1 ** 2).sum())

            lim = np.abs(np.abs((w1 * w).sum()) - 1)
            w = w1
            if lim < tol:
                break

        W[j, :] = w

    return W


def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    n, p = X.shape

    W = _sym_decorrelation(w_init)

    for _ in moves.xrange(max_iter):
        wtx = np.dot(W, X)
        gwtx, g_wtx = g(wtx, fun_args)

        W1 = (np.dot(gwtx, X.T) / float(p)
              - np.dot(np.diag(g_wtx.mean(axis=1)), W))

        W1 = _sym_decorrelation(W1)

        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break

    return W


# Some standard non-linear functions.
# XXX: these should be optimized, as they can be a bottleneck.
def _logcosh(x, fun_args):
    alpha = fun_args.get('alpha', 1.0)  # comment it out?
    gx = np.tanh(alpha * x)
    g_x = alpha * (1 - gx ** 2)
    return gx, g_x


def _exp(x, fun_args):
    exp = np.exp(-(x ** 2) / 2)
    gx = x * exp
    g_x = (1 - x ** 2) * exp
    return gx, g_x


def _cube(x, fun_args):
    return x ** 3, 3 * x ** 2


def fastica(X, n_components=None, algorithm="parallel", whiten=True,
            fun="logcosh", fun_args={}, max_iter=200, tol=1e-04, w_init=None,
            random_state=None):
    """Perform Fast Independent Component Analysis.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    n_components : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.
    algorithm : {'parallel', 'deflation'}, optional
        Apply a parallel or deflational FASTICA algorithm.
    whiten: boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
        Otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.
    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:

        def my_g(x):
            return x ** 3, 3 * x ** 2
    fun_args: dictionary, optional
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}
    max_iter: int, optional
        Maximum number of iterations to perform
    tol: float, optional
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged
    w_init: (n_components, n_components) array, optional
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used
    source_only: boolean, optional
        If True, only the sources matrix is returned.
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    K: (n_components, p) array or None.
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n.comp principal components. If whiten is 'False', K is
        'None'.

    W: (n_components, n_components) array
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::

            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I

    S: (n_components, n) array
        estimated source matrix

    Notes
    -----

    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    Implemented using FastICA:
    `A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430`

    """
    random_state = check_random_state(random_state)
    # make interface compatible with other decompositions
    X = array2d(X).T

    alpha = fun_args.get('alpha', 1.0)
    if not 1 <= alpha <= 2:
        raise ValueError('alpha must be in [1,2]')

    if fun == 'logcosh':
        g = _logcosh
    elif fun == 'exp':
        g = _exp
    elif fun == 'cube':
        g = _cube
    elif callable(fun):
        def g(x, fun_args):
            return fun(x, **fun_args)
    else:
        exc = ValueError if isinstance(fun, six.string_types) else TypeError
        raise exc("Unknown function %r;"
                  " should be one of 'logcosh', 'exp', 'cube' or callable"
                  % fun)

    n, p = X.shape

    if not whiten and n_components is not None:
        n_components = None
        warnings.warn('Ignoring n_components with whiten=False.')

    if n_components is None:
        n_components = min(n, p)
    if (n_components > min(n, p)):
        n_components = min(n, p)
        print("n_components is too large: it will be set to %s" % n_components)

    if whiten:
        # Centering the columns (ie the variables)
        X = X - X.mean(axis=-1)[:, np.newaxis]

        # Whitening and preprocessing by PCA
        u, d, _ = linalg.svd(X, full_matrices=False)

        del _
        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, X)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(p)
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = as_float_array(X, copy=True)

    if w_init is None:
        w_init = random_state.normal(size=(n_components, n_components))
    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_components, n_components):
            raise ValueError('w_init has invalid shape -- should be %(shape)s'
                             % {'shape': (n_components, n_components)})

    kwargs = {'tol': tol,
              'g': g,
              'fun_args': fun_args,
              'max_iter': max_iter,
              'w_init': w_init}

    if algorithm == 'parallel':
        W = _ica_par(X1, **kwargs)
    elif algorithm == 'deflation':
        W = _ica_def(X1, **kwargs)
    else:
        raise ValueError('Invalid algorithm: must be either `parallel` or'
                         ' `deflation`.')
    del X1

    if whiten:
        S = np.dot(np.dot(W, K), X)
        return K, W, S.T
    else:
        S = np.dot(W, X)
        return None, W, S.T


class FastICA(BaseEstimator, TransformerMixin):
    """FastICA: a fast algorithm for Independent Component Analysis

    Parameters
    ----------
    n_components : int, optional
        Number of components to use. If none is passed, all are used.
    algorithm : {'parallel', 'deflation'}
        Apply parallel or deflational algorithm for FastICA
    whiten : boolean, optional
        If whiten is false, the data is already considered to be
        whitened, and no whitening is performed.
    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:

        def my_g(x):
            return x ** 3, 3 * x ** 2
    fun_args: dictionary, optional
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}
    max_iter : int, optional
        Maximum number of iterations during fit
    tol : float, optional
        Tolerance on update at each iteration
    w_init : None of an (n_components, n_components) ndarray
        The mixing matrix to be used to initialize the algorithm.
    random_state: int or RandomState
        Pseudo number generator state used for random sampling.
    fit_inverse_transform: boolean, optional, default=False
        Whether to enable the inverse_transform method.

    Attributes
    ----------
    `components_` : 2D array, [n_components, n_features]
        The unmixing matrix
    `mixing_` : array, shape = [n_features, n_components]
        Mixing matrix. Only available when fit_inverse_transform is true.
    `sources_`: 2D array, [n_samples, n_components]
        The estimated latent sources of the data.

    Notes
    -----
    Implementation based on
    `A. Hyvarinen and E. Oja, Independent Component Analysis:
    Algorithms and Applications, Neural Networks, 13(4-5), 2000,
    pp. 411-430`
    """

    def __init__(self, n_components=None, algorithm='parallel', whiten=True,
                 fun='logcosh', fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None, fit_inverse_transform=False):
        super(FastICA, self).__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state
        self.fit_inverse_transform = fit_inverse_transform

    def fit_transform(self, X, y=None):
        """Fit the model and recover the sources from X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        fun_args = {} if self.fun_args is None else self.fun_args
        whitening_, unmixing_, sources_ = fastica(
            X=X, n_components=self.n_components, algorithm=self.algorithm,
            whiten=self.whiten, fun=self.fun, fun_args=fun_args,
            max_iter=self.max_iter, tol=self.tol, w_init=self.w_init,
            random_state=self.random_state)
        if self.whiten:
            self.components_ = np.dot(unmixing_, whitening_)
            self.mean_ = array2d(X).T.mean(axis=-1)
            self.whitening_ = whitening_
        else:
            self.components_ = unmixing_

        if hasattr(self, "mixing_"):
            del self.mixing_
        if self.fit_inverse_transform:
            self.mixing_ = linalg.pinv(self.components_)

        self.sources_ = sources_
        return sources_

    def fit(self, X, y=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def transform(self, X, y=None, copy=True):
        """Recover the sources from X (apply the unmixing matrix).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples
            and n_features is the number of features.
        copy : bool
            If False, data passed to fit are overwritten.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        X = array2d(X)
        if copy is True:
            X = X.copy()
        if self.whiten is True:
            X -= X.mean(axis=0)[np.newaxis]

        return np.dot(X, self.components_.T)

    @deprecated('To be removed in 0.16. Set fit_inverse_transform=True'
                ' and inspect the mixing_ attribute.')
    def get_mixing_matrix(self):
        """Compute the mixing matrix.

        Returns
        -------
        mixing_matrix : array, shape (n_features, n_components)
        """
        return (self.mixing_ if hasattr(self, 'mixing_')
                else linalg.pinv(self.components_))

    def inverse_transform(self, X, copy=True):
        """Transform the sources back to the mixed data (apply mixing matrix).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Sources, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
        """
        if not self.fit_inverse_transform:
            raise ValueError('inverse_transform not enabled.'
                             ' set fit_inverse_transform=True before fit')
        if copy is True:
            X = X.copy()

        X = np.dot(X, self.mixing_.T)
        if self.whiten:
            X += self.mean_

        return X
