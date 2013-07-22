"""Restricted Boltzmann Machine
"""

# Author: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
# License: BSD Style.

import time

import numpy as np

from ..base import BaseEstimator, TransformerMixin
from ..utils import array2d, check_random_state, as_float_array
from ..utils.extmath import safe_sparse_dot


def logistic_sigmoid(x):
    """
    Implements the logistic function, ``1 / (1 + e ** -x)``.

    This implementation is more stable by splitting on positive and negative
    values and computing::

        1 / (1 + exp(-x_i)) if x_i > 0
        exp(x_i) / (1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    x: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    res: array, shape (M, N)
        Value of the logistic function evaluated at every point in x

    Notes
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    positive_idx = x > 0
    out = np.empty_like(x)
    out[positive_idx] = 1. / (1 + np.exp(-x[positive_idx]))
    exp_x = np.exp(x[~positive_idx])
    out[~positive_idx] = exp_x / (1. + exp_x)
    return out


def log_logistic_sigmoid(x):
    """
    Implements the log of the logistic function, ``log(1 / (1 + e ** -x))``.

    This implementation is more stable by splitting on positive and negative
    values and computing::

        -log(1 + exp(-x_i)) if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    x: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    res: array, shape (M, N)
        Value of the logarithm of the logistic function evaluated at every
        point in x

    Notes
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    positive_idx = x > 0
    out = np.empty_like(x)
    out[positive_idx] = -np.log(1 + np.exp(-x[positive_idx]))
    out[~positive_idx] = (x[~positive_idx] -
                          np.log(1 + np.exp(x[~positive_idx])))
    return out


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """
    Bernoulli Restricted Boltzmann Machine (RBM)

    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML).

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units

    learning_rate : float, optional
        Learning rate to use during learning. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose: bool, optional
        The verbosity level.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features), optional
        Weight matrix, where n_features in the number of visible
        units and n_components is the number of hidden units.

    intercept_hidden_ : array-like, shape (n_components,), optional
        Biases of the hidden units

    intercept_visible_ : array-like, shape (n_features,), optional
        Biases of the visible units

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=False)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf
    """
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=False, random_state=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def _sample_binomial(self, p, rng):
        """
        Sample from a binomial distribution in place.

        Parameters
        ----------
        p: array-like, shape (M, N)
            Probabilities of the multivariate binomial to sample

        rng: RandomState
            Random number generator to use

        Notes
        -----
        This is equivalent to calling numpy.random.binomial(1, p) but is
        faster because it uses in-place operations on p.

        Returns
        -------
        x_new: array-like, shape (M, N)
        """
        p[rng.uniform(size=p.shape) < p] = 1.

        return np.floor(p, p)

    def transform(self, X):
        """
        Computes the probabilities ``P({\bf h}_j=1|{\bf v}={\bf X})``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The data to be transformed

        Returns
        -------
        h: array, shape (n_samples, n_components)
            Latent representations of the data
        """
        X = array2d(X)
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        """
        Computes the probabilities ``P({\bf h}_j=1|{\bf v})``.

        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            Values of the visible layer

        Returns
        -------
        h: array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer
        """
        return logistic_sigmoid(safe_sparse_dot(v, self.components_.T)
                                + self.intercept_hidden_)

    def _sample_hiddens(self, v, rng):
        """
        Sample from the distribution ``P({\bf h}|{\bf v})``.

        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from

        rng: RandomState
            Random number generator to use

        Returns
        -------
        h: array-like, shape (n_samples, n_components)
        """
        return self._sample_binomial(self._mean_hiddens(v), rng)

    def _mean_visibles(self, h):
        """
        Computes the probabilities ``P({\bf v}_i=1|{\bf h})``.

        Parameters
        ----------
        h: array-like, shape (n_samples, n_components)
            Values of the hidden layer to take the mean over

        Returns
        -------
        v: array-like, shape (n_samples, n_features)
        """
        return logistic_sigmoid(np.dot(h, self.components_)
                                + self.intercept_visible_)

    def _sample_visibles(self, h, rng):
        """
        Sample from the distribution ``P({\bf v}|{\bf h})``.

        Parameters
        ----------
        h: array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng: RandomState
            Random number generator to use

        Returns
        -------
        v: array-like, shape (n_samples, n_features)
        """
        return self._sample_binomial(self._mean_visibles(h), rng)

    def free_energy(self, v):
        """
        Computes the free energy
        ``\mathcal{F}({\bf v}) = - \log \sum_{\bf h} e^{-E({\bf v},{\bf h})}``.

        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            Values of the visible layer

        Returns
        -------
        free_energy: array-like, shape (n_samples,)
            The value of the free energy
        """
        return - np.dot(v, self.intercept_visible_) - np.log(1. + np.exp(
            safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_)) \
            .sum(axis=1)

    def gibbs(self, v):
        """
        Perform one Gibbs sampling step.

        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            Values of the visible layer to start from

        Returns
        -------
        v_new: array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step
        """
        rng = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, rng)
        v_ = self._sample_visibles(h_, rng)

        return v_

    def _fit(self, v_pos, rng):
        """
        Adjust the parameters to maximize the likelihood of ``{\bf v}``
        using Stochastic Maximum Likelihood (SML) [1].

        Parameters
        ----------
        v_pos: array-like, shape (n_samples, n_features)
            The data to use for training

        rng: RandomState
            Random number generator to use for sampling

        Returns
        -------
        pseudo_likelihood: array-like, shape (n_samples,)
            If verbose=True, Pseudo Likelihood estimate for this batch.

        References
        ----------
        [1] Tieleman, T. Training Restricted Boltzmann Machines using
            Approximations to the Likelihood Gradient. International Conference
            on Machine Learning (ICML) 2008
        """
        v_pos = as_float_array(v_pos)
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = self.learning_rate / self.batch_size
        v_pos *= lr
        v_neg *= lr
        self.components_ += safe_sparse_dot(v_pos.T, h_pos).T
        self.components_ -= np.dot(v_neg.T, h_neg).T
        self.intercept_hidden_ += lr * (h_pos.sum(0) - h_neg.sum(0))
        self.intercept_visible_ += (v_pos.sum(0) - v_neg.sum(0))

        self.h_samples_ = self._sample_binomial(h_neg, rng)

        if self.verbose:
            return self.pseudo_likelihood(v_pos)

    def pseudo_likelihood(self, v):
        """
        Compute the pseudo-likelihood of ``{\bf v}``.

        Parameters
        ----------
        v: array-like, shape (n_samples, n_features)
            Values of the visible layer

        Returns
        -------
        pseudo_likelihood: array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy to likelihood)
        """
        rng = check_random_state(self.random_state)
        fe = self.free_energy(v)

        v_ = v.copy()
        i_ = rng.randint(0, v.shape[1], v.shape[0])
        v_[np.arange(v.shape[0]), i_] = 1 - v_[np.arange(v.shape[0]), i_]
        fe_ = self.free_energy(v_)

        return v.shape[1] * log_logistic_sigmoid(fe_ - fe)

    def fit(self, X, y=None):
        """
        Fit the model to the data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        self:
            The fitted model
        """
        X = array2d(X)
        dtype = np.float32 if X.dtype.itemsize == 4 else np.float64
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            dtype=dtype,
            order='fortran')
        self.intercept_hidden_ = np.zeros(self.n_components, dtype=dtype)
        self.intercept_visible_ = np.zeros(X.shape[1], dtype=dtype)
        self.h_samples_ = np.zeros((self.batch_size, self.n_components),
                                   dtype=dtype)

        inds = np.arange(X.shape[0])
        rng.shuffle(inds)

        n_batches = int(np.ceil(len(inds) / float(self.batch_size)))

        verbose = self.verbose
        for iteration in xrange(self.n_iter):
            pl = 0.
            if verbose:
                begin = time.time()
            for minibatch in xrange(n_batches):
                pl_batch = self._fit(X[inds[minibatch::n_batches]], rng)

                if verbose:
                    pl += pl_batch.sum()

            if verbose:
                pl /= X.shape[0]
                end = time.time()
                print("Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                      % (iteration, pl, end - begin))

        return self

    def fit_transform(self, X, y=None):
        """
        Fit the model to the data X and transform it.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        X_transformed, array, shape (n_samples, n_components)
            Latent representations of the input data
        """
        X = array2d(X)
        self.fit(X, y)
        return self.transform(X)
