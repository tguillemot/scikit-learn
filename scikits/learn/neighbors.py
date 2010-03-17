"""
k-Nearest Neighbor Algorithm.

Uses BallTree algorithm, which is an efficient way to perform fast
neighbor searches in high dimensionality.
"""
import numpy as np
from scipy import stats
from BallTree import BallTree

class Neighbors:
  """
  Classifier implementing k-Nearest Neighbor Algorithm.

  Parameters
  ----------
  data : array-like, shape (n, k)
      The data points to be indexed. This array is not copied, and so
      modifying this data will result in bogus results.
  labels : array
      An array representing labels for the data (only arrays of
      integers are supported).
  k : int
      default number of neighbors.
  window_size : float
      the default window size.

  Examples
  --------
  >>> samples = [[0.,0.,1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
  >>> labels = [0,0,1,1]
  >>> neigh = Neighbors(k=3)
  >>> neigh.fit(samples, labels) #doctest: +ELLIPSIS
  <scikits.learn.neighbors.neighbors.Neighbors instance at 0x...>
  >>> print neigh.predict([[0,0,0]])
  [0]
  """

  def __init__(self, k = 5, window_size = 1.):
    """
    Internally uses scipy.spatial.KDTree for most of its algorithms.
    """
    self._k = k
    self.window_size = window_size

  def fit(self, X, y):
    self.ball_tree = BallTree(X)
    self.X = np.asarray(X)
    self.y = np.asarray(y)
    return self

  def kneighbors(self, data, k=None):
    """
    Finds the K-neighbors of a point.

    Parameters
    ----------
    point : array-like
        The new point.
    k : int
        Number of neighbors to get (default is the value
        passed to the constructor).

    Returns
    -------
    dist : array
        Array representing the lenghts to point.
    ind : array
        Array representing the indices of the nearest points in the
        population matrix.

    Examples
    --------
    In the following example, we construnct a Neighbors class from an
    array representing our data set and ask who's the closest point to
    [1,1,1]

    >>> import numpy as np
    >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    >>> labels = [0, 0, 1]
    >>> neigh = Neighbors(k=1)
    >>> neigh.fit(samples, labels) #doctest: +ELLIPSIS
    <scikits.learn.neighbors.neighbors.Neighbors instance at 0x...>
    >>> print neigh.kneighbors([1., 1., 1.])
    (0.5, 2)

    As you can see, it returns [0.5], and [2], which means that the
    element is at distance 0.5 and is the third element of samples
    (indexes start at 0). You can also query for multiple points:

    >>> print neigh.kneighbors([[0., 1., 0.], [1., 0., 1.]])
    (array([ 0.5       ,  1.11803399]), array([1, 2]))

    """
    if k is None: k = self._k
    return self.ball_tree.query(data, k=k)


  def predict(self, T, k=None):
    """
    Predict the class labels for the provided data.

    Parameters
    ----------
    test: array
        A 2-D array representing the test point.
    k : int
        Number of neighbors to get (default is the value
        passed to the constructor).

    Returns
    -------
    labels: array
        List of class labels (one for each data sample).

    Examples
    --------
    >>> import numpy as np
    >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    >>> labels = [0, 0, 1]
    >>> neigh = Neighbors(k=1)
    >>> neigh.fit(samples, labels) #doctest: +ELLIPSIS
    <scikits.learn.neighbors.neighbors.Neighbors instance at 0x...>
    >>> print neigh.predict([.2, .1, .2])
    0
    >>> print neigh.predict([[0., -1., 0.], [3., 2., 0.]])
    [0 1]
    """
    T = np.asanyarray(T)
    if k is None: k = self._k
    return _predict_from_BallTree(self.ball_tree, self.y, T, k=k)


def _predict_from_BallTree(ball_tree, Y, test, k):
    """
    Predict target from BallTree object containing the data points.

    This is a helper method, not meant to be used directly. It will
    not check that input is of the correct type.
    """
    Y_ = Y[ball_tree.query(test, k=k, return_distance=False)]
    if k == 1: return Y_hat
    # search most common values along axis 1 of labels
    # much faster than scipy.stats.mode
    return stats.mode(Y_, axis=1)[0]


def predict(X, Y, test, k=5):
  """
  Predict test using Nearest Neighbor Algorithm.
  """
  ball_tree  = BallTree(X)
  return _predict_from_BallTree(ball_tree, np.asarray(Y), test, k=k)
