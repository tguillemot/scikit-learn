"""
The :mod:`sklearn.neighbors` module implements the k-nearest neighbors
algorithm.
"""

from .ball_tree import BallTree
from .kd_tree import KDTree
from .dist_metrics import DistanceMetric
from .graph import kneighbors_graph, radius_neighbors_graph
from .unsupervised import NearestNeighbors
from .classification import KNeighborsClassifier, RadiusNeighborsClassifier
from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
from .nearest_centroid import NearestCentroid
from .kde import KernelDensity

__all__ = ['BallTree',
           'KDTree',
           'KNeighborsClassifier',
           'KNeighborsRegressor',
           'NearestCentroid',
           'NearestNeighbors',
           'RadiusNeighborsClassifier',
           'RadiusNeighborsRegressor',
           'kneighbors_graph',
           'radius_neighbors_graph',
           'KernelDensity']
