"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gmm import normalize, sample_gaussian, log_multivariate_normal_density
from .gmm import GMM, _distribute_covar_matrix_to_match_covariance_type
from .gmm import _validate_covars
from .dpgmm import DPGMM, VBGMM
