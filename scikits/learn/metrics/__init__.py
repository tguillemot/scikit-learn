"""
Metrics module with score functions, performance metrics and
pairwise metrics or distances computation
"""
from scipy.spatial import distance

from .metrics import confusion_matrix, roc_curve, auc, precision_score, \
                recall_score, fbeta_score, f1_score, zero_one_score, \
                precision_recall_fscore_support, classification_report, \
                precision_recall_curve, explained_variance_score, r2_score, \
                zero_one, mean_square_error, hinge_loss

from .cluster import homogeneity_completeness_v_measure
from .cluster import homogeneity_score
from .cluster import completeness_score
from .cluster import v_measure_score
from . import pairwise
from .pairwise import euclidean_distances, pairwise_distances
