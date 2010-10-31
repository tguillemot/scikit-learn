"""Utilities to evaluate the predictive performance of models"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
# License: BSD Style.

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix to evaluate the accuracy of a classification

    By definition a confusion matrix cm is such that cm[i, j] is equal
    to the number of observations known to be in group i but predicted
    to be in group j

    Parameters
    ==========

    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        estimated targets

    Returns
    =======
    cm : array, shape = [n_classes,n_classes]
        confusion matrix

    References
    ==========
    http://en.wikipedia.org/wiki/Confusion_matrix
    """
    if labels is None:
        # introspect the y signals to find the list of possible labels
        # ignoring possible NaNs
        clean_y_true = y_true[np.isfinite(y_true)].ravel()
        clean_y_pred = y_pred[np.isfinite(y_pred)].ravel()

        labels = np.r_[np.unique(clean_y_true).ravel(),
                       np.unique(clean_y_pred).ravel()]
        labels = np.unique(labels)

    n_labels = labels.size

    cm = np.empty((n_labels, n_labels))
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            cm[i, j] = np.sum(
                np.logical_and(y_true == label_i, y_pred == label_j))

    return cm


def roc_curve(y, probas_):
    """compute Receiver operating characteristic (ROC)

    Parameters
    ==========

    y : array, shape = [n_samples]
        true targets

    probas_ : array, shape = [n_samples]
        estimated probabilities

    Returns
    =======
    fpr : array, shape = [n]
        False Positive Rates

    tpr : array, shape = [n]
        True Positive Rates

    thresholds : array, shape = [n]
        Thresholds on proba_ used to compute fpr and tpr

    References
    ==========
    http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    y = y.ravel()
    probas_ = probas_.ravel()
    thresholds = np.sort(np.unique(probas_))[::-1]
    n_thresholds = thresholds.size

    tpr = np.empty(n_thresholds) # True positive rate
    fpr = np.empty(n_thresholds) # False positive rate
    n_pos = float(np.sum(y == 1)) # nb of true positive
    n_neg = float(np.sum(y == 0)) # nb of true negative

    for i, t in enumerate(thresholds):
        tpr[i] = np.sum(y[probas_ >= t] == 1) / n_pos
        fpr[i] = np.sum(y[probas_ >= t] == 0) / n_neg

    return fpr, tpr, thresholds


def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    Parameters
    ==========

    x : array, shape = [n]
        x coordinates

    y : array, shape = [n]
        y coordinates

    Returns
    =======
    auc : float

    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    h = np.diff(x)
    area = np.sum(h * (y[1:] + y[:-1])) / 2.0
    return area


def precision(y_true, y_pred):
    """Compute the precision

    The precision is the ratio :math:`tp / (tp + fp)` where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    =======
    precision : float
    """
    return precision_recall_fscore(y_true, y_pred)[0]


def recall(y_true, y_pred):
    """Compute the recall

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    =======
    recall : float
    """
    return precision_recall_fscore(y_true, y_pred)[1]


def fbeta_score(y_true, y_pred, beta):
    """Compute fbeta score

    The F_beta score can be interpreted as a weighted average of the precision
    and recall, where an F_beta score reaches its best value at 1 and worst
    score at 0.

    F_1 weights recall beta as much as precision.

    See: http://en.wikipedia.org/wiki/F1_score

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    beta: float

    Returns
    =======
    fbeta_score: float
    """
    return precision_recall_fscore(y_true, y_pred, beta=beta)[2]


def f1_score(y_true, y_pred):
    """Compute f1 score

    The F1 score can be interpreted as a weighted average of the precision
    and recall, where an F1 score reaches its best value at 1 and worst
    score at 0. The relative contribution of precision and recall to the f1
    score are equal.

        :math:`F_1 = 2 \cdot \frac{p \cdot r}{p + r}`

    See: http://en.wikipedia.org/wiki/F1_score

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    Returns
    =======
    f1_score: float

    References
    ==========
    http://en.wikipedia.org/wiki/F1_score
    """
    return fbeta_score(y_true, y_pred, 1)


def precision_recall_fscore(y_true, y_pred, beta=1.0):
    """Compute precision and recall and f-measure at the same time.

    The precision is the ratio :math:`tp / (tp + fp)` where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F_beta score can be interpreted as a weighted average of the precision
    and recall, where an F_beta score reaches its best value at 1 and worst
    score at 0.

    The F_1 score weights recall beta as much as precision.

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets

    y_pred : array, shape = [n_samples]
        predicted targets

    beta : float, 1.0 by default
        the strength of recall versus precision in the f-score

    Returns
    =======
    precision: float
    recall : float
    fscore : float

    References
    ==========
    http://en.wikipedia.org/wiki/Precision_and_recall
    """
    assert(beta > 0)

    true_pos = np.sum(y_true[y_pred == 1] == 1)
    false_pos = np.sum(y_true[y_pred == 1] == 0)
    false_neg = np.sum(y_true[y_pred == 0] == 1)

    # precision and recall
    precision = true_pos / float(true_pos + false_pos)
    recall = true_pos / float(true_pos + false_neg)

    # fbeta score
    beta2 = beta ** 2
    fscore = (1 + beta2) * (precision * recall) / (
        beta2 * precision + recall)
    return precision, recall, fscore


def precision_recall_curve(y_true, probas_pred):
    """Compute precision-recall pairs for different probability thresholds

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio :math:`tp / (tp + fp)` where tp is the number of
    true positives and fp the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio :math:`tp / (tp + fn)` where tp is the number of
    true positives and fn the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    Parameters
    ==========
    y_true : array, shape = [n_samples]
        true targets of binary classification in range {-1, 1} or {0, 1}

    probas_pred : array, shape = [n_samples]
        estimated probabilities

    Returns
    =======
    precision : array, shape = [n]
        Precision values

    recall : array, shape = [n]
        Recall values

    thresholds : array, shape = [n]
        Thresholds on proba_ used to compute precision and recall
    """
    y_true = y_true.ravel()
    labels = np.unique(y_true)
    if np.all(labels == np.array([-1, 1])):
        # convert {-1, 1} to boolean {0, 1} repr
        y_true[y_true == -1] = 0
        labels = np.array([0, 1])
    if not np.all(labels == np.array([0, 1])):
        raise ValueError("y_true contains non binary labels: %r" % labels)
    probas_pred = probas_pred.ravel()
    thresholds = np.sort(np.unique(probas_pred))
    n_thresholds = thresholds.size + 1
    precision = np.empty(n_thresholds)
    recall = np.empty(n_thresholds)
    for i, t in enumerate(thresholds):
        y_pred = np.ones(len(y_true))
        y_pred[probas_pred < t] = 0
        precision[i], recall[i], _ = precision_recall_fscore(y_true, y_pred)
    precision[-1] = 1.0
    recall[-1] = 0.0
    return precision, recall, thresholds


###############################################################################
# Loss functions


def zero_one(y_true, y_pred):
    """Zero-One classification loss

    Positive integer (number of misclassifications). The best score is 0.

    return the number of differences
    """
    return np.sum(y_pred != y_true)


def mean_square_error(y_true, y_pred):
    """Mean square error regression loss

    Positive floating point value: the best value is 0.0.

    return the mean square error
    """
    return np.linalg.norm(y_pred - y_true) ** 2


def explained_variance(y_true, y_pred):
    """Explained variance regression loss

    Best possible score is 1.0, lower values are worst.

    Note: the explained variance is not a symmetric function.

    return the explained variance
    """
    return 1 - np.var(y_true - y_pred) / np.var(y_true)

