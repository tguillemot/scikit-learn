""" Transformers to perform common preprocessing steps.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
# License: BSD

import numpy as np

from ..base import BaseEstimator, TransformerMixin


def _mean_and_std(X, axis=0, with_std=True):
    """Compute mean and std dev for centering, scaling

    Zero valued std components are reseted to 1.0 to avoid NaNs when scaling.
    """
    Xr = np.rollaxis(X, axis)
    mean_ = Xr.mean(axis=0)

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.0] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


def scale(X, axis=0, with_std=True, copy=True):
    """Method to standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.
    """
    mean_, std_ = _mean_and_std(X, axis, with_std)
    if copy:
        X = X.copy()
    Xr = np.rollaxis(X, axis)
    Xr -= mean_
    if with_std:
        Xr /= std_
    return X


class Scaler(BaseEstimator):
    """Object to standardize a dataset

    It centers the dataset and optionaly scales to fix the variance to 1 for
    each feature
    """

    def __init__(self, with_std=True):
        self.with_std = with_std

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        self.mean_, self.std_ = _mean_and_std(X, axis=0,
                                              with_std=self.with_std)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()
        # We are taking a view of the X array and modifying it
        X -= self.mean_
        if self.with_std:
            X /= self.std_
        return X


class SampleNormalizer(BaseEstimator):
    """Normalize samples individually to unit norm

    Each sample (e.g. each row) with at least one non zero component is
    rescaled independently of other samples so that its norm (l1 or l2)
    equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (most efficiently in CSR format).

    Parameters
    ----------
    copy : boolean, optional, default is True
        set to False to perform inplace row normalization

    norm : 'l1' or 'l2', optional ('l2' by default)
        the norm to use to normalize each non zero sample

    Note: this estimator is stateless (besides constructor parameters),
    the fit method does nothing but is useful when used in a pipeline:
    """

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self

    def transform(self, X, y=None):
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        """
        if self.copy:
            X = X.copy()

        if self.norm == 'l1':
            norms = np.abs(X).sum(axis=1)[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        else:
            raise ValueError("'%s' is not a supported norm" % self.norm)

        X /= norms

        return X


class Binarizer(BaseEstimator):
    """Binarize data according to a threshold"""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()

        cond = X > self.threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0

        return X


def _is_multilabel(y):
    return isinstance(y[0], tuple) or isinstance(y[0], list)


class LabelBinarizer(BaseEstimator, TransformerMixin):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Attributes
    ----------
    classes_ : array of shape [n_class]
        Holds the label for each class.

    Examples
    --------
    >>> from scikits.learn import preprocessing
    >>> clf = preprocessing.LabelBinarizer()
    >>> clf.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> clf.classes_
    array([1, 2, 4, 6])
    >>> clf.transform([1, 6])
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> clf.fit_transform([(1, 2), (3,)])
    array([[ 1.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> clf.classes_
    array([1, 2, 3])
    """

    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        self : returns an instance of self.
        """
        self.multilabel = _is_multilabel(y)
        if self.multilabel:
            # concatenation of the sub-sequences
            self.classes_ = np.unique(reduce(lambda a, b: a + b, y))
        else:
            self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """Transform multi-class labels to binary labels

        The output of transform is sometimes referred to by some authors as the
        1-of-K coding scheme.

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes]
        """

        if len(self.classes_) == 2:
            Y = np.zeros((len(y), 1))
        else:
            Y = np.zeros((len(y), len(self.classes_)))

        if self.multilabel:
            if not _is_multilabel(y):
                raise ValueError("y should be a list of label lists/tuples,"
                                 "got %r" % (y,))

            # inverse map: label => column index
            imap = dict((v, k) for k, v in enumerate(self.classes_))

            for i, label_tuple in enumerate(y):
                for label in label_tuple:
                    Y[i, imap[label]] = 1

            return Y

        elif len(self.classes_) == 2:
            Y[y == self.classes_[1], 0] = 1
            return Y

        elif len(self.classes_) >= 2:
            for i, k in enumerate(self.classes_):
                Y[y == k, i] = 1
            return Y

        else:
            raise ValueError("Wrong number of classes: %d"
                             % len(self.classes_))

    def inverse_transform(self, Y):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array of shape [n_samples, n_classes]
            Target values

        Returns
        -------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Note
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        if self.multilabel:
            Y = np.array(Y > 0, dtype=int)
            return [tuple(self.classes_[np.flatnonzero(Y[i])])
                    for i in range(Y.shape[0])]

        if len(Y.shape) == 1 or Y.shape[1] == 1:
            y = np.array(Y.ravel() > 0, dtype=int)

        else:
            y = Y.argmax(axis=1)

        return self.classes_[y]


class KernelCenterer(BaseEstimator, TransformerMixin):
    """Center a kernel matrix

    This is equivalent to centering phi(X) with
    scikits.learn.preprocessing.Scaler(with_std=False).
    """

    def fit(self, K):
        """Fit KernelCenterer

        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix

        Returns
        -------
        self : returns an instance of self.
        """
        n_samples = K.shape[0]
        self.K_fit_rows = np.sum(K, axis=0) / n_samples
        self.K_fit_all = K.sum() / (n_samples ** 2)
        return self

    def transform(self, K, copy=True):
        """Center kernel

        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix

        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """

        if copy:
            K = K.copy()

        K_pred_cols = (np.sum(K, axis=1) /
                       self.K_fit_rows.shape[0])[:, np.newaxis]

        K -= self.K_fit_rows
        K -= K_pred_cols
        K += self.K_fit_all

        return K
