# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
# License: BSD 3 clause

import warnings
import numbers
import math

import numpy as np
import numpy.ma as ma
import scipy.sparse as sp
import scipy.stats as st

from .base import BaseEstimator, TransformerMixin
from .utils import check_arrays
from .utils import array2d
from .utils import as_float_array
from .utils import atleast2d_or_csr
from .utils import atleast2d_or_csc
from .utils import safe_asarray
from .utils import warn_if_not_float
from .utils.fixes import unique
from .utils import deprecated

from .utils.multiclass import unique_labels
from .utils.multiclass import type_of_target

from .utils.sparsefuncs import inplace_csr_row_normalize_l1
from .utils.sparsefuncs import inplace_csr_row_normalize_l2
from .utils.sparsefuncs import inplace_csr_column_scale
from .utils.sparsefuncs import mean_variance_axis0
from .externals import six

zip = six.moves.zip
map = six.moves.map

__all__ = ['Binarizer',
           'Imputer',
           'KernelCenterer',
           'LabelBinarizer',
           'LabelEncoder',
           'MinMaxScaler',
           'Normalizer',
           'OneHotEncoder',
           'StandardScaler',
           'binarize',
           'normalize',
           'scale']


def _mean_and_std(X, axis=0, with_mean=True, with_std=True):
    """Compute mean and std deviation for centering, scaling.

    Zero valued std components are reset to 1.0 to avoid NaNs when scaling.
    """
    X = np.asarray(X)
    Xr = np.rollaxis(X, axis)

    if with_mean:
        mean_ = Xr.mean(axis=0)
    else:
        mean_ = None

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.0] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : array-like or CSR matrix.
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    See also
    --------
    :class:`sklearn.preprocessing.StandardScaler` to perform centering and
    scaling using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    if sp.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives.")
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0, "
                             " got axis=%d" % axis)
        warn_if_not_float(X, estimator='The scale function')
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()
            copy = False
        if copy:
            X = X.copy()
        _, var = mean_variance_axis0(X)
        var[var == 0.0] = 1.0
        inplace_csr_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        warn_if_not_float(X, estimator='The scale function')
        mean_, std_ = _mean_and_std(
            X, axis, with_mean=with_mean, with_std=with_std)
        if copy:
            X = X.copy()
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
        if with_std:
            Xr /= std_
    return X


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Standardizes features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The standardization is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This standardization is often used as an alternative to zero mean,
    unit variance scaling.

    Parameters
    ----------
    feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : boolean, optional, default is True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    `min_` : ndarray, shape (n_features,)
        Per feature adjustment for minimum.

    `scale_` : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        X = check_arrays(X, sparse_format="dense", copy=self.copy)[0]
        warn_if_not_float(X, estimator=self)
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        data_min = np.min(X, axis=0)
        data_range = np.max(X, axis=0) - data_min
        # Do not scale constant features
        data_range[data_range == 0.0] = 1.0
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range
        self.data_min = data_min
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        X = check_arrays(X, sparse_format="dense", copy=self.copy)[0]
        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        X = check_arrays(X, sparse_format="dense", copy=self.copy)[0]
        X -= self.min_
        X /= self.scale_
        return X


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    `mean_` : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    `std_` : array of floats with shape [n_features]
        The standard deviation for each feature in the training set.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_arrays(X, copy=self.copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            warn_if_not_float(X, estimator=self)
            self.mean_ = None

            if self.with_std:
                var = mean_variance_axis0(X)[1]
                self.std_ = np.sqrt(var)
                self.std_[var == 0.0] = 1.0
            else:
                self.std_ = None
            return self
        else:
            warn_if_not_float(X, estimator=self)
            self.mean_, self.std_ = _mean_and_std(
                X, axis=0, with_mean=self.with_mean, with_std=self.with_std)
            return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        X = check_arrays(X, copy=copy, sparse_format="csr")[0]
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if self.std_ is not None:
                warn_if_not_float(X, estimator=self)
                inplace_csr_column_scale(X, 1 / self.std_)
        else:
            warn_if_not_float(X, estimator=self)
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.std_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        if sp.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sp.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.std_ is not None:
                inplace_csr_column_scale(X, self.std_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.std_
            if self.with_mean:
                X += self.mean_
        return X


class Scaler(StandardScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        warnings.warn("Scaler was renamed to StandardScaler. The old name "
                      " will be removed in 0.15.", DeprecationWarning)
        super(Scaler, self).__init__(copy, with_mean, with_std)


def normalize(X, norm='l2', axis=1, copy=True):
    """Normalize a dataset along any axis

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Normalizer` to perform normalization
    using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_arrays(X, sparse_format=sparse_format, copy=copy)[0]
    warn_if_not_float(X, 'The normalize function')
    if axis == 0:
        X = X.T

    if sp.issparse(X):
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        elif norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        X /= norms

    if axis == 0:
        X = X.T

    return X


class Normalizer(BaseEstimator, TransformerMixin):
    """Normalize samples individually to unit norm

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1 or l2) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Parameters
    ----------
    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample.

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Notes
    -----
    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    See also
    --------
    :func:`sklearn.preprocessing.normalize` equivalent function
    without the object oriented API
    """

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        atleast2d_or_csr(X)
        return self

    def transform(self, X, y=None, copy=None):
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        """
        copy = copy if copy is not None else self.copy
        atleast2d_or_csr(X)
        return normalize(X, norm=self.norm, axis=1, copy=copy)


def binarize(X, threshold=0.0, copy=True):
    """Boolean thresholding of array-like or scipy.sparse matrix

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to binarize, element by element.
        scipy.sparse matrices should be in CSR or CSC format to avoid an
        un-necessary copy.

    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default is True
        set to False to perform inplace binarization and avoid a copy
        (if the input is already a numpy array or a scipy.sparse CSR / CSC
        matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Binarizer` to perform binarization
    using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    sparse_format = "csr"  # We force sparse format to be either csr or csc.
    if hasattr(X, "format"):
        if X.format in ["csr", "csc"]:
            sparse_format = X.format

    X = check_arrays(X, sparse_format=sparse_format, copy=copy)[0]
    if sp.issparse(X):
        if threshold < 0:
            raise ValueError('Cannot binarize a sparse matrix with threshold '
                             '< 0')
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        X.data[cond] = 1
        X.data[not_cond] = 0
        X.eliminate_zeros()
    else:
        cond = X > threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
    return X


class Binarizer(BaseEstimator, TransformerMixin):
    """Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurrences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modelled using the Bernoulli
    distribution in a Bayesian setting).

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default is True
        set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        atleast2d_or_csr(X)
        return self

    def transform(self, X, y=None, copy=None):
        """Binarize each element of X

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.
        """
        copy = copy if copy is not None else self.copy
        return binarize(X, threshold=self.threshold, copy=copy)


def _transform_selected(X, transform, selected="all", copy=True):
    """Apply a transform function to portion of selected features

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.

    transform : callable
        A callable transform(X) -> X_transformed

    copy : boolean, optional
        Copy X even if it could be avoided.

    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    if selected == "all":
        return transform(X)

    X = atleast2d_or_csc(X, copy=copy)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sp.issparse(X_sel) or sp.issparse(X_not_sel):
            return sp.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical integer features using a one-hot aka one-of-K scheme.

    The input to this transformer should be a matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix were each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Parameters
    ----------
    n_values : 'auto', int or array of ints
        Number of values per feature.

        - 'auto' : determine value range from training data.
        - int : maximum value for all features.
        - array : maximum value per feature.

    categorical_features: "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

    dtype : number type, default=np.float
        Desired dtype of output.

    Attributes
    ----------
    `active_features_` : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

    `feature_indices_` : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

    `n_values_` : array of shape (n_features,)
        Maximum number of values per feature.

    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], \
[1, 0, 2]])  # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
           n_values='auto')
    >>> enc.n_values_
    array([2, 3, 4])
    >>> enc.feature_indices_
    array([0, 2, 5, 9])
    >>> enc.transform([[0, 1, 1]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])

    See also
    --------
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """
    def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Input array of type int.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape
        if self.n_values == 'auto':
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected"
                                " 'auto', int or array of ints, got %r"
                                % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")
        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sp.coo_matrix((data, (row_indices, column_indices)),
                            shape=(n_samples, indices[-1]),
                            dtype=self.dtype).tocsr()

        if self.n_values == 'auto':
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features

        return out

    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        return _transform_selected(X, self._fit_transform,
                                   self.categorical_features, copy=True)

    def _transform(self, X):
        """Asssumes X contains only categorical features."""
        X = check_arrays(X, sparse_format='dense', dtype=np.int)[0]
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape

        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        n_values_check = np.max(X, axis=0) + 1
        if (n_values_check > self.n_values_).any():
            raise ValueError("Feature out of bounds. Try setting n_values.")

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sp.coo_matrix((data, (row_indices, column_indices)),
                            shape=(n_samples, indices[-1]),
                            dtype=self.dtype).tocsr()
        if self.n_values == 'auto':
            out = out[:, self.active_features_]
        return out

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input array of type int.

        Returns
        -------
        X_out : sparse matrix, dtype=int
            Transformed input.
        """
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.

    Attributes
    ----------
    `classes_`: array of shape [n_class]
        Holds the label for each class.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    """

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelNormalizer was not fitted yet.")

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.classes_ = np.unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self.classes_, y = unique(y, return_inverse=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self._check_fitted()

        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))

        return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        self._check_fitted()

        y = np.asarray(y)
        return self.classes_[y]


class LabelBinarizer(BaseEstimator, TransformerMixin):
    """Binarize labels in a one-vs-all fashion

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

    Parameters
    ----------

    neg_label: int (default: 0)
        Value with which negative labels must be encoded.

    pos_label: int (default: 1)
        Value with which positive labels must be encoded.

    Attributes
    ----------
    `classes_`: array of shape [n_class]
        Holds the label for each class.

    `multilabel_`: boolean
        True if the transformer was fitted on a multilabel rather than a
        multiclass set of labels.

    Examples
    --------
    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(neg_label=0, pos_label=1)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.multilabel_
    False
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])
    >>> lb.multilabel_
    True

    See also
    --------
    label_binarize : function to perform the transform operation of
        LabelBinarizer with fixed classes.
    """

    def __init__(self, neg_label=0, pos_label=1):
        if neg_label >= pos_label:
            raise ValueError("neg_label must be strictly less than pos_label.")

        self.neg_label = neg_label
        self.pos_label = pos_label

    @property
    @deprecated("Attribute 'multilabel' was renamed to 'multilabel_' in "
                "0.14 and will be removed in 0.16")
    def multilabel(self):
        return self.multilabel_

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelBinarizer was not fitted yet.")

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
        y_type = type_of_target(y)
        self.multilabel_ = y_type.startswith('multilabel')
        if self.multilabel_:
            self.indicator_matrix_ = y_type == 'multilabel-indicator'

        self.classes_ = unique_labels(y)

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
        self._check_fitted()

        y_is_multilabel = type_of_target(y).startswith('multilabel')

        if y_is_multilabel and not self.multilabel_:
            raise ValueError("The object was not fitted with multilabel"
                             " input.")

        return label_binarize(y, self.classes_,
                              multilabel=self.multilabel_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label)

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array of shape [n_samples, n_classes]
            Target values.

        threshold : float or None
            Threshold used in the binary and multi-label cases.

            Use 0 when:
                - Y contains the output of decision_function (classifier)
            Use 0.5 when:
                - Y contains the output of predict_proba

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        self._check_fitted()

        if threshold is None:
            half = (self.pos_label - self.neg_label) / 2.0
            threshold = self.neg_label + half

        if self.multilabel_:
            Y = np.array(Y > threshold, dtype=int)
            # Return the predictions in the same format as in fit
            if self.indicator_matrix_:
                # Label indicator matrix format
                return Y
            else:
                # Lists of tuples format
                return [tuple(self.classes_[np.flatnonzero(Y[i])])
                        for i in range(Y.shape[0])]

        if len(Y.shape) == 1 or Y.shape[1] == 1:
            y = np.array(Y.ravel() > threshold, dtype=int)

        else:
            y = Y.argmax(axis=1)

        return self.classes_[y]


def label_binarize(y, classes, multilabel=False, neg_label=0, pos_label=1):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels to encode.

    classes : array of shape [n_classes]
        Uniquely holds the label for each class.

    multilabel : boolean
        Set to true if y is encoding a multilabel tasks (with a variable
        number of label assignements per sample) rather than a multiclass task
        where one sample has one and only one label assigned.

    neg_label: int (default: 0)
        Value with which negative labels must be encoded.

    pos_label: int (default: 1)
        Value with which positive labels must be encoded.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    >>> label_binarize([(1, 2), (6,), ()], multilabel=True,
    ...                classes=[1, 6, 4, 2])
    array([[1, 0, 0, 1],
           [0, 1, 0, 0],
           [0, 0, 0, 0]])

    See also
    --------
    label_binarize : function to perform the transform operation of
        LabelBinarizer with fixed classes.
    """
    y_type = type_of_target(y)

    if multilabel or len(classes) > 2:
        if y_type == 'multilabel-indicator':
            # nothing to do as y is already a label indicator matrix
            return y

        Y = np.zeros((len(y), len(classes)), dtype=np.int)
    else:
        Y = np.zeros((len(y), 1), dtype=np.int)

    Y += neg_label

    y_is_multilabel = y_type.startswith('multilabel')

    if multilabel:
        if not y_is_multilabel:
            raise ValueError("y should be a list of label lists/tuples,"
                             "got %r" % (y,))

        # inverse map: label => column index
        imap = dict((v, k) for k, v in enumerate(classes))

        for i, label_tuple in enumerate(y):
            for label in label_tuple:
                Y[i, imap[label]] = pos_label

        return Y

    else:
        y = np.asarray(y)

        if len(classes) == 2:
            Y[y == classes[1], 0] = pos_label
            return Y

        elif len(classes) >= 2:
            for i, k in enumerate(classes):
                Y[y == k, i] = pos_label
            return Y

        else:
            # Only one class, returns a matrix with all negative labels.
            return Y


class KernelCenterer(BaseEstimator, TransformerMixin):
    """Center a kernel matrix

    Let K(x_i, x_j) be a kernel defined by K(x_i, x_j) = phi(x_i)^T phi(x_j),
    where phi(x) is a function mapping x to a hilbert space. KernelCenterer is
    a class to center (i.e., normalize to have zero-mean) the data without
    explicitly computing phi(x). It is equivalent equivalent to centering
    phi(x) with sklearn.preprocessing.StandardScaler(with_std=False).
    """

    def fit(self, K, y=None):
        """Fit KernelCenterer

        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.

        Returns
        -------
        self : returns an instance of self.
        """
        K = array2d(K)
        n_samples = K.shape[0]
        self.K_fit_rows_ = np.sum(K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        return self

    def transform(self, K, y=None, copy=True):
        """Center kernel

        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.

        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """
        K = array2d(K)
        if copy:
            K = K.copy()

        K_pred_cols = (np.sum(K, axis=1) /
                       self.K_fit_rows_.shape[0])[:, np.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K


def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------

    X : array or scipy.sparse matrix with shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.

    Examples
    --------

    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[ 1.,  0.,  1.],
           [ 1.,  1.,  0.]])
    """
    X = safe_asarray(X)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sp.issparse(X):
        if sp.isspmatrix_coo(X):
            # Shift columns to the right.
            col = X.col + 1
            # Column indices of dummy feature are 0 everywhere.
            col = np.concatenate((np.zeros(n_samples), col))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            row = np.concatenate((np.arange(n_samples), X.row))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sp.coo_matrix((data, (row, col)), shape)
        elif sp.isspmatrix_csc(X):
            # Shift index pointers since we need to add n_samples elements.
            indptr = X.indptr + n_samples
            # indptr[0] must be 0.
            indptr = np.concatenate((np.array([0]), indptr))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sp.csc_matrix((data, indices, indptr), shape)
        else:
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        return np.hstack((np.ones((n_samples, 1)) * value, X))


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


def _get_median(negative_elements, n_zeros, positive_elements):
    """Compute the median of the array formed by negative_elements,
       n_zeros zeros and positive_elements. This function is used
       to support sparse matrices."""
    negative_elements = np.sort(negative_elements, kind='heapsort')
    positive_elements = np.sort(positive_elements, kind='heapsort')

    n_elems = len(negative_elements) + n_zeros + len(positive_elements)
    if not n_elems:
        return np.nan

    median_position = (n_elems - 1) / 2.0

    if round(median_position) == median_position:
        median = _get_elem_at_rank(negative_elements, n_zeros,
                              positive_elements, median_position)
    else:
        a = _get_elem_at_rank(negative_elements, n_zeros,
                         positive_elements, math.floor(median_position))
        b = _get_elem_at_rank(negative_elements, n_zeros,
                         positive_elements, math.ceil(median_position))
        median = (a + b) / 2.0

    return median


def _get_elem_at_rank(negative_elements, n_zeros, positive_elements, k):
    """Compute the kth largest element of the array formed by
       negative_elements, n_zeros zeros and positive_elements."""
    len_neg = len(negative_elements)
    len_pos = len(positive_elements)

    if k < len_neg:
        return negative_elements[k]
    elif k >= len_neg + n_zeros:
        return positive_elements[k - len_neg - n_zeros]
    else:
        return 0


def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
       [extra_value] * n_repeat, where extra_value is assumed to be not part
       of the array."""
    # Compute the most frequent value in array only
    if array.size > 0:
        mode = st.mode(array)
        most_frequent_value = mode[0][0]
        most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # Ties the breaks. Copy the behaviour of scipy.stats.mode
        if most_frequent_value < extra_value:
            return most_frequent_value
        else:
            return extra_value


class Imputer(BaseEstimator, TransformerMixin):
    """Imputation transformer for completing missing values.

    Parameters
    ----------
    missing_values : integer or string, optional (default="NaN")
        The placeholder for the missing values. All occurences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".

    strategy : string, optional (default="mean")
        The imputation strategy.
          - If "mean", then replace missing values using the mean along
            the axis.
          - If "median", then replace missing values using the median along
            the axis.
          - If "most_frequent", then replace missing using the most frequent
            value along the axis.

    axis : integer, optional (default=0)
        The axis along which to impute.
         - If `axis=0`, then impute along columns.
         - If `axis=1`, then impute along rows.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place.

    Attributes
    ----------
    `statistics_` : array of shape (n_features,) or (n_samples,)
        The statistics along the imputation axis.

    Notes
    -----
    - When ``axis=0``, columns which only contained missing values at `fit`
      are discarded upon `transform`.
    - When ``axis=1``, an exception is raised if there are rows for which it is
      not possible to fill in the missing values (e.g., because they only
      contain missing values).
    """
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.axis = axis
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0, 1]:
            raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data will be computed in transform()
        # when the imputation is done per sample (i.e., when axis=1).
        if self.axis == 0:
            X = atleast2d_or_csc(X, dtype=np.float64, force_all_finite=False)

            if sp.issparse(X):
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    self.axis)
            else:
                self.statistics_ = self._dense_fit(X,
                                                   self.strategy,
                                                   self.missing_values,
                                                   self.axis)

        return self

    def _sparse_fit(self, X, strategy, missing_values, axis):
        """Fit the transformer on sparse data."""
        # Imputation is done "by column", so if we want to do it
        # by row we only need to convert the matrix to csr format.
        if axis == 1:
            X = X.tocsr()
        else:
            X = X.tocsc()

        # Count the zeros
        if missing_values == 0:
            n_zeros_axis = np.zeros(X.shape[not axis])
        else:
            n_zeros_axis = X.shape[axis] - np.diff(X.indptr)

        # Mean
        if strategy == "mean":
            if missing_values != 0:
                n_non_missing = n_zeros_axis

                # Mask the missing elements
                mask_missing_values = _get_mask(X.data, missing_values)
                mask_valids = np.logical_not(mask_missing_values)

                # Sum only the valid elements
                new_data = X.data.copy()
                new_data[mask_missing_values] = 0
                X = sp.csc_matrix((new_data, X.indices, X.indptr), copy=False)
                sums = X.sum(axis=0)

                # Count the elements != 0
                mask_non_zeros = sp.csc_matrix((mask_valids.astype(np.float64),
                                               X.indices,
                                               X.indptr), copy=False)
                s = mask_non_zeros.sum(axis=0)
                n_non_missing = np.add(n_non_missing, s)

            else:
                sums = X.sum(axis=axis)
                n_non_missing = np.diff(X.indptr)

            # Ignore the error, columns with a np.nan statistics_
            # are not an error at this point. These columns will
            # be removed in transform
            with np.errstate(all="ignore"):
                return np.ravel(sums) / np.ravel(n_non_missing)

        # Median + Most frequent
        else:
            # Remove the missing values, for each column
            columns_all = np.hsplit(X.data, X.indptr[1:-1])
            mask_missing_values = _get_mask(X.data, missing_values)
            mask_valids = np.hsplit(np.logical_not(mask_missing_values),
                                    X.indptr[1:-1])

            columns = [col[mask.astype(np.bool)]
                       for col, mask in zip(columns_all, mask_valids)]

            # Median
            if strategy == "median":
                median = np.empty(len(columns))
                for i, column in enumerate(columns):

                    negatives = column[column < 0]
                    positives = column[column > 0]
                    median[i] = _get_median(negatives,
                                            n_zeros_axis[i],
                                            positives)

                return median

            # Most frequent
            elif strategy == "most_frequent":
                most_frequent = np.empty(len(columns))

                for i, column in enumerate(columns):
                    most_frequent[i] = _most_frequent(column,
                                                      0,
                                                      n_zeros_axis[i])

                return most_frequent

    def _dense_fit(self, X, strategy, missing_values, axis):
        """Fit the transformer on dense data."""
        X = array2d(X, force_all_finite=False)
        mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=axis)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=axis)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmask(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # scipy.stats.mstats.mode cannot be used because it will no work
            # properly if the first element is masked and if it's frequency
            # is equal to the frequency of the most frequent valid element
            # See https://github.com/scipy/scipy/issues/2636

            # To be able access the elements by columns
            if axis == 0:
                X = X.transpose()
                mask = mask.transpose()

            most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(np.bool)
                row = row[row_mask]
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        X = as_float_array(X, copy=self.copy)

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data need to be recomputed
        # when the imputation is done per sample
        if self.axis == 1:
            X = atleast2d_or_csr(X, force_all_finite=False)

            if sp.issparse(X):
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    self.axis)

            else:
                self.statistics_ = self._dense_fit(X,
                                                   self.strategy,
                                                   self.missing_values,
                                                   self.axis)
        else:
            X = atleast2d_or_csc(X, force_all_finite=False)

        # Delete the invalid rows/columns
        invalid_mask = np.isnan(self.statistics_)
        valid_mask = np.logical_not(invalid_mask)
        valid_statistics = self.statistics_[valid_mask]
        valid_statistics_indexes = np.where(valid_mask)[0]
        missing = ",".join(
            map(str, np.arange(X.shape[not self.axis])[invalid_mask]))

        if self.axis == 0 and invalid_mask.any():
            if self.verbose:
                warnings.warn("Deleting features without "
                              "observed values: %s" % missing)
            X = X[:, valid_statistics_indexes]
        elif self.axis == 1 and invalid_mask.any():
            raise ValueError("Some rows only contain "
                             "missing values: %s" % missing)

        # Do actual imputation
        if sp.issparse(X) and self.missing_values != 0:
            if self.axis == 0:
                X = X.tocsr()
            else:
                X = X.tocsc()

            mask = _get_mask(X.data, self.missing_values)
            indexes = X.indices[mask]

            X.data[mask] = valid_statistics[indexes].astype(X.dtype)
        else:
            if sp.issparse(X):
                X = X.toarray()

            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=self.axis)
            values = np.repeat(valid_statistics, n_missing)

            if self.axis == 0:
                coordinates = np.where(mask.transpose())[::-1]
            else:
                coordinates = mask

            X[coordinates] = values

        return X
