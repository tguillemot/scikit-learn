# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <larsmans@gmail.com>
#
# Licence: BSD 3 clause

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

# Import CBLAS Functions
cdef extern from "cblas.h":

    # dot product of two n elemenet vectors x and y
    double ddot "cblas_ddot" (int n,
                              const double* x,
                              int incrx,
                              double* y,
                              int incry) nogil

    # scale the passed in n element vector x at scale
    void dscal "cblas_dscal" (int n,
                              double scale,
                              double* x,
                              int incrx) nogil

    # adds an n element vector x * scale to another n element vector y
    void daxpy "cblas_daxpy" (int n,
                              double scale,
                              const double* x,
                              int incrx,
                              double* y,
                              int incry) nogil


np.import_array()


cdef class WeightVector(object):
    """Dense vector represented by a scalar and a numpy array.

    The class provides methods to ``add`` a sparse vector
    and scale the vector.
    Representing a vector explicitly as a scalar times a
    vector allows for efficient scaling operations.

    Attributes
    ----------
    w : ndarray, dtype=double, order='C'
        The numpy array which backs the weight vector.
    w_data_ptr : double*
        A pointer to the data of the numpy array.
    wscale : double
        The scale of the vector.
    n_features : int
        The number of features (= dimensionality of ``w``).
    sq_norm : double
        The squared norm of ``w``.
    """

    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode='c'] w,
                  np.ndarray[double, ndim=1, mode='c'] aw):
        cdef double *wdata = <double *>w.data

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."
                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.w_data_ptr = wdata
        self.wscale = 1.0
        self.n_features = w.shape[0]
        self.sq_norm = ddot(<int>w.shape[0], wdata, 1, wdata, 1)

        self.aw = aw
        if self.aw is not None:
            self.aw_data_ptr = <double *>aw.data
            self.alpha = 0.0
            self.beta = 1.0

    cdef void add(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                  double c, double t) nogil:
        """Scales sample x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double mu = 1.0 / t
        cdef double alpha = self.alpha
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef double* w_data_ptr = self.w_data_ptr
        cdef double* aw_data_ptr
        if self.aw is not None:
            aw_data_ptr = self.aw_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            innerprod += (w_data_ptr[idx] * val)
            xsqnorm += (val * val)
            w_data_ptr[idx] += val * (c / wscale)
            if self.aw is not None:
                aw_data_ptr[idx] += (self.alpha * val * (-c / wscale))

        if self.aw is not None:
            if t > 1:
                self.beta /= (1 - mu)
            self.alpha += mu * self.beta * wscale

        self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

    cdef double dot(self, double *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil:
        """Computes the dot product of a sample x and the weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x`` (length of x_ind_ptr).

        Returns
        -------
        innerprod : double
            The inner product of ``x`` and ``w``.
        """
        cdef int j
        cdef int idx
        cdef double innerprod = 0.0
        cdef double* w_data_ptr = self.w_data_ptr
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            innerprod += w_data_ptr[idx] * x_data_ptr[j]
        innerprod *= self.wscale
        return innerprod

    cdef void scale(self, double c) nogil:
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c
        self.sq_norm *= (c * c)
        if self.wscale < 1e-9:
            self.reset_wscale()

    cdef void reset_wscale(self) nogil:
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        if self.aw is not None:
            daxpy(<int>self.aw.shape[0], self.alpha, <double *>self.w.data,
                  1, <double *>self.aw.data, 1)
            dscal(<int>self.aw.shape[0], 1.0 / self.beta,
                  <double *>self.aw.data, 1)
            self.alpha = 0.0
            self.beta = 1.0

        dscal(<int>self.w.shape[0], self.wscale, <double *>self.w.data, 1)
        self.wscale = 1.0

    cdef double norm(self) nogil:
        """The L2 norm of the weight vector. """
        return sqrt(self.sq_norm)
