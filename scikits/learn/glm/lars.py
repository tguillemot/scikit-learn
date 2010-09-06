# Least Angle Regression algorithm. See doc/module/glm for a
# complete discussion.
#
# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD Style.

import numpy as np
from scipy import linalg
from .base import LinearModel
import scipy.sparse as sp # needed by LeastAngleRegression
from .._minilearn import lars_fit_wrap
from ..utils.fixes import copysign

# Notes: np.ma.dot copies the masked array before doing the dot
# product. Maybe we should implement in C our own masked_dot that does
# not make unnecessary copies.

# all linalg.solve solve a triangular system, so this could be heavily
# optimized by binding (in scipy ?) trsv or trsm

def lars_path(X, y, max_iter=None, alpha_min=0, method="lar", precompute=True):
    """ Compute Least Angle Regression and LASSO path

        Parameters
        -----------
        X: array, shape: (n, p)
            Input data
        y: array, shape: (n)
            Input targets
        max_iter: integer, optional
            The number of 'kink' in the path
        alpha_min: float, optional
            The minimum correlation along the path. It corresponds
            to the regularization parameter alpha parameter in the Lasso.
        method: 'lar' or 'lasso'
            Specifies the problem solved: the LAR or its variant the LASSO-LARS
            that gives the solution of the LASSO problem for any regularization
            parameter.

        Returns
        --------
        alphas: array, shape: (k)
            The alphas along the path
        
        active: array, shape (?)
            Indices of active variables at the end of the path.
        
        coefs: array, shape (p,k)
            Coefficients along the path

        Notes
        ------
        XXX : add reference papers and wikipedia page
    
    TODOS:
    precompute : empty for now

    TODO: detect stationary points.
    Lasso variant
    store full path
    """

    X = np.atleast_2d(X)
    y = np.atleast_1d(y)

    n_samples, n_features = X.shape

    if max_iter is None:
        max_iter = min(n_samples, n_features)

    max_pred = max_iter # OK for now

    beta     = np.zeros ((max_iter + 1, X.shape[1]))
    alphas   = np.zeros (max_iter + 1)
    n_iter, n_pred = 0, 0
    active   = list()
    # holds the sign of covariance
    sign_active = np.empty (max_pred, dtype=np.int8)
    drop = False

    # will hold the cholesky factorization
    # only lower part is referenced. We do not create it as
    # empty array because chol_solve calls chkfinite on the
    # whole array, which can cause problems.
    L = np.zeros ((max_pred, max_pred), dtype=np.float64)

    Xt  = X.T
    Xna = Xt.view(np.ma.MaskedArray) # variables not in the active set
                                     # should have a better name

    Xna.soften_mask()

    while 1:


        # Calculate covariance matrix and get maximum
        res = y - np.dot (X, beta[n_iter]) # there are better ways
        Cov = np.ma.dot (Xna, res)

        imax    = np.ma.argmax (np.ma.abs(Cov)) #rename
        Cov_max =  Cov.data [imax]

        alpha = np.abs(Cov_max) #sum (np.abs(beta[n_iter]))
        alphas [n_iter] = alpha

        if (n_iter >= max_iter or n_pred >= max_pred ):
            break

        if (alpha < alpha_min): break


        if not drop:

            # Update the Cholesky factorization of (Xa * Xa') #
            #                                                 #
            #          ( L   0 )                              #
            #   L  ->  (       )  , where L * w = b           #
            #          ( w   z )    z = 1 - ||w||             #
            #                                                 #
            #   where u is the last added to the active set   #

            n_pred += 1
            active.append(imax)
            Xna[imax] = np.ma.masked
            Cov[imax] = np.ma.masked

            sign_active [n_pred-1] = np.sign (Cov_max)

            X_max = Xt[imax]

            c = np.dot (X_max, X_max)
            L [n_pred-1, n_pred-1] = c

            if n_pred > 1:
                b = np.dot (X_max, Xa.T)

                # please refactor me, using linalg.solve is overkill
                L [n_pred-1, :n_pred-1] = linalg.solve (L[:n_pred-1, :n_pred-1], b)
                v = np.dot(L [n_pred-1, :n_pred-1], L [n_pred - 1, :n_pred -1])
                L [n_pred-1,  n_pred-1] = np.sqrt (c - v)
        else:
            drop = False

        Xa = Xt[active] # also Xna[~Xna.mask]

        # Now we go into the normal equations dance.
        # (Golub & Van Loan, 1996)

        b = copysign (Cov_max.repeat(n_pred), sign_active[:n_pred])
        b = linalg.cho_solve ((L[:n_pred, :n_pred], True),  b)

        C = A = np.abs(Cov_max)
        u = np.dot (Xa.T, b)
        a = np.ma.dot (Xna, u)

        # equation 2.13, there's probably a simpler way
        g1 = (C - Cov) / (A - a)
        g2 = (C + Cov) / (A + a)

        # one for the border cases
        g = np.ma.concatenate((g1, g2))

        g = g[g > 0.]
        gamma_ = np.ma.min (g)

        if n_pred >= X.shape[1]:
            gamma_ = 1.

        if method == 'lasso':

            z = - beta[n_iter, active] / b
            z[z <= 0.] = np.inf

            idx = np.argmin(z)

            if z[idx] < gamma_:
                gamma_ = z[idx]
                drop = True

        n_iter += 1
        beta[n_iter, active] = beta[n_iter - 1, active] + gamma_ * b

        if drop:
            n_pred -= 1
            drop_idx = active.pop (idx)
            # please please please remove this masked arrays pain from me
            Xna[drop_idx] = Xna.data[drop_idx]
            print 'dropped ', idx, ' at ', n_iter, ' iteration'
            Xa = Xt[active] # duplicate
            L[:n_pred, :n_pred] = linalg.cholesky(np.dot(Xa, Xa.T), lower=True)
            sign_active = np.delete (sign_active, idx) # do an append to maintain size
            sign_active = np.append (sign_active, 0.)
            # should be done using cholesky deletes


    if alpha < alpha_min: # interpolate
        # interpolation factor 0 <= ss < 1
        ss = (alphas[n_iter-1] - alpha_min) / (alphas[n_iter-1] - alphas[n_iter])
        beta[n_iter] = beta[n_iter-1] + ss*(beta[n_iter] - beta[n_iter-1]);
        alphas[n_iter] = alpha_min
        alphas = alphas[:n_iter+1]
        beta = beta[:n_iter+1]

    return alphas, active, beta.T


class LARS (LinearModel):
    """ Least Angle Regression model a.k.a. LAR
    
    Parameters
    ----------
    n_features : int, optional
        Number of selected active features

    XXX : todo add fit_intercept
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    Attributes
    ----------
    `coef_` : array, shape = [n_features]
        parameter vector (w in the fomulation formula)

    XXX : add intercept_
    `intercept_` : float
        independent term in decision function.

    Examples
    --------
    >>> from scikits.learn import glm
    >>> clf = glm.LARS(n_features=1)
    >>> clf.fit([[-1,1], [0, 0], [1, 1]], [-1, 0, -1])
    LARS(normalize=True, n_features=1)
    >>> print clf.coef_
    [ 0.         -0.81649658]

    Notes
    -----
    See also scikits.learn.glm.LassoLARS that fits a LASSO model
    using a variant of Least Angle Regression
    
    XXX : add ref + wikipedia page
    
    See examples. XXX : add examples names
    """
    def __init__(self, n_features, normalize=True):
        self.n_features = n_features
        self.normalize = normalize
        self.coef_ = None

    def fit (self, X, y, **params):
        self._set_params(**params)
                # will only normalize non-zero columns

        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if self.normalize:
            self._xmean = X.mean(0)
            self._ymean = y.mean(0)
            X = X - self._xmean
            y = y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]

        method = 'lar'
        alphas_, active, coef_path_ = lars_path(X, y,
                                max_iter=self.n_features, method=method)
        self.coef_ = coef_path_[:,-1]
        return self


class LassoLARS (LinearModel):
    """ Lasso model fit with Least Angle Regression a.k.a. LARS
    
    It is a Linear Model trained with an L1 prior as regularizer.
    lasso).

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0

    XXX : todo add fit_intercept
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    Attributes
    ----------
    `coef_` : array, shape = [n_features]
        parameter vector (w in the fomulation formula)

    XXX : add intercept_
    `intercept_` : float
        independent term in decision function.

    Examples
    --------
    >>> from scikits.learn import glm
    >>> clf = glm.LassoLARS(alpha=0.1)
    >>> clf.fit([[-1,1], [0, 0], [1, 1]], [-1, 0, -1])
    LassoLARS(normalize=True, alpha=0.1, max_iter=None)
    >>> print clf.coef_
    [ 0.         -0.51649658]

    Notes
    -----
    See also scikits.learn.glm.Lasso that fits the same model using
    an alternative optimization strategy called 'coordinate descent.'
    """

    def __init__(self, alpha=1.0, max_iter=None, normalize=True):
        """ XXX : add doc
                # will only normalize non-zero columns
        """
        self.alpha = alpha
        self.normalize = normalize
        self.coef_ = None
        self.max_iter = max_iter

    def fit (self, X, y, **params):
        """ XXX : add doc
        """
        self._set_params(**params)

        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        n_samples = X.shape[0]
        alpha = self.alpha * n_samples # scale alpha with number of samples

        if self.normalize:
            self._xmean = X.mean(0)
            self._ymean = y.mean(0)
            X = X - self._xmean
            y = y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]

        method = 'lasso'
        alphas_, active, coef_path_ = lars_path(X, y,
                                            alpha_min=alpha, method=method,
                                            max_iter=self.max_iter)

        self.coef_ = coef_path_[:,-1]
        return self


#### OLD C-based LARS : will probably be removed


class LeastAngleRegression(LinearModel):
    """
    Least Angle Regression using the LARS algorithm.

    Attributes
    ----------
    `coef_` : array, shape = [n_features]
        parameter vector (w in the fomulation formula)

    `intercept_` : float
        independent term in decision function.

    `coef_path_` : array, shape = [max_features + 1, n_features]
         Full coeffients path.

    Notes
    -----
    predict does only work correctly in the case of normalized
    predictors.

    See also
    --------
    scikits.learn.glm.Lasso

    """

    def __init__(self):
        self.alphas_ = np.empty(0, dtype=np.float64)
        self._chol   = np.empty(0, dtype=np.float64)
        self.beta_    = np.empty(0, dtype=np.float64)

    def fit (self, X, Y, fit_intercept=True, max_features=None, normalize=True):
        """
        Fit the model according to data X, Y.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        Y : numpy array of shape [n_samples]
            Target values

        fit_intercept : boolean, optional
            wether to calculate the intercept for this model. If set
            to false, no intercept will be used in calculations
            (e.g. data is expected to be already centered).

        max_features : int, optional
            number of features to get into the model. The iterative
            will stop just before the `max_features` variable enters
            in the active set. If not specified, min(N, p) - 1
            will be used.

        normalize : boolean
            whether to normalize (make all non-zero columns have mean
            0 and norm 1).
        """
        ## TODO: resize (not create) arrays, check shape,
        ##    add a real intercept

        X  = np.asanyarray(X, dtype=np.float64, order='C')
        _Y = np.asanyarray(Y, dtype=np.float64, order='C')

        if Y is _Y: Y = _Y.copy()
        else: Y = _Y

        if max_features is None:
            max_features = min(*X.shape)-1

        sum_k = max_features * (max_features + 1) /2
        self.alphas_.resize(max_features + 1)
        self._chol.resize(sum_k)
        self.beta_.resize(sum_k)
        coef_row = np.zeros(sum_k, dtype=np.int32)
        coef_col = np.zeros(sum_k, dtype=np.int32)


        if normalize:
            # will only normalize non-zero columns
            self._xmean = X.mean(0)
            self._ymean = Y.mean(0)
            X = X - self._xmean
            Y = Y - self._ymean
            self._norms = np.apply_along_axis (np.linalg.norm, 0, X)
            nonzeros = np.flatnonzero(self._norms)
            X[:, nonzeros] /= self._norms[nonzeros]
        else:
            self._xmean = 0.
            self._ymean = 0.

        lars_fit_wrap(0, X, Y, self.beta_, self.alphas_, coef_row,
                      coef_col, self._chol, max_features)

        self.coef_path_ = sp.coo_matrix((self.beta_,
                                        (coef_row, coef_col)),
                                        shape=(X.shape[1], max_features+1)).todense()

        self.coef_ = np.ravel(self.coef_path_[:, max_features])

        if fit_intercept:
            self.intercept_ = self._ymean
        else:
            self.intercept_ = 0.

        return self


    def predict(self, X, normalize=True):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = np.asanyarray(X, dtype=np.float64, order='C')
        if normalize:
            X -= self._xmean
            X /= self._norms
        return  np.dot(X, self.coef_) + self.intercept_


    
