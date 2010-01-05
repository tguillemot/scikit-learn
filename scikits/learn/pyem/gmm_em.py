# /usr/bin/python
# Last Change: Tue Jun 12 08:00 PM 2007 J

"""Module implementing GMM, a class to estimate Gaussian mixture models using
EM, and EM, a class which use GMM instances to estimate models parameters using
the ExpectationMaximization algorithm."""

__docformat__ = 'restructuredtext'

# TODO:
#   - which methods to avoid va shrinking to 0 ? There are several options, 
#   not sure which ones are appropriates
#   - improve EM trainer
#   - online EM

import numpy as N
#import numpy.linalg as lin
from numpy.random import randn
#import _c_densities as densities
import densities
#from kmean import kmean
from scipy.cluster.vq import kmeans2 as kmean
from gauss_mix import GmParamError

#from misc import _DEF_ALPHA, _MIN_DBL_DELTA, _MIN_INV_COND

# Error classes
class GmmError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self):
        Exception.__init__(self)

class GmmParamError(GmmError):
    """Exception raised for errors in gmm params

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, message):
        GmmError.__init__(self)
        self.message    = message
    
    def __str__(self):
        return self.message

class MixtureModel(object):
    pass

class ExpMixtureModel(MixtureModel):
    """Class to model mixture of exponential pdf (eg Gaussian, exponential,
    Laplace, etc..). This is a special case because some parts of EM are common
    for those models..."""
    pass

class GMM(ExpMixtureModel):
    """ A class to model a Gaussian Mixture Model (GMM). An instance of this
    class is created by giving weights, mean and variances in the ctor.  An
    instanciated object can be sampled, trained by EM. """
    def init_kmean(self, data, niter = 5):
        """ Init the model with kmean."""
        k       = self.gm.k
        d       = self.gm.d
        init    = data[0:k, :]

        # XXX: This is bogus initialization should do better (in kmean or here,
        # do not know yet): should 
        (code, label)   = kmean(data, init, niter, minit = 'matrix')

        w   = N.ones(k) / k
        mu  = code.copy()
        if self.gm.mode == 'diag':
            va = N.zeros((k, d))
            for i in range(k):
                for j in range(d):
                    va[i, j] = N.cov(data[N.where(label==i), j], rowvar = 0)
        elif self.gm.mode == 'full':
            va  = N.zeros((k*d, d))
            for i in range(k):
                va[i*d:i*d+d, :] = \
                    N.cov(data[N.where(label==i)], rowvar = 0)
        else:
            raise GmmParamError("mode " + str(self.gm.mode) + \
                    " not recognized")

        self.gm.set_param(w, mu, va)

        self.isinit = True

    def init_random(self, data):
        """ Init the model at random."""
        k   = self.gm.k
        d   = self.gm.d
        w   = N.ones(k) / k
        mu  = randn(k, d)
        if self.gm.mode == 'diag':
            va  = N.fabs(randn(k, d))
        else:
            # If A is invertible, A'A is positive definite
            va  = randn(k * d, d)
            for i in range(k):
                va[i*d:i*d+d]   = N.dot( va[i*d:i*d+d], 
                    va[i*d:i*d+d].T)

        self.gm.set_param(w, mu, va)
        
        self.isinit = True

    def init_test(self, data):
        """Use values already in the model as initialization.
        
        Useful for testing purpose when reproducability is necessary. This does
        nothing but checking that the mixture model has valid initial
        values."""
        # We have
        try:
            self.gm.check_state()
        except GmParamError, e:
            print "Model is not properly initalized, cannot init EM."
            raise "Message was %s" % str(e)
        
    # TODO: 
    #   - format of parameters ? For variances, list of variances matrix,
    #   keep the current format, have 3d matrices ?
    #   - To handle the different modes, we could do something "fancy" such as
    #   replacing methods, to avoid checking cases everywhere and unconsistency.
    def __init__(self, gm, init = 'kmean'):
        """Initialize a mixture model.
        
        Initialize the model from a GM instance. This class implements all the
        necessary functionalities for EM.

        :Parameters:
            gm : GM
                the mixture model to train.
            init : string
                initialization method to use."""
        self.gm = gm

        # Possible init methods
        init_methods = {'kmean': self.init_kmean, 'random' : self.init_random,
                'test': self.init_test}

        if init not in init_methods:
            raise GmmParamError('init method %s not recognized' + str(init))

        self.init   = init_methods[init]
        self.isinit = False
        self.initst = init

    def compute_responsabilities(self, data):
        """Compute responsabilities.
        
        Return normalized and non-normalized respondabilities for the model.
        
        Note
        ----
        Computes the latent variable distribution (a posteriori probability)
        knowing the explicit data for the Gaussian model (w, mu, var): gamma(t,
        i) = P[state = i | observation = data(t); w, mu, va]

        This is basically the E step of EM for finite mixtures."""
        # compute the gaussian pdf
        tgd	= densities.multiple_gauss_den(data, self.gm.mu, self.gm.va)
        # multiply by the weight
        tgd	*= self.gm.w
        # Normalize to get a pdf
        gd	= tgd  / N.sum(tgd, axis=1)[:, N.newaxis]

        return gd, tgd

    def compute_log_responsabilities(self, data):
        """Compute log responsabilities.
        
        Return normalized and non-normalized responsabilities for the model (in
        the log domain)
        
        Note
        ----
        Computes the latent variable distribution (a posteriori probability)
        knowing the explicit data for the Gaussian model (w, mu, var): gamma(t,
        i) = P[state = i | observation = data(t); w, mu, va]

        This is basically the E step of EM for finite mixtures."""
        # compute the gaussian pdf
        tgd	= densities.multiple_gauss_den(data, self.gm.mu, self.gm.va, log = True)
        # multiply by the weight
        tgd	+= N.log(self.gm.w)
        # Normalize to get a pdf
        gd	= tgd  - densities.logsumexp(tgd)[:, N.newaxis]

        return gd, tgd

    def update_em(self, data, gamma):
        """Computes update of the Gaussian Mixture Model (M step)
        from the a posteriori pdf, computed by gmm_posterior
        (E step).
        """
        k = self.gm.k
        d = self.gm.d
        n = data.shape[0]
        invn = 1.0/n
        mGamma = N.sum(gamma, axis = 0)

        if self.gm.mode == 'diag':
            mu = N.zeros((k, d))
            va = N.zeros((k, d))
            gamma = gamma.T
            for c in range(k):
                x = N.dot(gamma[c:c+1, :], data)[0, :]
                xx = N.dot(gamma[c:c+1, :], data ** 2)[0, :]

                mu[c, :] = x / mGamma[c]
                va[c, :] = xx  / mGamma[c] - mu[c, :] ** 2
            w   = invn * mGamma

        elif self.gm.mode == 'full':
            # In full mode, this is the bottleneck: the triple loop
            # kills performances. This is pretty straightforward
            # algebra, so computing it in C should not be too difficult. The
            # real problem is to have valid covariance matrices, and to keep
            # them positive definite, maybe with special storage... Not sure
            # it really worth the risk
            mu  = N.zeros((k, d))
            va  = N.zeros((k*d, d))

            gamma = gamma.transpose()
            for c in range(k):
                #x   = N.sum(N.outer(gamma[:, c], 
                #            N.ones((1, d))) * data, axis = 0)
                x = N.dot(gamma[c:c+1, :], data)[0, :]
                xx = N.zeros((d, d))
                
                # This should be much faster than recursing on n...
                for i in range(d):
                    for j in range(d):
                        xx[i, j] = N.sum(data[:, i] * data[:, j] * gamma[c, :],
                                axis = 0)

                mu[c, :] = x / mGamma[c]
                va[c*d:c*d+d, :] = xx  / mGamma[c] \
                        - N.outer(mu[c, :], mu[c, :])
            w   = invn * mGamma
        else:
            raise GmmParamError("varmode not recognized")

        self.gm.set_param(w, mu, va)

    def likelihood(self, data):
        """ Returns the current log likelihood of the model given
        the data """
        assert(self.isinit)
        # compute the gaussian pdf
        tgd	= densities.multiple_gauss_den(data, self.gm.mu, self.gm.va)
        # multiply by the weight
        tgd	*= self.gm.w

        return N.sum(N.log(N.sum(tgd, axis = 1)), axis = 0)

    def bic(self, data):
        """ Returns the BIC (Bayesian Information Criterion), 
        also called Schwarz information criterion. Can be used 
        to choose between different models which have different
        number of clusters. The BIC is defined as:

        BIC = 2 * ln(L) - k * ln(n)

        where:
            * ln(L) is the log-likelihood of the estimated model
            * k is the number of degrees of freedom
            * n is the number of frames
        
        Not that depending on the literature, BIC may be defined as the opposite
        of the definition given here. """

        if self.gm.mode == 'diag':
            # for a diagonal model, we have k - 1 (k weigths, but one
            # constraint of normality) + k * d (means) + k * d (variances)
            free_deg    = self.gm.k * (self.gm.d * 2 + 1) - 1
        elif self.gm.mode == 'full':
            # for a full model, we have k - 1 (k weigths, but one constraint of
            # normality) + k * d (means) + k * d * d / 2 (each covariance
            # matrice has d **2 params, but with positivity constraint)
            if self.gm.d == 1:
                free_deg = self.gm.k * 3 - 1
            else:
                free_deg = self.gm.k * (self.gm.d + 1 + self.gm.d ** 2 / 2) - 1

        lk  = self.likelihood(data)
        n   = N.shape(data)[0]
        return bic(lk, free_deg, n)

    # syntactic sugar
    def __repr__(self):
        repre   = ""
        repre   += "Gaussian Mixture Model\n"
        repre   += " -> initialized by %s\n" % str(self.initst)
        repre   += self.gm.__repr__()
        return repre

class EM:
    """An EM trainer. An EM trainer
    trains from data, with a model
    
    Not really useful yet"""
    def __init__(self):
        pass
    
    def train(self, data, model, maxiter = 10, thresh = 1e-5):
        """Train a model using EM.

        Train a model using data, and stops when the likelihood increase
        between two consecutive iteration fails behind a threshold, or when the
        number of iterations > niter, whichever comes first

        :Parameters:
            data : ndarray
                contains the observed features, one row is one frame, ie one
                observation of dimension d
            model : GMM
                GMM instance.
            maxiter : int
                maximum number of iterations
            thresh : threshold
                if the slope of the likelihood falls below this value, the
                algorithm stops.

        :Returns:
            likelihood : ndarray
                one value per iteration.

        Note
        ----
        The model is trained, and its parameters updated accordingly, eg the
        results are put in the GMM instance.
        """
        if not isinstance(model, MixtureModel):
            raise TypeError("expect a MixtureModel as a model")

        # Initialize the data (may do nothing depending on the model)
        model.init(data)

        # Likelihood is kept
        like    = N.zeros(maxiter)

        # Em computation, with computation of the likelihood
        g, tgd      = model.compute_responsabilities(data)
        like[0]     = N.sum(N.log(N.sum(tgd, 1)), axis = 0)
        model.update_em(data, g)
        for i in range(1, maxiter):
            g, tgd      = model.compute_responsabilities(data)
            like[i]     = N.sum(N.log(N.sum(tgd, 1)), axis = 0)
            model.update_em(data, g)
            if has_em_converged(like[i], like[i-1], thresh):
                return like[0:i]

        return like
    
# Misc functions
def bic(lk, deg, n):
    """ Expects lk to be log likelihood """
    return 2 * lk - deg * N.log(n)

def has_em_converged(like, plike, thresh):
    """ given likelihood of current iteration like and previous
    iteration plike, returns true is converged: based on comparison
    of the slope of the likehood with thresh"""
    diff    = N.abs(like - plike)
    avg     = 0.5 * (N.abs(like) + N.abs(plike))
    if diff / avg < thresh:
        return True
    else:
        return False

if __name__ == "__main__":
    pass
    ## # #++++++++++++++++++
    ## # # Export the figure
    ## # #++++++++++++++++++
    ## # F   = P.gcf()
    ## # DPI = F.get_dpi()
    ## # DefaultSize = F.get_size_inches()
    ## # # the default is 100dpi for savefig:
    ## # F.savefig("example1.png")

    ## # # Now make the image twice as big, while keeping the fonts and all the
    ## # # same size
    ## # F.set_figsize_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    ## # Size = F.get_size_inches()
    ## # print "Size in Inches", Size
    ## # F.savefig("example2.png")
    ## P.show()
