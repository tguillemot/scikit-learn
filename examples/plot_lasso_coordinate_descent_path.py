"""
=====================
Lasso and Elastic Net
=====================

Lasso and elastic net (L1 and L2 penalisation) implemented using a
coordinate descent.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

# $Id$

from datetime import datetime
from itertools import cycle
import numpy as np
import pylab as pl

from scikits.learn.glm.coordinate_descent import Lasso, ElasticNet, lasso_path, \
                                    enet_path

n_samples, n_features, maxit = 5, 10, 30

np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)

################################################################################
# Fit models
################################################################################

################################################################################
# Demo path functions
################################################################################

print "Computing regularization path using the lasso..."
start = datetime.now()
alphas_lasso, weights_lasso = lasso_path(X, y, factor=0.97, n_alphas = 100)
print "This took ", datetime.now() - start

print "Computing regularization path using the elastic net..."
start = datetime.now()
alphas_enet, weights_enet = enet_path(X, y, factor=0.97, n_alphas = 100,
                                                beta=0.5)
print "This took ", datetime.now() - start


# Display results
color_iter = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for color, weight_lasso, weight_enet in zip(color_iter,
                            weights_lasso.T, weights_enet.T):
    pl.plot(-np.log(alphas_lasso), weight_lasso, color)
    pl.plot(-np.log(alphas_enet), weight_enet, color+'x')

pl.xlabel('-log(lambda)')
pl.ylabel('weights')
pl.title('Lasso and Elastic-Net Paths')
pl.legend(['Lasso','Elastic-Net'])
pl.axis('tight')
pl.show()

