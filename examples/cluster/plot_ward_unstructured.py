"""
===========================================================
A demo of hierarchical clustering - unstructured ward
===========================================================

Example builds a swiss roll dataset and runs the hierarchical
clustering on k-Nearest Neighbors graph. It's a hierarchical
clustering without structure prior.

"""

# Authors : Vincent Michel, 2010
#           Alexandre Gramfort, 2010
# License: BSD

print __doc__

import time as time
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from scikits.learn.cluster import Ward
from scikits.learn.datasets.samples_generator import swiss_roll

###############################################################################
# Generate data (swiss roll dataset)
n_samples = 500
noise = 0.05
X = swiss_roll(n_samples, noise)

###############################################################################
# Compute clustering
print "Compute unstructured hierarchical clustering..."
st = time.time()
ward = Ward(n_clusters=5).fit(X)
label = ward.labels_
print "Elapsed time: ", time.time() - st
print "Number of points: ", label.size
print "Number of clusters: ", np.unique(label).size

###############################################################################
# Plot result
fig = pl.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
              'o', color=pl.cm.jet(np.float(l) / np.max(label + 1)))
pl.show()
