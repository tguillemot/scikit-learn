"""
===========================================================
A demo of hierarchical clustering - structured ward
===========================================================

Example builds a swiss roll dataset and runs the hierarchical
clustering on k-Nearest Neighbors graph. It's a hierarchical
clustering with structure prior.

"""
# Authors : Vincent Michel, 2010
#           Alexandre Gramfort, 2010
# License: BSD

print __doc__

import time as time
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from scikits.learn.neighbors import kneighbors_graph
from scikits.learn.cluster import Ward
from scikits.learn.datasets.samples_generator import swiss_roll

###############################################################################
# Generate data (swiss roll dataset)
n_samples = 5000
noise = 0.05
X = swiss_roll(n_samples, noise)

###############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors
connectivity = kneighbors_graph(X, n_neighbors=10)

###############################################################################
# Compute clustering
print "Compute structured hierarchical clustering..."
st = time.time()
ward = Ward(n_clusters=10).fit(X, connectivity=connectivity)
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
              'o', color=pl.cm.jet(float(l) / np.max(label + 1)))
pl.show()
