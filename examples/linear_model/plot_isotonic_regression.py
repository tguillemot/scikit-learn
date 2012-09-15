"""
===================
Isotonic Regression
===================

An illustration of the isotonic regression on generated data.
The isotonic regression finds a non-decreasing approximation of a function
while minimizing the mean squared error on the training data.
For comparison a linear regression is also presented.
"""

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# Licence: BSD

import numpy as np
import pylab as pl
from matplotlib.collections import LineCollection

from sklearn.linear_model import IsotonicRegression, LinearRegression

n = 100
x = np.arange(n)
y = np.random.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))

###############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()
y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

###############################################################################
# plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(n))

fig = pl.figure()
pl.plot(x, y, 'r.', markersize=12)
pl.plot(x, y_, 'g.-', markersize=12)
pl.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
pl.gca().add_collection(lc)
pl.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
pl.title('Isotonic regression')
pl.show()
