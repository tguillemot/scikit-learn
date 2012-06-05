import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as pl

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

#----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T

# Observations
y = f(X).ravel()

dy = 1.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
xx = np.atleast_2d(np.linspace(0, 10, 1000)).T

clf = GradientBoostingRegressor(loss='quantile', alpha=0.9,
                                n_estimators=250, max_depth=3,
                                learn_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_upper = clf.predict(xx)

clf.set_params(alpha=0.1)
clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_lower = clf.predict(xx)

clf.set_params(loss='ls')
clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_pred = clf.predict(xx)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
pl.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
pl.plot(X, y, 'b.', markersize=10, label=u'Observations')
pl.plot(xx, y_pred, 'r-', label=u'Prediction')
pl.plot(xx, y_upper, 'k-')
pl.plot(xx, y_lower, 'k-')
pl.fill(np.concatenate([xx, xx[::-1]]),
        np.concatenate([y_upper, y_lower[::-1]]),
        alpha=.5, fc='b', ec='None', label='90% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')
pl.show()
