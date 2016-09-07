"""
=================================
Gaussian Mixture Model Sine Curve
=================================

This example highlights the advantages of the Dirichlet Process:
complexity control and dealing with sparse data. The dataset is formed
by 100 points loosely spaced following a noisy sine curve. The fit by
the GMM class, using the expectation-maximization algorithm to fit a
mixture of 10 Gaussian components, finds too-small components and very
little structure. The fits by the Dirichlet process, however, show
that the model can either learn a global structure for the data (small
alpha) or easily interpolate to finding relevant local structure
(large alpha), never falling into the problems shown by the GMM class.
"""

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, index, title):
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.04, top=0.95, hspace=.2, wspace=.05,
                    left=.03, right=.97)

# Number of samples per component
n_samples = 100


# Generate random sample following a sine curve
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(X.shape[0]):
    x = i * step - 6.
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))

# Fit a Gaussian mixture with EM using ten components
gmm = mixture.GaussianMixture(n_components=10, covariance_type='full',
                              max_iter=100).fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Expectation-maximization')

# Fit a Gaussian mixture models with a Dirichlet process prior
# with ten components
dpgmm = mixture.DirichletGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e-2,
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="random", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             r'Dirichlet Process with $\beta_0=0.01$')

# Sample the Dirichlet process Gaussian mixture
X_s, y_s = dpgmm.sample(n_samples=5000)
plot_samples(X_s, y_s, dpgmm.n_components, 0, "Sampling of model obtained "
             r"with a Dirichlet Process with $\beta_0=0.01$ and $5000$ samples")

# Fit a Gaussian mixture models with a Dirichlet process prior ten components
dpgmm = mixture.DirichletGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e+2,
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="kmeans", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 2,
             r'Dirichlet Process with $\beta_0=100$')

# Sample the Dirichlet process Gaussian mixture
X_s, y_s = dpgmm.sample(n_samples=5000)
plot_samples(X_s, y_s, dpgmm.n_components, 1, "Sampling of model obtained "
             r"with a Dirichlet Process with $\beta_0=100$ and $5000$ samples")

plt.show()
