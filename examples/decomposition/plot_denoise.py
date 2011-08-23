"""
=========================================
Image denoising using dictionary learning
=========================================
An example comparing the effect of reconstructing noisy fragments
of Lena using online :ref:`DictionaryLearning` and various transform methods.

The dictionary is fitted on the non-distorted left half of the image, and
subsequently used to reconstruct the right half.
"""
print __doc__

from time import time

import pylab as pl
import scipy as sp
import numpy as np

from scikits.learn.decomposition import DictionaryLearningOnline
from scikits.learn.feature_extraction.image import extract_patches_2d, \
                                                   reconstruct_from_patches_2d

###############################################################################
# Load Lena image and extract patches
lena = sp.lena() / 256.0

# downsample for higher speed
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena /= 16.0
height, width = lena.shape

# Distort the right half of the image
print "Distorting image..."
lena[:, height/2:] += 0.075 * np.random.randn(width, height/2)

# Extract all clean patches from the left half of the image
print "Extracting clean patches..."
patch_size = (4, 4)
data = extract_patches_2d(lena[:, :height/2], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, 0)
data -= intercept

###############################################################################
# Learn the dictionary from clean patches
t0 = time()
dico = DictionaryLearningOnline(n_atoms=100, alpha=1e-2, n_iter=300,
                                verbose=True, transform_algorithm='omp')
V = dico.fit(data).components_
dt = time() - t0
print dt

pl.figure(figsize=(4.5, 5))
for i, comp in enumerate(V):
    pl.subplot(10, 10, i + 1)
    pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r)
    pl.xticks(())
    pl.yticks(())
pl.suptitle("Dictionary learned from Lena patches\n Train time %.1fs" % dt,
            fontsize=16)
pl.subplots_adjust(0.02, 0.01, 0.98, 0.88, 0.08, 0.01)

###############################################################################
# Display the distorted image
i = 1
vmin, vmax = 0, 1
pl.figure(figsize=(7, 6))
pl.subplot(2, 2, i)
pl.title("Noisy image")
pl.imshow(lena, vmin=vmin, vmax=vmax, cmap=pl.cm.gray, interpolation='nearest')
pl.xticks(())
pl.yticks(())

###############################################################################
# Extract noisy patches and reconstruct them using the dictionary
print "Extracting noisy patches..."
data = extract_patches_2d(lena[:, height/2:], patch_size, random_state=0)
data = data.reshape(data.shape[0], -1) - intercept

transform_algorithms = [
    ('1-Orthogonal Matching Pursuit', 'omp',
     {'n_nonzero_coefs': 1, 'precompute_gram': True}),

    ('2-Orthogonal Matching Pursuit', 'omp',
     {'n_nonzero_coefs': 2, 'precompute_gram': True}),

    ('5-Least-angle regression', 'lars',
     {'max_iter': 5})]

reconstructions = {}
for title, transform_algorithm, fit_params in transform_algorithms:
    print title,
    reconstructions[title] = lena.copy()
    t0 = time()
    dico.transform_algorithm = transform_algorithm
    code = dico.transform(data, **fit_params)
    patches = np.dot(code, V) + intercept
    patches = patches.reshape(len(data), *patch_size)
    reconstructions[title][:, height/2:] = reconstruct_from_patches_2d(patches,
                                                           (width, height / 2))
    dt = time() - t0
    print dt
    i += 1
    pl.subplot(2, 2, i)
    pl.title(title + '\ntransform time: %.1f' % dt)
    pl.imshow(reconstructions[title], vmin=vmin, vmax=vmax, cmap=pl.cm.gray,
    interpolation='nearest')
    pl.xticks(())
    pl.yticks(())

pl.subplots_adjust(0.11, 0.04, 0.89, 0.84, 0.18, 0.26)
pl.suptitle('Transform methods for image denoising')
pl.show()
