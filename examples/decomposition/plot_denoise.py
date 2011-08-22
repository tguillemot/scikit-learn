"""
=========================================
Image denoising using dictionary learning
=========================================
An example comparing the effect of reconstructing noisy fragments
of Lena using a dictionary learned from the clear image.
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
# downsample
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena /= 16.0

height, width = lena.shape
print "Distorting image..."

lena[:, height/2:] += 0.075 * np.random.randn(width, height/2)

print "Extracting clean patches..."
patch_size = (4, 4)
data = extract_patches_2d(lena[:, :height/2], patch_size, random_state=0)
data = data.reshape(data.shape[0], -1)
#intercept = np.mean(data, 0)
#data -= intercept

###############################################################################
# Learn the dictionary from clean patches
dico = DictionaryLearningOnline(n_atoms=100, alpha=1e-5, n_iter=100,
                                verbose=True, transform_algorithm='omp')
V = dico.fit(data).components_

###############################################################################
# Generate noisy data and reconstruct using various methods
print ""  # a line break

transform_algorithms = [
    ('1-Orthogonal Matching Pursuit', 'omp',
     {'n_nonzero_coefs': 1, 'precompute_gram': True}),

    ('2-Orthogonal Matching Pursuit', 'omp',
     {'n_nonzero_coefs': 2, 'precompute_gram': True}),

    ('Least-angle regression', 'lars',
     {'max_iter': 2})]

######################
# Display the original
i = 1
vmin, vmax = 0, 1
pl.figure(figsize=(6, 5))
pl.subplot(2, 2, i)
pl.title("Noisy image")
pl.imshow(lena, vmin=vmin, vmax=vmax, cmap=pl.cm.gray, interpolation='nearest')
pl.xticks(())
pl.yticks(())

print "Extracting noisy patches..."
data = extract_patches_2d(lena[:, height/2:], patch_size, random_state=0)
data = data.reshape(data.shape[0], -1)
reconstructions = {}
for title, transform_algorithm, fit_params in transform_algorithms:
    print title,
    reconstructions[title] = lena.copy()
    t0 = time()
    dico.transform_algorithm = transform_algorithm
    code = dico.transform(data, **fit_params)
    patches = np.dot(code, V)  # + intercept
    patches = patches.reshape(len(data), *patch_size)
    reconstructions[title][:, height/2:] = reconstruct_from_patches_2d(patches,
                                                           (width, height / 2))
    print time() - t0

    i += 1
    pl.subplot(2, 2, i)
    pl.title(title)
    pl.imshow(reconstructions[title], vmin=vmin, vmax=vmax, cmap=pl.cm.gray,
    interpolation='nearest')
    pl.xticks(())
    pl.yticks(())

pl.subplots_adjust(0.09, 0.04, 0.92, 0.88, 0.18, 0.19)
pl.suptitle('Reconstruction methods for image denoising')
pl.show()
