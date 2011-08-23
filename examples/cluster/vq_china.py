# -*- coding: utf-8 -*-
"""
=========================================
Vector Quantization of Lena using k-means
=========================================

Performs a Vector Quantization of an image, reducing the number of colors
required to show the image.
"""
print __doc__
import numpy as np
import pylab as pl
from scikits.learn.cluster import KMeans
from scikits.learn.datasets import load_sample_images
from scikits.learn.utils import shuffle

# Get all sample images and obtain just china.jpg
sample_image_name = "china.jpg"
sample_images = load_sample_images()
index = None
for i, filename in enumerate(sample_images.filenames):
    if filename.endswith(sample_image_name):
        index = i
        break
if index is None:
    raise AttributeError("Cannot find sample image: %s" % sample_image_name)
image_data = sample_images.images[index]

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image_data.shape)
assert d == 3
image_array = np.reshape(image_data, (w * h, d))

print "Fitting estimator on a sub sample of the data"
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(k=10, max_iter=1000).fit(image_array_sample)
print "done."

# Get labels for all points
print "Predicting labels:"
labels = kmeans.predict(image_array)
print "done."

def recreate_image(codebook, labels, w, h):
    # Recreates the (compressed) image from the code book, labels and dimensions
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
pl.figure()
ax = pl.axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()
pl.imshow(image_data)

pl.figure()
ax = pl.axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()

pl.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

pl.show()
