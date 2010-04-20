"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of 
hand-written digits.

"""
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: Simplified BSD

# Standard scientific Python imports
import pylab as pl

# The digits dataset
from scikits.learn import datasets
digits = datasets.load_digits()

# The data that we are interesting in is made of 8x8 images of digits,
# let's have a look at the first 3 images. We know which digit they
# represent: it is given in the 'target' of the dataset.
for index, (image, label) in enumerate(zip(digits.images, digits.target)[:4]):
    pl.subplot(2, 4, index+1)
    pl.imshow(image, cmap=pl.cm.gray_r)
    pl.title('Training: %i' % label)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_features = len(digits.images)
data = digits.images.reshape((n_features, -1))

# Import a classifier:
from scikits.learn import svm
classifier = svm.SVC()

# We learn the digits on the first half of the digits
classifier.fit(data[:n_features/2], digits.target[:n_features/2])

# Now predict the value of the digit on the second half:
predicted = classifier.predict(data[n_features/2:])

for index, (image, prediction) in enumerate(zip(
                                       digits.images[n_features/2:], 
                                       predicted
                                    )[:4]):
    pl.subplot(2, 4, index+5)
    pl.imshow(image, cmap=pl.cm.gray_r)
    pl.title('Prediction: %i' % prediction)


pl.show()
