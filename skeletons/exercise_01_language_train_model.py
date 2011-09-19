"""Build a language detector model"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CharNGramAnalyzer
from sklearn.svm.sparse import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn import metrics

#
# New preprocessor better suited for language id than the default
# preprocessor
#

class LowerCasePreprocessor(object):
    """Simple preprocessor that should be available by default"""

    def preprocess(self, unicode_content):
        return unicode_content.lower()

    def __repr__(self):
        return "LowerCasePreprocessor()"

#
# The real code starts here
#

# the training data folder must be passed as first argument
languages_data_folder = sys.argv[1]
dataset = load_files(languages_data_folder, shuffle=True, random_state=42)

# split the dataset in training and test set:
n_samples_total = dataset.filenames.shape[0]
split = n_samples_total / 2

docs_train = dataset.data[:split]
docs_test = dataset.data[split:]

y_train = dataset.target[:split]
y_test = dataset.target[split:]


# Build a an analyzer that split strings into sequence of 1 to 3 characters
# after using the previously introduced LowerCasePreprocessor

# TODO

# Build a vectorizer / classifier pipeline using the previous analyzer

# TODO: the pipeline instance must be named 'clf'

# Fit the pipeline on the training set

# TODO

# Predict the outcome on the testing set

# TODO: the predicted outcome must be named 'y_predicted'


# TODO: uncomment the following once all of the above is implemented

## Print the classification report
#print metrics.classification_report(y_test, y_predicted,
#                                    target_names=dataset.target_names)
#
## Plot the confusion matrix
#cm = metrics.confusion_matrix(y_test, y_predicted)
#print cm
#
## import pylab as pl
##pl.matshow(cm)
##pl.show()
#
## Predict the result on some short new sentences:
#sentences = [
#    u'This is a language detection test.',
#    u'Ceci est un test de d\xe9tection de la langue.',
#    u'Dies ist ein Test, um die Sprache zu erkennen.',
#]
#predicted = clf.predict(sentences)
#
#for s, p in zip(sentences, predicted):
#    print u'The language of "%s" is "%s"' % (s, dataset.target_names[p])

