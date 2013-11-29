"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CharNGramAnalyzer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
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


# The training data folder must be passed as first argument
languages_data_folder = sys.argv[1]
dataset = load_files(languages_data_folder)

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_fraction=0.5)


# TASK: Build a an analyzer that split strings into sequence of 1 to 3
# characters after using the previous preprocessor
analyzer = CharNGramAnalyzer(
    min_n=1,
    max_n=3,
    preprocessor=LowerCasePreprocessor(),
)

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clf = Pipeline([
    ('vec', CountVectorizer(analyzer=analyzer)),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', Perceptron()),
])

# TASK: Fit the pipeline on the training set
clf.fit(docs_train, y_train)

# TASK: Predict the outcome on the testing set in a variable named y_predicted
y_predicted = clf.predict(docs_test)

# Print the classification report
print metrics.classification_report(y_test, y_predicted,
                                    target_names=dataset.target_names)

# Plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print cm

#import pylab as pl
#pl.matshow(cm, cmap=pl.cm.jet)
#pl.show()

# Predict the result on some short new sentences:
sentences = [
    u'This is a language detection test.',
    u'Ceci est un test de d\xe9tection de la langue.',
    u'Dies ist ein Test, um die Sprache zu erkennen.',
]
predicted = clf.predict(sentences)

for s, p in zip(sentences, predicted):
    print u'The language of "%s" is "%s"' % (s, dataset.target_names[p])
