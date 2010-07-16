"""
Machine Learning module in python
=================================

scikits.learn is a Python module integrating classique machine
learning algorithms in the tightly-nit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.sourceforge.net for complete documentation.
"""

import cross_val
import ball_tree
import gmm
import glm
import logistic
import lda
import metrics
import svm
import features

__all__ = ['cross_val', 'ball_tree', 'gmm', 'glm', 'logistic', 'lda',
           'metrics', 'svm', 'features']

__version__ = '0.5-git'

