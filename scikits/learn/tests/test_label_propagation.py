""" test the label propagation module """

import numpy as np

from .. import label_propagation
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from StringIO import StringIO

def test_label_propagation_fit():
    samples = [[1,0],[0,1],[1,3]]
    labels = [0,1,-1]
    lp = label_propagation.LabelPropagation()
    lp.fit(samples, labels, unlabeled_identifier=-1)
    assert lp.transduction[2] == 1

def test_label_spreading_fit():
    samples = [[1,0],[0,1],[1,3]]
    labels = [0,1,-1]
    lp = label_propagation.LabelSpreading()
    lp.fit(samples, labels, unlabeled_identifier=-1)
    assert lp.transduction[2] == 1

def test_string_labels():
    samples = [[1,0],[0,1],[1,3]]
    labels = ['banana', 'orange', 'unlabeled']
    lp = label_propagation.LabelPropagation()
    lp.fit(samples, labels, unlabeled_identifier='unlabeled')
    assert lp.transduction[2] == 'orange'

def test_distribution():
    samples = [[1,0],[0,1],[1,1]]
    labels = [0,1,-1]
    lp = label_propagation.LabelPropagation()
    lp.fit(samples, labels, unlabeled_identifier=-1)
    assert_array_almost_equal(np.asarray(lp._y[2]), np.array([[ 0.32243136,  0.32243136]]) )

if __name__ == '__main__':
    import nose
