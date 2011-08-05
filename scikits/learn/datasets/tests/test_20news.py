""" Test the 20news downloader, if the data is available.
"""
import numpy as np
import nose

from scikits.learn import datasets

def test_20news():
    try:
        data = datasets.fetch_20newsgroups(subset='all',
                        download_if_missing=False, 
                        shuffle=False)
    except IOError:
        # Data not there
        return

    # Extract a reduced dataset
    data2cats = datasets.fetch_20newsgroups(subset='all', 
                            categories=data.target_names[-1:-3:-1],
                            shuffle=False)
    # Check that the ordering of the target_names is the same
    # as the ordering in the full dataset
    nose.tools.assert_equal(data2cats.target_names, 
                            data.target_names[-2:])
    # Assert that we have only 0 and 1 as labels
    nose.tools.assert_equal(np.unique(data2cats.target).tolist(), [0, 1])

    # Check that the first entry of the reduced dataset corresponds to 
    # the first entry of the corresponding category in the full dataset
    entry1 = data2cats.data[0]
    category = data2cats.target_names[data2cats.target[0]]
    label = data.target_names.index(category)
    entry2 = data.data[np.where(data.target == label)[0][0]]
    nose.tools.assert_true(entry1 == entry2)

    

