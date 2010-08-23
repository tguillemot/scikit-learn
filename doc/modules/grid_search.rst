===================================================
Grid Search
===================================================

.. contents:: Tables of contents

`scikits.learn.grid_search` is a package to optimize
the parameters of a model (e.g. Support Vector Classifier)
using cross-validation

It is implemented in python, and uses the numpy and scipy
packages. The computation can be run in parallel using
the multiprocessing package.

GridSearchCV
====================

.. autoclass:: scikits.learn.grid_search.GridSearchCV
    :members:

Examples
--------

See :ref:`example_grid_search_digits.py` for an example of
Grid Search computation on the digits dataset.

