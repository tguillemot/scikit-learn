
.. _feature_extraction:

==================
Feature extraction
==================

.. currentmodule:: scikits.learn.feature_extraction

The :mod:`scikits.learn.feature_extraction` module can be used to extract
features in a format supported by machine learning algorithms from datasets
consisting of formats such as text and image.


Text feature extraction
=======================

.. currentmodule:: scikits.learn.feature_extraction.text

XXX: a lot to do here


Image feature extraction
========================

.. currentmodule:: scikits.learn.feature_extraction.image

Patch extraction
----------------

The :func:`extract_patches_2d` function extracts patches from an image stored
as a two-dimensional array, or three-dimensional with color information along
the third axis. For rebuilding an image from all its patches, use
:func:`reconstruct_from_patches_2d`. For example::

    >>> import numpy as np
    >>> import from scikits.learn.feature_extraction.image import \
            extract_patches_2d, reconstruct_from_patches_2d, PatchExtractor
    >>> one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
    >>> one_image[:, :, 0]
    array([[ 0,  3,  6,  9],
           [12, 15, 18, 21],
           [24, 27, 30, 33],
           [36, 39, 42, 45]])
    >>> patches = extract_patches_2d(one_image, (2, 2), max_patches=2, random_state=0)
    >>> patches.shape
    (2, 2, 2, 3)
    >>> patches[:, :, :, 0]
    array([[[ 0,  3],
            [12, 15]],
    <BLANKLINE>
           [[15, 18],
            [27, 30]]])
    >>> patches = extract_patches_2d(one_image, (2, 2))
    >>> patches.shape
    (9, 2, 2, 3)
    >>> patches[4, :, :, 0]
    array([[15, 18],
           [27, 30]])
    >>> reconstructed_image = reconstruct_from_patches_2d(patches, (4, 4, 3))
    >>> np.testing.assert_array_equal(one_image, reconstructed_image)

The :class:`PatchExtractor` class works in the same way as
:func:`extract_patches_2d`, only it supports multiple images as input. It is
implemented as an estimator, so it can be used in pipelines. See::

    >>> five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
    >>> patches = PatchExtractor((2, 2)).transform(five_images)
    >>> patches.shape
    (45, 2, 2, 3)
