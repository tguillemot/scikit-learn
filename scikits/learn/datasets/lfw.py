"""Loader for the Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/

Each picture is centered on a single face. The typical task is called
Face Verification: given a pair of two pictures, a binary classifier
must predict whether the two images are from the same person.

An alternative task, Face Recognition or Face Identification is:
given the picture of the face of an unknown person, identify the name
of the person by refering to a gallery of previously seen pictures of
identified persons.

Both Face Verification and Face Recognition are tasks that are typically
performed on the output of a model trained to perform Face Detection. The
most popular model for Face Detection is called Viola-Johns and is
implemented in the OpenCV library.
"""
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

from os.path import join
from os.path import exists
from os import listdir
from os import makedirs

import logging

from scipy.misc import imread
from scipy.misc import imresize
import numpy as np

from scikits.learn.externals.joblib import Memory
from .base import get_data_home
from .base import Bunch


BASE_URL = "http://vis-www.cs.umass.edu/lfw/"
ARCHIVE_NAME = "lfw.tgz"
FUNNELED_ARCHIVE_NAME = "lfw-funneled.tgz"
TARGET_FILENAMES = [
    'pairsDevTrain.txt',
    'pairsDevTest.txt',
    'pairs.txt',
    'people.txt',
]


def scale_face(face):
    """Scale back to 0-1 range in case of normalization for plotting"""
    scaled = face - face.min()
    scaled /= scaled.max()
    return scaled


def check_fetch_lfw(data_home=None, funneled=True):
    """Helper function to download any missing LFW data"""
    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, "lfw_home")

    if funneled:
        archive_path = join(lfw_home, FUNNELED_ARCHIVE_NAME)
        data_folder_path = join(lfw_home, "lfw_funneled")
        archive_url = BASE_URL + FUNNELED_ARCHIVE_NAME
    else:
        archive_path = join(lfw_home, ARCHIVE_NAME)
        data_folder_path = join(lfw_home, "lfw")
        archive_url = BASE_URL + ARCHIVE_NAME

    if not exists(lfw_home):
        makedirs(lfw_home)

    logging.info("LFW parent data folder: %s", lfw_home)

    if not exists(archive_path):
        import urllib
        logging.info("Downloading LFW data: %s", archive_url)
        downloader = urllib.urlopen(archive_url)
        open(archive_path, 'wb').write(downloader.read())

    for target_filename in TARGET_FILENAMES:
        target_filepath = join(lfw_home, target_filename)
        if not exists(target_filepath):
            url = BASE_URL + target_filename
            logging.info("Downloading LFW metadata: %s", url)
            downloader = urllib.urlopen(BASE_URL + target_filename)
            open(target_filepath, 'wb').write(downloader.read())

    if not exists(data_folder_path):
        import tarfile
        logging.info("Decompressing the data archive to %s", data_folder_path)
        tarfile.open(archive_path, "r:gz").extractall(path=lfw_home)

    return lfw_home, data_folder_path


def _load_lfw_pairs(index_file_path, data_folder_path, slice_=None,
                    center=True, normalize=True, resize=None):
    """Perform the actual data loading

    This operation is meant to be cached by a joblib wrapper.
    """
    with open(index_file_path) as f:
        splitted_lines = [l.strip().split('\t') for l in f.readlines()]
    filtered = [l for l in splitted_lines if len(l) > 2]
    n_pairs = len(filtered)

    default_slice = (slice(0, 250), slice(0, 250), slice(0, 3))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice, c_slice = slice_
    h = (h_slice.stop - h_slice.start) / (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) / (w_slice.step or 1)
    c = (c_slice.stop - c_slice.start) / (c_slice.step or 1)

    if resize is not None:
        h = int(resize * h)
        w = int(resize * w)

    target = np.zeros(n_pairs, dtype=np.int)
    pairs = np.zeros((n_pairs, 2, h, w, c), dtype=np.float32)

    for i, components in enumerate(filtered):
        if i % 1000 == 0:
            logging.info("Loading pair #%05d / %05d", i + 1, n_pairs)

        if len(components) == 3:
            target[i] = 1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[0], int(components[2]) - 1),
            )
        elif len(components) == 4:
            target[i] = -1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[2], int(components[3]) - 1),
            )
        else:
            raise ValueError("invalid line %d: %r" % (i + 1, components))
        for j, (name, idx) in enumerate(pair):
            person_folder = join(data_folder_path, name)
            filenames = list(sorted(listdir(person_folder)))
            filepath = join(person_folder, filenames[idx])
            face = np.asarray(imread(filepath)[slice_], dtype=np.float32)
            face /= 255.0 # scale uint8 coded colors to the [0.0, 1.0] floats
            if resize is not None:
                face = imresize(face, resize)
            face_shape = face.shape
            raveled_face = face.ravel()
            if center:
                raveled_face -= raveled_face.mean()
            stddev = raveled_face.std()
            if normalize and stddev != 0.0:
                raveled_face /= stddev
            pairs[i, j, :, :, :] = raveled_face.reshape(face_shape)

    return pairs, target


def load_lfw_pairs(subset='train', data_home=None,
                   slice_=(slice(50, 200), slice(75, 175), None),
                   center=True, normalize=True, funneled=True, resize=0.5):
    """Loader for the Labeled Faces in the Wild (LFW) dataset

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on the
    official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face. The task is called Face
    Verification: given a pair of two pictures, a binary classifier must
    predict whether the two images are from the same person.

    Parameters
    ----------
    subset: optional, default: 'train'
        Select the dataset to load: 'train' for the development
        training set, 'test' for the development test set, and '10_folds' for
        the official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home: optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit learn data is stored in '~/scikit_learn_data' subfolders.

    slice_: optional
        Provide a custom 3D slice (width, height, channels) to extract the
        'interesting' part of the jpeg files

    center: optional, True by default
        Locally center the each face by removing the mean.

    normalize: optional, True by default
        Perform local constrast normalization by dividing by the stddev.
    """

    parameters = locals().copy()
    del parameters['subset']
    del parameters['data_home']
    del parameters['funneled']

    lfw_home, data_folder_path = check_fetch_lfw(data_home=data_home,
                                                 funneled=funneled)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    m = Memory(cachedir=join(lfw_home, 'joblib'), mmap_mode='c', verbose=0)
    load_func = m.cache(_load_lfw_pairs)

    # select the right metadata file harcording to the requested subset
    label_filenames = {
        'train': 'pairsDevTrain.txt',
        'test': 'pairsDevTest.txt',
        '10_folds': 'pairs.txt',
    }
    if subset not in label_filenames:
        raise ValueError("subset='%s' is invalid: should be one of %r" % (
            subset, list(sorted(label_filenames.keys()))))
    index_file_path = join(lfw_home, label_filenames[subset])

    # load and memoize the pairs as np arrays
    pairs, target = load_func(index_file_path, data_folder_path, **parameters)

    # pack the results as a Bunch instance
    return Bunch(data=pairs, target=target,
                 DESCR="'%s' segment of the LFW pairs dataset" % subset)

