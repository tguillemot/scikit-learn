#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD style

"""
The :mod:`sklearn.gaussian_process` module implements scalar Gaussian Process
based predictions.
"""

from .gaussian_process import GaussianProcess
from . import correlation_models
from . import regression_models
