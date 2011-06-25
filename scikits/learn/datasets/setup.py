#!/usr/bin/env python

import numpy

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('datasets', parent_package, top_path)
    config.add_data_dir('data')
    config.add_data_dir('descr')


    config.add_extension('_svmlight_format',
                         sources=['_svmlight_format.cpp'],
                         include_dirs=[numpy.get_include()]
                         )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
