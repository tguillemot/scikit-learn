from os.path import join
import warnings
import sys
import numpy

from ConfigParser import ConfigParser

def configuration(parent_package='', top_path=None):
    
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, get_standard_file, \
         BlasNotFoundError

    config = Configuration('tree', parent_package, top_path)

    #config.add_subpackage('tests')

    libdecisiontree_sources = ['libdecisiontree.cpp', 'src/Node.cpp']
    libdecisiontree_depends = ['libdecisiontree_helper.cpp',
                               join('src', 'Histogram.h'),
                               join('src', 'Node.h'),
                               join('src', 'Node.cpp')]

    config.add_extension('libdecisiontree',
                         sources = libdecisiontree_sources,
                         include_dirs = [numpy.get_include(), 'src'],
                         depends = libdecisiontree_depends,
                         language="c++"
                         )

    config.add_extension('_tree',
                         sources=['_tree.c'],
                         include_dirs=[numpy.get_include()]
                         )

    config.add_subpackage('tests')

    return config

if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

