from os.path import join
import warnings
import numpy
import sys
if sys.version_info[0] < 3:
    from ConfigParser import ConfigParser
else:
    from configparser import ConfigParser

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    from numpy.distutils.system_info import get_standard_file
    from numpy.distutils.system_info import BlasNotFoundError

    config = Configuration('glm', parent_package, top_path)

    site_cfg  = ConfigParser()
    site_cfg.read(get_standard_file('site.cfg'))

    config.add_extension('cd_fast_sparse',
                         sources=[join('src', 'cd_fast_sparse.c')],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-std=c99'])

    # add other directories
    config.add_subpackage('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
