from os.path import join
import warnings
import numpy
from ConfigParser import ConfigParser

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, get_standard_file, BlasNotFoundError
    config = Configuration('learn',parent_package,top_path)

    site_cfg  = ConfigParser()
    site_cfg.read(get_standard_file('site.cfg'))

    config.add_subpackage('datasets')
    config.add_subpackage('features')
    config.add_subpackage('feature_selection')
    config.add_subpackage('utils')

    # libsvm
    libsvm_includes = [numpy.get_include()]
    libsvm_libraries = []
    libsvm_library_dirs = []
    libsvm_sources = [join('src', '_libsvm.c')]

    if site_cfg.has_section('libsvm'):
        libsvm_includes.append(site_cfg.get('libsvm', 'include_dirs'))
        libsvm_libraries.append(site_cfg.get('libsvm', 'libraries'))
        libsvm_library_dirs.append(site_cfg.get('libsvm', 'library_dirs'))
    else:
        libsvm_sources.append(join('src', 'svm.cpp'))

    config.add_extension('_libsvm',
                         sources=libsvm_sources,
                         include_dirs=libsvm_includes,
                         libraries=libsvm_libraries,
                         library_dirs=libsvm_library_dirs,
                         depends=[join('src', 'svm.h'),
                                 join('src', 'libsvm_helper.c'),
                                  ])

    ### liblinear module
    blas_sources = [join('src', 'blas', 'daxpy.c'),
                    join('src', 'blas', 'ddot.c'),
                    join('src', 'blas', 'dnrm2.c'),
                    join('src', 'blas', 'dscal.c')]

    liblinear_sources = [join('src', 'linear.cpp'),
                         join('src', '_liblinear.c'),
                         join('src', 'tron.cpp')]

    # we try to link agains system-wide blas
    blas_info = get_info('blas_opt', 0)

    extra_compile_args = blas_info.pop('extra_compile_args', [])

    if not blas_info:
        config.add_library('blas', blas_sources)
        warnings.warn(BlasNotFoundError.__doc__)

    config.add_extension('_liblinear',
                         sources=liblinear_sources,
                         libraries = blas_info.pop('libraries', ['blas']),
                         include_dirs=['src',
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         depends=[join('src', 'linear.h'),
                                  join('src', 'tron.h'),
                                  join('src', 'blas', 'blas.h'),
                                  join('src', 'blas', 'blasp.h')],
                         **blas_info)

    ## end liblinear module

    # minilear needs cblas, fortran-compiled BLAS will not be sufficient
    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or (
        ('NO_ATLAS_INFO', 1) in blas_info.get('define_macros', [])):
        config.add_library('cblas',
                           sources=[
                               join('src', 'cblas', '*.c'),
                               ]
                           )

    minilearn_sources = [
        join('src', 'minilearn', 'lars.c'),
        join('src', 'minilearn', '_minilearn.c')]


    config.add_extension('_minilearn',
                         sources=minilearn_sources,
                         libraries = blas_info.pop('libraries', 
                                                    ['cblas']),
                         include_dirs=[join('src', 'minilearn'),
                                       join('src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         extra_compile_args=['-std=c99'] + \
                                             blas_info.pop('extra_compile_args', []),
                         **blas_info
                         )

    config.add_extension('ball_tree',
                         sources=[join('src', 'BallTree.cpp')],
                         include_dirs=[numpy.get_include()]
                         )

    config.add_extension('cd_fast',
                         sources=[join('src', 'cd_fast.c')],
                         # libraries=['m'],
                         include_dirs=[numpy.get_include()])


    config.add_subpackage('utils')

    # add the test directory
    config.add_data_dir('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
