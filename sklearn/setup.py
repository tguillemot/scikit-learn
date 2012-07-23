import os
from os.path import join
import warnings


def get_blas_info():
    from numpy.distutils.system_info import get_info

    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])

    return cblas_libs, blas_info


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sklearn', parent_package, top_path)

    config.add_subpackage('__check_build')
    config.add_subpackage('svm')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('feature_extraction/tests')
    config.add_subpackage('cluster')
    config.add_subpackage('cluster/tests')
    config.add_subpackage('covariance')
    config.add_subpackage('covariance/tests')
    config.add_subpackage('decomposition')
    config.add_subpackage('decomposition/tests')
    config.add_subpackage("ensemble")
    config.add_subpackage("ensemble/tests")
    config.add_subpackage('feature_selection')
    config.add_subpackage('feature_selection/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')
    config.add_subpackage('externals')
    config.add_subpackage('mixture')
    config.add_subpackage('mixture/tests')
    config.add_subpackage('gaussian_process')
    config.add_subpackage('gaussian_process/tests')
    config.add_subpackage('neighbors')
    config.add_subpackage('manifold')
    config.add_subpackage('metrics')
    config.add_subpackage('semi_supervised')
    config.add_subpackage("tree")
    config.add_subpackage("tree/tests")
    config.add_subpackage('metrics/tests')
    config.add_subpackage('metrics/tests')
    config.add_subpackage('metrics/cluster')
    config.add_subpackage('metrics/cluster/tests')

    # add cython extension module for hmm
    config.add_extension(
        '_hmmc',
        sources=['_hmmc.c'],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
    )

    # some libs needs cblas, fortran-compiled BLAS will not be sufficient
    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or (
        ('NO_ATLAS_INFO', 1) in blas_info.get('define_macros', [])):
        config.add_library('cblas',
                           sources=[join('src', 'cblas', '*.c')])
        warnings.warn(BlasNotFoundError.__doc__)

    # the following packages depend on cblas, so they have to be build
    # after the above.
    config.add_subpackage('linear_model')
    config.add_subpackage('utils')

    # add the test directory
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
