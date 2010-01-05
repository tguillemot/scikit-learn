#! /usr/bin/env python
# Last Change: Mon Aug 21 12:00 PM 2006 J
# TODO:
#   - check how to handle cmd line build options with distutils and use
#   it in the building process

""" pyem is a small python package to estimate Gaussian Mixtures Models
from data, using Expectation Maximization"""
from distutils.core import setup, Extension
from pyem import version as pyem_version

# distutils does not update MANIFEST correctly, removes it
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import re

from numpy.distutils.misc_util import get_numpy_include_dirs
NUMPYINC    = get_numpy_include_dirs()[0]

# General variables:
#   - DISTNAME: name of the distributed package
#   - VERSION: the version reference is in pyem/__init__.py file
#   - other upper cased variables are the same than the corresponding 
#   keywords in setup call
DISTNAME    = 'pyem' 
VERSION     = pyem_version
DESCRIPTION ='A python module for Expectation Maximization learning of mixtures pdf',
AUTHOR      ='David Cournapeau',
AUTHOR_EMAIL='david@ar.media.kyoto-u.ac.jp',
URL         ='http://ar.media.kyoto-u.ac.jp/members/david',

# Source files for extensions

# Functions used to substitute values in File.
# Mainly use to replace config.h capabilities
def do_subst_in_file(sourcefile, targetfile, dict):
    """Replace all instances of the keys of dict with their values.
    For example, if dict is {'%VERSION%': '1.2345', '%BASE%': 'MyProg'},
    then all instances of %VERSION% in the file will be replaced with 1.2345 etc.
    """
    try:
        f = open(sourcefile, 'rb')
        contents = f.read()
        f.close()
    except:
        raise IOError, "Can't read source file %s"%sourcefile

    for (k,v) in dict.items():
        contents = re.sub(k, v, contents)
    try:
        f = open(targetfile, 'wb')
        f.write(contents)
        f.close()
    except:
        raise IOError, "Can't read source file %s"%sourcefile
    return 0 # success
 
class SetupOption:
    def __init__(self):
        self.kmean      = 'py'
        self.ext_modules= []
        self.cmdclass   = {}
        self.subsdic     = {'%KMEANIMPORT%': []}

    def _config_kmean(self):
        # Check in this order:
        #   - kmean in scipy.cluster,
        #   - custom vq with pyrex 
        #   - custom pure python vq
        #try:
        #    from scipy.cluster.vq import kmeans
        #    self.kmean  = 'scipy'
        #    #self.subsdic['%KMEANIMPORT%']   = scipy_kmean
        #except ImportError:
        #    try:
        #        from Pyrex.Distutils import build_ext
        #        self.kmean  = 'pyrex'
        #        self.ext_modules.append(Extension('pyem/c_gmm', 
        #            ['pyem/src/c_gmm.pyx'], include_dirs=[NUMPYINC]))
        #        self.cmdclass['build_ext']  = build_ext
        #        #self.subsdic['%KMEANIMPORT%']   = pyrex_kmean
        #    except ImportError:
        #        self.kmean  = 'py'
        #        #self.subsdic['%KMEANIMPORT%']   = pyrex_kmean
        try:
            from Pyrex.Distutils import build_ext
            self.kmean  = 'pyrex'
            self.ext_modules.append(Extension('pyem/c_gmm', 
                ['pyem/src/c_gmm.pyx'], include_dirs=[NUMPYINC]))
            self.cmdclass['build_ext']  = build_ext
            #self.subsdic['%KMEANIMPORT%']   = pyrex_kmean
        except ImportError:
            self.kmean  = 'py'
            #self.subsdic['%KMEANIMPORT%']   = pyrex_kmean
    def setup(self):
        self._config_kmean()
        import time
        #do_subst_in_file('pyem/kmean.py.in', 'pyem/kmean.py', self.subsdic)
        setup(name      = DISTNAME,
            version     = VERSION,
            description = DESCRIPTION,
            author      = AUTHOR,
            author_email= AUTHOR_EMAIL,
            url         = URL,
            packages    = ['pyem'],
            ext_modules = self.ext_modules,
            cmdclass    = self.cmdclass)

stpobj  = SetupOption()
stpobj.setup()
