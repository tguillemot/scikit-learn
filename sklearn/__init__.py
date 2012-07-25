"""
Machine Learning module in python
=================================

sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.sourceforge.net for complete documentation.
"""
import sys

try:
    __SKLEARN_SETUP__
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write('Running from sklearn source directory.\n')
else:
    from . import __check_build
    from .base import clone


    try:
        from numpy.testing import nosetester

        class _NoseTester(nosetester.NoseTester):
            """ Subclass numpy's NoseTester to add doctests by default
            """

            def test(self, label='fast', verbose=1, extra_argv=['--exe'],
                            doctests=True, coverage=False):
                """Run the full test suite

                Examples
                --------
                This will run the test suite and stop at the first failing
                example
                >>> from sklearn import test
                >>> test(extra_argv=['--exe', '-sx']) #doctest: +SKIP
                """
                return super(_NoseTester, self).test(label=label, verbose=verbose,
                                        extra_argv=extra_argv,
                                        doctests=doctests, coverage=coverage)

        try:
            test = _NoseTester(raise_warnings="release").test
        except TypeError:
            # Older versions of numpy do not have a raise_warnings argument
            test = _NoseTester().test
        del nosetester
    except:
        pass


    __all__ = ['cross_validation', 'cluster', 'covariance',
               'datasets', 'decomposition', 'feature_extraction',
               'feature_selection', 'semi_supervised',
               'gaussian_process', 'grid_search', 'hmm', 'lda', 'linear_model',
               'metrics', 'mixture', 'naive_bayes', 'neighbors', 'pipeline',
               'preprocessing', 'qda', 'svm', 'test', 'clone', 'pls']

    __version__ = '0.12-git'
