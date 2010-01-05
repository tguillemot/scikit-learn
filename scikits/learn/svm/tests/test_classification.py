from numpy.testing import *
import numpy as N

set_local_path('../..')
from svm.classification import *
from svm.dataset import LibSvmClassificationDataSet, LibSvmTestDataSet
from svm.kernel import *
restore_path()

class test_classification(NumpyTestCase):
    def check_basics(self):
        Model = LibSvmCClassificationModel
        kernel = LinearKernel()
        Model(kernel)
        Model(kernel, cost=1.0)
        weights = [(2, 10.0), (1, 20.0), (0, 30.0)]
        Model(kernel, weights=weights)
        Model(kernel, 1.0, weights)
        Model(kernel, cost=1.0, weights=weights)

        Model = LibSvmNuClassificationModel
        Model(kernel)
        Model(kernel, nu=0.5)
        Model(kernel, weights=weights)
        Model(kernel, 0.5, weights)

    def check_c_basics(self):
        ModelType = LibSvmCClassificationModel

        labels = [0, 1, 1, 2]
        x = [N.array([0, 0]),
             N.array([0, 1]),
             N.array([1, 0]),
             N.array([1, 1])]
        traindata = LibSvmClassificationDataSet(zip(labels, x))
        model = ModelType(RBFKernel(traindata.gamma))
        results = model.fit(traindata)
        testdata = LibSvmTestDataSet(x)
        results.predict(testdata)
        results.predict_values(testdata)

    def check_c_more(self):
        ModelType = LibSvmCClassificationModel

        labels = [0, 1, 1, 2]
        x = [N.array([0, 0]),
             N.array([0, 1]),
             N.array([1, 0]),
             N.array([1, 1])]
        traindata = LibSvmClassificationDataSet(zip(labels, x))
        cost = 10.0
        weights = [(1, 10.0)]
        testdata = LibSvmTestDataSet(x)

        kernels = [
            LinearKernel(),
            PolynomialKernel(3, traindata.gamma, 0.0),
            RBFKernel(traindata.gamma)
            ]
        expected_rhos = [
            [-0.999349, -1.0, -3.0],
            [0.375, -1.0, -1.153547],
            [0.671181, 0.0, -0.671133]
            ]
        expected_errors = [0, 1, 0]

        for kernel, expected_rho, expected_error in \
            zip(kernels, expected_rhos, expected_errors):
            model = ModelType(kernel, cost, weights)
            results = model.fit(traindata)

            self.assertEqual(results.labels, [0, 1, 2])
            self.assertEqual(results.nSV, [1, 2, 1])

            # use decimal=4 to suppress slight differences in values
            # calculated for rho on Windows with MSVC 7.1 and on
            # Fedora Core 4 with GCC 4.0.0.
            assert_array_almost_equal(results.rho, expected_rho, decimal=4)

            predictions = N.array(results.predict(testdata))
            self.assertEqual(N.sum(predictions != labels), expected_error)

    def check_c_probability(self):
        ModelType = LibSvmCClassificationModel

        labels = [0, 1, 1, 2]
        x = [N.array([0, 0]),
             N.array([0, 1]),
             N.array([1, 0]),
             N.array([1, 1])]
        traindata = LibSvmClassificationDataSet(zip(labels, x))
        cost = 10.0
        weights = [(1, 10.0)]
        testdata = LibSvmTestDataSet(x)

        kernels = [
            LinearKernel(),
            PolynomialKernel(3, traindata.gamma, 0.0),
            RBFKernel(traindata.gamma)
            ]

        for kernel in kernels:
            model = ModelType(kernel, cost, weights)
            results = model.fit(traindata)
            results.predict_probability(testdata)

    def check_cross_validate(self):
        labels = ([-1] * 50) + ([1] * 50)
        x = N.random.randn(len(labels), 10)
        traindata = LibSvmClassificationDataSet(zip(labels, x))
        kernel = LinearKernel()
        model = LibSvmCClassificationModel(kernel)
        nr_fold = 10
        pcorr = model.cross_validate(traindata, nr_fold)
        # XXX check cross-validation with and without probability
        # output enabled

    def check_nu_train(self):
        pass

if __name__ == '__main__':
    NumpyTest().run()
