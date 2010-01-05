from model import Model
from results import Results
import libsvm
import utils

class RegressionResults(Results):
    def __init__(self, dtype, model):
        Results.__init__(self, dtype, model)
        model = model.contents
        self.rho = model.rho[0]
        self.sv_coef = model.sv_coef[0][:model.l]

    def predict(self, x):
        x = self.dtype.convert_test_data(x)
        xptr = utils.array_as_ctype(x, libsvm.svm_node)
        return libsvm.svm_predict(self.model, xptr)

class EpsilonSVRModel(Model):
    """
    A model for epsilon-SV regression.

    See also:

    - Smola, Scholkopf: A Tutorial on Support Vector Regression
    - Gunn: Support Vector Machines for Classification and Regression
    - Muller, Vapnik: Using Support Vector Machines for Time Series
      Prediction
    """

    Results = RegressionResults

    def __init__(self, dtype, cost=1.0, epsilon=0.1, **kwargs):
        Model.__init__(self, dtype, **kwargs)
        self.svm_type = libsvm.EPSILON_SVR
        self.cost = cost
        self.epsilon = epsilon

class NuSVRModel(Model):
    """
    A model for nu-SV regression.

    See also: Scholkopf, et al.: New Support Vector Algorithms
    """

    Results = RegressionResults

    def __init__(self, dtype, cost=1.0, nu=0.5, **kwargs):
        Model.__init__(self, dtype, **kwargs)
        self.svm_type = libsvm.NU_SVR
        self.cost = cost
        self.nu = nu
