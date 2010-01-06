
# Matthieu Brucher

# Last Change : 2007-09-29 10:46

from numpy.ctypeslib import ndpointer, load_library
import numpy
import ctypes
import sys

# Load the library
#if sys.platform == 'win32':
  #_cost_function = load_library('cost_function', "\\".join(__file__.split("\\")[:-1]) + "\\release")
#else:
_cost_function = load_library('_cost_function', __file__)

_cost_function.allocate_cost_function.restype = ctypes.c_void_p
_cost_function.allocate_cost_function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_double, ctypes.c_double, ctypes.c_double]

_cost_function.call_cost_function.restype = ctypes.c_double
_cost_function.call_cost_function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_ulong]

ALLOCATOR = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_cost_function.gradient_cost_function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_ulong, ALLOCATOR]

class CostFunction:
  """
  Wrapper with ctypes around the cost function
  """
  def __init__(self, distances, nbCoords = 2, epsilon = 0.0000001, sigma = 1, x1 = 60, **kwargs):
    """
    Creates the correct cost function with the good arguments
    """
    sortedDistances = distances.flatten()
    sortedDistances.sort()
    sortedDistances = sortedDistances[distances.shape[0]:]

    self._nbCoords = nbCoords
    self._epsilon = epsilon
    self._sigma = sigma

    sigma = (sortedDistances[sigma * len(sortedDistances) // 100])
    self._x1 = x1
    x1 = (sortedDistances[x1 * len(sortedDistances) // 100]) ** 2
    del sortedDistances
    self.grad = None
    self.distances = distances.copy()
    self._cf = _cost_function.allocate_cost_function(self.distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), distances.shape[0], distances.shape[1], nbCoords, epsilon, sigma, x1)

  def __del__(self, close_func = _cost_function.delete_cost_function):
    """
    Deletes the cost function
    """
    if not (self._cf == 0):
      close_func(self._cf)
      self._cf = 0

  def __call__(self, parameters):
    """
    Computes the cost of a parameter
    """
    parameters = parameters.copy()
    return _cost_function.call_cost_function(self._cf, parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(parameters))

  def __getinitargs__(self):
    return(self.distances, self._nbCoords, self._epsilon, self._sigma, self._x1)

  def __getstate__(self):
    return ()

  def __setstate__(self, state):
    pass

  def gradient(self, parameters):
    """
    Computes the gradient of the function
    """
    self.grad = None
    parameters = parameters.copy()
    _cost_function.gradient_cost_function(self._cf, parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(parameters), ALLOCATOR(self.allocator))
    return self.grad

  def allocator(self, dim, shape):
    """
    Callback allocator
    """
    self.grad = numpy.zeros(shape[:dim], numpy.float64)
    ptr = self.grad.ctypes.data_as(ctypes.c_void_p).value
    return ptr
