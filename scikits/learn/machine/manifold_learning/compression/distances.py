
# Matthieu Brucher
# Last Change : 2008-04-07 13:35

"""
Allows to compute the nearest neighboors
"""

import numpy

def parzen(samples, neighboors, **kwargs):
  """
  Creates a list of the nearest neighboors in a Parzen window
  """
  l = []

  d = dist2hd(samples, samples)

  for dist in d:
    wi = numpy.where(dist < neighboors)[0]
    l.append(wi)

  return l

def kneigh(samples, neighboors, **kwargs):
  """
  Creates a list of the nearest neighboors in a K-neighboorhood
  """
  l = []

  d = dist2hd(samples, samples)

  for dist in d:
    indices = numpy.argsort(dist)
    l.append(indices[:neighboors])

  return l

def dist2hd(x,y):
   """
   Generate a 'coordinate' of the solution at a time
   """
   d = numpy.zeros((x.shape[0],y.shape[0]),dtype=x.dtype)
   for i in xrange(x.shape[1]):
       diff2 = x[:,i,None] - y[:,i]
       diff2 **= 2
       d += diff2
   numpy.sqrt(d,d)
   return d

def NumpyFloyd(distances):
  """
  Implementation with Numpy vector operations
  """
  for indice1 in xrange(len(distances)):
    for indice2 in xrange(len(distances)):
      distances[indice2, :] = numpy.minimum(distances[indice2, :], distances[indice2, indice1] + distances[indice1, :])
