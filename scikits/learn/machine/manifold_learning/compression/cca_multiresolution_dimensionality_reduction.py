
"""
Multireosolution optimization with a specific cost function
"""

# Matthieu Brucher
# Last Change : 2008-04-01 09:01

import numpy
import numpy.random
import math

from toolbox.optimizers import *
import cost_function

class Modifier(object):
  """
  Recenters the points on each axis
  """
  def __init__(self, nbCoords):
    self.nbCoords = nbCoords

  def __call__(self, parameters):
    points = parameters.reshape((-1, self.nbCoords))
    means = numpy.mean(points, axis = 0)
    return (points - means).ravel()

def optimize_cost_function(distances, function, nbCoords = 2, max_dist = 5, **kwargs):
  """
  Computes a new coordinates system that respects the distances between each point. Each iteration adds a new point in the process
  Parameters :
    - distances is the distances to respect
    - nbCoords is the number of remaining coordinates
    - epsilon is a small number
    - sigma is the percentage of distances below which the weight of the cost function is diminished
    - x1 is the percentage of distances '' which the weight of the cost function is diminished
    - x2 is the percentage of distances that indicates the limit when differences between estimated and real distances are too high and when the cost becomes quadratic
  """
  std = numpy.std(distances)
  x0 = numpy.random.normal(0., 0.1, size = (distances.shape[0], nbCoords))

  indices = numpy.array(range(0, distances.shape[0]))
  numpy.random.shuffle(indices)

  lineSearch = line_search.FibonacciSectionSearch(alpha_step = 1., min_alpha_step = 0.0001)

  fun = function((distances[indices[0:10]])[:, indices[0:10]], nbCoords, 100, **kwargs)
  optimi = optimizer.StandardOptimizerModifying(
    function = fun,
    step = step.GradientStep(),
    criterion = criterion.OrComposition(criterion.AbsoluteParametersCriterion(xtol = 0.001), criterion.IterationCriterion(iterations_max = 100)),
    x0 = x0[indices[0:10]].flatten(),
    line_search = lineSearch, post_modifier = Modifier(nbCoords))
  optimal = optimi.optimize()
  optimal = optimal.reshape(-1, nbCoords)
  x0[indices[0:10]] = optimal

  for i in xrange(11, distances.shape[0]+1):
    j = max(i-100, 0)
    print i
    maxdist = i * (max_dist - 99.9) / distances.shape[0] + 99.9

    fun = function((distances[indices[j:i]])[:, indices[j:i]], nbCoords, max_dist = 100, **kwargs)
    minc = numpy.min(x0[indices[0:i]], axis = 0)
    maxc = numpy.max(x0[indices[0:i]], axis = 0)

    optimi = optimizer.StandardOptimizerModifying(
      function = fun,
      step = step.PartialStep(step.GradientStep(), i - j, i - j - 1),
      criterion = criterion.OrComposition(criterion.AbsoluteParametersCriterion(xtol = 0.01 * numpy.mean(maxc-minc)), criterion.IterationCriterion(iterations_max = 100)),
      x0 = x0[indices[j:i]].flatten(),
      line_search = lineSearch, post_modifier = Modifier(nbCoords))
    optimal = optimi.optimize()
    optimal = optimal.reshape(-1, nbCoords)
    x0[indices[j:i]] = optimal
    fun = function((distances[indices[0:i]])[:, indices[0:i]], nbCoords, max_dist = maxdist, **kwargs)
    optimi = optimizer.StandardOptimizerModifying(
      function = fun,
      step = step.GradientStep(),
      criterion = criterion.OrComposition(criterion.AbsoluteParametersCriterion(xtol = 0.001 * numpy.mean(maxc-minc)), criterion.IterationCriterion(iterations_max = 10)),
      x0 = x0[indices[0:i]].flatten(),
      line_search = line_search.FixedLastStepModifier(line_search = lineSearch), post_modifier = Modifier(nbCoords))
    optimal = optimi.optimize()
    optimal = optimal.reshape(-1, nbCoords)
    x0[indices[0:i]] = optimal

  return x0
