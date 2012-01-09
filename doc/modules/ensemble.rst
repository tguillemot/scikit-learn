.. _ensemble:

================
Ensemble methods
================

.. currentmodule:: sklearn.ensemble

The goal of **ensemble methods** is to combine the predictions of several
models built with a given learning algorithm in order to improve
generalizability / robustness over a single model.

Two families of ensemble methods are usually distinguished:

  - In *averaging methods*, the driving principle is to build several models
    independently and then to average their predictions. On average, the
    combined model is usually better than any of the single model because
    its variance is reduced.

    **Examples:** Bagging methods, :ref:`Forests of randomized trees <forest>`, ...

  - By contrast, in *boosting methods*, models are built sequentially and one
    tries to reduce the bias of the combined model. The motivation is to combine
    several weak models to produce a powerful ensemble.

    **Examples:** AdaBoost, Least Squares Boosting, Gradient Tree Boosting, ...


.. _forest:

Forests of randomized trees
===========================

The :mod:`sklearn.ensemble` module includes two averaging algorithms based on
randomized :ref:`decision trees <tree>`: the RandomForest algorithm and the
Extra-Trees method. Both algorithms are perturb-and-combine techniques
specifically designed for trees::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[0, 0], [1, 1]]
    >>> Y = [0, 1]
    >>> clf = RandomForestClassifier(n_estimators=10)
    >>> clf = clf.fit(X, Y)

In random forests (see :class:`RandomForestClassifier` and
:class:`RandomForestRegressor` classes), each tree in the ensemble is built from
a sample drawn with replacement (i.e., a bootstrap sample) from the training
set. In addition, when splitting a node during the construction of the tree, the
split that is chosen is no longer the best split among all features. Instead,
the split that is picked is the best split among a random subset of the
features. As a result of this randomness, the bias of the forest usually
slightly increases (with respect to the bias of a single non-random tree) but,
due to averaging, its variance also decreases, usually more than compensating
for the increase in bias, hence yielding an overall better model.

In extra-trees (see :class:`ExtraTreesClassifier` and
:class:`ExtraTreesRegressor` classes), randomness goes one step further in the
way splits are computed. As in random forests, a random subset of candidate
features is used, but instead of looking for the most discriminative thresholds,
thresholds are drawn at random for each candidate feature and the best of these
randomly-generated thresholds is picked as the splitting rule. This usually
allows to reduce the variance of the model a bit more, at the expense of a
slightly greater increase in bias::

    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    ...     random_state=0)

    >>> clf = DecisionTreeClassifier(max_depth=None, min_split=1,
    ...     random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.978...

    >>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    ...     min_split=1, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.999...

    >>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    ...     min_split=1, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean() > 0.999
    True

The main parameters to adjust when using these methods is ``n_estimators`` and
``max_features``. The former is the number of trees in the forest. The larger
the better, but also the longer it will take to compute. In addition, note that
results will stop getting significantly better beyond a critical number of
trees. The latter is the size of the random subsets of features to consider when
splitting a node. The lower the greater the reduction of variance, but also the
greater the increase in bias. Empiricial good default values are
``max_features=n_features`` for regression problems, and
``max_features=sqrt(n_features)`` for classification tasks (where ``n_features``
is the number of features in the data). The best results are also usually
reached when setting ``max_depth=None`` in combination with ``min_split=1``
(i.e., when fully developping the trees). Bear in mind though that these values
are usually not optimal. The best parameter values should always be cross-
validated. In addition, note that bootstrap samples are used by default in
random forests (``bootstrap=True``) while the default strategy is to use the
original dataset for building extra-trees (``bootstrap=False``).

Finally, this module also features the parallel construction of the trees and
the parallel computation of the predictions through the ``n_jobs`` parameter. If
``n_jobs=k`` then computations are partitioned into ``k`` jobs, and run on ``k``
cores of the machine. If ``n_jobs=-1`` then all cores available on the machine
are used. Note that because of inter-process communication overhead, the speedup
might not be linear (i.e., using ``k`` jobs will unfortunately not be ``k``
times as fast). Significant speedup can still be achieved though when building a
large number of trees, or when building a single tree requires a fair amount of
time (e.g., on large datasets).

.. topic:: Examples:

 * :ref:`example_ensemble_plot_forest_iris.py`
 * :ref:`example_ensemble_plot_forest_importances_faces.py`

.. topic:: References

 * Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

 * Pierre Geurts, Damien Ernst., and Louis Wehenkel, "Extremely randomized
   trees", Machine Learning, 63(1), 3-42, 2006.


.. _gradient_boosting:

Gradient Tree Boosting
======================

`Gradient Tree Boosting <http://en.wikipedia.org/wiki/Gradient_boosting>`_ or Gradient Boosted Regression Trees (GBRT) is a generalization of boosting to arbitrary differentiable loss functions. GBRT is an accurate and effective off-the-shelf procedure that can be used for both regression and classification problems. Gradient Tree Boosting models are used in a variety of areas including Web search ranking and ecology.

The advantages of GBRT are:

  + Natural handling of data of mixed type (=heterogeneous features)

  + Predictive power

  + Robustness to outliers in input space (via robust loss functions)

The disadvantages of GBRT are:

  + Scalability, due to the sequential nature of boosting it can hardly be parallelized.

GBRT considers additive models of the following form:

  .. math::

    F(x) = \sum_{i=1}^{M} \gamma_i h_i(x)

where :math:`h_i(x)` are the basis functions which are usually called *weak learners* in the context of boosting. Gradient Tree Boosting uses :ref:`decision trees <tree>` of fixed size as weak learners. Decision trees have a number of abilities that make them valuable for boosting, namely the ability to handle data of mixed type and the ability to model complex functions. 

Similar to other boosting algorithms GBRT builds the additive model in a forward stagewise fashion: At each stage the decision tree :math:`h_m(x)` is choosen that maximally reduces the loss function :math:`L` given the current model :math:`F_{m-1}` and its fit :math:`F_{m-1}(x_i)`

  .. math::

    F_m(x) = F_{m-1}(x) + \arg_min_{h}  \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) - h(x))


TODO steepest descent - predict negative gradient of loss function w.r.t the current 

TODO table loss function - negative gradient - line search

TODO Shrinkage

TODO Stochastic Gradient Boosting

TODO Discussion - parameter tuning - other implementations


.. topic:: Examples:

 * :ref:`example_ensemble_plot_gradient_boosting_regression.py`

.. topic:: References

 * J. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", 
   The Annals of Statistics, Vol. 29, No. 5, 2001.

 * J. Friedman, "Stochastic Gradient Boosting", 1999

 * T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning 
   Ed. 2", Springer, 2009.
