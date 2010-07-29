from .externals.joblib import Parallel, delayed

try:
    from itertools import product
except:
    def product(*args, **kwds):
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)


def iter_grid(param_grid):
    """ Generators on the combination of the various parameter lists given.

        Parameters
        -----------
        kwargs: keyword arguments, lists
            Each keyword argument must be a list of values that should
            be explored.

        Returns
        --------
        params: dictionary
            Dictionnary with the input parameters taking the various
            values succesively.

        Examples
        ---------
        >>> param_grid = {'a':[1, 2], 'b':[True, False]}
        >>> list(iter_grid(param_grid))
        [{'a': 1, 'b': True}, {'a': 1, 'b': False}, {'a': 2, 'b': True}, {'a': 2, 'b': False}]
    """
    if hasattr(param_grid, 'has_key'):
        param_grid = [param_grid]
    for p in param_grid:
        keys = p.keys()
        for v in product(*p.values()):
            params = dict(zip(keys,v))
            yield params

def fit_grid_point(X, y, klass, orignal_params, clf_params, cv,
                                        loss_func, **fit_params):
    """Run fit on one set of parameters
    Returns the score and the instance of the classifier
    """
    params = orignal_params.copy()
    params.update(clf_params)
    n_samples, n_features = X.shape
    clf = klass(**params)
    score = 0
    for train, test in cv:
        clf.fit(X[train], y[train], **fit_params)
        if loss_func is not None:
            y_pred = clf.predict(X[test])
            score -= loss_func(y[test], y_pred)
        else:
            score += clf.score(X[test], y[test])

    return clf, score


class GridSearchCV(object):
    """
    Object to run a grid search on the parameters of a classifier.

    Important memmbers are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation

    Parameters
    ---------
    estimator: object type that implements the "fit" and "predict" methods
        A object of that type is instanciated for each grid point

    param_grid: dict
        a dictionary of parameters that are used the generate the grid

    cross_val_factory : a generator to run crossvalidation

    loss_func : function that takes 2 arguments and compares them in
        order to evaluate the performance of prediciton (small is good)

    n_jobs : int
        number of jobs to run in parallel (default 1)

    Optional Parameters
    -------------------

    Members
    -------

    Examples
    --------
    >>> import numpy as np
    >>> from scikits.learn.cross_val import LeaveOneOut
    >>> from scikits.learn.svm import SVC
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = SVC()
    >>> clf = GridSearchCV(svc, parameters, n_jobs=1)
    >>> print clf.fit(X, y).predict([[-0.8, -1]])
    [ 1.]
    """

    def __init__(self, estimator, param_grid, loss_func=None,
                        fit_params={}, n_jobs=1):
        assert hasattr(estimator, 'fit') and hasattr(estimator, 'predict'), (
            "estimator should a be an estimator implementing 'fit' and "
            "'predict' methods, %s (type %s) was passed" % (clf, type(clf))
            )
        if loss_func is None:
            assert hasattr(estimator, 'score'), ValueError(
                    "If no loss_func is specified, the estimator passed "
                    "should have a 'score' method. The estimator %s "
                    "does not." % estimator
                    )

        self.estimator = estimator
        self.param_grid = param_grid
        self.loss_func = loss_func
        self.n_jobs = n_jobs
        self.fit_params = fit_params


    def fit(self, X, y, cv=None, **kw):
        """Run fit with all sets of parameters
        Returns the best classifier
        """
        if cv is None:
            n_samples = y.size
            from scikits.learn.cross_val import KFold
            cv = KFold(n_samples, 2)

        grid = iter_grid(self.param_grid)
        klass = self.estimator.__class__
        orignal_params = self.estimator._get_params()
        out = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_grid_point)(X, y, klass, orignal_params, clf_params,
                    cv, self.loss_func, **self.fit_params)
                    for clf_params in grid)

        # Out is a list of pairs: estimator, score
        key = lambda pair: pair[1]
        best_estimator = max(out, key=key)[0] # get maximum score

        self.best_estimator = best_estimator
        self.predict = best_estimator.predict

        return self


if __name__ == '__main__':
    from scikits.learn.svm import SVC
    from scikits.learn import datasets

    iris = datasets.load_iris()

    # Add the noisy data to the informative features
    X = iris.data
    y = iris.target

    svc = SVC(kernel='linear')
    clf = GridSearchCV(svc, {'C':[1, 10]}, n_jobs=1)
    print clf.fit(X, y).predict([[-0.8, -1]])
