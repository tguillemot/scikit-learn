"""
===============================
Gradient Boosting OOB estimates
===============================

Out-of-bag estimates can be used to estimate the "optimal" number of
boosting iterations. The OOB estimator is a pessimistic estimator of the true
test loss, but remains a fairly good approximation for a small number of trees.

The figure shows the cumulative sum of the negative OOB improvements
as a function of the boosting iteration. As you can see, it tracks the test
loss for the first hundred iterations but then diverges in a
pessimistic way.
The figure also shows the performance of 3-fold cross validation which
usually gives a better estimate but is computationally more demanding.
"""
print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pylab as pl

from sklearn import ensemble
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

###############################################################################
# Generate data (adapted from G. Ridgeway's gbm example)

n = 1000
rs = np.random.RandomState(13)
x1 = rs.uniform(size=n)
x2 = rs.uniform(size=n)
x3 = rs.randint(0, 4, size=n)

p = 1 / (1.0 + np.exp(-(np.sin(3 * x1) - 4 * x2 + x3)))
y = rs.binomial(1, p, size=n)

X = np.c_[x1, x2, x3]

X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=9)

###############################################################################
# Fit regression model
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
clf = ensemble.GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("ACC: %.4f" % acc)

n_estimators = params['n_estimators']
x = np.arange(n_estimators) + 1


def heldout_score(clf, X_test, y_test):
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score


def cv_estimate(n_folds=3):
    cv = KFold(n=X_train.shape[0], n_folds=n_folds)
    cv_clf = ensemble.GradientBoostingClassifier(**params)
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv:
        cv_clf.fit(X_train[train], y_train[train])
        val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
    val_scores /= n_folds
    return val_scores


test_score = heldout_score(clf, X_test, y_test)
cv_score = cv_estimate(3)

cumsum = -np.cumsum(clf.oob_improvement_)
oob_best_iter = x[np.argmin(cumsum)]
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]

oob_color = map(lambda x: x / 256.0, (190, 174, 212))
test_color = map(lambda x: x / 256.0, (127, 201, 127))
cv_color = map(lambda x: x / 256.0, (253, 192, 134))

pl.plot(x, cumsum, label='OOB loss', color=oob_color)
pl.plot(x, test_score, label='Test loss', color=test_color)
pl.plot(x, cv_score, label='CV loss', color=cv_color)
pl.axvline(x=oob_best_iter, color=oob_color)
pl.axvline(x=test_best_iter, color=test_color)
pl.axvline(x=cv_best_iter, color=cv_color)

xticks = pl.xticks()
xticks_pos = np.array(xticks[0].tolist() +
                      [oob_best_iter, cv_best_iter, test_best_iter])
xticks_label = np.array(map(lambda t: int(t), xticks[0]) +
                        ['OOB', 'CV', 'Test'])
ind = np.argsort(xticks_pos)
xticks_pos = xticks_pos[ind]
xticks_label = xticks_label[ind]
pl.xticks(xticks_pos, xticks_label)

pl.legend(loc='upper right')
pl.ylabel('normalized loss')
pl.xlabel('number of iterations')

pl.show()
