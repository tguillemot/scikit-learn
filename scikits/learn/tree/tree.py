# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution
"""
================
Tree Classifier
================

A decision tree classifier

Implements Classification and Regression Trees (Breiman et al. 1984)

"""

from __future__ import division
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ._tree import eval_gini, eval_entropy, eval_miss, eval_mse
import random

__all__ = [
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    ]

lookup_c = \
      {'gini': eval_gini,
       'entropy': eval_entropy,
       'miss': eval_miss,
       }
lookup_r = \
      {'mse': eval_mse,
      }


class Leaf(object):
    '''
        v : target value
            Classification: array-like, shape = [n_features]
                Histogram of target values
            Regression:  real number
                Mean for the region
    '''

    def __init__(self, v):
        self.v = v

    def __str__(self):
        return 'Leaf(%s)' % (self.v)


class Node(object):
    '''
        feat : target value
            Classification: array-like, shape = [n_features]
                Histogram of target values
            Regression:  real number
                Mean for the region
    '''
    
    def __init__(self, dimension, value, error, left, right):
        self.dimension = dimension
        self.value = value
        self.error = error
        self.left = left
        self.right = right
        
    def __str__(self):
        return 'x[%s] < %s, \\n error = %s' % \
            (self.dimension, self.value, self.error)

def _find_best_split(features, labels, criterion):
    n_samples, n_features = features.shape
    K = np.abs(labels.max()) + 1
    pm = np.zeros((K,), dtype=np.float64)
        
    best = None
    split_error = criterion(labels, pm)
    for i in xrange(n_features):
        features_at_i = features[:, i]
        domain_i = sorted(set(features_at_i))
        for d1, d2 in zip(domain_i[:-1], domain_i[1:]):
            t = (d1 + d2) / 2.
            cur_split = (features_at_i < t)
            left_labels = labels[cur_split]
            right_labels = labels[~cur_split]
            e1 = len(left_labels) / n_samples * \
                criterion(left_labels, pm)              
            e2 = len(right_labels) / n_samples * \
                criterion(right_labels, pm)
            error = e1 + e2
            if error < split_error:
                split_error = error
                best = i, t, error
    return best

def _build_tree(is_classification, features, labels, criterion, \
               max_depth, min_split, F, K):
    
    if len(labels) != len(features):
        raise ValueError("Number of labels does not match " + \
                          "number of features\n" + 
                         "num labels is %s and num features is %s " % 
                         (len(labels), len(features)))
        
    sample_dims = np.array(xrange(features.shape[1]))    
    if F is not None:
        if F <= 0:
            raise ValueError("F must be > 0.\n" + 
                             "Did you mean to use None to signal no F?")
        if F > features.shape[1]:
            raise ValueError("F must be < num dimensions of features.\n" + 
                             "F is %s, n_dims = %s " % (F, features.shape[1]))            
        
        sample_dims = np.sort(np.array( \
                        random.sample(xrange(features.shape[1]), F)))
        features = features[:, sample_dims]

    if min_split <= 0:
        raise ValueError("min_split must be greater than zero.\n" + 
                         "min_split is %s." % min_split)
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero.\n" + 
                         "max_depth is %s." % max_depth)

    def recursive_partition(features, labels, depth):
        is_split_valid = True
        
        if depth >= max_depth:
            is_split_valid = False
        
        S = _find_best_split(features, labels, criterion)
        if S is not None:
            dim, thresh, error = S
            split = features[:, dim] < thresh
            if len(features[split]) < min_split or \
                len(features[~split]) < min_split:
                is_split_valid = False            
        else:
            is_split_valid = False
        
        if is_split_valid == False:
            if is_classification:
                a = np.zeros((K,))
                t = labels.max() + 1
                a[:t] = np.bincount(labels)
                return Leaf(a) 
            else:
                return Leaf(np.mean(labels))            
            
        return Node(dimension=sample_dims[dim],
                    value=thresh,
                    error=error,
                    left=recursive_partition(features[split], \
                                              labels[split], depth + 1),
                    right=recursive_partition(features[~split], \
                                              labels[~split], depth + 1))
        
    return recursive_partition(features, labels, 0)


def _apply_tree(tree, features):
    '''
    conf = apply_tree(tree, features)

    Applies the decision tree to a set of features.
    '''
    if type(tree) is Leaf:
        return tree.v
    if features[tree.dimension] < tree.value:
        return _apply_tree(tree.left, features)
    return _apply_tree(tree.right, features)


def _graphviz(tree):
    '''Print decision tree in .dot format
    '''
    if type(tree) is Leaf:
        return ""
    left = "\"" + str(tree) + "\" -> \"" + str(tree.left) + "\";\n"
    right = "\"" + str(tree) + "\" -> \"" + str(tree.right) + "\";\n"

    return left + _graphviz(tree.left) + right + _graphviz(tree.right)


class BaseDecisionTree(BaseEstimator):
    '''
    Should not be used directly, use derived classes instead
    '''

    _dtree_types = ['classification', 'regression']    
    
    def __init__(self, K, impl, criterion, max_depth, min_split, F, seed):
        
        if not impl in self._dtree_types:
            raise ValueError("impl should be one of %s, %s was given" % (
                self._dtree_types, impl))                    
            
        self.type = impl
                         
        self.K = K
                
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_split = min_split        
        self.F = F
        
        if seed is not None:
            random.seed(seed)
        
        self.n_features = None
        self.tree = None

    def export_to_graphviz(self):
        with open("tree.dot", 'w') as f:
            f.write("digraph Tree {\n")
            f.write(_graphviz(self.tree))
            f.write("\n}\n")        

    def fit(self, X, y):
        """
        Fit the tree model according to the given training data and
        parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values (integers in classification, real numbers in
            regression)
            For classification, labels must correspond to classes 0,1,...,K-1

        Returns
        -------
        self : object
            Returns self.
        """               
        X = np.asanyarray(X, dtype=np.float64, order='C')       
        _, self.n_features = X.shape    
        
        if self.type == 'classification':
            y = np.asanyarray(y, dtype=np.int, order='C') 
            if self.K is None:
                self.K = y.max() + 1
            if y.max() >= self.K or y.min() < 0:
                raise ValueError("Labels must be in the range [0 to %s)",
                                 self.K)  
            self.tree = _build_tree(True, X, y, lookup_c[self.criterion], \
                                    self.max_depth, self.min_split, self.F, \
                                    self.K)
        else: #regression
            y = np.asanyarray(y, dtype=np.float64, order='C')               
            self.tree = _build_tree(False, X, y, lookup_r[self.criterion], \
                                    self.max_depth, self.min_split, self.F, \
                                    None)                  
        return self  
        
    def predict(self, X):
        """
        This function does classification or regression on an array of
        test vectors X.

        For a classification model, the predicted class for each
        sample in X is returned.  For a regression model, the function
        value of X calculated is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
        """

        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
            
        if self.tree is None:
            raise Exception('Tree not initialized. Perform a fit first')
        
        if self.n_features != n_features:
            raise ValueError("Number of features of the model must match the input.\n" + 
                             "Model n_features is %s and input n_features is %s " % \
                             (self.n_features, n_features))        
        
        C = np.zeros(n_samples, dtype=int)
        for idx, sample in enumerate(X):
            if self.type == 'classification':
                C[idx] = np.argmax(_apply_tree(self.tree, sample))
            else:
                C[idx] = _apply_tree(self.tree, sample)            
        
        return C 


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """Classify a multi-labeled dataset with a decision tree.

    Parameters
    ----------
    K : integer, mandatory 
        number of classes    

    criterion : string
        function to measure goodness of split

    max_depth : integer
        maximum depth of the tree  
              
    min_split : integer
        minimum size to split on
        
    F : integer, optional
        if given, then, choose F features

    seed : integer or array_like, optional
        seed the random number generator

    
    #Example
    #-------
    #>>> import numpy as np
    #>>> from scikits.learn.datasets import load_iris
    #>>> from scikits.learn.cross_val import StratifiedKFold
    #>>> from scikits.learn import tree
    #>>> data = load_iris()
    #>>> skf = StratifiedKFold(data.target, 10)
    #>>> for train_index, test_index in skf:
    #...     tree = tree.DecisionTreeClassifier(K=3)
    #...     tree.fit(data.data[train_index], data.target[train_index])
    #...     #print np.mean(tree.predict(data.data[test_index]) == data.target[test_index])
    #... 

    """

    def __init__(self, K=None, criterion='gini', max_depth=10, \
                  min_split=1, F=None, seed=None):
        BaseDecisionTree.__init__(self, K, 'classification', criterion, \
                                  max_depth, min_split, F, seed)
    
    def predict_proba(self, X):
        """
        This function does classification on a test vector X
        given a model with probability information.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        P : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered by arithmetical
            order.

        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
            
        if self.tree is None:
            raise Exception('Tree not initialized. Perform a fit first')
        
        if self.n_features != n_features:
            raise ValueError("Number of features of the model must match the input.\n" + 
                             "Model n_features is %s and input n_features is %s " % \
                             (self.n_features, n_features))        
        
        P = np.zeros((n_samples, self.K))
        for idx, sample in enumerate(X):
            P[idx,:] = _apply_tree(self.tree, sample)
            P[idx,:] /= np.sum(P[idx,:]) 
        return P    

    def predict_log_proba(self, X):
        """
        This function does classification on a test vector X

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        P : array-like, shape = [n_samples, n_classes]
            Returns the log-probabilities of the sample for each class in
            the model, where classes are ordered by arithmetical
            order.

        """
        
        return np.log(self.predict_proba(X))


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """Perform regression on dataset with a decision tree.
    
    Parameters
    ----------

    criterion : string
        function to measure goodness of split

    max_depth : integer
        maximum depth of the tree  
              
    min_split : integer
        minimum size to split on
        
    F : integer, optional
        if given, then, choose F features

    seed : integer or array_like, optional
        seed the random number generator    
    
    #Example
    #-------
    #>>> import numpy as np
    #>>> from scikits.learn.datasets import load_boston
    #>>> from scikits.learn.cross_val import KFold
    #>>> from scikits.learn import tree
    #>>> data = load_boston()
    #>>> np.random.seed([1]) 
    #>>> perm = np.random.permutation(data.target.size / 8)
    #>>> data.data = data.data[perm]
    #>>> data.target = data.target[perm]
    #>>> kf = KFold(len(data.target), 2)
    #>>> for train_index, test_index in kf:
    #...     tree = tree.DecisionTreeRegressor()
    #...     tree.fit(data.data[train_index], data.target[train_index])
    #...     #print np.mean(np.power(tree.predict(data.data[test_index]) - data.target[test_index], 2))
    #... 

    """

    def __init__(self, criterion='mse', max_depth=10, \
                  min_split=1, F=None, seed=None):       
        BaseDecisionTree.__init__(self, None, 'regression', criterion, \
                                  max_depth, min_split, F, seed)
