# -*- coding: utf-8 -*-
# Copyright (C) 2010-2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution
"""
================
Random Forest
================

    A Random Forest classifier 
    
    Implements Random Forests (Breiman 2001)

"""

from __future__ import division
import numpy as np
import copy
import random
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..externals.joblib import Parallel, delayed

__all__ = [
    'RandomForestClassifier',
    'RandomForestRegressor',    
    ]
          
def _train_tree(X, y, i, r, t):   
    X = np.asanyarray(X, dtype=np.float64, order='C')       
    n_samples, n_features = X.shape    

    if r <= 0 or r > 1 :
        raise ValueError("r must be in 0 <= r < 1.\n" +
                         "r is %s" % r)
    n = int(r*n_samples)
              
    tree = copy.copy(t)
        
    sample_rows = np.sort(np.array(random.sample(xrange(n_samples), n)))
    X = X[sample_rows, :]
        
    if type(t) == DecisionTreeClassifier:
        y = np.asanyarray(y, dtype=np.int, order='C')  
    elif type(t) == DecisionTreeRegressor:                 
        y = np.asanyarray(y, dtype=np.float64, order='C')  
    else:
        raise Exception("base tree type not valid")
    
    y_in = y[sample_rows] 
    y_out = y[~sample_rows]
            
    tree.fit(X, y_in)
    
    """
    @TODO Compute the out-of-bag error using y_out
    """
    
    return tree            
            
            
class BaseRandomForest(BaseEstimator):
    '''
    Should not be used directly, use derived classes instead
    '''
    
    def __init__(self, seed, base_tree,  n_trees, r, n_jobs):
        if seed is not None:
            random.seed(seed)        
        self.base_tree = base_tree
        self.n_trees = n_trees
        self.r = r  

        if n_jobs <= 0:
            raise ValueError("n_jobs must be >= 0")           
        self.n_jobs = n_jobs
        self.forest = None

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

        if self.base_tree.K is None:
            y = np.asanyarray(y, dtype=np.int, order='C') 
            self.base_tree.K = y.max() + 1            

        forest = []
        if self.n_jobs > 1:
            forest = Parallel(self.n_jobs) \
                             (delayed(_train_tree)(X,y,i,self.r,self.base_tree) \
                              for i in range(self.n_trees))   
        else:
            forest = [self._train_tree(X,y,i,self.r,self.base_tree) for i in range(self.n_trees)]


            
        self.forest = forest
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
            
        if self.forest is None:
            raise Exception('Random Forest not initialized. Perform a fit first')       
        
        C = np.zeros(n_samples, dtype=int)
        for idx, sample in enumerate(X): 
            if type(self.base_tree) == DecisionTreeClassifier:
                preds = [int(t.predict(sample)) for t in self.forest]
                C[idx] = np.argmax(np.bincount(preds))
            elif type(self.base_tree) == DecisionTreeRegressor:
                preds = [float(t.predict(sample)) for t in self.forest]
                C[idx] = np.mean(preds)           
            else:
                raise Exception("base tree type not valid")

        return C 
    

class RandomForestClassifier(BaseRandomForest, ClassifierMixin):
    """Classify a multi-labeled dataset with a random forest.

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

    n_trees : integer, optional
        the number of trees in the forest
        
    r : float, optional
        the ratio of training samples used per tree 0 < r <= r
    
    n_jobs : integer, optional
        the number of processes to use for parallel computation
    
    #Example
    #-------
    #>>> import numpy as np
    #>>> from scikits.learn.datasets import load_iris
    #>>> from scikits.learn.cross_val import StratifiedKFold
    #>>> from scikits.learn import ensemble
    #>>> data = load_iris()
    #>>> skf = StratifiedKFold(data.target, 10)
    #>>> for train_index, test_index in skf:
    #...     rf = ensemble.RandomForestClassifier(K=3)
    #...     rf.fit(data.data[train_index], data.target[train_index])
    #...     #print np.mean(tree.predict(data.data[test_index]) == data.target[test_index])
    #... 

    
    """     
    def __init__(self, K=None, criterion='gini', max_depth=10,\
                  min_split=1, F=None, seed=None, n_trees=10, r=0.7, \
                  n_jobs=2):
        base_tree = DecisionTreeClassifier( K=K, criterion=criterion, \
            max_depth=max_depth, min_split=min_split, F=F, seed=seed)
        BaseRandomForest.__init__(self, seed, base_tree, n_trees, r, n_jobs)
           
    def predict_proba(self, X):
        """
        This function does classification or regression on an array of
        test vectors X.

        For a classification model, the probability of each class for each
        sample in X is returned.  

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, K]
        """
        
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape    
            
        if self.forest is None:
            raise Exception('Random Forest not initialized. Perform a fit first')       
        
        P = np.zeros((n_samples, self.base_tree.K), dtype=float)
        for t in self.forest:
            P += t.predict_proba(X)
        P /= self.n_trees

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
    
class RandomForestRegressor(BaseRandomForest, RegressorMixin):
    """Perform regression on dataset with a random forest.

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

    n_trees : integer, optional
        the number of trees in the forest
        
    r : float, optional
        the ratio of training samples used per tree 0 < r <= r
    
    n_jobs : integer, optional
        the number of processes to use for parallel computation
    
    #Example
    #-------
    #>>> import numpy as np
    #>>> from scikits.learn.datasets import load_boston
    #>>> from scikits.learn.cross_val import KFold
    #>>> from scikits.learn import ensemble
    #>>> data = load_boston()
    #>>> np.random.seed([1]) 
    #>>> perm = np.random.permutation(data.target.size / 8)
    #>>> data.data = data.data[perm]
    #>>> data.target = data.target[perm]    
    #>>> kf = KFold(len(data.target), 2)
    #>>> for train_index, test_index in kf:
    #...     rf = ensemble.RandomForestRegressor(n_jobs=2)
    #...     rf.fit(data.data[train_index], data.target[train_index])
    #...     #print np.mean(np.power(tree.predict(data.data[test_index]) - data.target[test_index], 2)) 
    #... 


   
    """ 
     
    def __init__(self, criterion='mse', max_depth=10,\
                  min_split=1, F=None, seed=None, n_trees=10, r=0.7, \
                   n_jobs=2):       
        base_tree = DecisionTreeRegressor(criterion=criterion, \
            max_depth=max_depth, min_split=min_split, F=F, seed=seed)
        BaseRandomForest.__init__(self, seed, base_tree, n_trees, r, n_jobs)
              