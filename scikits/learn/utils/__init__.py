
import numpy as np
import scipy.sparse as sp

def safe_asanyarray(X, dtype=None, order=None):
    if sp.issparse(X):
        return X.__class__(X, dtype)
    else:
        return np.asanyarray(X, dtype, order)

