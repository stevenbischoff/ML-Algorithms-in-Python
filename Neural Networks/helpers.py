"""
This module creates some package settings and defines miscellaneous useful
functions

Author: Steve Bischoff
Version: April 16, 2023
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed = 1679685223
np.random.seed(seed)
random.seed(seed)

def sigmoid(W, X):
    """
    Args:
        W: 1D array
        X: 1D or 2D array
    Returns:
        Value or array
    """
    return 1/(1 + np.exp(-np.dot(W, X.T)))

def softmax(X):
    """
    Args:
        X: 1D array
    Returns:
        1D array taking the softmax over X
    """
    return np.exp(X)/sum(np.exp(X))

def mse(Y1, Y2):
    """
    Args:
        Y1: value or 1D array
        Y2: value or 1D array
    Returns:
        value or 1D array
    """
    return np.mean((Y1 - Y2)**2)

def accuracy(Y, Y_hat):
    """
    Args:
        Y: 1D array
        Y_hat: 1D array
    Returns:
        Percent matching
    """
    return (Y == Y_hat).mean()

def get_X_Y(df, feature_list, class_label):
    """
    A helper function to extract X (with bias) and Y arrays from a dataframe
    given labels.
    
    Args:
        df: Pandas dataframe
        feature_list: list-like of column labels
        class_label: column label
    Returns:
        A set of (X, Y) Numpy arrays
    """

    X = np.array(df[feature_list])
    X = np.insert(X,0,1.0,1)
    Y = np.array(df[class_label])

    return X, Y

def shuffled_copies(X, Y):
    """
    Shuffles two arrays simultaneously, keeping correspondence.
    
    Args:
        X: array
        Y: array
    Returns:
        Set of (X, Y) arrays
    """
    p = np.random.permutation(len(X))
    return X[p], Y[p]
