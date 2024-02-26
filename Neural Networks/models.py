"""
Neural network parent classes

This module contains abstract Classifier, Regressor, and Autoencoder classes.
These parent classes contain methods shared by the classes in the classifiers
and regressors modules.

Author: Steve Bischoff
Version: April 16, 2023
"""
from helpers import *

class Classifier():
    """
    A parent class for Linear, Multi-Layer, and Autoencoder classifiers.
    
    Attributes:
        class_list: list-like
        feature_list: list-like
        lr: float, learning rate
        n_classes: int
        W1_size: int
        train_epochs: int
    """

    def __init__(self, class_list, feature_list, lr=0.01):
        
        self.class_list = class_list
        self.feature_list = feature_list
        self.lr = lr

        self.n_classes = len(self.class_list)
        self.W1_size = len(self.feature_list)+1

        self.train_epochs = 0
        

    def get_Y(self, Y_input):
        """
        Transforms a 1-D set of class values into a multi-dimensional one-hot
        encoding.
        
        Args:
            Y_input: Pandas Series of class values.
        Returns:
            A multi-dimensional array that one-hot encodes the input.
        """
        Y = []
        for cls in self.class_list:
            cls_series = pd.Series(index=Y_input.index, dtype='float64')
            cls_series[Y_input == cls] = 1
            cls_series[Y_input != cls] = 0
            Y.append(cls_series)

        return np.array(Y).T
        
    def train(self, X, Y, max_epochs=100, method='incremental'):
        """
        Trains the model given X and Y data for a specified number of epochs
        using a specified training method.
        
        Args:
            X: Numpy array
            Y: Numpy array
            max_epochs: int
            method: 'incremental' or 'batch'
        """
        for i in range(max_epochs): # upper limit on training epochs
            if method == 'incremental':
                self.incremental_epoch(X, Y)
                X, Y = shuffled_copies(X, Y)
            elif method == 'batch':
                self.batch_epoch(X, Y)
        
########################################
class Regressor():
    """
    A parent class for Linear, Multi-Layer, and Autoencoder regressors.
    
    Attributes:
        y_label: column label
        feature_list: list-like
        lr: float, learning rate
        W1_size: int
        train_epochs: int
    """

    def __init__(self, y_label, feature_list, lr):

        self.y_label = y_label
        self.feature_list = feature_list
        self.lr = lr

        self.W1_size = len(feature_list)+1

        self.train_epochs = 0

    def train(self, X, Y, max_epochs=100, method='incremental'):
        """
        Trains the model given X and Y data for a specified number of epochs
        using a specified training method.
        
        Args:
            X: Numpy array
            Y: Numpy array
            max_epochs: int
            method: 'incremental' or 'batch'
        """
        for i in range(max_epochs): # upper limit on training epochs
            if method == 'incremental':
                self.incremental_epoch(X, Y)
                X, Y = shuffled_copies(X, Y)
            elif method == 'batch':
                self.batch_epoch(X, Y)

########################################               
class Autoencoder():
    """
    A parent class for Autoencoder classifiers and regressors.
    """

    def encode_decode(self, X):
        """
        Args:
            X: Numpy array
        Returns:
            Predicted array, hidden layer array
        """
        Z = sigmoid(self.W1, X)
        X_hat = np.dot(self.D, Z)

        return X_hat, Z 

    def incremental_encoder_epoch(self, X):
        """
        Performs one epoch of incremental training on the autoencoder.
        
        Args:
            X: Numpy array
        """

        # iterate through training data
        for i in range(len(X)):
            
            Xi = X[i]
            X_hat, Z = self.encode_decode(Xi)
            
            e0 = Xi[1:] - X_hat
            D_update = np.dot(e0.reshape(self.D.shape[0], 1),
                              Z.reshape(1, self.D.shape[1]))

            e1 = np.dot(e0, self.D)
            W1_update = np.dot((e1*Z*(1-Z)).reshape(self.W2_size-1, 1),
                               Xi.reshape(1, self.W1_size))

            self.D += self.lr*D_update
            self.W1 += self.lr*W1_update

        self.encode_train_epochs += 1

    def batch_encoder_epoch(self, X):
        """
        Performs one epoch of batch training on the autoencoder.
        
        Args:
            X: Numpy array
        """

        X_hat, Z = self.encode_decode(X)

        e2 = X.T[1:] - X_hat
        D_update = np.dot(e2*X_hat*(1-X_hat), Z.T)

        e1 = np.dot(self.D.T, e2)
        W1_update = np.dot(e1*Z*(1-Z), X)

        self.D += self.lr*D_update
        self.W1 += self.lr*W1_update

        self.encode_train_epochs += 1

    def train_encoder(self, X, max_epochs=100, method='incremental'):
        """
        Trains the autoencoder given X data for a specified number of epochs
        using a specified training method.
        
        Args:
            X: Numpy array
            max_epochs: int
            method: 'incremental' or 'batch'
        """

        if method == 'batch':
            for i in range(max_epochs):
                self.batch_encoder_epoch(X)
        elif method == 'incremental':
            for i in range(max_epochs):                
                self.incremental_encoder_epoch(X)
