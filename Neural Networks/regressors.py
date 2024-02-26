"""
Neural network regressors

This module contains classes for Linear, Multi-Layer Perceptron, and Autoencoder
regressors. These classes are instantiated and their performance evaluated in
the regression module.

Author: Steve Bischoff
Version: April 16, 2023
"""
from models import *

class LinearRegressor(Regressor):
    """
    Implements training and prediction functionality for simple linear models.
    """

    def __init__(self, y_label, feature_list, lr=0.001, loc=0.0, scale=0.01):

        super().__init__(y_label, feature_list, lr)
        self.set_weights(loc, scale)


    def set_weights(self, loc=0.0, scale=0.01):
        """
        Randomly initializes the ouput layer weights.
        Args:
            loc: float, weigh center
            scale: float, weight standard deviation
        """
        self.W = np.random.normal(loc=loc, scale=scale,
                                  size=(1, self.W1_size))

    def feedforward(self, X):
        """
        Args:
            X: 2D array
        Returns:
            1D array
        """
        return np.dot(self.W, X.T)[0]

    def predict(self, X):
        """
        Args:
            X: 1D or 2D array
        Returns:
            value or 1D array
        """
        return self.feedforward(X)

    def incremental_epoch(self, X, Y):
        """
        Performs one epoch of incremental training.
        
        Args:
            X: Numpy array
            Y: Numpy array
        """
        
        for i in range(len(X)):
            Xi = X[i]
            yi = Y[i]
            y_pred = self.feedforward(Xi.T)
            W_update = np.dot((yi - y_pred), Xi)
            self.W += self.lr*W_update

        self.train_epochs += 1
            

    def batch_epoch(self, X, Y):
        """
        Performs one epoch of batch training.
        
        Args:
            X: Numpy array
            Y: Numpy array
        """

        Y_pred = self.feedforward(X)
        W_update = np.dot((Y - Y_pred), X)/len(X)
        self.W += self.lr*W_update

        self.train_epochs += 1

################################################################################
class MLPRegressor(Regressor): # 2 hidden layers
    """
    Implements training and prediction functionality for multi-layer perceptron
    regressors.
    """

    def __init__(self, y_label, feature_list, lr=0.001, W2_size=None,
                 V_size=None, loc=0.0, scale=0.01):

        super().__init__(y_label, feature_list, lr)
        
        if W2_size == None:
            self.W2_size = self.W1_size*2//3 + 1
        else:
            self.W2_size = W2_size
        if V_size == None:
            self.V_size = self.W1_size//3 + 1
        else:
            self.V_size = V_size

        self.set_weights(loc=loc, scale=scale)


    def set_weights(self, loc=0.0, scale=0.01):
        """
        Randomly initializes the hidden layer weights (W1 and W2) and the
        ouput layer weights (V).
        
        Args:
            loc: float, weigh center
            scale: float, weight standard deviation
        """
        self.W1 = np.random.normal(loc=loc, scale=scale,
                                  size=(self.W2_size-1, self.W1_size))
        self.W2 = np.random.normal(loc=loc, scale=scale,
                                  size=(self.V_size-1, self.W2_size))
        self.V = np.random.normal(loc=loc,scale=scale,size=(1, self.V_size))


    def feedforward(self, X):
        """
        Args:
            X: 2D array
        Returns:
            1D array
        """
        
        Z1 = sigmoid(self.W1, X)
        Z1 = np.insert(Z1,0,1.0,0)

        Z2 = sigmoid(self.W2, Z1.T)
        Z2 = np.insert(Z2,0,1.0,0)

        Y = np.dot(self.V, Z2)

        return Y, Z2, Z1

    def predict(self, X):
        """
        Args:
            X: 1D or 2D array
        Returns:
            value or 1D array
        """
        return self.feedforward(X)[0][0]

    def incremental_epoch(self, X, Y):
        """
        Performs one epoch of incremental training.
        
        Args:
            X: Numpy array
            Y: Numpy array
        """
        for i in range(len(X)):
            Xi = X[i]
            yi = Y[i]
            
            y_pred, Z2, Z1 = self.feedforward(Xi)
            e0 = yi - y_pred
            V_update = (e0*Z2).reshape(1,self.V_size)
            
            e2 = e0*self.V            
            W2_update = np.dot((e2*Z2*(1-Z2))[:,1:].T, Z1.reshape(1,self.W2_size))

            e1 = np.dot(e2[:,1:],self.W2)
            W1_update = np.dot((e1*Z1*(1-Z1))[:,1:].T, Xi.reshape(1,self.W1_size))

            self.V += self.lr*V_update
            self.W2 += self.lr*W2_update
            self.W1 += self.lr*W1_update

        self.train_epochs += 1
        

    def batch_epoch(self, X, Y):
        """
        Performs one epoch of batch training.
        
        Args:
            X: Numpy array
            Y: Numpy array
        """      
        Y_pred, Z2, Z1 = self.feedforward(X)

        e0 = Y - Y_pred
        V_update = np.sum(e0*Z2,axis=1)/len(X) 

        e2 = np.dot(self.V.T, e0) 
        W2_update = np.dot(e2*Z2*(1-Z2), Z1.T)[1:]/len(X)

        e1 = np.dot(self.W2.T, e2[1:]) 
        W1_update = np.dot(e1*Z1*(1-Z1), X)[1:]/len(X)
        
        self.V += self.lr*V_update    
        self.W2 += self.lr*W2_update
        self.W1 += self.lr*W1_update

        self.train_epochs += 1


class AERegressor(Autoencoder, MLPRegressor):
    """
    Implements training functionality for Autoencoder regression
    """

    def __init__(self, y_label, feature_list, lr=0.001, W2_size=None, V_size=None, loc=0.0, scale=0.01):

        self.y_label = y_label
        self.feature_list = feature_list
        self.lr = lr
        self.encode_train_epochs = 0
        self.train_epochs = 0
        self.test_scores = []

        self.W1_size = len(feature_list)+1 # W1_size+1 if bias term

        if W2_size == None:
            self.W2_size = self.W1_size*2//3 + 1
        else:
            self.W2_size = W2_size
        if V_size == None:
            self.V_size = self.W1_size//3 + 1
        else:
            self.V_size = V_size

        self.set_weights(loc, scale)        

    def set_weights(self, loc=0.0, scale=0.01):
        """
        Randomly initializes the hidden layer weights (W1 and W2), the
        autoencoder output weights (D), and the regressor ouput layer weights
        (V).
        
        Args:
            loc: float, weigh center
            scale: float, weight standard deviation
        """
        self.W1 = np.random.normal(loc=loc, scale=scale,
                                  size=(self.W2_size-1, self.W1_size)) 
        self.D = np.random.normal(loc=loc, scale=scale,
                                  size=(self.W1_size-1, self.W2_size-1))
        self.W2 = np.random.normal(loc=loc, scale=scale,
                                  size=(self.V_size-1, self.W2_size))
        self.V = np.random.normal(loc=loc,scale=scale,size=(1, self.V_size))

    def train(self, X, Y, max_epochs=100, method='incremental', X_test=[], Y_test=[], verbose=False):
        """
        Overrides MLPClassifier.train(), training the autoencoder first.
        """
        self.train_encoder(X, max_epochs=max_epochs, method=method)
        super().train(X, Y, max_epochs=max_epochs, method=method)
