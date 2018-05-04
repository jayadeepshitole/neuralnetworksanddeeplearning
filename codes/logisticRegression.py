# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:48:56 2018

@author: jayadeep
"""

import numpy as np
from generalFunctions import sigmoid
import matplotlib.pyplot as plt

class LogisticRegression(object):
    def __init__(self, number_features):
        self.number_features = number_features
        self.weight = np.zeros((number_features, 1))
        self.bias = 0

    def gradients(self, X, y):
        """Return the gradient of weight and bias, X is numpy.ndarray of shape (num_features, num_training_exaples), y is of shape 
        (1, num_training_examples)"""
    
        m = X.shape[1]
        # FORWARD PROPAGATION
        Z = np.dot(self.weight.T, X) + self.bias
        A = sigmoid(Z)
        
        dZ = A - y
        dW = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)
        return(dW, db)
    
    def train(self, X, y, learning_rate = 0.01, epochs = 100, print_cost = True, plot_cost = True):
        m = X.shape[1]
        
        epoch_costs = []
        for i in range(epochs + 1):
            dW, db = self.gradients(X, y)
            self.weight = self.weight - learning_rate * dW
            self.bias = self.bias - learning_rate * db
            
            if print_cost == True:
                if (i % 50 == 0):
                    A = sigmoid(np.dot(self.weight.T, X) + self.bias)
                    cost = (- 1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A))) 
                    epoch_costs.append([i, cost])
                    print(f"cost at epoch: {i} is {np.round(cost, 4)}")
        costs = np.array(epoch_costs)
        if plot_cost:
            plt.plot(costs[:, 0], costs[:, 1])
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.title("Logistic curve fitting loss function")
            plt.show()
        
    def predict(self, X):
        """
        compute predicted value of class probability using fitted looistic regression model
        
        Argument:
        X -- a numpy ndarray of shape (num_features, n) where n is the number of samples j
        
        Return:
        p -- a numpy ndarray of shape (1, n), prediction of prob(class = 1)
        """
        
        p = sigmoid(np.matmul(self.weight.T , X)  + self.bias)
        return p
    
        
        
        
        
        
        
        
        
        