# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:48:56 2018

@author: jayadeep
"""

import os
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


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
    
    def train(self, X, y, learning_rate = 0.01, epochs = 10, verbose = True):
        X = X.T
        y = y.T
        m = X.shape[1]
        
        for i in range(epochs):
            dW, db = self.gradients(X, y)
            self.weight = self.weight - learning_rate * dW
            self.bias = self.bias - learning_rate * db
            
            if verbose == True:
                if (i % 10 == 0):
                    A = sigmoid(np.dot(self.weight.T, X) + self.bias)
                    cost = (- 1 / m) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A))) 
                    print(f"cost at epoch: {i} is {cost}")
        

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")            

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

LR = LogisticRegression(number_features = 12288)
X = train_set_x.T
y = train_set_y.T
LR.train(X, y, learning_rate=0.009, epochs=100)
LR.bias

LR = LogisticRegression(number_features = 2)
LR.weight, LR.bias, X, y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]).T, np.array([[1, 0]]).T
LR.gradients(X, y)
LR.train(X, y, learning_rate=0.009, epochs=100)
LR.bias

    
###Miscellenous Functions
def sigmoid(z):
    """Return sigmoid of z"""
    return (1/(1 + np.exp(-1* z)))

def sigmoid_deri(z):
    """Return derivative of sigmoid function at z"""
    return (np.exp(z)/pow((1 + np.exp(z)), 2))