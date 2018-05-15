# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:26:30 2018

@author: jayadeep

A module to implement stochastic gradient descent algorithm for feedforward neural network. 
Gradients are calculated using backpropagation. 
Reference: http://neuralnetworksanddeeplearning.com/chap1.html
"""

import os
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
import numpy as np
from generalFunctions import sigmoid, sigmoid_deri

class Network(object):

    def __init__(self, sizes):
        """
        Argument:
        sizes -- list of number of units in each layer of the neural network
        
        Return:
        num_layers -- total number of layer in the neural network
        sizes -- same as the argument
        weights -- numpy.ndarray of weights for each layer 
        bias -- numpy.ndarray of biases for each layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
    def calculate_gradients(self, X, Y):
        """
        This function calculates the cost function and the gradient of the cost function with 
        respect to weights and biases
        
        Argument:
        X -- data of shape (number of features, number of examples)
        Y -- true "label" of shape (number of categories, number of examples)
        
        Return:
        grads -- a dictionary of dw, db where dw is the gradient of cost function wrt weights
        of shape same as w and db is the gradient of cost function wrt biases of shape same as b
        """
        Z1 = np.matmul(self.weights[0], X) + self.biases[0]       #(30, m)
        A1 = sigmoid(Z1)                                          #(30, m)
        Z2 = np.matmul(self.weights[1], A1) + self.biases[1]       #(10, m)
        A2 = sigmoid(Z2)                                          #(10, m)
        # number of examples
        m = X.shape[1]
        dZ2 = A2 - Y                                              #(784, m)
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)                      #(10, 30)
        db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)    #(10, 1)
        dZ1 = np.multiply(np.matmul(self.weights[1].T, dZ2), sigmoid_deri(Z1)) #(30, m)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)                       #(30, 784)
        db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)    #(30, 1)
        
        grads = {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}        
        return grads
        
    def train(self, X_train, Y_train, X_test = None, Y_test = None, epochs = 100, batch_size = 32, learning_rate = 0.005):
        """
        Updates weights and biases of neural network using batch gradient descent. This function prints loss function value and test accuracy after every 10 epochs.
		
        Arguments:
        X_train -- training data, a numpy.ndarray of shape (num_px*num*px, m_train)
        Y_train -- training data labels, a numpy.ndarray of shape (total number of categories, m_train)
        X_test -- test data, a numpy.ndarray of shape (num_px*num*px, m_test)
        Y_test -- test data labels, a numpy.ndarray of shape (total number of categories, m_test)
        epochs -- number of learning iterations over the whole training data set
        batch_size	-- number of training examples used in batch gradient descent to update weights
        learning_rate -- learning rate used in updating the weights and biases
        """
        m_train = X_train.shape[1]
        for epoch in range(epochs + 1):
            batch = np.arange(0, m_train)
            np.random.shuffle(batch)
            for k in range(m_train // batch_size + 1):
                if k * batch_size <  m_train:
                    X_mini_batch = X_train[:,batch[k * batch_size:(k + 1) * batch_size]]
                    Y_mini_batch = Y_train[:,batch[k * batch_size:(k + 1) * batch_size]]
                    self.update_weights(X_mini_batch, Y_mini_batch, learning_rate)
            
            if epoch % 10 == 0:          
                # Loss function
                A2 = self.feedforward(X_train)
                cost = (1 / m_train) * np.sum(-np.multiply(Y_train, np.log(A2)) - np.multiply(1 - Y_train, np.log(1 - A2)))
                print(f"epoch:{epoch}, Cost: {cost}, ", end = '')
                # Accutacy on training data
                if X_test is not None and Y_test is not None:
                    A2_test = self.feedforward(X_test)
                    class_pred = A2_test.argmax(axis = 0)
                    class_actual = Y_test.argmax(axis = 0)
                    acc = sum(class_actual == class_pred)
                    print(f"accuracy:{acc}/{X_test.shape[1]}")
                
    def update_weights(self, X, Y, learning_rate):
        """
        This function calculates gradients and updates weights
		
        Arguments:
        X -- training data, mini batch of size (num_px*num*px, batch_size)
        Y -- training labels, mini batch of size (number of categories, batch_size)
        learning_rate -- learning rate 
        """
        grads = self.calculate_gradients(X, Y)
        #update weights and biases
        self.weights[0] = self.weights[0] - learning_rate * grads["dW1"]
        self.weights[1] = self.weights[1] - learning_rate * grads["dW2"]
        self.biases[0] = self.biases[0] - learning_rate * grads["db1"]
        self.biases[1] = self.biases[1] - learning_rate * grads["db2"]
        
    
    def feedforward(self, X):
        """
        This function calculates the Y_pred, feedforward propagation.
    		
        Argument:
        X -- training data, a numpy.ndarray of shape (num_px*num*px, total number of examples)
    		
        Return:
        A -- feedforward output, Y_pred of size (number of categories, total number of examples)
        """
        A = X
        for b, W in zip(self.biases, self.weights):
            A = sigmoid(np.matmul(W, A) + b)
        return A