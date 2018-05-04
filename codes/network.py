# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:26:30 2018

@author: jayadeep

A module to implement stochastic gradient descent algorithm for feedforward neural network. 
Gradients are calculated using backpropagation. 
"""

import os
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
import numpy as np
from mnist_loader import load_data_wrapper
from generalFunctions import sigmoid, sigmoid_deri

class Network(object):
    """The list sizes contains the number of neurons in the respecitve layers of the netdwork. The
    biases and weights are initialised randomly using gaussian distribution with mean 0 and variance 1.
    Note that we assume the first layer as input layer without any biases.
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
#net = Network([2, 3, 1])
    
    def feedforward(self, a):
        """Return an numpy.ndarray of shape (num_layers - 1, 2), the value at [i, 0] denotes the z values 
        and [i, 1] denotes the activation a for hidden layer i.
        Where, z = wa + b and a = sigmoid(z).
        For network of size 3, function will return [[z1, a1], [z2, a2]]
        The input a is the training data - a numpy.ndarray of shape (n, 1) or (n, m)"""
        za = []
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, a) + b
            a = sigmoid(z)
            za.append([z,a])
        return(np.array(za))
    
    def gradientDescent(self, training_data, learning_rate = 0.01):
        
        training_inputs, training_labels = zip(*training_data)
        training_inputs = np.array(training_inputs)
        training_inputs = training_inputs.reshape((training_inputs.shape[0], training_inputs.shape[1])).T
        training_labels = np.array(training_labels)
        training_labels = training_labels.reshape((training_labels.shape[0], training_labels.shape[1])).T
        
        grads_w = [np.zeros(w.shape) for w in self.weights]
        grads_b = [np.zeros(b.shape) for b in self.biases]
        
        za = self.feedforward(training_inputs)
        z = za[-1, 0]
        a = za[-1, 1]
        dz = a - training_labels
        dw = da * np.dot(sigmoid_deri(z).shape)
        
        z2 = za[1, 0]
        a2 = za[1, 1]
        da2 = da3 * self.weights[2] * sigmoid_deri(z2)
        
        #a2 = za[1, 1]
        #z2 = za[1, 0]
        #da2 = -1* np.mean(training_data[1] - a2)

training_data, validation_data, test_data = load_data_wrapper()
net = Network([784, 10, 10, 1])
za = net.GD(training_data = training_data)