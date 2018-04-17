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
    
    def GD(self, learning_rate = 0.01, training_data):
        
        training_inputs, training_labels = zip(*training_data)
        training_inputs = np.array(training_inputs)
        training_inputs = training_inputs.reshape((50000, 784)).T
        training_labels = np.array(training_labels)

        za = self.feedforward(self, training_inputs)
        a2 = za[1, 1]
        z2 = za[1, 0]
        da2 = -1* np.mean(training_data[1] - a2)
        dw2 = da2 * 

training_data, validation_data, test_data = load_data_wrapper()
net = Network([784, 10, 1])
net.feedforward(training_inputs)

net = Network([2,3,1])
a = np.array([1,1])
a = a.reshape((2,1))
za = net.feedforward(a)
za.shape
za[0]
sigmoid(np.matmul(net.weights[0], a) + net.biases[0])

###Miscellenous Functions
def sigmoid(z):
    """Return sigmoid of z"""
    return (1/(1 + np.exp(-1* z)))

def sigmoid_deri(z):
    """Return derivative of sigmoid function at z"""
    return (np.exp(z)/pow((1 + np.exp(z)), 2))