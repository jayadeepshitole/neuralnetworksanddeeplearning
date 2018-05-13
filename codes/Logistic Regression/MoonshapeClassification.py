# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:33:22 2018

@author: sangi
Reference: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import os
# set current directory
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
from logisticRegression import LogisticRegression
from generalFunctions import plot_decision_boundary

######################################################################################

## Generate a dataset a plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(x = X[:,0], y = X[:,1], c = y, cmap = plt.cm.Spectral)

# The data is not linearly separable, this means that linear classifiers like logistic regression wont be 
# able to fit the data unless you hand-engineer non-linear features (such as polynomials).
# In fact, that's one of the major advantages of Neural Networks. You don't need to worry about the
# feature engineering. The hidden layer of neural network will learn the features for you.
######################################################################################

## Logistic Regression
num_features = 2
LR = LogisticRegression(dim = num_features)
grads, costs = LR.train(X = X.T, Y = y.reshape(1, 200), print_cost = True, num_iterations = 5000, learning_rate = 0.01, plot_cost = True)
plot_decision_boundary(lambda x: LR.predict(x), X = X, y = y) 
plt.title("Logistic Regression") 

# The graph shows the decision boundary learned by our logistic regression classifier. It separates
# the data as good as it can using a straight line, but it's unable to capure the "moon-shape" of
# our data.
