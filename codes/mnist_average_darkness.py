# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:43:19 2018

@author: jayadeep
"""

###################################################################################################
## 1. Packages
import os
# set current directory
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
import numpy as np
from mnist_loader import load_data
###################################################################################################

##Data
# Loading the Zipcode data
"""
The data contains:
    - a training set of m_train images labeled as cat (y = 1) or non-cat (y = 0) 
    - a test set of m_test images labeled as cat or non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the three channels (RGB). Thus each image is square (height = num_px) and (width = num_px) 
"""
training_data, validation_data, test_data = load_data()
X_train = training_data[0].T
Y_train = training_data[1]
X_test = test_data[0].T
Y_test = test_data[1]
X_valid = validation_data[0].T
Y_valid = validation_data[1]
###################################################################################################

def calculate_avg_darkness(X_train, Y_train):
    avg_darkness = []
    for digit in range(0,10):
        index_digit = (Y_train == digit)
        X_digit = X_train[:,index_digit]
        darkness_digit = X_digit.sum() / (X_digit.shape[0] * X_digit.shape[1])
        avg_darkness.append(darkness_digit)
    return np.array(avg_darkness)
        
def calculate_acc(X_train, Y_train, X_test, Y_test):    
    avg_darkness = calculate_avg_darkness(X_train, Y_train)
    X_test_avg_darkness = X_test.sum(axis = 0) / X_test.shape[0]
    X_test_avg_darkness_dist = abs(X_test_avg_darkness.reshape(1, X_test_avg_darkness.shape[0]) - avg_darkness.reshape(len(avg_darkness), 1))
    Y_test_pred = X_test_avg_darkness_dist.argmin(axis = 0)
    acc = 100 * sum(Y_test == Y_test_pred)/len(Y_test)
    print(f"accuracy is: {acc}%")

avg_darkness = calculate_avg_darkness(X_train, Y_train)
calculate_acc(X_train, Y_train, X_train, Y_train)
calculate_acc(X_train, Y_train, X_test, Y_test)
calculate_acc(X_train, Y_train, X_valid, Y_valid)