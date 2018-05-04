# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:07:35 2018

@author: jayadeep
"""

import os
# set current directory
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
import numpy as np
import matplotlib.pyplot as plt
from generalFunctions import load_cat_dataset
from logisticRegression import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cat_dataset()

# Example of a picture
index = 5
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
X = train_set_x
y = train_set_y
LR.train(X, y, learning_rate=0.009, epochs = 250, print_cost = True, plot_cost = False)

y_pred = LR.predict(test_set_x_flatten)
y_class_pred = (y_pred >= 0.50) + 0
np.sum(abs(test_set_y - y_class_pred))
confusion_matrix(test_set_y[0], y_class_pred[0])
fpr, tpr, thresholds = roc_curve(test_set_y[0], y_pred[0])
roc_auc_score(test_set_y[0], y_pred[0])