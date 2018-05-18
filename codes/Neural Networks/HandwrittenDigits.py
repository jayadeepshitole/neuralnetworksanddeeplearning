# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:07:35 2018

@author: jayadeep
"""

###################################################################################################
## 1. Packages
import os
# set current directory
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
from network import Network
import matplotlib.pyplot as plt
import numpy as np
from mnist_loader import load_data, imageprepare
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

## Example of a picture
index = 25
plt.imshow(X_train[:,index].reshape(28, 28))
print ( "it's a " + "y = " + str(Y_train[index])) 

m_train = X_train.shape[1]
m_test = X_test.shape[1]
num_px = X_train.shape[0]
print(f"Number of training examples: m_train = {m_train}")
print(f"Number of testing examples: m_train = {m_test}")
print(f"Height/Width of each image: {int(np.sqrt(num_px))}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Convert Y to oneHotEncoding 
Y_train = np.eye(10)[Y_train].T
Y_test = np.eye(10)[Y_test].T

# Reshape the training and test examples
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))
###################################################################################################

# Fit Logistic Model
NN = Network(sizes=[784,30,10])
NN.train(X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, epochs = 100,learning_rate = 3)
###################################################################################################

## (PUT YOUR IMAGE NAME) 
my_image = "three_paint.jpg"   # change this to the name of your image file 
fname = "..\\images\\" + my_image

my_image = np.array([imageprepare(fname)])#file path here
my_predicted_image = NN.feedforward(my_image.T).argmax(axis = 0)
plt.imshow(my_image.reshape((28,28)))
print(my_predicted_image)

plt.imshow(X_train[:,0].reshape(28,28))

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")