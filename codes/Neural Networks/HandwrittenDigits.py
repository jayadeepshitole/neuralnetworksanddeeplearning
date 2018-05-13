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
from scipy import ndimage
import scipy
from mnist_loader import load_data
###################################################################################################

##Data
# Loading the data (cat/non-cat)
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
NN.train(X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, epochs = 500,learning_rate = 0.5)

m_train = X_train.shape[1]
batch_size = 1500
for k in m_train // batch_size + 1:
    if k * batch_size <  m_train:
        mini_batch = X_train[:,batch[k * batch_size:(k + 1) * batch_size]]
    
m_train = 4
batch_size = 2    
for k in range(m_train // batch_size + 1):
    if k * batch_size <  m_train:
        print(f"{k * batch_size}:{(k + 1) * batch_size}")

grads, costs = LR.train(X = train_set_x, Y = train_set_y, num_iterations = 2000, learning_rate=0.005, print_cost = True, plot_cost = True)

# Prediction Accuracy for training data
y_pred = LR.predict(X = train_set_x)
LR.accuracy_stats(train_set_y, y_pred)

# Prediction Accuracy for test data
y_test_pred = LR.predict(X = test_set_x)
LR.accuracy_stats(test_set_y, y_test_pred)

###################################################################################################

# Experiment with different learning rates
learning_rates = [0.01, 0.001, 0.0001]
models_costs = {}
for l_rate in learning_rates:
    print (f"learning rate is :{l_rate}")
    # Fit Logistic Model
    LR = LogisticRegression(dim = 12288)
    grads, costs = LR.train(X = train_set_x, Y = train_set_y, num_iterations = 2000, learning_rate=l_rate, print_cost = False, plot_cost = True)
    models_costs[str(l_rate)] = costs
    
    # Prediction Accuracy for training data
    y_pred = LR.predict(X = train_set_x)
    LR.accuracy_stats(train_set_y, y_pred, confusion_mat=False, ROC=False)
    
    # Prediction Accuracy for test data
    y_test_pred = LR.predict(X = test_set_x)
    LR.accuracy_stats(test_set_y, y_test_pred, confusion_mat=False, ROC = False)

for i in learning_rates:
    plt.plot(np.squeeze(models_costs[str(i)]), label= str(i))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
###################################################################################################

## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "dog_image.jpg"   # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "..\\images\\" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = LR.predict(my_image) >= 0.50 + 0 

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")