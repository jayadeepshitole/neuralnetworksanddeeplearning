# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:07:35 2018

@author: jayadeep
Reference: https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Logistic%20Regression%20with%20a%20Neural%20Network%20mindset.ipynb
"""

###################################################################################################
## 1. Packages
import os
# set current directory
os.chdir("F:\\neuralnetworksanddeeplearning\\codes")
from generalFunctions import load_cat_dataset
from logisticRegression import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy
###################################################################################################

##Data
# Loading the data (cat/non-cat)
"""
The data contains:
    - a training set of m_train images labeled as cat (y = 1) or non-cat (y = 0) 
    - a test set of m_test images labeled as cat or non-cat
    - each image is of shape (num_px, num_px, 3) where 3 is for the three channels (RGB). Thus each image is square (height = num_px) and (width = num_px) 
"""
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_cat_dataset()

## Example of a picture
#index = 25
#plt.imshow(train_set_x_orig[index])
#print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")            

m_train = train_set_x_orig.shape[1]
m_test = test_set_x_orig.shape[1]
num_px = train_set_x_orig.shape[1]
print(f"Number of training examples: m_train = {m_train}")
print(f"Number of testing examples: m_train = {m_test}")
print(f"Height/Width of each image: {num_px}")
print(f"train_set_x shape: {train_set_x_orig.shape}")
print(f"train_set_y shape: {train_set_y.shape}")
print(f"test_set_x shape: {test_set_x_orig.shape}")
print(f"test_set_y shape: {test_set_y.shape}")

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# Standerdize the dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
###################################################################################################

# Fit Logistic Model
LR = LogisticRegression(dim = 12288)
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