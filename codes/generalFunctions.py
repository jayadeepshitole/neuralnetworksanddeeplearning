import numpy as np
import h5py
from PIL import Image

###Miscellenous Functions
def sigmoid(z):
    """
	Compute sigmoid of z
	
	Argumets:
	z -- A scalar or numpy array of any size
	
	Return:
	s -- sigmoid(z) 
	"""
    s = (1/(1 + np.exp(-1* z)))
    return s

def sigmoid_deri(z):
    """
	Compute derivative of sigmoid function at z
	
	Argumets:
	z -- A scalar or numpy array of any size
	
	Return:
	s -- derivative of sigmoid function at z	
	"""
    s = (np.exp(z)/pow((1 + np.exp(z)), 2))
    return s
	
def load_cat_dataset():
    train_dataset = h5py.File('..\\data\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('..\\data\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes	
	
def mnistAsImage(x, file = "..\\images\\number.png"):
    """
    Convert array of numbers to image on the disk

    Argumets:
    x -- a numpy array of dim (784, 1) or (784, ), with values from 0 to 255
    file -- location where we want to save the image
	 """
    x = np.reshape((28, 28)) 
    im = Image.fromarray(x).convert('L')
    im.save(file)