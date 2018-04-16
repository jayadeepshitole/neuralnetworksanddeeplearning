# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:41:44 2018

@author: jayadeep
"""

import _pickle as cPickle
import gzip
from PIL import Image
import numpy as np

def load_data():
    """Return the MNIST data containing the training_data, the validation_data, the test_data
    
    The ``training_data`` is a tuple with two entries. The first entry contains actual training images. This is a numpy ndarray with 
    50,000 entries. Each entry is in turn a numpy ndarray with 784 values, representing the 28*28 = 784 pixels in a single MNSIT image.
    
    The second entry in the ``training_data`` tuple is a numpy ndarray containing 50,000 entries. Those entries are just the digit values
    (0,...,9) of the corresponding training images in the first entry of the tuple.
    
    The ``validation_data`` and the ``test_data`` are the same as the ``training_data`` except they contain 10,000 images.
    
    """
    with gzip.open("..\\data\\mnist.pkl.gz", "rb") as data:
        training_data, validation_data, test_data = cPickle.load(data, encoding='latin1')
    return (training_data, validation_data, test_data)

def to_image(array_of_numbers, img_name = "img"):
    """saves a png image given an array of shape (784, ) or (784, 1)
    """
    
    array_of_numbers = np.reshape(array_of_numbers,(28,28))
    array_of_numbers = array_of_numbers * 255
    im = Image.fromarray(array_of_numbers).convert('L')
    im.save(f"..\\image\\{img_name}.png")

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data, test_data)``. Based in ``load_data``, but the format is
    more convenient in implementing neural networks.
    
    In particular, ``training_data`` is a list containing 50,000 2 tuples ``(x,y)``. ``x`` is a 784-dimensional numpy.ndarray
    containing input image. ``y`` is 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct digit 
    for ``x``.
    
    ``validation_data`` and ``test_data`` are lists containing 10,000 2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    """
    tr_d, va_d, te_d = load_data()
    training_input = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_input, training_results)
    validation_input = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_data = zip(validation_input, va_d[1])
    test_input = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_data = zip(test_input, te_d[1])
    return (training_data, validation_data, test_data)

def vectorize(j):
    """Return a 10-dimensional unit vector with 1.0 at the jth position and 0s at all other positions. 
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def test_code():
    """
    loads mnist data, and saves first test_image in png format    
    """
    training_data, validation_data, test_data = load_data()
    temp = test_data[0][0]
    to_image(temp, img_name = "image")