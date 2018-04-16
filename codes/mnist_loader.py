# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:41:44 2018

@author: jayadeep
"""

import _pickle as cPickle
import gzip

with gzip.open("mnist.pkl.gz", "rb") as data:
    training_data, validation_data, test_data = cPickle.load(data, encoding='latin1')


