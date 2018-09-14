#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:16:02 2018

@author: mansour
"""
# Converting text samples in to input_tensor with shape:
# (n=number_of_samples,
# max_height=maximum allowed number of words in a sample,
# max_width = maximum length of word dictionary.)

#import numpy as np
#
#samples = ['The cat sat on my mat.','The dog ate my homework']
#word_index = dict()
#
#for sample in samples:
#    for word in sample.split():
#        if word not in word_index.keys():
#            word_index[word] = len(word_index) + 1
#
#
#numb_samples = len(samples)
#max_height = 10                 # Based on maximum number of words in a sample.
#max_width = len(word_index) + 1 # Based on the number of keys (unique) in dictionary
#
#input_tensor = np.zeros(shape=(numb_samples, max_height, max_width))ight
#
#for n, sample in enumerate(samples):
#    for i, word in enumerate(sample.split()):
#        if word in word_index:
#            input_tensor[n,i,word_index[word]] = 1.


# Converting text samples into tensors using Keras
            
import numpy as np
from keras.preprocessing import text

samples = ['The cat sat on my mat.','The dog ate my homework']
tokenizer = text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(samples)
matrix = tokenizer.texts_to_matrix(samples)

                           























        




        

