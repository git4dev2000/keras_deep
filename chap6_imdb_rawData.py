#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:45:43 2018

@author: mansour
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence 
from keras.preprocessing import text
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import os
import matplotlib.pyplot as plt

# Loading file contents (raw data) into a list of iterables...
train_dir = '/home/mansour/keras_deep/aclImdb/aclImdb/train'
text_data = []
labels = []

for folder in [os.path.join(train_dir, 'pos'), os.path.join(train_dir, 'neg')]:
    fnames = os.listdir(folder)
    for fname in fnames:
        if fname[-4:] == '.txt':
            f = open(os.path.join(folder,fname))
            text_data.append(f.read())
            f.close()
            if 'pos' in folder:
                labels.append(1)
            else:
                labels.append(0)

# Preprocessing of data using keras...
dict_size = 10000
max_comment_len = 100
training_sample_size = 200
val_size = 10000

tokenizer = text.Tokenizer(num_words=dict_size) #Instantiating a Tokenizer
tokenizer.fit_on_texts(text_data)
sequences = tokenizer.texts_to_sequences(text_data)
fixed_size_seq = sequence.pad_sequences(sequences, maxlen=max_comment_len)

data = fixed_size_seq
labels = np.array(labels)

# Shuffeling and spliting to train and validation
random_indices = np.arange(len(data))
np.random.shuffle(random_indices)

x_train, y_train = data[:training_sample_size], labels[:training_sample_size]
x_val= data[training_sample_size:training_sample_size+val_size]
y_val = labels[training_sample_size:training_sample_size+val_size]








