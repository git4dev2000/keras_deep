#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:53:46 2018

@author: mansour
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence 
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

max_length = 20
max_dic_length = 10000
embedding_dim = 8

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_dic_length)
x_train, x_test = [sequence.pad_sequences(elm, maxlen=max_length) for elm in (x_train, x_test)]

# Building a network model...
nn_model = models.Sequential()
nn_model.add(layer=layers.Embedding(input_dim=max_dic_length, output_dim=embedding_dim,
                                    input_length=max_length))
nn_model.add(layer=layers.Flatten())
nn_model.add(layer=layers.Dense(units=1, activation='sigmoid'))

nn_model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
                 loss=losses.binary_crossentropy,
                 metrics=[metrics.binary_accuracy])

history = nn_model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_split=0.2)
