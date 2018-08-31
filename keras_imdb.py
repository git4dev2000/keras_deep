#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:21:22 2018

@author: mansour
"""
# Notes:
# 1) Data type (both inputs and labels) must be of tensor type with dtype=float32
# 2) one-hot encoding must be used for Dense layers.
# 3) The first hiddne layes should have argument: inpust_shape i.e. a tuple tupe

import numpy as np
from keras import models
from keras import layers
from keras import utils
from keras.datasets import imdb
from keras import activations
from keras import optimizers
from keras import metrics
from keras import losses
from matplotlib import pyplot as plt

(train_inputs, train_labels), (test_inputs, test_labels) = imdb.load_data(num_words=10000)

print(sum(train_inputs[0]))
# Data preparation functions...
def seq_to_vector(seq, n_col = 10000):
    res = np.zeros(shape=(len(seq), n_col))
    
    for idx, list_elm in enumerate(seq):
        res[idx, list_elm] = 1.
        return res


train_inputs, test_inputs = [(seq_to_vector(elm)).astype("float32") for elm in (train_inputs, test_inputs)]
train_labels, test_labels = [elm.astype("float32") for elm in (train_labels, test_labels)]
train_labels, test_labels = [elm.astype("float32") for elm in (train_labels, test_labels)]



print(test_inputs.dtype, type(train_inputs), train_inputs.dtype, type(train_inputs))
print(test_labels.dtype, type(test_labels), train_labels.dtype, type(train_labels))


# Considering validation...
val_inputs = train_inputs[:10000]
val_labels = train_labels[:10000]

train_inputs_subset = train_inputs[10000:] 
train_labels_subset = train_labels[10000:]


print(train_inputs_subset.shape, train_labels_subset.shape, train_labels_subset.dtype, type(train_inputs_subset))
print(val_inputs.shape, val_labels.shape, val_inputs.dtype, val_labels.dtype)

# Settig up network...
network = models.Sequential()
network.add(layer=layers.Dense(units=16, activation="relu", input_shape=(10000,)))
network.add(layer=layers.Dense(units=16, activation="relu"))
network.add(layer=layers.Dense(units=1, activation="sigmoid"))


network.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

# Training and saving history...
history  = network.fit(train_inputs_subset,
                       train_labels_subset,
                       epochs=20,
                       batch_size=512,
                       validation_data=(val_inputs, val_labels))

history_dict = history.history

print(history_dict.keys())

# Plotting history...
val_loss = history_dict.get("val_loss")
train_sub_loss = history_dict.get("loss")

epoches = range(1, len(val_loss) + 1)
plt.plot(epoches, val_loss, "b", label="Validation loss")
plt.plot(epoches, train_sub_loss, "bo", label = "Training loss")
plt.title("History of Training")
plt.xlabel = "epoches"
plt.ylabel = "Loss"
plt.legend()
plt.show()

# Evaluating performance...
a = network.evaluate(test_inputs, test_labels)

print(train_inputs_subset.shape, train_labels_subset.shape)

for idx in range(len(train_inputs_subset[:,0])):
    print(np.sum(train_inputs_subset[idx]))
        
