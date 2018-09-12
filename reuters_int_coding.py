#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:58:40 2018

@author: mansour
"""
import numpy as np
from keras.datasets import reuters
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import to_categorical
from matplotlib import pyplot as plt


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

def input_2_tensor(x, n_col=10000):
    res = np.zeros(shape=(len(x), n_col))
    for idx, elm in enumerate(x):
        res[idx,elm] = 1.
    return res

x_train, x_test = [input_2_tensor(elm) for elm in (x_train, x_test)]
#y_train, y_test = [to_categorical(elm.astype("float32")) for elm in (y_train, y_test)]


x_val, y_val = [elm[:1000] for elm in (x_train, y_train)]
x_train_sub, y_train_sub = [elm[1000:] for elm in (x_train, y_train)]

nn_model = models.Sequential()
nn_model.add(layer=Dense(units=64, activation='relu', input_shape = (10000,)))
nn_model.add(layer=Dense(units=64, activation='relu'))
nn_model.add(layer=Dense(units=46, activation='softmax'))

nn_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                 loss=losses.sparse_categorical_crossentropy, # labels are encoded as integer not one-hot encoding.
                 metrics=[metrics.sparse_categorical_accuracy]) # labesl are integers

history = nn_model.fit(x=x_train_sub, y=y_train_sub,
             batch_size=512, epochs=20,
             validation_data=(x_val,y_val))

history_dict = history.history

epoches = range(1,len(history_dict.get('loss'))+1)
plt.plot(epoches, history_dict.get('loss'),'bo',label="Loss")
plt.plot(epoches, history_dict.get('val_loss'),'b',label="Validation_Loss")
plt.xlabel = "Epoches"
plt.ylabel = "Loss"
plt.legend()
plt.show()

# Getting the minimum validation losss
val_loss = history_dict.get('val_loss')
best_epoch = val_loss.index(min(val_loss)) + 1

# Retraining using optimum epoch number and making predictions...
nn_model = models.Sequential()
nn_model.add(layer=Dense(units=64, activation='relu', input_shape = (10000,)))
nn_model.add(layer=Dense(units=64, activation='relu'))
nn_model.add(layer=Dense(units=46, activation='softmax'))

nn_model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                 loss=losses.sparse_categorical_crossentropy, # labels as integers...
                 metrics=[metrics.sparse_categorical_accuracy]) # lables as integers

history = nn_model.fit(x=x_train_sub, y=y_train_sub,
             batch_size=512, epochs=best_epoch,
             validation_data=(x_val,y_val))

history_dict = history.history
epoches = range(1,len(history_dict.get('loss'))+1)
plt.plot(epoches, history_dict.get('loss'),'bo',label="Loss")
plt.plot(epoches, history_dict.get('val_loss'),'b',label="Validation_Loss")
plt.xlabel = "Epoches"
plt.ylabel = "Loss"
plt.legend()
plt.show()

# plotting accuracy graphs for training 
plt.clf()
epoches = range(1,len(history_dict.get('loss'))+1)
plt.plot(epoches, history_dict.get('sparse_categorical_accuracy'),'bo',label="Accuracy")
plt.plot(epoches, history_dict.get('val_sparse_categorical_accuracy'),'b',label="Validation_Accuracy")
plt.xlabel = "Epoches"
plt.ylabel = "Loss"
plt.legend()
plt.show()


# Evaluating performance...
loss, accuracy = nn_model.evaluate(x=x_test, y=y_test)
loss
accuracy
# Making predictions using trained network...
pred = nn_model.predict(x=x_test)







