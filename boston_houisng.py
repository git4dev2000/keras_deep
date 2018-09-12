#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:53:53 2018

@author: mansour
"""
import numpy as np
from keras.datasets import boston_housing
from keras import layers
from keras import models
from keras import losses
from keras import metrics
from keras import optimizers
from keras import preprocessing
from matplotlib import pyplot as plt




(x_train, y_train) , (x_test, y_test) = boston_housing.load_data()

# Normalizing training and test inputs using mean and std of training inputs...
mean_input = np.mean(x_train, axis=0)
std_input = np.std(x_train, axis=0)

x_train, x_test = [(elm - mean_input)/std_input for elm in (x_train, x_test)]

def build_model():
    nn_model = models.Sequential()
    nn_model.add(layer=layers.Dense(units=64, activation='relu',  input_shape=(13,)))
    nn_model.add(layer=layers.Dense(units=64, activation='relu'))
    nn_model.add(layer=layers.Dense(units=1))
    
    nn_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return nn_model


k = 4
total_sample_size = x_train.shape[0]
val_size = total_sample_size // k
idx_seq = np.arange(total_sample_size)
mae_scores = []
history_list = []
num_epoch = 500
batch_size = 8
for i in range(k):
    val_idx = idx_seq[i*val_size:(i+1)*val_size]
    train_subset_idx = np.setdiff1d(idx_seq, val_idx)
    
    x_val = x_train[val_idx]
    x_train_subet = x_train[train_subset_idx]
    
    y_val = y_train[val_idx]
    y_train_subset = y_train[train_subset_idx]             
    
    network = build_model()
    history = network.fit(x=x_train_subet, y=y_train_subset, batch_size=batch_size,
                          epochs=num_epoch, verbose=0,
                          validation_data=(x_val, y_val))
    
    history_list.append(history.history.get('val_mean_absolute_error'))


average_mae_history = [np.mean([x[i] for x in history_list]) for i in range(num_epoch)]
epoches =  np.arange(1,(len(average_mae_history)+1))

plt.plot(epoches[10:],
            np.asarray(average_mae_history[10:]),
            'b', label='val_MAE')
plt.legend()
plt.show()

# Define an optional smoothing function... 
def exp_smooth(iterable, factor=0.9):
    smooth_output=[]
    for point in iterable:
        if smooth_output:
            prev_point = smooth_output[-1]
            new_point = prev_point*factor + (1-factor)*point
            smooth_output.append(new_point)
        else:
            smooth_output.append(point)
    return smooth_output


# Finding the optimum epoch...
average_mae_history_smooth = exp_smooth(average_mae_history) 
optimum_epoch = average_mae_history.index(min(average_mae_history)) + 1

# Retraining using optimum epoch
batch_size = 16
best_model = build_model()
best_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=80)

# Evaluation final final model using test data...
mse, mae = best_model.evaluate(x=x_test, y=y_test)
mse
mae








    