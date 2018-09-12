#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:28:53 2018

@author: mansour
"""
# No augmented data...

import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
from keras import models
from keras import layers
from keras import metrics
from keras import losses
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator 
from keras.applications import VGG16


conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))

# Peparing data to feed to the network...
base_dir = '/home/mansour/keras_deep/cats_vs_dogs/all'
train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')





def extract_feature(directory, numb_sample, batch_size=20):
    feature_data = np.zeros(shape=(numb_sample,4,4,512))
    label_data = np.zeros(shape=(numb_sample,))
    
    # Creating image data generator
    datagen = ImageDataGenerator(rescale=1./255)
    input_label_batch_tups = datagen.flow_from_directory(
            directory=directory,
            target_size=(150,150),
            class_mode='binary',
            batch_size=batch_size)
    # Truncating generator
    i=0
    for input_batch, label_batch in input_label_batch_tups:
        conv_base_output = conv_base.predict(input_batch)
        feature_data[i*batch_size:(i+1)*batch_size] = conv_base_output
        label_data[i*batch_size:(i+1)*batch_size] = label_batch
        print(i)
        i+=1
        if i*batch_size>=numb_sample:
            break
    return feature_data, label_data

# Using extracted features as inputs for a new nn model with Dense layers
train_features = extract_feature(train_dir, 2000)
val_features = extract_feature(val_dir, 1000)
test_features = extract_feature(test_dir, 1000)

# Training a Dense network that uses feature data
nn_model = models.Sequential()
nn_model.add(layer=layers.Dense(units=2**8, activation='relu', input_shape=(512*4*4,)))
nn_model.add(layer=layers.Dropout(0.5))
nn_model.add(layer=layers.Dense(units=1, activation='sigmoid'))

nn_model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                 loss=losses.binary_crossentropy,
                 metrics=[metrics.binary_accuracy])


history = nn_model.fit(x=train_features[0].reshape(2000,-1),
                       y=train_features[1],
                       epochs=30,
                       batch_size=20,
                       validation_data=(val_features[0].reshape(1000,-1),val_features[1]))

# Plotting results
epoches = np.arange(1,31)
plt.plot(epoches, history.history.get('loss'),'bo',label='Training_Loss')
plt.plot(epoches, history.history.get('val_loss'),'b',label='Validation_Loss')
plt.legend()        

plt.clf()
epoches = np.arange(1,31)
plt.plot(epoches, history.history.get('binary_accuracy'),'bo',label='Training_Loss')
plt.plot(epoches, history.history.get('val_binary_accuracy'),'b',label='Validation_Loss')
plt.legend()        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    