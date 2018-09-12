#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:22:56 2018

@author: mansour
"""
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

# Building a conv network
conv_nn = models.Sequential()
conv_nn.add(layer=layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150,150,3)))
conv_nn.add(layer=layers.MaxPool2D(pool_size=(2,2)))
conv_nn.add(layer=layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
conv_nn.add(layer=layers.MaxPool2D(pool_size=(2,2)))
conv_nn.add(layer=layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
conv_nn.add(layer=layers.MaxPool2D(pool_size=(2,2)))
conv_nn.add(layer=layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
conv_nn.add(layer=layers.MaxPool2D(pool_size=(2,2)))
conv_nn.add(layer=layers.Flatten())
conv_nn.add(layer=layers.Dropout(0.5))
conv_nn.add(layer=layers.Dense(units=512, activation='relu')) 
conv_nn.add(layer=layers.Dense(units=1, activation='sigmoid'))
 
conv_nn.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

# Preprocessing of data
#
train_path = '/home/mansour/keras_deep/cats_vs_dogs/all/train'
validation_path = '/home/mansour/keras_deep/cats_vs_dogs/all/validation' 
    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(150,150),
        class_mode='binary',
        batch_size=32,
        interpolation='nearest')

val_generator = val_datagen.flow_from_directory(
        directory=validation_path,
        target_size=(150,150),
        class_mode='binary',
        batch_size=32,
        interpolation='nearest')

# Training the network using generators...
history = conv_nn.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_generator,
        validation_steps=50,
        workers=8)

conv_nn.save('cat_vs_dog_augnebt_model.h5')

# Plotting history...
epoches = np.arange(1,100+1)
plt.plot(epoches, history.history.get('loss'),'bo',label='Training_Loss')
plt.plot(epoches, history.history.get('val_loss'),'b',label='Validation_Loss')
plt.legend()

plt.clf()
plt.plot(epoches, history.history.get('binary_accuracy'),'ro',label='Training_Accuracy')
plt.plot(epoches, history.history.get('val_binary_accuracy'), 'r',label='Validation_Accuracy')
plt.legend()












