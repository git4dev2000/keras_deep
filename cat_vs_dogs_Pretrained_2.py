#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 17:19:25 2018

@author: mansour
"""

# Including data augmentation
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

# Loading a pre-trained network and freez its weights
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))
conv_base.trainable = False

# Peparing data to feed to the network...
base_dir = '/home/mansour/keras_deep/cats_vs_dogs/all'
train_dir = os.path.join(base_dir,'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Preprocessing of data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=40,
        shear_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generating batches of input_data and labels
train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150,150),
        class_mode='binary',
        batch_size=20)

val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=(150,150),
        class_mode='binary',
        batch_size=20)

# Extending cov_nn model and training it using train and validation generator
nn_model = models.Sequential()
nn_model.add(layer=conv_base)
nn_model.add(layer=layers.Flatten())
nn_model.add(layer=layers.Dense(units=2**8, activation='relu'))
nn_model.add(layer=layers.Dense(units=1, activation='sigmoid'))

nn_model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                 loss=losses.binary_crossentropy,
                 metrics=[metrics.binary_accuracy])

history = nn_model.fit_generator(generator=train_generator,
                       steps_per_epoch=100,
                       epochs=30,
                       validation_data=val_generator,
                       validation_steps=50)

nn_model.save('cat_vs_dogs_pretrained_model_2.h5')

# Plotting
epoches = np.arange(1,31)
plt.plot(epoches, history.history.get('loss'), 'bo', label='Training_Loss')
plt.plot(epoches, history.history.get('val_loss'), 'b', label='Validation_Loss')

plt.clf()
plt.plot(epoches, history.history.get('binary_accuracy'), 'bo', label='Training_Metric')
plt.plot(epoches, history.history.get('val_binary_accuracy'), 'b', label='Validation_Metric')













# Creating 