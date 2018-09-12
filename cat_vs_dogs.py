#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 16:23:22 2018

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





all_data_folder = '/home/mansour/keras_deep/cats_vs_dogs/all'

# Creating three subfolders containing training, val and test:
sub_folder_names = ['train', 'validation', 'test']
catg_names = ['dogs', 'cats']

#for elm in sub_folder_names:
#    for names in catg_names:
#        os.makedirs(os.path.join(all_data_folder, elm, names),exist_ok=False)
#
## Copying data to their appropriate folder:
#train_cat_names  = ['cat.{}.jpg'.format(i) for i in range(1000)] 
#val_cat_names = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
#test_cat_names = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
#
#train_dog_names  = ['dog.{}.jpg'.format(i) for i in range(1000)] 
#val_dog_names = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
#test_dog_names = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
#
## Cats...
#for elm in train_cat_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'train', 'cats', elm)
#    shutil.copyfile(src, dest)
#
#for elm in val_cat_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'validation', 'cats', elm)
#    shutil.copyfile(src, dest)
#
#for elm in test_cat_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'test', 'cats', elm)
#    shutil.copyfile(src, dest)
## Dogs...
#for elm in train_dog_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'train', 'dogs', elm)
#    shutil.copyfile(src, dest)
#
#for elm in val_dog_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'validation', 'dogs', elm)
#    shutil.copyfile(src, dest)
#
#for elm in test_dog_names:
#    src = os.path.join(all_data_folder, elm)
#    dest = os.path.join(all_data_folder, 'test', 'dogs', elm)
#    shutil.copyfile(src, dest)
###############################################################################
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
conv_nn.add(layer=layers.Dense(units=512, activation='relu')) 
conv_nn.add(layer=layers.Dense(units=1, activation='sigmoid'))

conv_nn.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy])

# preprocessing of image data...
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory='/home/mansour/keras_deep/cats_vs_dogs/all/train',
        batch_size=20,
        target_size=(150,150),
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        directory='/home/mansour/keras_deep/cats_vs_dogs/all/validation',
        batch_size=20,
        target_size=(150,150),
        class_mode='binary')

# training network using generator data...
history = conv_nn.fit_generator(
        generator=train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

# Saving model...
conv_nn.save('conv_nn_cats_vs_dogs.h5')

# Plotting history
epoches = np.arange(1,len(history.history.get('loss'))+1)
plt.plot(epoches, history.history.get('loss'), 'bo',label='Loss')
plt.plot(epoches, history.history.get('val_loss'),'b', label='Validation_Loss')
plt.legend()

plt.clf()
plt.plot(epoches, history.history.get('binary_accuracy'),'ro',label='Acuuracy')
plt.plot(epoches, history.history.get('val_binary_accuracy'),'r',label='Val_Accuracy')
plt.legend()
###############################################################################
# Augment image data and see how they look...
from keras.preprocessing import image

datagen = image.ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

cat_dir = '/home/mansour/keras_deep/cats_vs_dogs/all/train/cats'
file_paths = [os.path.join(cat_dir, elm) for elm in os.listdir(cat_dir)]

# Picking an image
img = image.load_img(file_paths[13], target_size=(150,150))
x = image.img_to_array(img=img)

# Reshaping to be used for imageDataGenerator instance...
x = x.reshape((1,)+x.shape)

#
i=0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    np_array_image = image.img_to_array(batch[0])
    imgplot =plt.imshow(np_array_image[:,:,1])
    i+=1
    if i>5:
        break
plt.show()
        































