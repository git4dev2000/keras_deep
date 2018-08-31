#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:31:56 2018

@author: mansour
"""

import numpy as np
from keras import losses
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import metrics
from keras import optimizers


a = optimizers.RMSprop()
