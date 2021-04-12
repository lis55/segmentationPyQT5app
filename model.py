import numpy as np
import os

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow as tf
from functools import partial
from tensorflow.keras import backend as K
import tensorflow as tf


import random
import re
import shutil, os
import numpy as np
from tensorflow.keras.utils import Sequence
import vtk
from vtk.util import numpy_support

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


def unet(pretrained_weights=None, input_size=(512, 512, 1),act = 'relu'):
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation=act, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation=act, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation=act, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation=act, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation=act, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation=act, padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation=act, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation=act, padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation=act, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation=act, padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation=act, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation=act, padding='same', kernel_initializer='he_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation=act, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation=act, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_scheduler), loss=combo_loss(alpha=0.2, beta=0.4), metrics=[dice_accuracy])
    #model.compile(optimizer=RMSprop(lr=0.00001), loss=combo_loss, metrics=[dice_accuracy])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5,decay_steps=2720,decay_rate=0.9)

def combo_loss(alpha=0.2, beta=0.4): # beta before 0.4
    def loss(y_true, y_pred):
        return alpha*tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=beta)+((1-alpha)*dice_coefficient_loss(y_true, y_pred))
    return loss

def dice_coefficient(y_true, y_pred, smooth=0.01):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_accuracy(y_true, y_pred, smooth=0.01):
    return dice_coefficient(y_true, y_pred, smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def load_dicom(foldername, doflipz = True):
    reader = vtk.vtkDICOMImageReader()
    reader.SetFileName(foldername)
    reader.Update()

    #note: It workes when the OS sorts the files correctly by itself. If the files weren’t properly named, lexicographical sorting would have given a messed up array. In that
    # case you need to loop and pass each file to a separate reader through the SetFileName method, or you’d have to create a vtkStringArray, push the sorted filenames, and use
    # the vtkDICOMImageReader.SetFileNames method.

    # Load meta data: dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')


    return ArrayDicom
from PIL import Image
def make_overlay(overlay, background):
    background = background[:, :] / np.max(background[:, :])
    background = Image.fromarray((background * 255).astype('uint8'))
    background = background.convert("RGBA")
    overlay = Image.fromarray((overlay * 255).astype('uint8'))
    overlay = overlay.convert("RGBA")

    # Split into 3 channels
    r, g, b, a = overlay.split()

    # Increase Reds
    g = b.point(lambda i: i * 0)

    # Recombine back to RGB image
    overlay = Image.merge('RGBA', (r, g, b, a))

    new_img = Image.blend(background, overlay, 0.3)

    return new_img