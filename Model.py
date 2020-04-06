import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import class_weight

import numpy as np


tf.keras.backend.set_floatx('float64')

drop = 0.5
img_size = (128, 128)

model = Sequential([
    Conv2D(8, 5, activation = 'relu', input_shape = (img_size[0], img_size[1], 1)),
    MaxPool2D(3),
    Conv2D(16, 4, activation = 'relu'),
    MaxPool2D(2),
    Conv2D(32, 3, activation = 'relu'),
    Flatten(),
    Dense(32, activation = 'relu'),
    Dropout(drop),
    Dense(8, activation = 'relu'),
    Dense(3, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    shear_range = 0.2,
    zoom_range = 0.05,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.05,
    brightness_range = [1, 1.5],
    horizontal_flip = True,
    validation_split = 0.1,
    dtype = tf.float64)

train_generator = datagen.flow_from_directory(
    'Dataset/Train',
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 32,
    shuffle = True,
    subset = "training",
    class_mode='categorical')


test_datagen = ImageDataGenerator(
    rescale = 1. / 255.,
    dtype = tf.float64)

test_generator = test_datagen.flow_from_directory(
    'Dataset/Test',
    target_size = img_size,
    color_mode = 'grayscale',
    batch_size = 16,
    shuffle = True,
    class_mode='categorical')


class_weights = class_weight.compute_class_weight(
                   'balanced',
                   np.unique(train_generator.classes), 
                   train_generator.classes)


model.fit(train_generator, 
          epochs = 10,
          shuffle = True,
          validation_data = test_generator,
          class_weight = class_weights,
          workers = 8,
          max_queue_size = 512)


model.save('saved/saved.h5')