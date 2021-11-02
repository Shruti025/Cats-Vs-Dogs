# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:04:00 2021

@author: LENOVO
"""

# %% Training Dataset

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255 , 
    shear_range = 0.2 , 
    zoom_range = 0.2 , 
    horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/Lenovo/Downloads/Cats_and_Dogs/train',
    target_size = (256, 256),
    batch_size = 64,
    class_mode = 'binary'
    )

#%% Test Dataset
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = test_datagen.flow_from_directory(
    'C:/Users/Lenovo/Downloads/Cats_and_Dogs/test1',
    target_size = (256,256),
    batch_size = 64,
    class_mode = 'binary'
    )

#%% Initialising CNN model

cnn = tf.keras.models.Sequential()

#%% Convolution Layer

cnn.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation = 'relu', input_shape = [256 , 256 , 3]))

#%% Pooling    
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2 ))

#%% Repeating layers    
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
    
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))    

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))
#%% Flattening

cnn.add(tf.keras.layers.Flatten())

#%% Dense

cnn.add(tf.keras.layers.Dense(units = 512 , activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))

#%% Compiling the Model

cnn.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#%% Train the model

cnn.fit(x = train_generator , validation_data = validation_generator , epochs = 20)

#%% Testing for individual cases
#Making prediction for single image

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('C:/Users/Lenovo/Downloads/predict4.jpg' , target_size = (256, 256))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image , axis = 0)
#result = cnn.predict(test_image)

#train_generator.class_indices

#if result[0][0] == 1:
#    prediction = "It's a Dog! "
#else:
#    prediction = "It's a Cat! "

#print(prediction)