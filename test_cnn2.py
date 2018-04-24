from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from skimage import io
from skimage import color
from skimage import exposure
from skimage import transform

import os

import pandas as pd
import numpy as np

test_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# Initialising the CNN
cnn = Sequential()

# Step 1 - Convolution
cnn.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), padding = 'same', activation = 'relu'))

# Step 2 - Pooling
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Add third convolutional layer
cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full connection
cnn.add(Dense(units = 1024, activation = 'relu'))
cnn.add(Dropout(.5))
cnn.add(Dense(units = 43, activation = 'softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


cnn.load_weights("cnn_weights4.h5")

	
test_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
		horizontal_flip=True,
        fill_mode='nearest')
		
test_set = test_datagen.flow_from_directory('/home/csc2168/hw/cis/CIS_520/predicted_cropped_images/',
											class_mode = None,
											shuffle = False,
                                            target_size = (32, 32),
                                            batch_size = 32)
						
											
preds = cnn.predict_generator(test_set)
labels = preds.argmax(axis = 1);
print(labels)
df = pd.DataFrame({"filename": test_set.filenames, "label" : labels})
df.to_csv("cnn_result.csv", index = False)


