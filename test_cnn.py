# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
from skimage import io, color, exposure, transform
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


test = pd.read_csv('/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Test/GT-final_test.csv',sep=';')


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), padding = 'same', activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(.5))
classifier.add(Dense(units = 43, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.load_weights("cnn_weights.h5")


x_test = []
y_test = []

for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
	y_test.append(class_id)

	
test_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
		horizontal_flip=True,
        fill_mode='nearest')
		
test_set = test_datagen.flow_from_directory('/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Test/',
											class_mode = None,
											shuffle = False,
                                            target_size = (64, 64),
                                            batch_size = 32)
		

preds = classifier.predict_generator(test_set)
labels = preds.argmax(axis = 1);

accuracy = (labels == y_test)
accuracy = accuracy.astype(int)
num = (np.shape(labels))
accuracy = accuracy.sum()
print("Correctly Classified:")
print(accuracy)
print("Out of:")
print(num[0])

