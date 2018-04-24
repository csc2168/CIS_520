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


# CNN Initialization
cnn = Sequential()

# 1st convolutional layer & pool
cnn.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd layer
cnn.add(Conv2D(64, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# 3rd layer
cnn.add(Conv2D(128, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Flatten())

# Full connection and dropout
cnn.add(Dense(units = 1024, activation = 'relu'))
cnn.add(Dropout(.5))
cnn.add(Dense(units = 43, activation = 'softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


test_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
		horizontal_flip=True,
        fill_mode='nearest')

training_set = train_datagen.flow_from_directory('/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Training/Images/',
                                                 target_size = (32, 32),
                                                 batch_size = 32)

val_set = test_datagen.flow_from_directory('/home/csc2168/hw/cis/CIS_520/GTSRB/Final_Training/val_images/',
                                            target_size = (32, 32),
                                            batch_size = 32)

cnn.fit_generator(training_set,
                         steps_per_epoch = 10000,
                         epochs = 6,
                         validation_data = val_set,
validation_steps = 2000)

cnn.save_weights('cnn_weights5.h5');