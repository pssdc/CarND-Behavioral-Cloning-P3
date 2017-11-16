import os
import csv
# This code is more or less same as sample code provided in the lab. 
from sklearn.utils import shuffle
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split the code into training and validation samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import pandas as pd
import matplotlib.image as mpimg
from scipy.misc import imresize

# define generator to process the data on the fly instead of loading the whole data set in the memory.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = mpimg.imread(name)
                # downsize the image to speed up the training and have a smaller model.h5 
                center_image = imresize(center_image, (30, 60, 3))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # create the array 
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

###################################################################################################
# Starting Keras network
###################################################################################################

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(BatchNormalization(input_shape=(30, 60, 3)))

#-------------------------------------------------------------------------
#Convolution Layer 1 
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
# RELU Activation
model.add(Activation('relu'))
# Dropout
model.add(Dropout(0.2))

#-------------------------------------------------------------------------
# Convolution Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
# RELU Activation
model.add(Activation('relu'))

#-------------------------------------------------------------------------
# Convolution Layer 3
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
# RELU Activation
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#-------------------------------------------------------------------------
# Flatten 
model.add(Flatten())
# Dropout
model.add(Dropout(0.4))

#-------------------------------------------------------------------------
# Fully Connected Layer 1
model.add(Dense(128))
# RELU Activation
model.add(Activation('relu'))
# Dropout
model.add(Dropout(0.4))

sudo apt-get install pandoc#-------------------------------------------------------------------------
# Fully connected layer 2
model.add(Dense(64))
# RELU Activation
model.add(Activation('relu'))

#-------------------------------------------------------------------------
# Fully connected layer 3
model.add(Dense(32))
# RELU Activation
model.add(Activation('relu'))

model.add(Dense(1))
#-------------------------------------------------------------------------

############################ Ending Keras network ##################################################

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')

model.summary()
