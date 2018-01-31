import keras
import numpy as np

from keras import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Convolution1D, BatchNormalization, Activation, \
    LocallyConnected1D, Dropout
from keras.optimizers import SGD, Adam

from util import load_single_train_data

model = Sequential()

# 1
model.add(BatchNormalization(input_shape=[16, 8], momentum=0.9))
model.add(Convolution1D(filters=64, kernel_size=3))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 2
model.add(Convolution1D(filters=64, kernel_size=3))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 3
model.add(LocallyConnected1D(filters=64, kernel_size=1))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 4
model.add(LocallyConnected1D(filters=64, kernel_size=1))
model.add(Dropout(0.5))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 5
model.add(Flatten())
model.add(Dense(units=512))
model.add(Dropout(0.5))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 6
model.add(Dense(units=512))
model.add(Dropout(0.5))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 7
model.add(Dense(units=128))
model.add(BatchNormalization(momentum=0.9))
model.add(Activation('relu'))

# 8
model.add(Dense(units=8, activation='softmax'))
# adam = Adam(lr=0.1, decay=0.0)
sgd = SGD(lr=0.1, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=sgd, metrics=['accuracy'])
train_x, train_y, test_x, test_y = load_single_train_data('.cache/dba/data', 5)
train_x = train_x.reshape(train_x.shape[0], 16, 8)
test_x = test_x.reshape(train_x.shape[0], 16, 8)
train_y = keras.utils.to_categorical(train_y - 1, 8)
test_y = keras.utils.to_categorical(test_y - 1, 8)

model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=28)