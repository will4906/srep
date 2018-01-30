import keras
import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from data import load_whole_train_data

model = Sequential()

model.add(Dense(input_dim=160, units=1024, activation='relu'))
model.add(Dense(units=8, activation='softmax'))
sgd = SGD(lr=0.1, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=sgd, metrics=['accuracy'])
train_x, train_y, test_x, test_y = load_whole_train_data()

train_y = keras.utils.to_categorical(train_y - 1, 8)
test_y = keras.utils.to_categorical(test_y - 1, 8)

model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=50)
