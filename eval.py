import os

import keras

from keras.models import load_model

from util import load_single_train_data

aver = 0.0
subindex = 0
for i in range(1, 19):
    subindex += 1
    train_x, train_y, test_x, test_y = load_single_train_data('.cache2/dba/data', i)
    train_x = train_x.reshape(train_x.shape[0], 16, 8)
    test_x = test_x.reshape(train_x.shape[0], 16, 8)
    train_y = keras.utils.to_categorical(train_y - 1, 8)
    test_y = keras.utils.to_categorical(test_y - 1, 8)

    model = load_model('save_single' + os.sep + 'srep_' + str(i) + '.h5')
    result = model.evaluate(test_x, test_y)
    aver += result[1]

print(aver / subindex)