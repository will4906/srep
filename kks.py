import keras
import sys

import os
from keras.models import load_model

from util import load_single_train_data

if __name__ == '__main__':

    if (len(sys.argv) != 2):
        print('请输入subject id')

    if os.path.exists('save_single') is False:
        os.mkdir('save_single')

    if os.path.exists('save_single' + os.sep + 'srep_' + sys.argv[1] + '.h5') is False:
        model = load_model('srep_all.h5')

        train_x, train_y, test_x, test_y = load_single_train_data('.cache2/dba/data', int(sys.argv[1]))
        train_x = train_x.reshape(train_x.shape[0], 16, 8)
        test_x = test_x.reshape(train_x.shape[0], 16, 8)
        train_y = keras.utils.to_categorical(train_y - 1, 8)
        test_y = keras.utils.to_categorical(test_y - 1, 8)

        model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=55)
        model.save('save_single' + os.sep + 'srep_' + sys.argv[1] + '.h5')