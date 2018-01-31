import keras
import sys
from keras.models import load_model

from util import load_single_train_data

if __name__ == '__main__':

    if (len(sys.argv) != 2):
        print('请输入subject id')

    model = load_model('srep_all.h5')

    train_x, train_y, test_x, test_y = load_single_train_data('.cache2/dba/data', int(sys.argv[1]))
    train_x = train_x.reshape(train_x.shape[0], 16, 8)
    test_x = test_x.reshape(train_x.shape[0], 16, 8)
    train_y = keras.utils.to_categorical(train_y - 1, 8)
    test_y = keras.utils.to_categorical(test_y - 1, 8)

    model.fit(train_x, train_y, batch_size=1000, validation_data=(test_x, test_y), epochs=100)
    model.save('srep_all.h5')