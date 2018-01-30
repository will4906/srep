import keras

from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, LocallyConnected2D, Activation, BatchNormalization, Dropout, Dense, Flatten, \
    MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import load_model

from data import load_whole_train_data


def learning_rate_func(index):
    if index >= 16 and index < 24:
        return 0.01
    elif index >= 24:
        return 0.001
    else:
        return 0.1


def get_srep_model():
    '''
    根据论文实现的模型
    TODO:Locallyconnected2D可能不尽人意
    '''
    learning_rate_scheduler = LearningRateScheduler(learning_rate_func)
    model = Sequential()

    model.add(BatchNormalization(input_shape=[1,20, 8,], momentum=0.9))

    model.add(Conv2D(64, (3, 3), use_bias=False,
                     padding='same', strides=(1, 1)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), use_bias=False,
                     padding='same', strides=(1, 1)))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(LocallyConnected2D(64, (1, 1), use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(LocallyConnected2D(64, (1, 1), use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(units=512, use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=512, use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=128, use_bias=False))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Dense(units=8, use_bias=True))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=0.0)

    adam = Adam(lr=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model, learning_rate_scheduler


def load_exit_model(path):
    return load_model(path), LearningRateScheduler(learning_rate_func)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_whole_train_data()

    train_y = keras.utils.to_categorical(train_y - 1, 8)
    test_y = keras.utils.to_categorical(test_y - 1, 8)
    model, learning_rate_scheduler = get_srep_model()

    model.fit(train_x, train_y, batch_size=1000, epochs=28, validation_data=(test_x, test_y),
              callbacks=[learning_rate_scheduler])