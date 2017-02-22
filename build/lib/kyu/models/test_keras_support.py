# from kyu.utils.cov_reg import FobNormRegularizer
from keras.layers.secondstat import SecondaryStatistic, WeightedVectorization

from keras.models import Sequential
from keras.layers import Input, Convolution2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import ActivityRegularization
from keras.layers import Embedding
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers
import keras.backend as K
import numpy as np


# input image dimensions
img_rows, img_cols = 28, 28
nb_classes = 10
batch_size = 128
nb_epoch = 5
weighted_class = 9
standard_weight = 1
high_weight = 5
max_train_samples = 5000
max_test_samples = 1000


def get_data():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:max_train_samples]
    X_test = X_test[:max_test_samples]
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_train = y_train[:max_train_samples]
    y_test = y_test[:max_test_samples]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    test_ids = np.where(y_test == np.array(weighted_class))[0]
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    return (X_train, Y_train), (X_test, Y_test), test_ids, input_shape


def create_model(input_shape, weight_reg=None, activity_reg=None, cov_reg=None):
    model = Sequential()
    model.add(Convolution2D(16,3,3, input_shape=input_shape, W_regularizer='l1'))
    model.add(Activation('relu'))
    model.add(SecondaryStatistic(cov_regularizer=cov_reg))
    model.add(WeightedVectorization(10))
    model.add(Dense(10, W_regularizer=weight_reg,
                    activity_regularizer=activity_reg))
    model.add(Activation('softmax'))
    return model

def test_cov_reg():
    (X_train, Y_train), (X_test, Y_test), test_ids, input_shape = get_data()
    for reg in [None, 'Fob']:
        model = create_model(input_shape, cov_reg=reg)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.summary()
        # assert len(model.losses) == 1
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch, verbose=1)
        model.evaluate(X_test[test_ids, :], Y_test[test_ids, :], verbose=0)


if __name__ == '__main__':
    test_cov_reg()
