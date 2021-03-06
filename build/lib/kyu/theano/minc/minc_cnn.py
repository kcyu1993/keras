'''Trains a simple convnet on the minc dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
19 seconds per epoch on a Titan Z GPU with CuDNN and CnMeM
39 seconds per epoch on a Titan Z GPU
=================
auther: Kaicheng Yu
Modification on network structure, add the two additional layer,
1 conv to cov
1 cov to prob
Basic for debugging purpose.

Issue 2016.10.11
After various test, the model didn't work when combing the two layers
however, it is working only with Second-order statistics.

Suspection is the correctness of the layer weighted parameters.
It is necessary to test, with

Issue 2016.10.24
Trying the new experiments:
    Pre-trained the model, save the point

'''

from __future__ import print_function
import numpy as np


np.random.seed(1337)  # for reproducibility

from keras.datasets import minc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import SecondaryStatistic, WeightedVectorization
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project
from keras import backend as K
from keras import optimizers

batch_size = 128
nb_classes = 10
nb_epoch = 50

ratio = 0.1     # Sepcify the ratio of true data used.

# input image dimensions
img_rows, img_cols = 362, 362
# input image channel
img_channel = 3
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

''' Modify this to read local data, reference from minc example.'''
# the data, shuffled and split between train and test sets
loader = minc.Minc2500()
data = loader.loadfromfile("minc-2500.plk.hdf5")

if len(data) == 3:
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data
else:
    (X_train, y_train), (X_test, y_test) = data

nb_train = len(X_train)
nb_valid = 200 # len(X_valid)
nb_test = len(X_test)


# X_train = X_train[:nb_train,]
# y_train = y_train[:nb_train,]
# X_valid = X_valid[:nb_valid,]
# y_valid = y_valid[:nb_valid,]
# X_test = X_test[:nb_test,]
# y_test = y_test[:nb_test,]
input_shape = (img_channel, img_rows, img_cols)
# Data loading and manipulation.
# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (img_channel, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, img_channel)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print("Load in total {} train sample and {} test sample".format(len(X_train), len(X_test)))

def minc_model1():
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # Start of secondary layer
    model.add(Flatten())

    model.add(Dense(nb_classes))
    # model.add(Activation('relu'))
    model.add(Activation('softmax'))

    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def model_without_dense():
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # Start of secondary layer
    model.add(SecondaryStatistic(activation='linear'))
    model.add(WeightedVectorization(10, activation='linear', init='normal'))

    model.add(Activation('softmax'))
    # opt = optimizers.sgd(lr=0.01)
    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def test_more_parameters():
    print("Model 1 ")


def model_parametrized():
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # Start of secondary layer
    # print('Adding secondary statistic layer ')
    model.add(SecondaryStatistic(activation='linear', parametrized=True, output_dim=10, init='normal'))
    model.add(WeightedVectorization(10, activation='linear', init='normal'))

    model.add(Activation('softmax'))
    # Define the optimizers:
    # opt = optimizers.sgd(lr=0.01)
    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def main_loop():
    print("fitting the whole model ")
    model = minc_model1()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Save the model to another location.
    output_name = 'model_saved/minc_cnn_sndstat.weights'
    out_dir = get_absolute_dir_project(output_name)
    print('saving model to location -> {} '.format(out_dir))
    model.save_weights(out_dir)
    model2 = model_without_dense()
    model2.load_weights(out_dir, by_name=True)
    model2.summary()
    model2.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    score = model2.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def minc_baseline_comparison():
    """
    Comparison CNN + FC 10 vs CNN + Cov WP.
    :return:
    """
    print('BEGIN TEST 1')
    model1 = minc_model1()
    model1.summary()
    model1.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    score = model1.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # print("BEGIN TEST 2 ")
    # model2 = model_without_dense()
    # model2.summary()
    # model2.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    # score = model2.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])


def evaluate_model():
    model = minc_model1()
    output_name = 'model_saved/minc_cnn_sndstat.weights'
    out_dir = get_absolute_dir_project(output_name)
    model.load_weights(out_dir)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def test_loader():
    # Save the model to another location.
    output_name = 'model_saved/minc_cnn_sndstat.weights'
    out_dir = get_absolute_dir_project(output_name)
    print('saving model to location -> {} '.format(out_dir))


def test_model(model):
    model.summary()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def test_para():
    model = model_parametrized()
    test_model(model)


def test_non_para():
    model = model_without_dense()
    test_model(model)

if __name__ == '__main__':

    # main_loop()
    # test_loader()
    # evaluate_model()

    # minc_baseline_comparison()
    print("test parametrized layer false")
    test_non_para()
    # test_para()

