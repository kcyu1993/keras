'''Trains a simple convnet on the MNIST dataset.

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

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import SecondaryStatistic, WeightedProbability, O2Transform
from keras.layers import PReLU
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project
from keras import backend as K
from keras import optimizers

batch_size = 128
nb_classes = 10
nb_epoch = 50

ratio = 0.1     # Sepcify the ratio of true data used.

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

''' Modify this to read local data, reference from MNIST example.'''
# the data, shuffled and split between train and test sets
useInfiMNIST = False
if useInfiMNIST:
    print('Use infi mnist / 80k mnist')
    X_train, y_train = mnist.load_infimnist('mnist8m')
    X_test, y_test = mnist.load_infimnist('t10k')
else:
    print('Use original keras mnist')
    data = mnist.load_data()

    if len(data) == 3:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data
    else:
        (X_train, y_train), (X_test, y_test) = data

nb_train = len(X_train)
nb_valid = 200 # len(X_valid)
nb_test = len(X_test)


X_train = X_train[:nb_train,]
y_train = y_train[:nb_train,]
# X_valid = X_valid[:nb_valid,]
# y_valid = y_valid[:nb_valid,]
X_test = X_test[:nb_test,]
y_test = y_test[:nb_test,]

# Data loading and manipulation.
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print("Load in total {} train sample and {} test sample".format(len(X_train), len(X_test)))


def mnist_fitnet_v1():
    """
    Implement the fit model has 205K param
    Without any Maxout design in this version
    Just follows the general architecture

    :return: model sequential
    """
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,4), strides=(2,2)))
    # model.add(Dropout(0.25))
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

    model.add(Convolution2D(12, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(12, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dense(500))
    model.add(Dense(10, activation='softmax'))
    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def mnist_model1():
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
    # model.add(SecondaryStatistic(activation='linear'))
    # model.add(WeightedProbability(10, activation='linear', init='normal'))
    model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    # model.add(Activation('relu'))
    model.add(Activation('softmax'))


    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    #
    # model.add(Flatten())

    # Define the optimizers:
    # opt = optimizers.sgd(lr=0.01)
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

    model.add(SecondaryStatistic(activation='linear'))
    model.add(WeightedProbability(10, activation='linear', init='normal'))
    model.add(Activation('softmax'))
    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def model_with_merge():
    """
    Build a model concat both O1 and O2 information
    :return: compiled model
    """
    img_input = Input(shape=input_shape)



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
    model.add(SecondaryStatistic(activation='linear', parametrized=False, output_dim=10, init='normal'))
    model.add(O2Transform(activation='linear', output_dim=100))
    model.add(Activation('relu'))
    # model.add(PReLU(weights=[0.25]))
    model.add(WeightedProbability(10, activation='relu', init='normal'))
    # model.add(Flatten())

    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('relu'))
    model.add(Activation('softmax'))

    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    #
    # model.add(Flatten())
    # Define the optimizers:
    # opt = optimizers.sgd(lr=0.01)
    opt = optimizers.rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def main_loop():
    print("fitting the whole model ")
    model = mnist_model1()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Save the model to another location.
    output_name = 'model_saved/mnist_cnn_sndstat.weights'
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


def mnist_baseline_comparison():
    """
    Comparison CNN + FC 10 vs CNN + Cov WP.
    :return:
    """
    print('BEGIN TEST 1')
    model1 = mnist_model1()
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
    model = mnist_model1()
    output_name = 'model_saved/mnist_cnn_sndstat.weights'
    out_dir = get_absolute_dir_project(output_name)
    model.load_weights(out_dir)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def test_loader():
    # Save the model to another location.
    output_name = 'model_saved/mnist_cnn_sndstat.weights'
    out_dir = get_absolute_dir_project(output_name)
    print('saving model to location -> {} '.format(out_dir))


def test_para():
    model = model_parametrized()
    model.summary()
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':

    # main_loop()
    # test_loader()
    # evaluate_model()

    mnist_baseline_comparison()
    # print("test parametrized layer false")
    # test_para()
