'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SecondaryStatistic, WeightedProbability
from keras.layers import Convolution2D, MaxPooling2D, O2Transform
from keras.optimizers import SGD, rmsprop
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project, get_weight_path
from keras.utils.logger import Logger
from keras.applications.resnet50 import ResNet50CIFAR


import sys
import os

from example_engine import ExampleEngine

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_fitnet.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

input_shape = X_train.shape[1:]


def cifar_fitnet_v1(second=False, parametric=True):
    """
    Implement the fit model has 205K param
    Without any Maxout design in this version
    Just follows the general architecture

    :return: model sequential
    """
    model = Sequential(name='fitnet_v1')
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(48, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    if not second:
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dense(nb_classes, activation='softmax'))
        model.name = model.name + "_original"
    else:
        model.add(SecondaryStatistic(name='second_layer'))
        if parametric:
            model.add(O2Transform(output_dim=100, name='O2transform_1'))
            model.add(O2Transform(output_dim=50, name='O2transform_2'))
        model.add(WeightedProbability(output_dim=nb_classes))
        model.add(Activation('softmax'))
        model.name = model.name + "_second"

    opt = rmsprop()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model




def model_original():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def model_snd(parametric=True):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(SecondaryStatistic(activation='linear'))
    if parametric:
        model.add(O2Transform(activation='relu', output_dim=100))
    model.add(WeightedProbability(10,activation='linear', init='normal'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def resnet50_original():
    # img_input = Input(shape=(img_channels, img_rows, img_cols))
    # model = ResNet50(True, weights='imagenet')
    model = ResNet50CIFAR(nb_class=nb_classes)
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def resnet50_snd(parametric=False):
    x, img_input = ResNet50CIFAR(False, nb_class=nb_classes)
    x = SecondaryStatistic(parametrized=False)(x)
    if parametric:
        x = O2Transform(output_dim=100, activation='relu')(x)
        x = O2Transform(output_dim=10, activation='relu')(x)
    x = WeightedProbability(output_dim=nb_classes, activation='softmax')(x)
    model = Model(img_input, x, name='ResNet50CIFAR_snd')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def test_original(load=False, save=True, verbose=1):
    model = model_original()
    fit_model(model, load=load, save=save, verbose=verbose)


def test_snd_layer(load=False, save=True, parametric=True):
    model = model_snd(parametric)
    fit_model(model, load=load, save=save)


def test_fitnet_layer(load=False, save=True, second=False, parametric=True, verbose=1):
    model = cifar_fitnet_v1(second=second, parametric=parametric)
    fit_model(model, load=load, save=save, verbose=verbose)


def test_resnet50_original(verbose=1):
    model = resnet50_original()
    fit_model(model, verbose=verbose)


def test_resnet_snd(parametric=True, verbose=1):
    model = resnet50_snd(parametric)
    fit_model(model, load=True, save=True, verbose=verbose)

def test_merge_model():
    raise NotImplementedError


def fit_model(model, load=False, save=True, verbose=1):
    engine = ExampleEngine([X_train, Y_train], model, [X_test, Y_test],
                           load_weight=load, save_weight=save,
                           batch_size=batch_size, nb_epoch=nb_epoch, title='cifar10', verbose=verbose)
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation)
    score = engine.model.evaluate(X_test, Y_test, verbose=0)
    # engine.plot_result('loss')
    # engine.plot_result('acc')
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))


def test_routine1():
    print('Test original')
    test_original()
    # print("Test snd model without pre-loading data")
    # test_snd_layer()
    # print('Test snd model with pre-trained')
    # test_snd_layer(loads=True)

def test_routine2():
    """
    Train model snd_layer without pretraining
    :return:
    """
    print("test snd layer")
    sys.stdout = Logger(LOG_PATH + '/cifar_routine2.log')
    test_snd_layer(load=False)


def test_routine3():
    sys.stdout = Logger(LOG_PATH + '/cifar_routine3.log')
    test_snd_layer(load=True, save=False)


def test_routine4():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    test_resnet50_original(2)


def test_routine5():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    # test_fitnet_layer(load=True, verbose=1)
    test_fitnet_layer(second=True, load=False, verbose=2)

if __name__ == '__main__':

    nb_epoch = 50
    # test_routine1()
    # print('test')
    # test_routine1()
    test_routine5()
    # test_resnet50_original(2)
    # test_resnet_snd(True, verbose=1)
