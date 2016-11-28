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

import logging

from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SecondaryStatistic, WeightedProbability
from keras.layers import Convolution2D, MaxPooling2D, O2Transform
from keras.optimizers import SGD, rmsprop
from keras.utils import np_utils
from keras.utils.data_utils import get_absolute_dir_project
from keras.utils.logger import Logger
from keras.applications.resnet50 import ResNet50CIFAR, ResCovNet50CIFAR, covariance_block_original

import sys
import os

from example_engine import ExampleEngine

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

BASELINE_PATH = get_absolute_dir_project('model_saved/cifar10_baseline.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_cnn_sndstat.weights')
SND_PATH = get_absolute_dir_project('model_saved/cifar10_fitnet.weights')
LOG_PATH = get_absolute_dir_project('model_saved/log')

cifar_10 = True
if cifar_10:
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
else:
    # the data, shuffled and split between train and test sets
    # label_mode = 'fine'
    label_mode = 'fine'
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    if label_mode is 'fine':
        nb_classes = 100
    elif label_mode is 'coarse':
        nb_classes = 20

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


def cifar_fitnet_v1(second=False, parametric=[]):
    """
    Implement the fit model has 205K param
    Without any Maxout design in this version
    Just follows the general architecture

    :return: model sequential
    """
    basename = 'fitnet_v1'


    model = Sequential()
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
        basename += "_fc"
    else:
        model.add(SecondaryStatistic(name='second_layer'))
        basename += "_snd"
        if parametric is not []:
            basename += '_para-'
            for para in parametric:
                basename += str(para) + '_'
        for ind, para in enumerate(parametric):
            model.add(O2Transform(output_dim=para, name='O2transform_{}'.format(ind)))
        model.add(WeightedProbability(output_dim=nb_classes))
        model.add(Activation('softmax'))

    model.name = basename
    return model


def cifar_fitnet_v2(parametrics=[], mode=0):
    """
        Implement the fit model has 205K param
        Without any Maxout design in this version
        Just follows the general architecture

        :return: model sequential
        """
    nb_class = nb_classes
    basename = 'fitnet_v2'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'

    input_tensor = Input(input_shape)
    x = Convolution2D(16, 3, 3, border_mode='same')(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(16, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    block1_x = x

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    block2_x = x
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    block3_x = x

    cov_input = block3_x
    if mode == 0: # Original Network
        x = Flatten()(x)
        x = Dense(500)(x)
        x = Dense(nb_classes)(x)
        x = Activation('softmax')(x)
    elif mode == 1: # Original Cov_Net
        x = covariance_block_original(x, nb_class, stage=4, block='a', parametric=parametrics)
        x = Activation('softmax')(x)

    elif mode == 2: # Concat balanced
        cov_branch = covariance_block_original(cov_input, nb_class, stage=4, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 3: # Concat two softmax
        cov_branch = covariance_block_original(cov_input, nb_class, stage=4, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        cov_branch = Activation('softmax')(cov_branch)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 4: # Concat multiple branches (balanced)
        cov_branch1 = covariance_block_original(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_original(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_original(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class)(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 5: # Concat multiple 'softmax' final layers
        cov_branch1 = covariance_block_original(block1_x, nb_class, stage=2, block='a',
                                                parametric=parametrics, activation='softmax')
        cov_branch2 = covariance_block_original(block2_x, nb_class, stage=3, block='b',
                                                parametric=parametrics, activation='softmax')
        cov_branch3 = covariance_block_original(block3_x, nb_class, stage=4, block='c',
                                                parametric=parametrics, activation='softmax')
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc_softmax')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 6: # Average multiple softmax
        cov_branch1 = covariance_block_original(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_original(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_original(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='ave', name='average')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(input_tensor, x)


    model.name = basename
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
    return model


def resnet50_original():
    # img_input = Input(shape=(img_channels, img_rows, img_cols))
    # model = ResNet50(True, weights='imagenet')
    model = ResNet50CIFAR(nb_class=nb_classes)
    # let's train the model using SGD + momentum (how original).
    return model


def resnet50_snd(parametric=False):
    x, img_input = ResNet50CIFAR(False, nb_class=nb_classes)
    x = SecondaryStatistic(parametrized=False)(x)
    if parametric:
        x = O2Transform(output_dim=100, activation='relu')(x)
        x = O2Transform(output_dim=10, activation='relu')(x)
    x = WeightedProbability(output_dim=nb_classes, activation='softmax')(x)
    model = Model(img_input, x, name='ResNet50CIFAR_snd')
    return model


def run_original(load=False, save=True, verbose=1):
    model = model_original()
    fit_model(model, load=load, save=save, verbose=verbose)


def run_snd_layer(load=False, save=True, parametric=True):
    model = model_snd(parametric)
    fit_model(model, load=load, save=save)


def run_fitnet_layer(load=False, save=True, second=False, parametric=True, verbose=1):
    model = cifar_fitnet_v1(second=second, parametric=parametric)
    fit_model(model, load=load, save=save, verbose=verbose)


def run_resnet50_original(verbose=1):
    model = resnet50_original()
    fit_model(model, verbose=verbose)


def run_resnet_snd(parametric=True, verbose=1):
    model = resnet50_snd(parametric)
    fit_model(model, load=True, save=True, verbose=verbose)


def run_resnet_merge(parametrics=[], verbose=1, start=0, stop=3):
    for mode in range(start, stop):
        model = ResCovNet50CIFAR(parametrics=parametrics, nb_class=nb_classes, mode=mode)
        fit_model(model, load=False, save=True, verbose=verbose)

def run_fitnet_merge(parametrics=[], verbose=1, start=0, stop=6):
    for mode in range(start, stop):
        model = cifar_fitnet_v2(parametrics, mode)
        fit_model(model, load=False, save=True, verbose=verbose)


def fit_model(model, load=False, save=True, verbose=1):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    save_log = True
    engine = ExampleEngine([X_train, Y_train], model, [X_test, Y_test],
                           load_weight=load, save_weight=save, save_log=save_log,
                           lr_decay=True, early_stop=False,
                           batch_size=batch_size, nb_epoch=nb_epoch, title='cifar10', verbose=verbose)

    if save_log:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=batch_size, nb_epoch=nb_epoch, augmentation=data_augmentation)
    score = engine.model.evaluate(X_test, Y_test, verbose=0)
    engine.plot_result('loss')
    engine.plot_result('acc')
    print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if save_log:
        sys.stdout = engine.stdout.close()


def run_routine1():
    print('Test original')
    run_original()
    # print("Test snd model without pre-loading data")
    # run_snd_layer()
    # print('Test snd model with pre-trained')
    # run_snd_layer(loads=True)

def run_routine2():
    """
    Train model snd_layer without pretraining
    :return:
    """
    print("test snd layer")
    sys.stdout = Logger(LOG_PATH + '/cifar_routine2.log')
    run_snd_layer(load=False)


def run_routine3():
    sys.stdout = Logger(LOG_PATH + '/cifar_routine3.log')
    run_snd_layer(load=True, save=False)


def run_routine4():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    run_resnet50_original(2)


def run_routine5():
    # sys.stdout = Logger(LOG_PATH + '/cifar_routine4.log')
    # run_fitnet_layer(load=True, verbose=1)
    run_fitnet_layer(second=True, load=False, verbose=2)


def run_routine6():
    logging.info("Run holistic test for merged model")
    print("Run holistic test for merged model")
    run_resnet_merge([],2)
    run_resnet_merge([50],2)
    run_resnet_merge([100],2)
    run_resnet_merge([100, 50],2)


def run_routine7():
    logging.info("Run holistic test for complex merged model 2")
    print("Run holistic test for complex merged model 2")
    print("mode 5 for para reset")
    # run_resnet_merge([],2, 5, 6)
    run_resnet_merge([50],2, 5, 6)
    run_resnet_merge([100],2, 5, 6)
    run_resnet_merge([100, 50],2, 5, 6)


def run_routine8():
    """ Fitnet complete trial """
    # paras = [[100,], [50,], [100,50]]
    # paras = [[50,], [100,50]]
    paras = [[100,50]]
    print("Run routine 8 Epoch {} for para {}".format(nb_epoch, paras))
    list_model = []
    for para in paras:
        model = cifar_fitnet_v1(True, para)
        list_model.append(model)
    list_model.append(cifar_fitnet_v1(False, []))
    for model in list_model:
        print('test model {}'.format(model.name))
        fit_model(model, load=False, save=True, verbose=2)


def run_routine9():
    """
    Fitnet v2 complete test

    Returns
    -------

    """
    print("routine 9")
    run_fitnet_merge([])


if __name__ == '__main__':
    nb_epoch = 400
    # run_routine1()
    # print('test')
    # run_routine1()
    # run_routine5()
    # run_routine6()
    # sys.stdout = logging
    # run_routine7()
    # run_resnet50_original(2)
    # run_resnet_snd(True, verbose=1)
    # plot_models()
    # run_routine8()
    run_routine9()
    # plot_results()
