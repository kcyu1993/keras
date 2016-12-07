import pytest
import numpy as np
import keras.backend as K
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Activation, Flatten
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.utils.test_utils import get_test_data, keras_test
from kyu.utils.examples.example_engine import ExampleEngine, CVEngine

input_shape = (3, 28, 28)
nb_classes = 10


def create_model(kernel_size, nb_filters):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))

    # Start of secondary layer
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


def create_models():
    """
    Create some arbitrary easy models

    Returns
        [models]
    -------

    """
    models = []
    models.append(create_model([3,3],1))
    models.append(create_model([3,3],2))
    models.append(create_model([3,3],3))
    models.append(create_model([3,3],4))
    return models


def creat_mnist_data():
    img_rows, img_cols = 28, 28
    data = mnist.load_data()

    if len(data) == 3:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data
    else:
        (X_train, y_train), (X_test, y_test) = data

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

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print("Load in total {} train sample and {} test sample".format(len(X_train), len(X_test)))
    return (X_train, Y_train), (X_test, Y_test)


def create_test_data():
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=500,
                                                         nb_test=200,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         nb_class=nb_classes)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def test_cv_engine():
    train, valid = create_test_data()
    models = create_models()
    engine = CVEngine(models, train, validation=valid, load_weight=True, save_weight=True,
                      verbose=2, save_log=False)



@keras_test
def test_mnist_engine():
    train, test = creat_mnist_data()
    model = create_model([3,3], 4)
    # Holistic test for engine, including multiple parts
    engine = ExampleEngine(train, model, validation=test,
                           load_weight=True, save_weight=True,
                           lr_decay=True, early_stop=True,
                           save_per_epoch=True, title='MNIST_trial_engine')
    engine.fit(batch_size=32, nb_epoch=20, augmentation=True)
    engine.plot_result('acc')
    engine.plot_result('loss')
    tmpfile = engine.save_history(engine.history)
    history = engine.load_history(tmpfile)
    assert history.history == engine.history.history


if __name__ == '__main__':
    pytest.main([__file__])



