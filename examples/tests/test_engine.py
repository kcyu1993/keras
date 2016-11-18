import pytest
import numpy as np


from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Activation, Flatten
from keras.utils.np_utils import to_categorical
from keras.utils.test_utils import get_test_data
from examples.example_engine import ExampleEngine, CVEngine

input_shape = (3, 16, 16)
nb_classes = 4


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


if __name__ == '__main__':
    pytest.main([__file__])



