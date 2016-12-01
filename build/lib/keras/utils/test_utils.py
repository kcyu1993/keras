import tempfile

import numpy as np
from numpy.testing import assert_allclose
import inspect
import functools

from .np_utils import to_categorical
from ..engine import Model, Input
from ..models import Sequential, model_from_json
from .. import backend as K


def get_test_data(nb_train=1000, nb_test=500, input_shape=(10,),
                  output_shape=(2,),
                  classification=True, nb_class=2):
    '''
        classification=True overrides output_shape
        (i.e. output_shape is set to (1,)) and the output
        consists in integers in [0, nb_class-1].

        Otherwise: float output with shape output_shape.
    '''
    nb_sample = nb_train + nb_test
    if classification:
        y = np.random.randint(0, nb_class, size=(nb_sample,))
        X = np.zeros((nb_sample,) + input_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((nb_sample,))
        X = np.zeros((nb_sample,) + input_shape)
        y = np.zeros((nb_sample,) + output_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:nb_train], y[:nb_train]), (X[nb_train:], y[nb_train:])


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, fixed_batch_size=False):
    '''Test routine for a layer with a single input tensor
    and single output tensor.
    '''
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data = (10 * np.random.random(input_shape)).astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape

    if expected_output_dtype is None:
        if input_dtype is None:
            input_dtype = input_data.dtype
        expected_output_dtype = input_dtype

    # instantiation
    layer = layer_cls(**kwargs)
    # test get_weights , set_weights
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    if fixed_batch_size:
        x = Input(batch_shape=input_shape, dtype=input_dtype)
    else:
        x = Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)

    print("layer eps {}".format(layer.eps))

    assert K.dtype(y) == expected_output_dtype

    model = Model(input=x, output=y)
    model.compile('rmsprop', 'mse')

    expected_output_shape = layer.get_output_shape_for(input_shape)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert expected_output_shape == actual_output_shape
    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization
    model_config = model.get_config()
    model = Model.from_config(model_config)
    model.compile('rmsprop', 'mse')

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    model = Sequential()
    model.add(layer)
    model.compile('rmsprop', 'mse')
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert expected_output_shape == actual_output_shape
    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test JSON serialization
    json_model = model.to_json()
    model = model_from_json(json_model)

    # test save weights to file
    _, fname = tempfile.mkstemp(suffix='.weights')
    model.save_weights(fname)
    model.load_weights(fname)
    model.load_weights(fname, by_name=True)

    # for further checks in the caller function
    return actual_output


def keras_test(func):
    '''Clean up after tensorflow tests.
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if K._BACKEND == 'tensorflow':
            K.clear_session()
        return output
    return wrapper


def image_classification(model, input_shape, nb_class=23,
                         fit=False,
                         nb_train=500, nb_test=200,
                         nb_epoch=10, batch_size=16):
    """
    Classify random color images into several classes using logistic regression
    with convolutional hidden layer.
    Implement the test_image classification via basic version.

    :param model:       Any classification model
    :param input_shape: Raw image input
    :param nb_class:    output prediction classes
    :return:
    """
    '''
    '''
    np.random.seed(1337)
    if input_shape is None:
        input_shape = (16, 16, 3)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=nb_train,
                                                         nb_test=nb_test,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         nb_class=nb_class)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    if fit:
        history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            verbose=0)
        assert history is not None
