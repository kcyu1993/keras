
"""
Test the minc related
"""
import pytest
import numpy as np
# from numpy.testing import assert_allclose

from examples import minc_vgg19
from keras.utils.np_utils import to_categorical
from keras.utils.test_utils import get_test_data, keras_test


def image_classification(model, input_shape, nb_class=23, fit=False):
    '''
    Classify random 16x16 color images into several classes using logistic regression
    with convolutional hidden layer.
    Implement the test_image classification via basic version.

    '''
    np.random.seed(1337)
    if input_shape is None:
        input_shape = (16, 16, 3)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=100,
                                                         nb_test=50,
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
        history = model.fit(X_train, y_train, nb_epoch=1, batch_size=16,
                            validation_data=(X_test, y_test),
                            verbose=0)
        # assert(history.history['val_acc'][-1] > 0.85)


def minc_models(second=False, parametric=False):
    input_shape = (3, 224, 224)
    model = minc_vgg19.create_ResNet50(second=second, parametric=parametric)
    image_classification(model, input_shape, fit=False)


@keras_test
def test_minc_ResCov_non_para():
    print("test Resnet 50, with cov and no para")
    minc_models(True, False)


@keras_test
def test_minc_ResCov_para():
    print("test Resnet 50, with cov and no para")
    minc_models(True, True)


@keras_test
def test_minc_ResNet():
    print("test Resnet 50, with cov and no para")
    minc_models(True, False)


@keras_test
def test_minc_VGG19():
    print("test VGG19")
    model = minc_vgg19.create_VGG_original2()
    image_classification(model, input_shape=(3,224,224))


@keras_test
def test_minc_VGG19_snd():
    print("test VGG19 snd")
    model = minc_vgg19.create_VGG_snd()
    image_classification(model, input_shape=(3,224,224))

if __name__ == '__main__':
    pytest.main([__file__])
