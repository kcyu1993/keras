import pytest
import numpy as np
# from numpy.testing import assert_allclose

from examples.cifar10_sndstat import cifar_fitnet_v1
from keras.utils.test_utils import image_classification, keras_test
from keras.applications.resnet50 import ResCovNet50CIFAR


nb_class = 10
input_shape = (3, 32, 32)


@keras_test
def test_ResCovNetCIFAR_nonparam():
    model = ResCovNet50CIFAR([], nb_class=10, mode=0)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR([], nb_class=10, mode=1)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR([], nb_class=10, mode=2)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR([], nb_class=10, mode=3)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)


@keras_test
def test_ResCovNetCIFAR_param():
    param = [100, 100]
    model = ResCovNet50CIFAR(param, nb_class=10, mode=0)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR(param, nb_class=10, mode=1)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR(param, nb_class=10, mode=2)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = ResCovNet50CIFAR(param, nb_class=10, mode=3)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)



if __name__ == '__main__':
    pytest.main([__file__])

