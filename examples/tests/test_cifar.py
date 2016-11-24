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
    for i in range(4, 7):
        model = ResCovNet50CIFAR([], nb_class=10, mode=i)
        image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    # assert 0


@keras_test
def test_ResCovNetCIFAR_param():
    param = [100, 100]
    for i in range(4, 7):
        model = ResCovNet50CIFAR(param, nb_class=10, mode=i)
        image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    # assert 0


@keras_test
def test_fitnet_param():
    paras = [[100, ], [50, ], [100, 50]]
    for para in paras:
        model = cifar_fitnet_v1(True, para)
        image_classification(model, nb_class=10, input_shape=input_shape, fit=False)
    model = cifar_fitnet_v1(False)
    image_classification(model, nb_class=10, input_shape=input_shape, fit=False)

if __name__ == '__main__':
    pytest.main([__file__])

