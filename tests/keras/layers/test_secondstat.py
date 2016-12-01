import pytest
import numpy as np
from scipy.linalg import logm
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras import backend as K
from keras.layers import SecondaryStatistic, WeightedProbability, O2Transform


def test_matrix_logrithm():
    data = np.random.randn(3, 10, 10)
    result = logm(data)

@keras_test
def test_secondstat():
    nb_samples = 2
    nb_steps = 8
    cols = 10
    rows = 12
    nb_filter = 3
    epsilon = 1e-4
    if K.image_dim_ordering() == 'th':
        rowvar = True
        sum_axis = 2
        input_shape = (nb_samples, nb_filter, cols, rows)
    elif K.image_dim_ordering() == 'tf':
        rowvar = False
        input_shape = (nb_samples, cols, rows, nb_filter)
        sum_axis = 1
    else:
        raise ValueError('Image Dim Ordering error {}'.format(K.image_dim_ordering()))
    data = np.random.randn(*input_shape).astype(K.floatx())
    assert len(data.shape) == 4

    a = layer_test(SecondaryStatistic,
                   input_data=data,
                   kwargs={
                       'output_dim':None,
                       'parametrized':False,
                       'init':'glorot_uniform',
                       'activation':'linear',
                       'eps':epsilon
                   },
                   input_shape=input_shape)
    data2 = data.reshape((nb_samples, nb_filter, cols * rows))
    mean = np.mean(data2, axis=sum_axis, keepdims=True)
    data2_normal = data2 - mean
    cov = np.zeros((nb_samples, nb_filter, nb_filter))
    for i in range(nb_samples):
        cov[i, :, :] = np.cov(data2_normal[i, :, :], rowvar=rowvar)
        print(cov[i, :, :].shape)
    identity = epsilon * np.eye(nb_filter)
    cov += np.expand_dims(identity, axis=0)
    # print(cov)
    # print(a)
    print(a - cov)
    assert_allclose(a, cov, rtol=1e-4)
    assert_allclose(a, cov, rtol=1e-5)
    assert_allclose(a, cov, rtol=1e-6)

#
# @keras_test
# def test_WeightProbability():
#     raise NotImplementedError
#
#
# @keras_test
# def test_O2Transform():
#     raise NotImplementedError

if __name__ == '__main__':
    # TODO Finish the testing cases for self defined layers
    pytest.main([__file__])
    # test_secondstat()