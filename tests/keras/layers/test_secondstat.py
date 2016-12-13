import pytest
import numpy as np
from scipy.linalg import logm
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras import backend as K
from keras.layers import SecondaryStatistic, WeightedProbability, O2Transform


# def test_matrix_logrithm():
#     data = np.random.randn(3, 10, 10)
#     result = logm(data)

@keras_test
def test_secondstat():
    nb_samples = 2
    nb_steps = 8
    cols = 10
    rows = 12
    nb_filter = 3
    epsilon = 0
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
    print('image ordering {}'.format(K.image_dim_ordering()))
    data = np.random.randn(*input_shape).astype(K.floatx())
    assert len(data.shape) == 4

    a = layer_test(SecondaryStatistic,
                   input_data=data,
                   kwargs={
                       'init':'glorot_uniform',
                       'activation':'linear',
                       'eps':epsilon,
                       'cov_mode':'channel'
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


def test_variance():
    nb_samples = 2
    nb_steps = 8
    cols = 10
    rows = 12
    nb_filter = 3
    epsilon = 0
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
    print('image ordering {}'.format(K.image_dim_ordering()))
    data = np.random.randn(*input_shape).astype(K.floatx())
    assert len(data.shape) == 4

    # Stimulate the theano backend operation
    a = K.variable(data)
    ta = K.reshape(a, (-1, nb_filter, cols*rows))
    ta_mean = K.mean(ta, axis=2, keepdims=True)
    ta_normal = ta - ta_mean
    taa = K.sum(K.multiply(K.expand_dims(ta_normal, dim=1), K.expand_dims(ta_normal, dim=2)), axis=3)
    taa /= cols * rows - 1

    if K.backend() == 'tensorflow':
        sess = K.get_session()
        sess.as_default()
        res = sess.run(taa)
    else:
        res = taa.eval()
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
    print(res - cov)
    assert_allclose(res, cov, rtol=1e-4)
    assert_allclose(res, cov, rtol=1e-5)
    assert_allclose(res, cov, rtol=1e-6)

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
    import os
    os.environ["KERAS_BACKEND"] = 'tensorflow'
    pytest.main([__file__])
    # test_secondstat()