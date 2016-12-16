import pytest
import numpy as np
from scipy.linalg import logm
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras import backend as K
from keras.layers import SecondaryStatistic, WeightedProbability, O2Transform, LogTransform

epsilon = 1e-4

def get_mat_log(data):
    if K.backend() == 'theano':
        # construct theano tensor operation
        from theano.tensor.nlinalg import svd, diag
        from theano.tensor.elemwise import Elemwise
        from theano.scalar import log
        import theano.tensor as T
        # assert u.eval() == v.eval()
        da = K.variable(data)
        u,d,v = svd(da)
        d += epsilon
        inner = diag(T.log(d))
        # print(inner.eval())
        result2 = T.dot(u, T.dot(inner, v))

        print("U shape {} V shape {}".format(u.eval().shape, v.eval().shape))
        print("D matrix {}".format(d.eval()))
        assert np.allclose(u.eval(), v.eval().transpose())
        return result2.eval()

#
# def test_matrix_logrithm():
#
#     # Creation of data
#     data = np.random.randn(10, 10)
#     # make sure the data is symmetric
#     data = np.dot(data.transpose(), data)
#
#     # Assure the numerical stability and generate the result
#     data1 = data + epsilon * np.eye(10)
#     result = logm(data1)
#     # numpy operation (data = u * diag(d) * v
#     u, d, v = np.linalg.svd(data)
#     d += epsilon
#     d = np.diag(np.log(d))
#     result3 = np.dot(u, np.dot(d, v))
#     result2 = get_mat_log(data)
#     # print (result2, result)
#     print(result - result2)
#     assert np.allclose(result3, result) # pass
#     assert np.allclose(result2, result3, atol=1e-6)
#     assert np.allclose(result, result2, atol=1e-6)


def test_log_transform_layer():
    # generate 3D data tensor
    input_shape = (2,10,10)
    data = np.random.randn(*input_shape).astype(K.floatx())
    for i in range(data.shape[0]):
        data[i] = data[i].transpose().dot(data[i])
    # Generate corresponding 3D Tensor
    # k_tensor = K.variable(data)
    res = layer_test(LogTransform,
                     input_data=data,
                     kwargs={'epsilon': epsilon},
                     input_shape=input_shape
                     )
    res2 = np.zeros(shape=input_shape)
    for i in range(input_shape[0]):
        res2[i] = logm(data[i] + epsilon * np.eye(input_shape[1]))

    assert np.allclose(res, res2)

#
# @keras_test
# def test_secondstat():
#     nb_samples = 2
#     nb_steps = 8
#     cols = 10
#     rows = 12
#     nb_filter = 3
#     epsilon = 0
#     if K.image_dim_ordering() == 'th':
#         rowvar = True
#         sum_axis = 2
#         input_shape = (nb_samples, nb_filter, cols, rows)
#     elif K.image_dim_ordering() == 'tf':
#         rowvar = False
#         input_shape = (nb_samples, cols, rows, nb_filter)
#         sum_axis = 1
#     else:
#         raise ValueError('Image Dim Ordering error {}'.format(K.image_dim_ordering()))
#     print('image ordering {}'.format(K.image_dim_ordering()))
#     data = np.random.randn(*input_shape).astype(K.floatx())
#     assert len(data.shape) == 4
#
#     a = layer_test(SecondaryStatistic,
#                    input_data=data,
#                    kwargs={
#                        'init':'glorot_uniform',
#                        'activation':'linear',
#                        'eps':epsilon,
#                        'cov_mode':'channel'
#                    },
#                    input_shape=input_shape)
#
#     data2 = data.reshape((nb_samples, nb_filter, cols * rows))
#     mean = np.mean(data2, axis=sum_axis, keepdims=True)
#     data2_normal = data2 - mean
#     cov = np.zeros((nb_samples, nb_filter, nb_filter))
#     for i in range(nb_samples):
#         cov[i, :, :] = np.cov(data2_normal[i, :, :], rowvar=rowvar)
#         print(cov[i, :, :].shape)
#     identity = epsilon * np.eye(nb_filter)
#     cov += np.expand_dims(identity, axis=0)
#     # print(cov)
#     # print(a)
#     print(a - cov)
#     assert_allclose(a, cov, rtol=1e-4)
#     assert_allclose(a, cov, rtol=1e-5)
#     assert_allclose(a, cov, rtol=1e-6)
#
#
# def test_variance():
#     nb_samples = 2
#     nb_steps = 8
#     cols = 10
#     rows = 12
#     nb_filter = 3
#     epsilon = 0
#     if K.image_dim_ordering() == 'th':
#         rowvar = True
#         sum_axis = 2
#         input_shape = (nb_samples, nb_filter, cols, rows)
#     elif K.image_dim_ordering() == 'tf':
#         rowvar = False
#         input_shape = (nb_samples, cols, rows, nb_filter)
#         sum_axis = 1
#     else:
#         raise ValueError('Image Dim Ordering error {}'.format(K.image_dim_ordering()))
#     print('image ordering {}'.format(K.image_dim_ordering()))
#     data = np.random.randn(*input_shape).astype(K.floatx())
#     assert len(data.shape) == 4
#
#     # Stimulate the theano backend operation
#     a = K.variable(data)
#     ta = K.reshape(a, (-1, nb_filter, cols*rows))
#     ta_mean = K.mean(ta, axis=2, keepdims=True)
#     ta_normal = ta - ta_mean
#     taa = K.sum(K.multiply(K.expand_dims(ta_normal, dim=1), K.expand_dims(ta_normal, dim=2)), axis=3)
#     taa /= cols * rows - 1
#
#     if K.backend() == 'tensorflow':
#         sess = K.get_session()
#         sess.as_default()
#         res = sess.run(taa)
#     else:
#         res = taa.eval()
#     data2 = data.reshape((nb_samples, nb_filter, cols * rows))
#     mean = np.mean(data2, axis=sum_axis, keepdims=True)
#     data2_normal = data2 - mean
#     cov = np.zeros((nb_samples, nb_filter, nb_filter))
#     for i in range(nb_samples):
#         cov[i, :, :] = np.cov(data2_normal[i, :, :], rowvar=rowvar)
#         print(cov[i, :, :].shape)
#     identity = epsilon * np.eye(nb_filter)
#     cov += np.expand_dims(identity, axis=0)
#     # print(cov)
#     # print(a)
#     print(res - cov)
#     assert_allclose(res, cov, rtol=1e-4)
#     assert_allclose(res, cov, rtol=1e-5)
#     assert_allclose(res, cov, rtol=1e-6)

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
    # os.environ["KERAS_BACKEND"] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ["KERAS_BACKEND"] = 'theano'
    pytest.main([__file__])
    # test_secondstat()