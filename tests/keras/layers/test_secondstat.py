import pytest
import numpy as np
from keras.engine import Model
from scipy.linalg import logm
from numpy.testing import assert_allclose
import tensorflow as tf
from keras.utils.test_utils import layer_test, keras_test
from keras import backend as K
from keras.engine import merge
from keras.layers import SecondaryStatistic, WeightedVectorization, O2Transform, LogTransform, \
    MatrixReLU, Convolution2D, Regrouping, SeparateConvolutionFeatures, Input, Dense, O2Transform_v2

# def test_matrix_logrithm():
#     data = np.random.randn(3, 10, 10)
#     result = logm(data)
#
from kyu.tensorflow.ops.svd_gradients import matrix_symmetric, svd_v2


def get_covariance_matrices(batch_size, nb_channel, dim, rank):
    input_shape = (batch_size, nb_channel, dim, rank)
    data = np.random.randn(*input_shape).astype(K.floatx())
    mean = np.mean(data, axis=3, keepdims=True)
    data_norm = data - mean
    cov = np.zeros((batch_size, nb_channel, dim, dim))
    for i in range(batch_size):
        for j in range(nb_channel):
            cov[i,j,:,:] = np.cov(data_norm[i,j,:,:], rowvar=True)
            print(cov[i,j,:,:].shape)
    cov = np.transpose(cov, [0,2,3,1])
    return cov


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
        rowvar = True
        input_shape = (nb_samples, cols, rows, nb_filter)
        sum_axis = 2
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
                       'cov_mode':'channel',
                       'robust':False,
                       'normalization':'mean',
                   },
                   input_shape=input_shape,
                   input_dtype=K.floatx())

    if K.image_dim_ordering() == 'tf':
        data2 = data.reshape((nb_samples, cols * rows, nb_filter))
        data2 = data2.transpose(0,2,1)
    else:
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


@keras_test
def test_encode_mean_cov():
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
        rowvar = True
        input_shape = (nb_samples, cols, rows, nb_filter)
        sum_axis = 2
    else:
        raise ValueError('Image Dim Ordering error {}'.format(K.image_dim_ordering()))
    print('image ordering {}'.format(K.image_dim_ordering()))
    data = np.random.randn(*input_shape).astype(K.floatx())
    assert len(data.shape) == 4

    a = layer_test(SecondaryStatistic,
                   input_data=data,
                   kwargs={
                       'init': 'glorot_uniform',
                       'activation': 'linear',
                       'eps': epsilon,
                       'cov_mode': 'mean',
                   },
                   input_shape=input_shape,
                   input_dtype=K.floatx())

    if K.image_dim_ordering() == 'tf':
        data2 = data.reshape((nb_samples, cols * rows, nb_filter))
        data2 = data2.transpose(0, 2, 1)
    else:
        data2 = data.reshape((nb_samples, nb_filter, cols * rows))
    mean = np.mean(data2, axis=sum_axis, keepdims=True)
    data2_normal = data2 - mean
    cov = np.zeros((nb_samples, nb_filter, nb_filter))
    for i in range(nb_samples):
        cov[i, :, :] = np.cov(data2_normal[i, :, :], rowvar=rowvar)
        print(cov[i, :, :].shape)
    identity = epsilon * np.eye(nb_filter)
    cov += np.expand_dims(identity, axis=0)
    print(cov)
    print(mean)
    print(a)
    assert 0
    # print(a - cov)
    # assert_allclose(a, cov, rtol=1e-4)
    # assert_allclose(a, cov, rtol=1e-5)
    # assert_allclose(a, cov, rtol=1e-6)

# if __name__ == '__main__':
#     # import os
#     # os.environ["KERAS_BACKEND"] = 'tensorflow'
#     pytest.main([__file__])
#     # test_secondstat()


def test_logtransform():
    input_shape = (4,2048,2048)
    data = np.random.randn(*input_shape).astype(K.floatx())
    data = K.batch_dot(data, data.transpose(0,2,1))
    sess = K.get_session()
    with sess.as_default():
        data = K.eval(data)
    # print(data)
    # traditional log
    from scipy.linalg import logm
    res = np.zeros(shape=input_shape)
    for i in range(data.shape[0]):
        res[i,:,:] = logm(data[i,:,:])

    # log layer
    from keras.layers.secondstat import LogTransform
    a = layer_test(LogTransform,
                   input_data=data,
                   kwargs={
                       'epsilon':1e-4,
                   },
                   input_shape=input_shape,
                   input_dtype=K.floatx())

    # print(a - res)
    diff = np.linalg.norm(a - res)
    print(diff)
    assert_allclose(a, res, rtol=1e-4)
    # assert_allclose(a, res, rtol=1e-3)
    # assert_allclose(a, res, rtol=1e-4)


def compare_with_matlab_version():
    """
    with the data saved in
    /tmp/test.mat

    Result is

    ans(:,:,1) =

   -4.8908   -0.0121    0.0948
   -0.0121   -4.4502   -0.0098
    0.0948   -0.0098   -4.3483


    ans(:,:,2) =

   -4.4925   -0.0594    0.0041
   -0.0594   -4.5666    0.1034
    0.0041    0.1034   -4.4904


    [[[ -4.87092209e+00  -1.21317953e-02   9.48495865e-02]
  [ -1.21318027e-02  -4.43027258e+00  -9.79678333e-03]
  [  9.48495865e-02  -9.79675353e-03  -4.32838297e+00]]

 [[ -4.47253942e+00  -5.94185591e-02   4.10652161e-03]
  [ -5.94185591e-02  -4.54664803e+00   1.03441119e-01]
  [  4.10652161e-03   1.03441238e-01  -4.47046089e+00]]]
    Returns
    -------

    """
    from scipy.io import loadmat
    data = loadmat('/tmp/test.mat')
    data = data['x']
    input_shape = (None, 10, 10, 3)
    tf_input = tf.placeholder(K.floatx(), shape=input_shape, name='tf_input')
    x = SecondaryStatistic(normalization=None, name='second')(tf_input)
    cov_mat = x
    x = LogTransform(1e-4, name='log')(x)

    sess = K.get_session()
    with sess.as_default():
        result = x.eval({tf_input:data})
        cov_mat_eval = cov_mat.eval({tf_input:data})

    # print(result)
    # print(cov_mat_eval[0])
    # verify batch_1
    d1 = data[0]
    d1 /= 99
    d1 = np.reshape(d1, (d1.shape[0] * d1.shape[1], d1.shape[2]))
    # print(d1.shape)
    cov_d1 = np.sum((np.expand_dims(d1, 1) * np.expand_dims(d1, 2)), axis=0)
    # print(cov_d1.shape)
    # print(result)

    assert_allclose(cov_d1, cov_mat_eval[0], rtol=1e-5)

    # compare the log
    from scipy.linalg import svd
    u, s, v = svd(cov_d1)
    log_d1 = np.dot(u, np.dot(np.diag(np.log(s + 1e-4)), u.transpose()))

    assert_allclose(log_d1, result[0], rtol=1e-5)

    # check for gradients
    mat_grads = loadmat('/tmp/gradients.mat')
    mat_grads = mat_grads['lower2']['dzdx']
    mat_grads = mat_grads[0][0]

    print(mat_grads.shape)
    # y_grads = tf.Variable(np.ones((2, 3, 3)).astype(np.float32))
    y_grads = tf.placeholder(tf.float32, shape=(None,3,3))
    tf.global_variables_initializer()

    tf_grad = tf.gradients(x, [tf_input], grad_ys=y_grads)
    tf_grad_res = sess.run(tf_grad, {tf_input: data, y_grads: np.ones((2,3,3), dtype=np.float32)})
    # print(tf_grad_res)
    mat_grads = np.transpose(mat_grads, [3,0,1,2])
    mat_grads = np.expand_dims(mat_grads, 0)
    assert_allclose(tf_grad_res, mat_grads)

    ## Get detailed gradients


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])
    return res



INPUT_SHAPE = (None, 10, 10, 3)


def gradient_svd_for_log_v2(op, grad_s, grad_u, grad_v):
    s, u, v = op.outputs
    diagS = tf.matrix_diag(s)
    inner = diagS
    inner = tf.matrix_diag_part(inner) + 1e-4
    log_inner = tf.log(inner)
    inverse_inner = 1. / inner
    # inner = tf.matrix_diag(inner)
    dLdC = tf.matmul(grad_u, tf.matrix_inverse(tf.matmul(u, tf.matrix_diag(log_inner)))) / 2

    grad_S = tf.matmul(
        2 * tf.sqrt(diagS),
        tf.matmul(
            tf.matrix_diag(1. / inner),
            tf.matmul(
                tf.transpose(u, [0, 2, 1]),
                tf.matmul(dLdC, u))))
    diag_grad_S = tf.matrix_diag_part(grad_S)
    K = tf.transpose(get_eigen_K(tf.sqrt(s), True), [0, 2, 1])
    tmp = K * tf.matmul(tf.transpose(u, [0, 2, 1]), grad_u)

    dxdz = tmp + tf.matrix_diag(diag_grad_S)
    dxdz = tf.matmul(dxdz, tf.transpose(u, [0, 2, 1]))
    dxdz = tf.matmul(u, dxdz)
    # dxdz = tf.matmul(
    #     v, tf.matmul(tf.matrix_diag(2*tf.sqrt(s)), tmp) + tf.matrix_diag(diag_grad_S),
    #     )
    return dxdz


# @tf.RegisterGradient('Svd')
def gradient_eig_for_log(op, grad_s, grad_u, grad_v):
    """ U == V in such operation """
    s, u, v = op.outputs
    u_t = tf.transpose(u, [0,2,1])
    diagS = tf.matrix_diag(s)
    inner = diagS
    log_inner = tf.log(s)

    dLdC = tf.matmul(grad_u, tf.matrix_inverse(tf.matmul(u, tf.matrix_diag(log_inner)))) / 2
    tmp = tf.matmul(u_t, tf.matmul(dLdC, u))
    grad_S = tf.matmul(tf.matrix_diag(1. / s), tmp)

    # K = tf.transpose(get_eigen_K(s, square=False), [0, 2, 1])
    K = get_eigen_K(s, square=False)
    dzdx = K * tf.matmul(u_t, grad_u) + tf.matrix_diag(tf.matrix_diag_part(grad_S))
    dzdx = tf.matmul(u, tf.matmul(dzdx, u_t))
    return dzdx


# @tf.RegisterGradient('Svd')
def gradient_svd_for_log(op, grad_s, grad_u, grad_v):
        s, u, v = op.outputs
        diagS = tf.matrix_diag(s)  # Check

        inner = tf.matmul(tf.transpose(diagS, [0, 2, 1]), diagS)
        inner = tf.matrix_diag_part(inner) + 1e-4
        log_inner = tf.log(inner)
        inverse_inner = 1. / inner
        # inner = tf.matrix_diag(inner)
        test = tf.matrix_inverse(tf.matmul(v, tf.matrix_diag(log_inner)))
        dLdC = tf.matmul(grad_v, test) / 2

        grad_S = tf.matmul(
            2 * diagS,
            tf.matmul(
                tf.matrix_diag(inverse_inner),
                tf.matmul(
                    tf.transpose(v, [0, 2, 1]),
                    tf.matmul(dLdC, v))))

        diag_grad_S = tf.matrix_diag_part(grad_S)
        K = get_eigen_K(s, True)

        tmp = matrix_symmetric(K * tf.matmul(tf.transpose(v, [0, 2, 1]), grad_v))

        # Create the shape accordingly.
        u_shape = u.get_shape()[1].value
        v_shape = v.get_shape()[1].value

        eye_mat = tf.eye(v_shape, u_shape)
        realS = tf.matmul(tf.reshape(diagS, [-1, v_shape]), eye_mat)
        realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])

        real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(diag_grad_S), [-1, v_shape]), eye_mat)
        real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])

        tmp = 2 * tf.matmul(realS, tmp)

        dxdz = tmp + real_grad_S
        # return new_id
        dxdz = tf.matmul(dxdz, tf.transpose(v, [0, 2, 1]))
        dxdz = tf.matmul(u, dxdz)
        return dxdz


def pesudo_gradient(s,u, v, grad_s, grad_v):
    diagS = tf.matrix_diag(s)   # Check

    inner = tf.matmul(tf.transpose(diagS, [0,2,1]), diagS)
    inner = tf.matrix_diag_part(inner) + 1e-4
    log_inner = tf.log(inner)
    inverse_inner = 1./ inner
    # inner = tf.matrix_diag(inner)
    test = tf.matrix_inverse(tf.matmul(v, tf.matrix_diag(log_inner)))
    dLdC = tf.matmul(grad_v, test) / 2

    grad_S = tf.matmul(
        2 * diagS,
        tf.matmul(
            tf.matrix_diag(inverse_inner),
            tf.matmul(
                tf.transpose(v, [0, 2, 1]),
                tf.matmul(dLdC, v))))

    diag_grad_S = tf.matrix_diag_part(grad_S)
    K = get_eigen_K(s, True)

    tmp = matrix_symmetric(K * tf.matmul(tf.transpose(v, [0, 2, 1]), grad_v))

    # Create the shape accordingly.
    u_shape = u.get_shape()[1].value
    v_shape = v.get_shape()[1].value

    eye_mat = tf.eye(v_shape, u_shape)
    realS = tf.matmul(tf.reshape(diagS, [-1, v_shape]), eye_mat)
    realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0,2,1])

    real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(diag_grad_S), [-1, v_shape]), eye_mat)
    real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0,2,1])

    tmp = 2 * tf.matmul(realS, tmp)

    dxdz = tmp + real_grad_S
    # return new_id
    dxdz = tf.matmul(dxdz, tf.transpose(v, [0, 2, 1]))
    dxdz = tf.matmul(u, dxdz)
    # dxdz = tf.matmul(
    #     v, tf.matmul(tf.matrix_diag(2*tf.sqrt(s)), tmp) + tf.matrix_diag(diag_grad_S),
    #     )
    return dxdz


    # grad_S = tf.matmul(
    #     2*tf.sqrt(diagS),
    #     tf.matmul(
    #         tf.matrix_diag( 1./ inner),
    #         tf.matmul(
    #             tf.transpose(v, [0,2,1]),
    #             tf.matmul(dLdC, v))))
    # diag_grad_S = tf.matrix_diag_part(grad_S)
    # K = get_eigen_K(tf.sqrt(s))
    #
    # tmp = K * tf.matmul(tf.transpose(v, [0,2,1]), grad_v)
    # tmp = matrix_symmetric(tmp)
    # tmp = 2 * tf.matmul(tf.sqrt(diagS), tmp)
    # dxdz = tmp + tf.matrix_diag(diag_grad_S)
    # # return dxdz
    # dxdz = tf.matmul(dxdz, tf.transpose(v, [0,2,1]))
    # dxdz = tf.matmul(u, dxdz)
    # # dxdz = tf.matmul(
    # #     v, tf.matmul(tf.matrix_diag(2*tf.sqrt(s)), tmp) + tf.matrix_diag(diag_grad_S),
    # #     )
    # return grad_S


def gradient_eig_comparision():
    epsilon = 1e-4


    from scipy.io import loadmat
    data = loadmat('/tmp/test.mat')
    data = data['x']
    input_shape = (None, 10, 10, 3)

    tf_input = tf.placeholder(K.floatx(), shape=input_shape, name='tf_input')
    x = SecondaryStatistic(normalization=None, name='second')(tf_input)
    cov_mat = x
    # x = LogTransform(1e-4, name='log')(x)
    s, u, v = tf.svd(x)
    inner = s + epsilon
    inner = tf.log(inner)
    inner = tf.matrix_diag(inner)
    tf_log = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0,2,1])))
    y_grads = tf.placeholder(tf.float32, shape=(None, 3, 3))

    grad_s = tf.gradients(tf_log, s, grad_ys=y_grads)[0]
    # grad_v = tf.gradients(tf_log, v, grad_ys=y_grads) # not used!, so no gradient is calculated.
    grad_u = tf.gradients(tf_log, u, grad_ys=y_grads)[0]
    grad_x = tf.gradients(tf_log, tf_input, grad_ys=y_grads)[0]

    grad_S = pesudo_gradient(s,u,v, grad_s, grad_u)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = K.get_session(config=config)
    with sess.as_default():
        result = x.eval({tf_input: data})
        cov_mat_eval = cov_mat.eval({tf_input: data})
        grad_s_eval = grad_s.eval({tf_input:data, y_grads: np.ones((2,3,3), dtype=np.float32)})
        grad_u_eval = grad_u.eval({tf_input:data, y_grads: np.ones((2,3,3), dtype=np.float32)})
        grad_input_eval = grad_x.eval({tf_input:data, y_grads: np.ones((2,3,3), dtype=np.float32)})
        grad_S_eval = grad_S.eval({tf_input:data, y_grads: np.ones((2,3,3), dtype=np.float32)})
    # print(grad_s_eval)
    # print(grad_u_eval)
    # print(grad_input_eval)
        # check for gradients
    print(grad_S_eval)
    mat_grads = loadmat('/tmp/gradients.mat')
    mat_grads = mat_grads['lower2']['dzdx']
    mat_grads = mat_grads[0][0]
    mat_grads = np.transpose(mat_grads, [3, 0, 1, 2])
    assert_allclose(grad_input_eval, mat_grads, rtol=1e-4)


def gradient_svd_comparision(path=''):
    """
    Testing function to compare SVD gradients.

    Returns
    -------

    """
    epsilon = 1e-5
    path = '/cvlabdata1/home/kyu/git/matrix_backprop/test/test2.mat'

    from scipy.io import loadmat
    data = loadmat(path)
    input_x = np.asanyarray(data['lower']['x'][0][0], dtype=np.float32)
    input_x = np.transpose(input_x, [3,0,1,2])

    input_shape = input_x.shape

    input_grad = np.asanyarray(data['upper']['dzdx'][0][0], dtype=np.float32)
    input_grad = np.reshape(input_grad, [input_shape[3], input_shape[3],-1])
    input_grad = np.transpose(input_grad, [2,1,0])


    tf_input = tf.placeholder(K.floatx(), shape=input_shape, name='tf_input')
    x = tf.reshape(tf_input, (-1, input_shape[1]*input_shape[2], input_shape[3]))
    cov_mat = x
    x /= input_shape[1] * input_shape[2]

    s, u, v = tf.svd(x, full_matrices=True)
    # s,u,v = svd_v2(x, full_matrices=True)

    inner = tf.square(s) + epsilon
    inner = tf.log(inner)
    inner = tf.matrix_diag(inner)
    tf_log = tf.matmul(v, tf.matmul(inner, tf.transpose(v, [0,2,1])))

    y_grads = tf.placeholder(tf.float32, shape=(None, input_shape[3], input_shape[3]))
    grad_s = tf.gradients(tf_log, s, grad_ys=y_grads)[0]
    grad_v = tf.gradients(tf_log, v, grad_ys=y_grads)[0] # not used!, so no gradient is calculated.
    # grad_u = tf.gradients(tf_log, u, grad_ys=y_grads)[0]
    grad_x = tf.gradients(tf_log, tf_input, grad_ys=y_grads)[0]

    grad_S = pesudo_gradient(s, u, v, grad_s, grad_v)
    feed_dict = {tf_input:input_x, y_grads: input_grad}
    sess = K.get_session()
    with sess.as_default():
        result = x.eval({tf_input: input_x})
        cov_mat_eval = cov_mat.eval({tf_input: input_x})
        grad_s_eval = grad_s.eval(feed_dict)
        grad_v_eval = grad_v.eval(feed_dict)
        grad_input_eval = grad_x.eval(feed_dict)
        grad_S_eval = grad_S.eval(feed_dict)
    # print(grad_s_eval)
    # print(grad_u_eval)
    # print(grad_input_eval)
        # check for gradients
    # print(grad_S_eval)
    mat_grads = np.asanyarray(data['lower']['dzdx'][0][0], dtype=np.float32)
    mat_grads = np.transpose(mat_grads, [3, 0, 1, 2])
    assert_allclose(grad_input_eval, mat_grads)
    print(np.allclose(grad_input_eval, mat_grads))


def test_matrixrelu():
    input_shape = (4, 100, 10)
    epsilon = 1e-4
    data = np.random.randn(*input_shape).astype(K.floatx())
    k_data = K.batch_dot(data, data.transpose(0, 2, 1))
    sess = K.get_session()
    with sess.as_default():
        data = K.eval(k_data)
    # print(data)
    print(data.shape)
    # traditional log
    res = np.zeros(shape=(4, 100, 100))
    from numpy.linalg import eig
    for i in range(data.shape[0]):
        tmp_d = data[i, :, :]
        s, u = eig(tmp_d)
        comp = np.zeros_like(s) + epsilon
        s[np.where(s < epsilon)] = epsilon
        s = np.diag(s)
        tmp_res = np.matmul(u, np.matmul(s, u.transpose()))
        res[i, :, :] = tmp_res

    import tensorflow as tf
    tf_input = tf.placeholder(K.floatx(), (4,100,100))
    tf_s, tf_u = tf.self_adjoint_eig(tf_input)
    comp = tf.zeros_like(tf_s) + epsilon
    comp = tf.Print(comp, [comp], message='comp:')
    tf_s = tf.Print(tf_s, [tf_s], message='tf_s:', summarize=400)
    inner = tf.where(tf.less(tf_s, comp), comp, tf_s)
    inner = tf.Print(inner, [inner], 'inner:')
    inner = tf.matrix_diag(inner)
    tf_relu = tf.matmul(tf_u, tf.matmul(inner, tf.transpose(tf_u, [0,2,1])))

    with sess.as_default():

        tf_result = tf_relu.eval({tf_input:data})

    # log layer
    from keras.layers.secondstat import LogTransform
    a = layer_test(MatrixReLU,
                   input_data=data,
                   kwargs={
                       'epsilon': epsilon,
                   },
                   input_shape=(4, 100, 100),
                   input_dtype=K.floatx())

    # a = tf_result
    # print(a - res)
    diff = np.linalg.norm(a - res)
    print(diff)
    assert_allclose(a, res, rtol=1e-4)
    # assert_allclose(a, res, rtol=1e-3)
    # assert_allclose(a, res, rtol=1e-4)


def simple_second_model():
    # Define and create a simple Conv2D model
    n = 8
    input_tensor = Input(INPUT_SHAPE[1:])
    x = Convolution2D(1024, 3,3)(input_tensor)
    x = Convolution2D(2048, 3,3)(x)

    list_covs = SeparateConvolutionFeatures(n)(x)
    list_covs = Regrouping(None)(list_covs)
    list_outputs = []
    for cov in list_covs:
        cov = SecondaryStatistic()(cov)
        cov = O2Transform(100)(cov)
        cov = O2Transform(100)(cov)
        list_outputs.append(WeightedVectorization(10)(cov))

    x = merge(list_outputs, mode='concat')
    x = Dense(10)(x)

    model = Model(input_tensor, x)

    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()
    return model


def test_o2transform_v2():
    cov = get_covariance_matrices(2, 3, 4, 2)
    a = layer_test(O2Transform_v2,
                   input_data=cov,
                   kwargs={
                       'output_dim': 8,
                   },
                   input_shape=(2,3,4, 4),
                   input_dtype=K.floatx())

    print(a)


if __name__ == '__main__':
    # test_secondstat()
    # test_encode_mean_cov()
    # test_logtransform()
    # compare_with_matlab_version()
    gradient_svd_comparision()
    # gradient_eig_comparision()
    # test_matrixrelu()
    # simple_second_model()
    # test_secondstat()