"""
Implement Gradient for tf.svd

"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import function


def matrix_symmetric(x):
    return (x + tf.transpose(x, [0,2,1])) / 2


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    x -= 1e-4
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res


@tf.RegisterGradient('Svd')
def gradient_eig(op, grad_s, grad_u, grad_v):
    """
    Implementation of EIG operation gradients.

    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------

    """
    s, u, v = op.outputs
    u_t = tf.transpose(u, [0, 2, 1])
    with tf.name_scope('K'):
        K = get_eigen_K(s, square=False)

    dzdx1 = 2 * K * tf.matmul(u_t, grad_u) + tf.matrix_diag(grad_s, name='dzdx1')
    dzdx1 = tf.matmul(dzdx1, u_t)
    dzdx_final = tf.matmul(u, dzdx1, name='dzdx_final')
    dzdx_final = tf.where(tf.is_nan(dzdx_final), tf.zeros_like(dzdx_final), dzdx_final)
    return dzdx_final


# @tf.RegisterGradient('Svd')
def gradient_eig_for_log(op, grad_s, grad_u, grad_v):
    """ U == V in such operation """
    s, u, v = op.outputs
    # with tf.name_scope('SVD_GRADIENT'):
    u_t = tf.transpose(u, [0,2,1])
    # log_inner = 1. / tf.log(tf.abs(s) + 0.001)
    # s = tf.Print(s, [s], summarize=64, message='s')
    # log_inner = 1. / tf.log(s)
    # log_inner = tf.Print(log_inner, [log_inner], summarize=64, message='log_inner')
    # log_inner = tf.where(tf.is_nan(log_inner), tf.zeros_like(log_inner), log_inner)
    # log_inner = tf.matrix_diag(log_inner)
    # sess2 = tf.Session()
    # dLdC = tf.matmul(grad_u, tf.matrix_inverse(tf.matmul(u, tf.matrix_diag(log_inner)))) / 2
    grad_u = tf.Print(grad_u, [grad_u], summarize=10, message='grad_u')
    grad_s = tf.Print(grad_s, [grad_s], message='grad_s')
    # grad_v = tf.Print(grad_v, [grad_v], message='grad_v')

    # dLdC = tf.matmul(grad_u, (tf.matmul(log_inner, u_t)), name='dLdC') / 2
    # dLdC = tf.Print(dLdC, [dLdC], summarize=64*64, message='dLdC')

    # tmp = tf.matmul(u_t, tf.matmul(dLdC, u), name='tmp')
    # tmp = tf.Print(tmp, [tmp], summarize=64*64, message='tmp')

    # inv_diag_S = (1. / s)
    # inv_diag_S = tf.where(tf.is_inf(inv_diag_S), tf.zeros_like(inv_diag_S), inv_diag_S)
    # # inv_diag_S = tf.Print(inv_diag_S, [inv_diag_S], summarize=64, message='inv_diag_S')
    # inv_diag_S = tf.matrix_diag(inv_diag_S)
    # grad_S = tf.matmul(inv_diag_S, tmp, name='grad_S')
    # grad_S = tf.Print(grad_S, [grad_S],summarize=64, message='grad_S')

    with tf.name_scope('K'):
        K = get_eigen_K(s, square=False)
    # K = tf.Print(K, [K], summarize=64*64, message='K')

    # dzdx1 = K * tf.matmul(u_t, grad_u) + tf.matrix_diag(tf.matrix_diag_part(grad_S), name='dzdx1')
    dzdx1 = K * tf.matmul(u_t, grad_u) + tf.matrix_diag(grad_s, name='dzdx1')
    # dzdx1 = tf.Print(dzdx1, [dzdx1],  summarize=64*64, message='dzdx1')
    dzdx1 = tf.matmul(dzdx1, u_t)

    # u = tf.Print(u, [u], summarize=64*64, message='u')

    dzdx_final = tf.matmul(u, dzdx1, name='dzdx_final')
    dzdx_final = tf.where(tf.is_nan(dzdx_final), tf.zeros_like(dzdx_final), dzdx_final)
    dzdx_final = tf.Print(dzdx_final, [dzdx_final],  summarize=10, message='dzdx_final')
    # return tf.ones_like(dzdx) / 10
    # dzdx_final = tf.Print(dzdx, [dzdx], message='dzdx ')
    return dzdx_final


# @tf.RegisterGradient('Svd')
def gradient_svd_for_log(op, grad_s, grad_u, grad_v):
    """
    Verified version of SVD implementation for matrix log.
    Only valid when using the following ways to calculate log.

    tf_input = tf.placeholder(K.floatx(), shape=input_shape, name='tf_input')
    x = tf.reshape(tf_input, (-1, 10 * 10, 3))
    cov_mat = x
    x /= 10 * 10
    s, u, v = tf.svd(x, full_matrices=True)
    inner = tf.square(s) + epsilon
    inner = tf.log(inner)
    inner = tf.matrix_diag(inner)
    tf_log = tf.matmul(v, tf.matmul(inner, tf.transpose(v, [0,2,1])))

    Parameters
    ----------
    op: tensorflow.Op
    grad_s : gradient regarding to s
    grad_u : gradient regarding to u
    grad_v : gradietn regarding to v

    Returns
    -------

    """
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


def grad_batch_matrix_logs(op, grads):
    inputs = op.inputs
    outputs = op.outputs

    for inp in inputs:
        print(inp)
    for output in outputs:
        print(output)

    for grad in grads:
        print(grad)
    return tf.zeros_like(op.inputs)


@function.Defun(tf.float32, tf.float32, python_grad_func=grad_batch_matrix_logs)
def batch_matrix_log(x, epsilon):
    """
    Matrix log with epsilon to ensure stability.
    Input must be a Symmetric matrix.

    Parameters
    ----------
    x : tf.Tensor with [..., dim1, dim2]

    epsilon

    Returns
    -------
    log of eigen-values.

    """
    s, u, v = tf.svd(x)
    # print(s.eval())
    inner = s + epsilon
    inner = tf.log(inner)
    inner = tf.matrix_diag(inner)
    return tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0,2,1])))


if __name__ == '__main__':
    """ Verification of gradients """
    x = tf.Variable(np.random.rand(2,3,3), dtype=tf.float32)
    x = tf.identity(x)
    grad = tf.gradients(batch_matrix_log(x), [x])[0]

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
        res = sess.run(grad)
        print(res)
        print(sess.run(batch_matrix_log(x)))
    sess.close()

