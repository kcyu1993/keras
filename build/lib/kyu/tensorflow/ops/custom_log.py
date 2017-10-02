import tensorflow as tf
from tensorflow.python.framework.function import Defun

from kyu.tensorflow.ops.math import _EPSILON
from kyu.tensorflow.ops.svd_gradients import get_eigen_K

_EPSILON = 1e-5


def gradient_matrix_log_eig(op, grad_s, grad_u):
    """ U == V in such operation """
    s, u = op.outputs
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
    # grad_u = tf.Print(grad_u, [grad_u], summarize=10, message='grad_u')
    # grad_s = tf.Print(grad_s, [grad_s], message='grad_s')
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
    # dzdx_final = tf.Print(dzdx_final, [dzdx_final],  summarize=10, message='dzdx_final')
    # return tf.ones_like(dzdx) / 10
    # dzdx_final = tf.Print(dzdx, [dzdx], message='dzdx ')
    return dzdx_final


def matrix_log(x, eps=1e-5):
    """
    Define the matrix logarithm with the gradients

    Parameters
    ----------
    x

    Returns
    -------

    """
    s, u = tf.self_adjoint_eig(x)
    s = tf.abs(s)
    inner = s + eps
    inner = tf.log(inner)
    inner = tf.where(tf.is_nan(inner), tf.zeros_like(inner), inner)
    inner = tf.matrix_diag(inner)
    tf_log = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0,2,1])))
    return tf_log


@Defun(tf.float32, python_grad_func=gradient_matrix_log_eig)
def safe_matrix_log_op(x):
    """ Safe matrix log for ill-conditioned cases """
    return matrix_log(x, _EPSILON)