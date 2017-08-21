"""
Test Defun in Tensorflow

"""


import tensorflow as tf

from kyu.tensorflow.ops.custom_sqrt import *


def test_defun_sign_sqrt():
    """ Test Sign-Sqrt with custom gradient """

    x = tf.Variable([3., 2., 1e-7, 1e-9, 0., -1, -2], dtype=tf.float32)

    y = custom_sign_sqrt(x)
    custom_y = custom_sign_sqrt_op(x)

    grad_y = tf.gradients(y, x)
    custom_grad_y = tf.gradients(custom_y, x)

    # applied again
    y = custom_sign_sqrt(y)
    custom_y = custom_sign_sqrt_op(custom_y)
    # applied again
    y = custom_sign_sqrt(y)
    custom_y = custom_sign_sqrt_op(custom_y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Original y {} and its gradient {}".format(sess.run(y), sess.run(grad_y)))
        print('Custom y {} and its gradient {}'.format(sess.run(custom_y), sess.run(custom_grad_y)))


def test_defun_safe_sqrt():
    x = tf.Variable([3., 2., 1e-7, 1e-9, 0., -1, -2], dtype=tf.float32)

    y = custom_safe_sqrt(x)
    custom_y = safe_abs_sqrt(x)

    grad_y = tf.gradients(y, x)
    custom_grad_y = tf.gradients(custom_y, x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Original y {} and its gradient {}".format(sess.run(y), sess.run(grad_y)))
        print('Custom y {} and its gradient {}'.format(sess.run(custom_y), sess.run(custom_grad_y)))


def test_truncated_safe_sqrt():
    x = tf.Variable([3., 2., 1e-7, 1e-9, 0., -1, -2], dtype=tf.float32)

    y = truncated_sqrt(x)
    custom_y = truncated_safe_sqrt(x)

    grad_y = tf.gradients(y, x)
    custom_grad_y = tf.gradients(custom_y, x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Original y {} and its gradient {}".format(sess.run(y), sess.run(grad_y)))
        print('Custom y {} and its gradient {}'.format(sess.run(custom_y), sess.run(custom_grad_y)))


if __name__ == '__main__':
    # test_defun_sign_sqrt()
    # test_defun_safe_sqrt()
    test_truncated_safe_sqrt()