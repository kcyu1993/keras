"""
Test Defun in Tensorflow

"""


import tensorflow as tf
from tensorflow.python.framework.function import Defun


def custom_sign_sqrt_grad(op, grad):
    x = op.inputs[0]
    dx = 0.5 / tf.sqrt(tf.abs(x) + 1e-5)
    res_grad = dx * grad
    return res_grad


def custom_sign_sqrt(x):
    return tf.sign(x) * tf.sqrt(tf.abs(x))


@Defun(tf.float32, python_grad_func=custom_sign_sqrt_grad)
def custom_sign_sqrt_op(x):
    y = custom_sign_sqrt(x)
    return y


def test_defun_sign_sqrt():
    """ Test Sign-Sqrt with custom gradient """

    x = tf.Variable([3., 2., 1., 0.], dtype=tf.float32)

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


if __name__ == '__main__':
    test_defun_sign_sqrt()