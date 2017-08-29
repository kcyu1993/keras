"""
Define the custom sqrt as tensorflow function

"""
import tensorflow as tf
from tensorflow.python.framework.function import Defun

_EPSILON = 1e-8


def custom_sign_sqrt_grad(op, grad):
    x = op.inputs[0]
    eps = tf.zeros_like(x) + _EPSILON
    x = tf.where(tf.less(tf.abs(x), eps), eps, x)
    dx = 0.5 / tf.sqrt(tf.abs(x))
    res_grad = dx * grad
    return res_grad


def custom_sign_sqrt(x):
    return tf.sign(x) * tf.sqrt(tf.abs(x))


@Defun(tf.float32, python_grad_func=custom_sign_sqrt_grad)
def custom_sign_sqrt_op(x):
    y = custom_sign_sqrt(x)
    return y


def custom_safe_sqrt_grad(op, grad):
    x = op.inputs[0]
    eps = tf.zeros_like(x) + _EPSILON
    x = tf.where(tf.less(tf.abs(x), eps), eps, x)
    dx = tf.sign(x) * 0.5 / tf.sqrt(tf.abs(x))
    return dx * grad


def custom_safe_sqrt(x):
    return tf.sqrt(tf.abs(x))


@Defun(tf.float32, python_grad_func=custom_safe_sqrt_grad)
def custom_safe_sqrt_op(x):
    return custom_safe_sqrt(x)


def truncated_sqrt(x):
    eps = tf.zeros_like(x)
    x = tf.where(tf.less(x, eps), eps, x)
    # return safe_abs_sqrt(x)
    return tf.sqrt(x)


def truncated_safe_sqrt(x):
    eps = tf.zeros_like(x)
    x = tf.where(tf.less(x, eps), eps, x)
    return safe_abs_sqrt(x)

# Alias
safe_sign_sqrt = custom_sign_sqrt_op
safe_abs_sqrt = custom_safe_sqrt_op
safe_truncated_sqrt = truncated_safe_sqrt
