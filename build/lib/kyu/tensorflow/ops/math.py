"""
Implement math operations

"""
import tensorflow as tf


def get_matrix_norm(x, mode='Frob', keep_dim=False):
    """
    Tensorflow Frob norm

    Parameters
    ----------
    x : Tensor with shape [..., M, M]
    mode : string Frob at the time being.

    Returns
    -------
    tf.scalar
    """

    if mode == 'Frob':
        if keep_dim:
            return tf.sqrt(tf.trace(tf.batch_matmul(x, tf.transpose(x, [0,2,1]))))
        else:
            return tf.reduce_sum(tf.pow(x, tf.ones_like(x) * 2))
    else:
        raise RuntimeError("Not supported norm: " + mode)