"""
Define the testing file for matrix logarithm and exponential, with related SVD gradients.
"""

import numpy as np
import tensorflow as tf
from kyu.tensorflow.ops.test_normalization import get_covariance_matrices

def check_svd_with_singular_values():
    # s = np.asanyarray([5,4,3,2,1,0,0,0,-1,-2])
    # u = np.random.randn(np.shape(s)[0], np.shape(s)[0])
    # v = np.random.randn(np.shape(s)[0], np.shape(s)[0])
    # s = tf.constant(s, dtype=tf.float32)
    # u = tf.constant(u, dtype=tf.float32)
    # v = tf.constant(v, dtype=tf.float32)
    epsilon = 1e-10
    # A = tf.matmul(u,tf.matmul(tf.matrix_diag(s),v, transpose_b=True))
    A = get_covariance_matrices(2, 1, 10, 2)
    A = tf.constant(A, dtype=tf.float32)
    ns, nu, nv = tf.svd(A, full_matrices=True)

    # ns is not invertable
    dim = tf.where(ns > tf.zeros_like(ns), tf.ones_like(ns), tf.zeros_like(ns))
    d = tf.reduce_sum(dim, axis=1)