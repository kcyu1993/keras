"""
Implement the test matlab gradients

"""

from scipy.io import loadmat
from os.path import join
import numpy as np
import keras.backend as K
import tensorflow as tf

TESTGRAD_PATH = '/cvlabdata1/home/kyu/git/bcnn_kyu/data/testGradient/'


def gradient_for_gsp_layer_matlab_implementation():
    filename = 'test_gsp_gradient.mat'
    mat = loadmat(join(TESTGRAD_PATH, filename))

    # Data loading
    x = np.asanyarray(mat['res']['x'][0][0])
    x = np.transpose(x, [3,0,1,2])

    y = np.asanyarray(mat['res']['y'][0][0])
    y = np.squeeze(y)
    y = np.transpose(y, [1,0])

    grad_x = np.asanyarray(mat['res']['grad_x'][0][0])
    grad_x = np.transpose(grad_x, [3, 0, 1, 2])

    grad_y = np.asanyarray(mat['res']['grad_y'][0][0])
    grad_y = np.squeeze(grad_y)
    grad_y = np.transpose(grad_y)

    def gsp_op(x):
        shape = K.int_shape(x)
        return K.mean(K.pow(K.reshape(x, [shape[0], shape[1]*shape[2],shape[3]]), 2), axis=1)

    # Compute gsp layer
    k_x = K.variable(x)
    k_y = gsp_op(k_x)
    k_grad_y = K.variable(grad_y)
    k_grad_x = K.gradients(k_y, [k_x,])

    sess = K.get_session()
    sess.run(tf.global_variables_initializer())

    # Get result for tensorflow.
    ts_y = sess.run(k_y)
    ts_grad_x = sess.run(k_grad_x)

    # Compose the complete gradient.
    final_ts_grad_x = ts_grad_x[0] * np.expand_dims(np.expand_dims(grad_y, 1), 1)

    assert np.allclose(final_ts_grad_x, grad_x)
    assert np.allclose(ts_y, y)
