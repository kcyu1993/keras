"""
Define the second order batch normalization

"""
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np

from kyu.tensorflow.ops.math import matrix_log, matrix_exp


class SPDBatchNormalization(Layer):
    """
    Define the SPD Batch Normalization testing version
    v0.1

    """
    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(BatchNormalization, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer


class SecondOrderBatchNormalization(BatchNormalization):
    """
    Second-order batch normalization

    log-map into Euclidean space
    BatchNormalization as normal
    Exp-map into manifold space

    Input:
        3D tensor with (B, D1, D2)

    Output:
        3D tensor with (B, D1, D2)

    Base implementation: keras.layers.BatchNormalization

    mode:
        0 (version 1. without any log-map and exp map. direct vectorization and normalization to check).
        1 (version 1. without any log exp map, triu vectorization and normalization.
    """

    def __init__(self, so_mode=0,
                 # epsilon=1e-3, mode=0, axis=-1, momentum=0.99,
                 # weights=None, beta_init='zero', gamma_init='one',
                 # gamma_regularizer=None, beta_regularizer=None,
                 **kwargs):
        self.so_mode = so_mode
        print(" ##### SecondOrderBatchNormalization with mode {} \n".format(so_mode))
        super(SecondOrderBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """ Test and build the normalization shape. """
        # intermediate size should be vectorized

        # if self.so_mode == 0:
        #     super(SecondOrderBatchNormalization, self).build(input_shape)
        if self.so_mode == 1 or self.so_mode == 0 or self.so_mode == 2:
            self.input_spec = [InputSpec(shape=input_shape)]
            intermediate_shape = (input_shape[0], input_shape[1] * input_shape[2])
            shape = (intermediate_shape[self.axis],)
            self.gamma = self.add_weight(shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         name='{}_gamma'.format(self.name))
            self.beta = self.add_weight(shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        name='{}_beta'.format(self.name))
            self.running_mean = self.add_weight(shape, initializer='zero',
                                                name='{}_running_mean'.format(self.name),
                                                trainable=False)
            self.running_std = self.add_weight(shape, initializer='one',
                                               name='{}_running_std'.format(self.name),
                                               trainable=False)

            # if self.initial_weights is not None:
            #     self.set_weights(self.initial_weights)
            #     del self.initial_weights
            self.built = True

    def call(self, x, mask=None):
        # if self.so_mode == 0:
        #     """ treat as a normal matrix """
        shape = x.get_shape().as_list()
        if self.so_mode == 0 or self.so_mode == 1:
            """ triu mapping """
            if self.so_mode == 1:
                x = tf.matrix_band_part(x, 0, -1)
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x_norm = super(SecondOrderBatchNormalization, self).call(x, mask)
            """ enable exp mapping """
            x_norm = tf.reshape(x_norm, [-1,]+shape[1:])
            if self.so_mode == 1:
                x_norm = tf.matrix_band_part(x_norm, 0, -1)
                x_norm = x_norm + K.transpose(x_norm, [0, 2, 1]) - \
                         tf.matrix_diag(tf.matrix_diag_part(x_norm))
        elif self.so_mode == 2:
            """ Apply the log and exp mapping of the matrix """
            x = matrix_log(x, self.epsilon)
            # x = tf.Print(x, [x, ], name='log_transform')
            x = tf.matrix_band_part(x, 0, -1)
            x = tf.reshape(x, [-1, np.prod(shape[1:])])
            x_norm = super(SecondOrderBatchNormalization, self).call(x, mask)
            x_norm = tf.reshape(x_norm, [-1,] + shape[1:])
            x_norm = matrix_exp(x_norm)
            x_norm = tf.matrix_band_part(x_norm, 0, -1)
            x_norm = x_norm + K.transpose(x_norm, [0, 2, 1]) - \
                     tf.matrix_diag(tf.matrix_diag_part(x_norm))
        elif self.so_mode == 3:
            pass
        else:
            raise ValueError("SecondBatchNorm: so_mode not supported {}".format(self.so_mode))
        return x_norm

    def get_config(self):

        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  'momentum': self.momentum,
                  'so_mode': self.so_mode}
        base_config = super(SecondOrderBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
