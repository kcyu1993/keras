# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..engine import Layer, InputSpec
from ..utils.np_utils import conv_output_length, conv_input_length

from theano.tensor.elemwise import *
from theano import scalar

import logging


class WeightedProbability(Layer):
    ''' Probability weighted vector layer for secondary image statistics
    neural networks. It is simple at this time, just v_c.T * Cov * v_c, with
    basic activitation function such as ReLU, softmax, thus the
    Version 0.1: Implement the basic weighted probablitity coming from cov-layer
    Version 0.2: Implement trainable weights to penalized over-fitting

    '''


    def __init__(self, output_dim, input_dim=None, init='glorot_uniform', activation='linear', weights=None,
                 # W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 # W_constraint=None, b_constraint=None,
                 bias=False,  **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim      # Squared matrix input, as property of cov matrix
        self.output_dim = output_dim    # final classified categories number

        self.bias = bias
        self.initial_weights = weights

        super(WeightedProbability, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build function
        :param input_shape:
        :return:
        """
        # 3D tensor (nb_samples, n_cov, n_cov)
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]

        input_dim = input_shape[1]

        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        '''
        The calculation of call function is not trival.
        sum( self.W .* ( x * self.W) ) along axis 1
        :param x:
        :param mask:
        :return: final output vector with w_i^T * W * w_i as item i, and propagate to all
            samples. Output Shape (nb_samples, vector c)
        '''
        logging.debug("prob_out: x_shape {}".format(x.shape))
        # new_W = K.expand_dims(self.W, dim=1)
        output = K.sum(Elemwise(scalar_op=scalar.mul)(self.W, K.dot(x, self.W)), axis=1)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        # 3D tensor (nb_samples, n_cov, n_cov)
        '''
        :param input_shape: 3D tensor where item 1 and 2 must be equal.
        :return: (nb_samples, number C types)
        '''
        logging.debug(input_shape)
        assert input_shape and len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  # 'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  # 'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  # 'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  # 'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  # 'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(WeightedProbability, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
