# -*- coding: utf-8 -*-
from __future__ import absolute_import

from theano import scalar
from theano.tensor import elemwise, Elemwise

from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K
from keras import activations

from theano import tensor as T
import logging


class SecondaryStatistic(Layer):
    ''' This layer shall compute the image secondary statistics and
    output the probabilities. Two computation in one single layer.
    Input to this layer shall come from an output of a convolution layer.
    To be exact, from Convolution2D
    Thus shall be in format of

    # Input shape
        (samples, nb_filter, rows, cols)

    # Output shape
        3D tensor with
            (samples, out_dim, out_dim)
        This is just the 2D covariance matrix for all samples feed in.

    # Arguments
        out_dim         weight matrix, if none, make it align with nb filters
        weights         initial weights.
        W_regularizer   regularize the weight if possible in future
        init:           initialization of function.
        activation      test activation later


    '''

    def __init__(self, output_dim=None, parametrized=False,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, dim_ordering='default', **kwargs):
        self.out_dim = output_dim
        self.parametrized = parametrized

        # input parameter preset
        self.nb_filter = 0
        self.cols = 0
        self.rows = 0
        self.nb_samples = 0

        self.activation = activations.get(activation)

        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)
        self.dim_ordering = 'th'

        self.input_spec = [InputSpec(ndim=4)]
        super(SecondaryStatistic, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the model based on input shape
        Should not set the weight vector here.
        :param input_shape:
        :return:
        """
        # print('secondary_stat: input shape lenth', len(input_shape))

        if self.dim_ordering == 'th':

            self.cov_dim = input_shape[1]

            self.nb_samples = input_shape[0]
            self.nb_filter = input_shape[1]
            self.rows = input_shape[2]
            self.cols = input_shape[3]
            # print("Second layer build: input shape is {}".format(input_shape))
            # Set out_dim accordingly.
            if self.parametrized:
                if self.out_dim is None:
                    self.out_dim = self.cov_dim
                # Create the weight vector
                self.W_shape = (self.cov_dim, self.out_dim)
                if self.initial_weights is not None:
                    self.set_weights(self.initial_weights)
                    del self.initial_weights
                else:
                    self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
                self.trainable_weights = [self.W]

            else:
                # No input parameters, set the weights to identity matrix
                self.W_shape = (self.cov_dim, self.cov_dim)
                # print('second_stat: weight shape: ',self.W_shape)
                self.out_dim = self.cov_dim
                self.W = K.eye(self.cov_dim, name='{}_W'.format(self.name))
                self.non_trainable_weights = [self.W]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering
                            + ' tensorflow not supported')

        self.built = True

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.out_dim, self.out_dim

    def call(self, x, mask=None):
        if not self.built:
            raise Exception("Secondary stat layer not built")
        logging.debug('Secondary_stat parameter', type(x))  # Confirm the type of x is indeed tensor4D
        cov_mat = self.calculate_pre_cov(x)

        return cov_mat

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  }
        base_config = super(SecondaryStatistic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reshape_tensor2d(self, x):
        # given a 3D tensor, reshape it to 2D.
        return K.reshape(K.flatten(x.T),
                         (self.cols * self.rows,
                          self.nb_filter))

    def reshape_tensor3d(self, x):
        # Given a 4D tensor, reshape to 3D
        return K.reshape(x, (-1, self.nb_filter, self.cols * self.rows))

    def calculate_pre_cov(self, x):
        """
        4D tensor to 3D (N, nb_filter, col* row)
        :param x:
        :return:
        """
        xf = self.reshape_tensor3d(x)
        xf_mean = K.mean(xf, axis=2, keepdims=2)
        xf_normal = xf - xf_mean
        tx = K.sum(elemwise.Elemwise(scalar_op=scalar.mul)(
            xf_normal.dimshuffle([0, 'x', 1, 2]),
            xf_normal.dimshuffle([0, 1, 'x', 2])
        ), axis=3)
        cov = tx / (self.rows * self.cols - 1)
        return cov

    def calculate_covariance(self, x):
        """
        Input shall be 3D tensor (nb_filter,ncol,nrow)
        Return just (nb_filter, nb_filter)
        :param x:   data matrix (nb_filter, ncol, nrow)
        :return:    Covariance matrix (nb_filter, nb_filter)
        """

        tx = self.reshape_tensor2d(x)
        # Calcualte the covariance
        tx_mean = K.mean(tx, axis=0)
        # return tx_mean
        tx_normal = tx - tx_mean
        # return tx_normal
        tx_cov = K.dot(tx_normal.T, tx_normal) / (self.cols * self.rows - 1)
        return tx_cov


class O2Transform(Layer):
    ''' This layer shall stack one trainable weights out of previous input layer.


        # Input shape
            3D tensor with
            (samples, input_dim, input_dim)
            Note the input dim must align, i.e, must be a square matrix.

        # Output shape
            3D tensor with
                (samples, out_dim, out_dim)
            This is just the 2D covariance matrix for all samples feed in.

        # Arguments
            out_dim         weight matrix, if none, make it align with nb filters
            weights         initial weights.
            W_regularizer   regularize the weight if possible in future
            init:           initialization of function.
            activation      test activation later (could apply some non-linear activation here
        '''

    def __init__(self, output_dim=None,
                 init='glorot_uniform', activation='relu', weights=None,
                 W_regularizer=None, dim_ordering='default', **kwargs):
        self.out_dim = output_dim

        # input parameter preset
        self.nb_samples = 0

        self.activation = activations.get(activation)

        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)
        self.dim_ordering = 'th'

        self.input_spec = [InputSpec(ndim=3)]
        # Set out_dim accordingly.

        super(O2Transform, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the model based on input shape
        Should not set the weight vector here.
        :param input_shape: (nb-sample, input_dim, input_dim)
        :return:
        """
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]

        if self.dim_ordering == 'th':
            # Create the weight vector
            self.W_shape = (input_shape[1], self.out_dim)
            if self.initial_weights is not None:
                self.set_weights(self.initial_weights)
                del self.initial_weights
            else:
                self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
            self.trainable_weights = [self.W]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering
                            + ' tensorflow not supported')

        self.built = True

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        return input_shape[0], self.out_dim, self.out_dim

    def call(self, x, mask=None):
        # result, updates = scan(fn=lambda tx: K.dot(self.W.T, K.dot(tx, self.W)),
        #                         outputs_info=None,
        #                         sequences=[x],
        #                         non_sequences=None)
        #
        com = K.dot(T.transpose(K.dot(x, self.W),[0,2,1]), self.W)
        # print("O2Transform shape" + com.eval().shape)
        return com

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  }
        base_config = super(O2Transform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
