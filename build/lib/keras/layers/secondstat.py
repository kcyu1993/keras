# -*- coding: utf-8 -*-
from __future__ import absolute_import

from theano import scalar
from theano.tensor import elemwise

from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K
from keras import activations

from theano import tensor as T
from theano import scan


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

        self.activition = activations.get(activation)

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
        # TODO Weight is not supported at this moment, just write them as exp
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
        # TODO Check the validity.
        return (input_shape[0], self.out_dim, self.out_dim)

    def call(self, x, mask=None):
        if not self.built:
            raise Exception("Secondary stat layer not built")
        print('Secondary_stat parameter', type(x))  # Confirm the type of x is indeed tensor4D

        # TODO Compute the mean vector
        # Pesudo data creation
        # x = np.random.rand(*shape)

        # Step 1: reshape the 3D array into 2D array

        # type (theano.config.floatX, matrix)
        # Compute the covariance matrix Y, by sum( <x_ij - x, x_ij.T - x.T> )
        # cov_mat, updates = scan(fn=lambda tx:  self.calculate_covariance(tx),
        # cov_mat, updates = scan(fn=lambda tx: K.dot(self.W.T, K.dot(self.calculate_covariance(tx), self.W)),
        #                         outputs_info=None,
        #                         sequences=[x],
        #                         non_sequences=None)
        # print(components.type)
        # print(components.eval().shape)
        # print(components.eval())

        # Times the weight vector
        cov_mat = self.calculate_pre_cov(x)
        # result = K.dot(K.eye(self.out_dim), K.dot(cov_mat, K.eye(self.out_dim)))
        # return the (samples, cov-mat) as 3D tensor.
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
        # TODO Finish this
        self.out_dim = output_dim

        # input parameter preset
        self.nb_samples = 0

        self.activition = activations.get(activation)

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