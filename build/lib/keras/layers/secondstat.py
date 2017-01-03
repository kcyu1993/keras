# -*- coding: utf-8 -*-
from __future__ import absolute_import


from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K
from keras import activations

# TODO Remove this theano import to prevent any usage in tensorflow backend
# Potentially check Keras backend then import relevant libraries


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
        eps             weight of elipson * I to add to cov matrix, default 0
        out_dim         weight matrix, if none, make it align with nb filters
        weights         initial weights.
        W_regularizer   regularize the weight if possible in future
        init:           initialization of function.
        activation      test activation later


    '''
    def __init__(self, eps=0,
                 cov_mode='channel',
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        if dim_ordering == 'th':
            self.axis_filter = 1
            self.axis_row = 2
            self.axis_col = 3
        else:
            self.axis_filter = 3
            self.axis_row = 1
            self.axis_col = 2

        # if cov_mode is 'channel':
        #     self.axis_cov = (self.axis_filter,)
        #     self.axis_non_cov = (self.axis_row, self.axis_col)
        # elif cov_mode is 'feature':
        #     self.axis_cov = (self.axis_row, self.axis_col)
        #     self.axis_non_cov = (self.axis_filter,)
        if cov_mode not in ['channel', 'feature']:
            raise ValueError('only support cov_mode across channel and features, given {}'.format(cov_mode))

        self.cov_mode = cov_mode

        # input parameter preset
        self.nb_filter = 0
        self.cols = 0
        self.rows = 0
        self.nb_samples = 0
        self.eps = eps

        self.activation = activations.get(activation)

        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        super(SecondaryStatistic, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the model based on input shape,
        Should not set the weight vector here.
        Add the cov_mode in 'channel' or 'feature',
            by using self.cov_axis.

        dim-ordering is only related to the axes
        :param input_shape:
        :return:
        """
        print('secondary_stat: input shape lenth', len(input_shape))
        print('second_stat: input shape {}'.format(input_shape))
        print('second_stat: axis filter {}'.format(self.axis_filter))
        self.nb_samples = input_shape[0]
        self.nb_filter = input_shape[self.axis_filter]
        self.rows = input_shape[self.axis_row]
        self.cols = input_shape[self.axis_col]

        # Calculate covariance axis
        if self.cov_mode is 'channel':
            self.cov_dim = self.nb_filter
        else:
            self.cov_dim = self.rows * self.cols

        # Set out_dim accordingly.
        self.out_dim = self.cov_dim
        print('output_dim:' + str(self.out_dim))

        self.W_shape = (self.cov_dim, self.cov_dim)
        print('second_stat: weight shape: ',self.W_shape)
        self.W = K.eye(self.cov_dim, name='{}_W'.format(self.name))
        self.non_trainable_weights = [self.W]

        self.b_shape = (self.cov_dim, self.cov_dim)
        # TODO should nout use expand dim here
        self.b = K.expand_dims(K.eye(self.cov_dim, name="{}_b".format(self.name)), 0)
        # self.non_trainable_weights += [self.b,]
        self.built = True

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.out_dim, self.out_dim

    def call(self, x, mask=None):
        if not self.built:
            raise Exception("Secondary stat layer not built")
        logging.debug('Secondary_stat parameter', type(x))  # Confirm the type of x is indeed tensor4D
        cov_mat = self.calculate_pre_cov(x)
        print('call during second {}'.format(self.eps))
        cov_mat += self.eps * self.b
        return cov_mat

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'eps': self.eps
                  }
        base_config = super(SecondaryStatistic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reshape_tensor3d(self, x):
        """
        Transpose and reshape to a format
        (None, cov_axis, data_axis)
        Parameters
        ----------
        x : tensor  (None, filter, cols, rows) for th,
                    (None, cols, rows, filter) for tf
        Returns
        -------
        """
        if self.dim_ordering == 'th':
            tx = K.reshape(x, (-1, self.nb_filter, self.cols * self.rows))
        else:
            tx = K.reshape(x, (-1, self.cols * self.rows, self.nb_filter))
            tx = K.transpose(tx, (0,2,1))
        if self.cov_mode is 'channel':
            return tx
        else:
            return K.transpose(tx, (0,2,1))

    def calculate_pre_cov(self, x):
        """
        4D tensor to 3D (N, nb_filter, col* row)
        :param x: Keras.tensor  (N, nb_filter, col, row) data being called
        :return: Keras.tensor   (N, nb_filter, col* row)
        """
        xf = self.reshape_tensor3d(x)
        xf_mean = K.mean(xf, axis=2, keepdims=True)
        xf_normal = xf - xf_mean

        tx = K.sum(K.multiply(K.expand_dims(xf_normal, dim=1),
                              K.expand_dims(xf_normal, dim=2)),
                   axis=3)
        if self.cov_mode is 'channel':
            cov = tx / (self.rows * self.cols - 1)
        else:
            cov = tx / (self.nb_filter - 1)

        return cov

    # Deprecated method

    def calculate_covariance(self, x):
        """
        Input shall be 3D tensor (nb_filter,ncol,nrow)
        Return just (nb_filter, nb_filter)
        :param x:   data matrix (nb_filter, ncol, nrow)
        :return:    Covariance matrix (nb_filter, nb_filter)
        """
        # tx = self.reshape_tensor2d(x)
        # Calcualte the covariance
        # tx_mean = K.mean(tx, axis=0)
        # return tx_mean
        # tx_normal = tx - tx_mean
        # return tx_normal
        # tx_cov = K.dot(tx_normal.T, tx_normal) / (self.cols * self.rows - 1)
        # return tx_cov
        raise DeprecationWarning("deprecated, should use calculate_pre_cov to do 4D direct computation")

    def reshape_tensor2d(self, x):
        # given a 3D tensor, reshape it to 2D.
        raise DeprecationWarning("no longer support")
        # return K.reshape(K.flatten(x.T),
        #                  (self.cols * self.rows,
        #                   self.nb_filter))


class FlattenSymmetric(Layer):
    """
    Flatten Symmetric is a layer to flatten the previous layer with symmetric matrix.

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            2D tensor with (samples, input_dim * (input_dim +1) / 2 )
            Drop the duplicated terms
        # Arguments
            name            name of the model
    """

    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        super(FlattenSymmetric, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                            'Make sure to pass a complete "input_shape" '
                            'or "batch_input_shape" argument to the first '
                            'layer in your model.')
        assert input_shape[1] == input_shape[2]
        return input_shape[0], input_shape[1]*(input_shape[1]+1)/2

    def call(self, x, mask=None):
        return K.batch_flatten(x)


class LogTransform(Layer):
    """
    LogTranform layer supports the input of a 3D tensor, output a corresponding 3D tensor in
        Log-Euclidean space

    It implement the Matrix Logarithm with a small shift (epsilon)

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            3D tensor with (samples, input_dim, input_dim)
        # Arguments
            epsilon

    """

    def __init__(self, epsilon=0, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        super(LogTransform, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "LogTransform'
                            'is not fully defined '
                            '(got ' + str( input_shape[1:]) + '. ')
        assert input_shape[1] == input_shape[2]
        return input_shape

    def get_config(self):
        """ Get config for model save and reload """
        config = {'epsilon':self.eps}
        base_config = super(LogTransform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        """
        2016.12.15 Implement with the theano.scan

        Returns
        -------
        3D tensor with same shape as input
        """
        if K.backend() == 'theano':
            from theano import scan
            components, update = scan(fn=lambda tx: self.logm(tx),
                                      outputs_info=None,
                                      sequences=[x],
                                      non_sequences=None)

            return components
        else:
            raise NotImplementedError

    def logm(self, x):
        """
        Log-transform of a 2D tensor, assume it is square and symmetric positive definite.

        Parameters
        ----------
        x : 2D square tensor

        Returns
        -------
        result : 2D square tensor with same shape
        """

        if K.backend() == 'theano':
            # construct theano tensor operation
            from theano.tensor.nlinalg import svd, diag
            from theano.tensor.elemwise import Elemwise
            from theano.scalar import log
            import theano.tensor as T
            # This implementation would be extremely slow. but efficient?
            u, d, v = svd(x)
            d += self.eps
            inner = diag(T.log(d))
            # print(inner.eval())
            res = T.dot(u, T.dot(inner, v))
            # print("U shape {} V shape {}".format(u.eval().shape, v.eval().shape))
            # print("D matrix {}".format(d.eval()))
            # assert np.allclose(u.eval(), v.eval().transpose())
            return res
        else:
            # support tensorflow implementation
            raise NotImplementedError


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
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        # Set out_dim accordingly.
        self.out_dim = output_dim

        # input parameter preset
        self.nb_samples = 0
        self.activation = activations.get(activation)
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)
        self.input_spec = [InputSpec(ndim=3)]
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

        # Create the weight vector
        self.W_shape = (input_shape[1], self.out_dim)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        else:
            self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
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
        com = K.dot(K.transpose(K.dot(x, self.W),[0,2,1]), self.W)
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
        output = K.sum(K.multiply(self.W, K.dot(x, self.W)), axis=1)
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
