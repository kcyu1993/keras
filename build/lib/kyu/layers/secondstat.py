# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

import tensorflow as tf

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers import Flatten
from kyu.tensorflow.ops.cov_reg import FrobNormRegularizer, VonNeumannDistanceRegularizer, robust_estimate_eigenvalues
from kyu.tensorflow.ops.math import safe_matrix_eig_op
# TODO Remove this theano import to prevent any usage in tensorflow backend
# Potentially check Keras backend then import relevant libraries
from kyu.utils.inspect_util import get_default_args


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
        Fob_regularizer Fob norm regularizer
        init:           initialization of function.
        activation      test activation later
        cov_alpha       Use for robust estimation
        cov_beta        Use for parametric mean
        kwargs          goes into Layer construction
    '''
    def __init__(self,
                 eps=1e-5,
                 cov_mode='channel',
                 activation='linear',
                 normalization='mean',
                 cov_regularizer=None,
                 cov_alpha=0.01,
                 cov_beta=0.3,
                 use_kernel=False,
                 kernel_initializer='ones',
                 kernel_regularizer=None,
                 kernel_constraint='NonNeg',
                 alpha_initializer='ones',
                 alpha_constraint=None,
                 dim_ordering='default',
                 robust=False,
                 **kwargs):

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

        if cov_mode not in ['channel', 'feature', 'mean', 'pmean']:
            raise ValueError('only support cov_mode across channel and features and mean, given {}'.format(cov_mode))

        self.cov_mode = cov_mode

        if normalization not in ['mean', None]:
            raise ValueError('Only support normalization in mean or None, given {}'.format(normalization))
        self.normalization = normalization

        # input parameter preset
        self.nb_filter = 0
        self.cols = 0
        self.rows = 0
        self.nb_samples = 0
        self.eps = eps

        self.activation = activations.get(activation)

        self.use_kernel = use_kernel
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_constraint = constraints.get(alpha_constraint)

        ## Add the fob regularizer.
        self.cov_regulairzer = cov_regularizer
        self.cov_alpha = cov_alpha
        self.cov_beta = cov_beta
        self.robust = robust

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
        self.nb_samples = input_shape[0]
        self.nb_filter = input_shape[self.axis_filter]
        self.rows = input_shape[self.axis_row]
        self.cols = input_shape[self.axis_col]

        # Calculate covariance axis
        if self.cov_mode == 'channel' or self.cov_mode == 'mean' or self.cov_mode == 'pmean':
            self.cov_dim = self.nb_filter
            kernel_shape = (self.rows * self.cols,)
        else:
            self.cov_dim = self.rows * self.cols
            kernel_shape = (self.nb_filter, )

        # Set out_dim accordingly.
        if self.cov_mode == 'mean' or self.cov_mode == 'pmean':
            self.out_dim = self.cov_dim + 1
        else:
            self.out_dim = self.cov_dim

        if self.cov_mode == 'pmean':
            self.mean_p = self.cov_beta
            self.name += '_pm_{}'.format(self.mean_p)
            print("use parametric non_trainable {}".format(self.mean_p))

        if self.robust:
            print('use robust estimation with cov_alpha {}'.format(self.cov_alpha))
            self.name += '_rb'

        if self.use_kernel:
            self.kernel = self.add_weight(shape=kernel_shape,
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            self.kernel = None

        if self.cov_regulairzer == 'Fob':
            self.C_regularizer = FrobNormRegularizer(self.out_dim, self.cov_alpha)
            self.activity_regularizer = self.C_regularizer
        elif self.cov_regulairzer == 'vN':
            self.C_regularizer = VonNeumannDistanceRegularizer(self.out_dim, self.cov_alpha, self.eps)
            self.activity_regularizer = self.C_regularizer

        # add the alpha
        # self.alpha = self.add_weight(
            # shape=d
        # )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.out_dim, self.out_dim

    def call(self, x, mask=None):
        if not self.built:
            raise Exception("Secondary stat layer not built")
        logging.debug('Secondary_stat parameter', type(x))  # Confirm the type of x is indeed tensor4D
        cov_mat, x_mean = self.calculate_pre_cov(x)
        # print('call during second {}'.format(self.eps))
        # cov_mat += self.eps * self.b
        if self.robust:
            """ Implement the robust estimate, by apply an elementwise function to it. """
            if K.backend() != 'tensorflow':
                raise RuntimeError("Not support for theano now")
            import tensorflow as tf
            # with tf.device('/cpu:0'):
            s, u = tf.self_adjoint_eig(cov_mat)
            comp = tf.zeros_like(s)
            s = tf.where(tf.less(s, comp), comp, s)
            # s = tf.Print(s, [s], message='s:', summarize=self.out_dim)
            inner = robust_estimate_eigenvalues(s, alpha=self.cov_alpha)
            inner = tf.identity(inner, 'RobustEigen')
            # inner = tf.Print(inner, [inner], message='inner:', summarize=self.out_dim)
            cov_mat = tf.matmul(u, tf.matmul(tf.matrix_diag(inner), tf.transpose(u, [0,2,1])))

        if self.cov_mode == 'mean' or self.cov_mode == 'pmean':
            # Encode mean into Cov mat.
            addition_array = K.mean(x_mean, axis=1, keepdims=True)
            addition_array /= addition_array # Make it 1
            if self.cov_mode == 'pmean':
                x_mean = self.mean_p * x_mean
                new_cov = K.concatenate([cov_mat + K.batch_dot(x_mean, K.transpose(x_mean, (0,2,1))), x_mean])
            else:
                new_cov = K.concatenate([cov_mat, x_mean])
            tmp = K.concatenate([K.transpose(x_mean, axes=(0,2,1)), addition_array])
            new_cov = K.concatenate([new_cov, tmp], axis=1)
            cov_mat = K.identity(new_cov, 'final_cov_mat')

        return cov_mat

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'activation': self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'eps': self.eps,
                  'cov_mode': self.cov_mode
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
        if self.cov_mode == 'channel' or self.cov_mode =='mean' or self.cov_mode =='pmean':
            return tx
        else:
            return K.transpose(tx, (0,2,1))

    def calculate_pre_cov(self, x):
        """1
        4D tensor to 3D (N, nb_filter, col* row)
        :param x: Keras.tensor  (N, nb_filter, col, row) data being called
        :return: Keras.tensor   (N, nb_filter, col* row)
        """
        xf = self.reshape_tensor3d(x)
        xf_mean = K.mean(xf, axis=2, keepdims=True)
        if self.normalization == 'mean':
            xf_normal = xf - xf_mean
        else:
            xf_normal = xf
        if self.use_kernel:
            # Parametric Covariance matrix computation
            tx = K.dot(xf_normal, tf.matrix_diag(self.kernel))
            tx = K.batch_dot(tx, K.permute_dimensions(xf_normal, [0,2,1]))
        else:
            tx = K.batch_dot(xf_normal, K.transpose(xf_normal, [0,2,1]))
        # tx = K.sum(K.multiply(K.expand_dims(xf_normal, dim=1),
        #                       K.expand_dims(xf_normal, dim=2)),
        #            axis=3)
        if self.cov_mode == 'channel' or self.cov_mode == 'mean' or self.cov_mode == 'pmean':
            cov = tx / (self.rows * self.cols - 1)
            # cov = tx / (self.rows * self.cols )
        else:
            cov = tx / (self.nb_filter - 1)

        if self.normalization == None:
            # cov /= (self.rows * self.cols - 1)
            cov /= (self.rows * self.cols)
        cov = K.identity(cov, 'pre_cov')
        return cov, xf_mean

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


class Correlation(Layer):
    """
    Computer correlation layer as normalization
    
    It implements correlation computation based on a input tensor.

    by the following formula.  Corr = Cov / (std * std')
    
    """

    def __init__(self, epsilon=1e-5, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        self.out_dim = None
        super(Correlation, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.out_dim = input_shape[1]
        self.built = True

    def call(self, x, mask=None):
        """
        Call the correlation matrix.
        
        Parameters
        ----------
        x
        mask

        Returns
        -------

        """
        variance = tf.matrix_diag_part(x)
        std = tf.expand_dims(tf.sqrt(variance), axis=2)
        outer = tf.matmul(std, tf.transpose(std, [0,2,1]))
        corr = tf.div(x, outer)
        return corr
        # inner = tf.where(tf.is_nan(inner), tf.zeros_like(inner), inner)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        return input_shape

    def get_config(self):
        config = {'epsilon':self.eps}
        base_config = super(Correlation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def __init__(self, epsilon=1e-4, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        self.out_dim = None
        # self.b = None
        super(LogTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        """
                Build the model based on input shape
                Should not set the weight vector here.
                :param input_shape: (nb-sample, input_dim, input_dim)
                :return:
                """
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.out_dim = input_shape[2]
        # self.b = K.eye(self.out_dim, name='strange?')
        self.built = True

    def compute_output_shape(self, input_shape):
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
            if self.built:
                import tensorflow as tf
                s, u = safe_matrix_eig_op(x)
                eps = tf.identity(tf.zeros_like(s) + self.eps, name='eps')
                inner = tf.where(tf.less(s, eps), eps, s, name='eps_inner')
                inner = tf.log(inner, name='log_inner')
                inner = tf.matrix_diag(inner)
                tf_log = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])), name='tf_log')
                return tf_log

            else:
                raise RuntimeError("Log transform layer should be built before using")

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
            res = T.dot(u, T.dot(inner, v))
            return res
        else:
            from kyu.tensorflow.ops.svd_gradients import batch_matrix_log
            return batch_matrix_log(x, self.eps)


class PowTransform(Layer):
    """
    PowTranform layer supports the input of a 3D tensor, output a corresponding 3D tensor in
        Power Euclidean space

    References:
        Is second-order information really helpful in large-scale visual recognition?

    It implement the Matrix Logarithm with a small shift (epsilon)

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            3D tensor with (samples, input_dim, input_dim)
        # Arguments
            epsilon

    """

    def __init__(self, alpha=0.5, epsilon=1e-7, normalization='frob', **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        self.out_dim = None
        self.alpha = alpha
        self.norm = normalization
        # self.b = None
        super(PowTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        """
                Build the model based on input shape
                Should not set the weight vector here.
                :param input_shape: (nb-sample, input_dim, input_dim)
                :return:
                """
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.out_dim = input_shape[2]
        # self.b = K.eye(self.out_dim, name='strange?')
        self.built = True

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "LogTransform'
                            'is not fully defined '
                            '(got ' + str( input_shape[1:]) + '. ')
        assert input_shape[1] == input_shape[2]
        return input_shape

    def get_config(self):
        """ Get config for model save and reload """
        config = {'epsilon':self.eps}
        base_config = super(PowTransform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        """

        Returns
        -------
        3D tensor with same shape as input
        """
        if K.backend() == 'theano' or K.backend() == 'CNTK':
            raise NotImplementedError("This is not implemented for theano anymore.")
        else:
            if self.built:
                import tensorflow as tf
                from kyu.tensorflow.ops import safe_truncated_sqrt
                with tf.device('/cpu:0'):
                    s, u = safe_matrix_eig_op(x)
                inner = safe_truncated_sqrt(s)
                if self.norm == 'l2':
                    inner /= tf.reduce_max(inner)
                elif self.norm == 'frob' or self.norm == 'Frob':
                    inner /= tf.sqrt(tf.reduce_sum(s))
                elif self.norm is None:
                    pass
                else:
                    raise ValueError("PowTransform: Normalization not supported {}".format(self.norm))
                # inner = tf.Print(inner, [inner], message='power inner', summarize=65)
                inner = tf.matrix_diag(inner)
                tf_pow = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
                return tf_pow
            else:
                raise RuntimeError("PowTransform layer should be built before using")


class O2Transform(Layer):
    """ This layer shall stack one trainable weights out of previous input layer.
        Update for Keras 2. API.
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
    """

    def __init__(self,
                 output_dim,    # Cannot be None!
                 activation='relu',
                 # activation_regularizer=None,
                 # weights=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 # use_bias=False,
                 **kwargs):

        # Set out_dim accordingly.
        self.out_dim = output_dim

        # input parameter preset
        self.nb_samples = 0
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.initial_weights = weights
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
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
        kernel_shape = (input_shape[1], self.out_dim)
        # if self.initial_weights is not None:
        #     self.set_weights(self.initial_weights)
        #     del self.initial_weights
        # else:
        #     self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint
                                      )
        # self.trainable_weights = [self.W]
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        return input_shape[0], self.out_dim, self.out_dim

    def call(self, inputs):
        # result, updates = scan(fn=lambda tx: K.dot(self.W.T, K.dot(tx, self.W)),
        #                         outputs_info=None,
        #                         sequences=[x],
        #                         non_sequences=None)
        #
        com = K.dot(K.transpose(K.dot(inputs, self.kernel),[0,2,1]), self.kernel)
        # print("O2Transform shape" + com.eval().shape)
        return com

    def get_config(self):
        config = {'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'activation': activations.serialize(self.activation),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  }
        base_config = super(O2Transform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class O2Transform_v2(Layer):
    """ This layer shall stack one trainable weights out of previous input layer.

            # Input shape
                4D tensor with
                (samples, input_dim, input_dim, nb_channel)
                Note the input dim must align, i.e, must be a square matrix.

            # Output shape
                4D tensor with
                    (samples, out_dim, out_dim, nb_channel)
                This is just the 2D covariance matrix for all samples feed in, 
                with multiple channel design.

            # Arguments
                out_dim         weight matrix, if none, make it align with nb filters
                weights         initial weights.
                W_regularizer   regularize the weight if possible in future
                init:           initialization of function.
                activation      test activation later (could apply some non-linear activation here
        """

    def __init__(self, output_dim=None,
                 init='glorot_uniform', activation='relu', weights=None,
                 W_regularizer=None, dim_ordering='default',
                 W_constraint=None,
                 **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering

        # Set out_dim accordingly.
        self.out_dim = output_dim
        self.out_channel = 1
        # input parameter preset
        self.nb_samples = 0
        self.activation = activations.get(activation)
        self.init = initializers.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.input_spec = [InputSpec(ndim=3)]
        super(O2Transform_v2, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the model based on input shape
        
        :param input_shape: (nb-sample, input_dim, input_dim, nb_channel)
        :return:
        """
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2]
        self.out_channel = input_shape[3]
        # Create the weight vector
        self.W_shape = (input_shape[1], self.out_dim)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        else:
            self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        self.built = True

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == input_shape[2]
        return input_shape[0], self.out_dim, self.out_dim, self.out_channel

    def call(self, x, mask=None):
        # result, updates = scan(fn=lambda tx: K.dot(self.W.T, K.dot(tx, self.W)),
        #                         outputs_info=None,
        #                         sequences=[x],
        #                         non_sequences=None)
        #
        # com = K.dot(K.transpose(K.dot(x, self.W), [0, 2, 1]), self.W)
        batch_fn = lambda x: self.o2transform(x, self.W)
        com = tf.map_fn(batch_fn, x)
        # print("O2Transform shape" + com.eval().shape)
        return com

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  }
        base_config = super(O2Transform_v2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def o2transform(self, x, w):
        """
        take a 2D matrix as well as a weight vector.
        Parameters
        ----------
        x
        w

        Returns
        -------
    
        """

        o2t = lambda x, w: K.dot(w, K.dot(x, K.transpose(w)))
        return tf.map_fn(o2t, [x, w])


class WeightedVectorization(Layer):
    """ Probability weighted vector layer for secondary image statistics
    neural networks. It is simple at this time, just v_c.T * Cov * v_c, with
    basic activitation function such as ReLU, softmax, thus the

        Version 0.1: Implement the basic weighted probablitity coming from cov-layer
        Version 0.2: Implement trainable weights to penalized over-fitting
        Version 0.3: Change to Keras 2 API.

    """

    def __init__(self, output_dim,
                 input_dim=None,
                 activation='linear',
                 eps=1e-8,
                 output_sqrt=False,     # Normalization
                 normalization=False,   # normalize to further fit Chi-square distribution
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 use_bias=False,  # use bias for normalization additional
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 use_gamma=False,  # use gamma for general gaussian distribution
                 gamma_initializer='ones',
                 gamma_regularizer='l2',
                 gamma_constraint=None,
                 activation_regularizer=None,
                 **kwargs):

        # self parameters
        self.output_sqrt = output_sqrt
        self.normalization = normalization
        self.eps = eps
        self.input_dim = input_dim  # Squared matrix input, as property of cov matrix
        self.output_dim = output_dim  # final classified categories number
        if output_dim is None:
            raise ValueError("Output dim must be not None")

        if activation_regularizer in ('l2', 'l1',None):
            self.activation_regularizer = regularizers.get(activation_regularizer)
        else:
            raise ValueError("Activation regularizer only support l1, l2, None. Got {}".format(activation_regularizer))

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_gamma = use_gamma
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)

        self.activation = activations.get(activation)
        super(WeightedVectorization, self).__init__(**kwargs)

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
        if self.output_dim is None:
            print("Wrong ! Should not be a None for output_dim")
        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel'
                                      )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias'
                                        )

        else:
            self.bias = None
        if self.use_gamma:
            self.gamma = self.add_weight(shape=(self.output_dim,),
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='gamma'
                                         )
        else:
            self.gamma = None
        self.built = True

    def call(self, inputs):
        '''
        The calculation of call function is not trival.
        sum( self.W .* ( x * self.W) ) along axis 1
        :param x:
        :param mask:
        :return: final output vector with w_i^T * W * w_i as item i, and propagate to all
            samples. Output Shape (nb_samples, vector c)
        '''
        # logging.debug("prob_out: x_shape {}".format(K.shape(inputs)))
        # new_W = K.expand_dims(self.W, dim=1)
        if K.backend() == 'tensorflow':
            output = K.sum((self.kernel * K.dot(inputs, self.kernel)), axis=1)
        else:
            raise NotImplementedError("Not support for other backend. ")

        if self.normalization:
            # make kernel
            output /= K.sum(K.pow(self.kernel, 2), axis=0)

        if self.output_sqrt:
            from kyu.tensorflow.ops import safe_sign_sqrt
            output = safe_sign_sqrt(output)
        if self.use_gamma:
            output *= self.gamma

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=K.image_data_format())

        if self.activation_regularizer:
            return self.activation_regularizer(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
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
                  'input_dim': self.input_dim,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'use_gamma': self.use_gamma,
                  'normalization': self.normalization,
                  'output_sqrt': self.output_sqrt,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'gamma_initializer': initializers.serialize(self.gamma_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'activation_regularizer': regularizers.serialize(self.activation_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'gamma_constraint': constraints.serialize(self.gamma_constraint)
                  }
        base_config = super(WeightedVectorization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalSquarePooling(Layer):
    """
    develop the equivalent format of BN-Cov-PV
    """

    def __init__(self, output_dim,
                 input_dim=None,
                 activation='linear',
                 eps=1e-8,
                 output_sqrt=False,  # Normalization
                 normalization=False,  # normalize to further fit Chi-square distribution
                 # kernel_initializer='glorot_uniform',
                 # kernel_constraint=None,
                 # kernel_regularizer=None,
                 use_bias=False,  # use bias for normalization additional
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 use_gamma=False,  # use gamma for general gaussian distribution
                 gamma_initializer='ones',
                 gamma_regularizer='l2',
                 gamma_constraint=None,
                 activation_regularizer=None,
                 **kwargs):
        self.data_format = K.image_data_format()
        # self parameters
        self.output_sqrt = output_sqrt
        self.normalization = normalization
        self.eps = eps
        self.input_dim = input_dim  # Squared matrix input, as property of cov matrix
        self.output_dim = output_dim  # final classified categories number
        if output_dim is None:
            raise ValueError("Output dim must be not None")

        if activation_regularizer in ('l2', 'l1', None):
            self.activation_regularizer = regularizers.get(activation_regularizer)
        else:
            raise ValueError("Activation regularizer only support l1, l2, None. Got {}".format(activation_regularizer))

        # self.kernel_initializer = initializers.get(kernel_initializer)
        # self.kernel_constraint = constraints.get(kernel_constraint)
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.use_beta = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_gamma = use_gamma
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)

        self.activation = activations.get(activation)
        super(GlobalSquarePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build function
        :param input_shape:
        :return:
        """
        # 3D tensor (nb_samples, n_cov, n_cov)
        assert len(input_shape) == 4

        input_dim = input_shape[1]
        if self.output_dim is None:
            print("Wrong ! Should not be a None for output_dim")

        if self.use_beta:
            self.beta = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias'
                                        )

        else:
            self.beta = None
        if self.use_gamma:
            self.gamma = self.add_weight(shape=(self.output_dim,),
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='gamma'
                                         )
        else:
            self.gamma = None
        self.built = True

    def call(self, inputs):
        '''
        The calculation of call function is not trival.
        sum( self.W .* ( x * self.W) ) along axis 1
        :param x:
        :param mask:
        :return: final output vector with w_i^T * W * w_i as item i, and propagate to all
            samples. Output Shape (nb_samples, vector c)
        '''
        # logging.debug("prob_out: x_shape {}".format(K.shape(inputs)))
        # new_W = K.expand_dims(self.W, dim=1)
        if K.backend() == 'tensorflow':
            output = K.square(x=inputs)
        else:
            raise NotImplementedError("Not support for other backend. ")

        # Implement the Global Average Pooling
        if self.data_format == 'channels_last':
            output = K.mean(output, axis=[1,2])
        else:
            output = K.mean(output, axis=[2,3])

        # if self.normalization:
        #     make kernel
            # output /= K.sum(K.pow(self.kernel, 2), axis=0)

        if self.output_sqrt:
            from kyu.tensorflow.ops import safe_sign_sqrt
            output = safe_sign_sqrt(output)
        if self.use_gamma:
            output *= self.gamma

        if self.use_beta:
            output = K.bias_add(output, self.beta, data_format=K.image_data_format())

        if self.activation_regularizer:
            return self.activation_regularizer(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
        # same as GlobalPooling2D
        logging.debug(input_shape)
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_beta,
                  'use_gamma': self.use_gamma,
                  'normalization': self.normalization,
                  'output_sqrt': self.output_sqrt,
                  # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'gamma_initializer': initializers.serialize(self.gamma_initializer),
                  # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'activation_regularizer': regularizers.serialize(self.activation_regularizer),
                  # 'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'gamma_constraint': constraints.serialize(self.gamma_constraint)
                  }
        base_config = super(GlobalSquarePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiLinear(Layer):
    """
    Define the BiLinear layer for comparison
    It operates similar to Cov layer, but without removing the mean.
    It then passed it into a square root operation of the output matrix and then to
        
    """
    def __init__(self, eps=1e-5,
                 bilinear_mode='channel',
                 activation='linear',
                 dim_ordering='default',
                 **kwargs):
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
        if bilinear_mode not in ['channel', 'feature', 'mean', 'pmean']:
            raise ValueError('only support cov_mode across channel and features and mean, given {}'.
                             format(bilinear_mode))

        self.bilinear_mode = bilinear_mode

        # input parameter preset
        self.nb_filter = 0
        self.cols = 0
        self.rows = 0
        self.nb_samples = 0
        self.eps = eps

        self.activation = activations.get(activation)

        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        super(BiLinear, self).__init__(**kwargs)

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
        # print('secondary_stat: input shape lenth', len(input_shape))
        # print('second_stat: input shape {}'.format(input_shape))
        # print('second_stat: axis filter {}'.format(self.axis_filter))
        self.nb_samples = input_shape[0]
        self.nb_filter = input_shape[self.axis_filter]
        self.rows = input_shape[self.axis_row]
        self.cols = input_shape[self.axis_col]

        # Calculate covariance axis
        if self.bilinear_mode == 'channel' or self.bilinear_mode == 'mean' or self.bilinear_mode == 'pmean':
            self.cov_dim = self.nb_filter
        else:
            self.cov_dim = self.rows * self.cols

        # Set out_dim accordingly.
        self.out_dim = self.cov_dim
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.out_dim*self.out_dim

    def get_config(self):
        config = {
            'eps':1e-5,
            'activation':self.activation.__name__,
            'dim_ordering':self.dim_ordering,
            'bilinear_mode':self.bilinear_mode,
        }
        base_config = super(BiLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        if not self.built:
            raise Exception("BiLinear layer not built")
        xf = self.reshape_tensor3d(x)
        tx = K.batch_dot(xf, K.transpose(xf, [0,2,1]))
        if self.bilinear_mode == 'channel' or self.bilinear_mode == 'mean' or self.bilinear_mode == 'pmean':
            bilinear_output = tx / (self.rows * self.cols - 1)
        #     # cov = tx / (self.rows * self.cols )
        else:
            bilinear_output = tx / (self.nb_filter - 1)
        from kyu.tensorflow.ops import safe_sign_sqrt
        bilinear_output = safe_sign_sqrt(bilinear_output)
        # bilinear_output = K.sign(bilinear_output) * K.sqrt(K.abs(bilinear_output))
        bilinear_output = Flatten()(bilinear_output)
        bilinear_output = K.l2_normalize(bilinear_output, axis=1)
        return bilinear_output

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
            tx = K.transpose(tx, (0, 2, 1))
        if self.bilinear_mode == 'channel' or self.bilinear_mode == 'mean' or self.bilinear_mode == 'pmean':
            return tx
        else:
            return K.transpose(tx, (0, 2, 1))


class BiLinear_v2(Layer):
    """
    BiLinear v2 is a layer which can take two inputs and group them in the interaction style, that can 
    take from multiple layers like Merge.
    
    Add support for two input
    
    References : keras.layer.Regroup

    into groups equally.

        # Input shape
            n 4D tensor with (nb_sample, x, y, z)
        # Output shape
            C(n,2) = n*(n-1)/2 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
        # Arguments
               should make z/n an integer

    """
    def __init__(self, inputs, mode='bilinear', concat_axis=-1,
                 output_shape=None, output_mask=None,
                 arguments=None, node_indices=None, tensor_indices=None,
                 name=None, version=1,
                 ):
        if K.backend() == 'theano' or K.image_dim_ordering() == 'th':
            raise RuntimeError("Only support tensorflow backend or image ordering")

        self.inputs = inputs
        self.mode = mode
        self.concat_axis = concat_axis
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask
        self.arguments = arguments if arguments else {}

        # Layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None

        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))

        self.name = name

        if inputs:
            # The inputs is a bunch of nodes shares the same input.
            if not node_indices:
                node_indices = [0 for _ in range(len(inputs))]
            self.built = True
            # self.add_inbound_node(inputs, node_indices, tensor_indices)
        else:
            self.built = False

    # def build(self, input_shape):
    #     self.output_dim = input_shape[3] / self.n
    #     self.out_shape = input_shape[:2] + (self.output_dim,)
    #     self.split_loc = [self.output_dim * i for i in range(self.n)]
    #     self.split_loc.append(self.output_dim * self.n)
    #     self.built = True

    def call(self, inputs, mask=None):
        import tensorflow as tf
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError("Regrouping must be taking more than one "
                            "tensor, Got: " + str(inputs))
        # Case: 'mode' is a lambda function or function
        if callable(self.mode):
            import inspect
            arguments = self.arguments
            arg_spec = inspect.getargspec(self.mode)
            if 'mask' in arg_spec.args:
                arguments['mask'] = mask
            return self.mode(inputs, **arguments)

        if self.mode == 'group':

            outputs = []
            n_inputs = len(inputs)
            for i in range(n_inputs - 1):
                for j in range(i + 1, n_inputs):
                    with tf.device('/gpu:0'):
                        outputs.append(K.concatenate([tf.identity(inputs[i]),tf.identity(inputs[j])], self.concat_axis))
            # for i in range(0, n_inputs - 1, 2):
            #     with tf.device('/gpu:0'):
            #         conc = K.concatenate([tf.identity(inputs[i]), tf.identity(inputs[i+1])])
            #     outputs.append(conc)
            return outputs
        else:
            raise RuntimeError("Mode not recognized {}".format(self.mode))

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        n_inputs = len(input)
        if input_mask is None or all([m is None for m in input_mask]):
            # return [None for _ in range(0, n_inputs - 1, 2)]
            return [None for _ in range(n_inputs * (n_inputs - 1) / 2)]
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))

    def compute_output_shape(self, input_shape):
        """ Return a list """
        assert isinstance(input_shape, list)

        output_shape = []
        n_inputs = len(input_shape)
        for i in range(0, n_inputs - 1):
            for j in range(i, n_inputs - 1):
                tmp_shape = list(input_shape[i])
                tmp_shape[self.concat_axis] += input_shape[j][self.concat_axis]
                output_shape.append(tmp_shape)
            # tmp_shape = list(input_shape[i])
            # tmp_shape[self.concat_axis] += input_shape[i+1][self.concat_axis]
            # output_shape.append(tmp_shape)
        return output_shape

# Alias
Cov = SecondaryStatistic
O2T = O2Transform
PV = WeightedVectorization


def get_custom_objects():
    """
    Get custom objects for Keras pipeline

    Returns
    -------

    """
    return globals()


def get_default_secondstat_args(identifier, custom_object=None, method_name='__init__'):
    """
    Obtain the default second_stat layers args

    Parameters
    ----------
    identifier
    custom_object
    method_name

    Returns
    -------

    """
    return get_default_args(identifier,
                            module_object=get_custom_objects(),
                            custom_object=custom_object,
                            method_name=method_name)

