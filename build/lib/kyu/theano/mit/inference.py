


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


    '''
    def __init__(self, eps=1e-5,
                 cov_mode='channel',
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, dim_ordering='default',
                 normalization='mean',
                 cov_regularizer=None, cov_alpha=0.01, cov_beta=0.3,
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

        # if cov_mode is 'channel':
        #     self.axis_cov = (self.axis_filter,)
        #     self.axis_non_cov = (self.axis_row, self.axis_col)
        # elif cov_mode is 'feature':
        #     self.axis_cov = (self.axis_row, self.axis_col)
        #     self.axis_non_cov = (self.axis_filter,)
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

        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.initial_weights = weights
        self.W_regularizer = regularizers.get(W_regularizer)

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
        # print('secondary_stat: input shape lenth', len(input_shape))
        # print('second_stat: input shape {}'.format(input_shape))
        # print('second_stat: axis filter {}'.format(self.axis_filter))
        self.nb_samples = input_shape[0]
        self.nb_filter = input_shape[self.axis_filter]
        self.rows = input_shape[self.axis_row]
        self.cols = input_shape[self.axis_col]

        # Calculate covariance axis
        if self.cov_mode == 'channel' or self.cov_mode == 'mean' or self.cov_mode == 'pmean':
            self.cov_dim = self.nb_filter
        else:
            self.cov_dim = self.rows * self.cols

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

        if self.cov_regulairzer == 'Fob':
            self.C_regularizer = FrobNormRegularizer(self.out_dim, self.cov_alpha)
            self.activity_regularizer = self.C_regularizer
        elif self.cov_regulairzer == 'vN':
            self.C_regularizer = VonNeumannDistanceRegularizer(self.out_dim, self.cov_alpha, self.eps)
            self.activity_regularizer = self.C_regularizer
        self.built = True

    def get_output_shape_for(self, input_shape):
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
            s, u = tf.self_adjoint_eig(cov_mat)
            comp = tf.zeros_like(s)
            s = tf.where(tf.less(s, comp), comp, s)
            # s = tf.Print(s, [s], message='s:', summarize=self.out_dim)
            inner = robust_estimate_eigenvalues(s, alpha=self.cov_alpha)
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
            cov_mat = new_cov

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
        """
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
            cov /= (self.rows * self.cols )
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
