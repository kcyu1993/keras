import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers import BatchNormalization


def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
          matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
      Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].

      """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
              [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
               [(row_before_length, row_after_length)]],
              axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked


class SignedSqrt(Layer):

    def __init__(self, scale=1, **kwargs):
        super(SignedSqrt, self).__init__(**kwargs)
        self.input_spec = [InputSpec(min_ndim=2)]
        self.scale = scale

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        from kyu.tensorflow.ops import safe_sign_sqrt
        return safe_sign_sqrt(self.scale * inputs)


class L2Norm(Layer):

    def __init__(self, axis=1, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.axis = axis
        self.input_spec = [InputSpec(min_ndim=2)]

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.l2_normalize(inputs, self.axis)

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {'axis': self.axis}
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    def build(self, input_shape):
        # create and store the mask
        assert input_shape[1] == input_shape[2]
        self.upper_triangular_mask = tf.constant(
            np.triu(
                np.ones((input_shape[1], input_shape[2]), dtype=np.bool_),
            0),
            dtype=tf.bool
            )
        self.built = True

    def compute_output_shape(self, input_shape):
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
        fn = lambda x : tf.boolean_mask(x, self.upper_triangular_mask)
        return tf.map_fn(fn, x)


class TransposeFlattenSymmetric(Layer):
    """
    Implement the Transposed operation for FlattenSymmetric
    """
    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='2+')]
        self.input_dim = None
        self.batch_size = None
        super(TransposeFlattenSymmetric, self).__init__(**kwargs)

    def build(self, input_shape):
        """ Input a batch-vector form """
        self.input_dim = input_shape[1]
        self.batch_size = input_shape[0]
        self.built = True

    def call(self, x, mask=None):
        """ Call the to symmetric matrices. """


class SeparateConvolutionFeatures(Layer):
    """
    SeparateConvolutionFeatures is a layer to separate previous convolution feature maps
    into groups equally.

        # Input shape
            4D tensor with (nb_sample, x, y, z)
        # Output shape
            n 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
        # Arguments
            n   should make z/n an integer

    """
    def __init__(self, n, **kwargs):
        if K.backend() == 'theano' or K.image_data_format() == 'channels_first':
            raise RuntimeError("Only support tensorflow backend or image ordering")
        self.n = n
        self.input_spec = [InputSpec(ndim='4+')]
        self.split_axis = 3
        self.output_dim = None
        self.out_shape = None
        self.split_loc = None
        super(SeparateConvolutionFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[3] / self.n
        self.out_shape = input_shape[:2] + (self.output_dim,)
        self.split_loc = [self.output_dim * i for i in range(self.n)]
        self.split_loc.append(self.output_dim * self.n)
        self.built = True

    def compute_output_shape(self, input_shape):
        """ Return a list """
        output_shape = []
        for i in range(self.n):
            output_shape.append(input_shape[:3] + (input_shape[3]/ self.n,))
        return output_shape

    def call(self, x, mask=None):

        # import tensorflow as tf
        # return tf.split(self.split_axis, self.n, x)
        outputs = []
        for i in range(self.n):
            outputs.append(x[:,:,:, self.split_loc[i]:self.split_loc[i+1]])
        return outputs

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        if input_mask is None:
            return [None for i in range(self.n)]
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))


class Regrouping(Layer):
    """
    Regrouping layer is a layer to provide different combination of given layers.

    References : keras.layer.Merge

    into groups equally.

        # Input shape
            n 4D tensor with (nb_sample, x, y, z)
        # Output shape
            C(n,2) = n*(n-1)/2 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
        # Arguments
               should make z/n an integer

    """
    def __init__(self, inputs, mode='group', concat_axis=-1,
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
                            "tensor, Got: "+ str(inputs))
        # Case: 'mode' is a lambda function or function
        if callable(self.mode):
            arguments = self.arguments
            import inspect
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


class MatrixConcat(Layer):
    """
        Regrouping layer is a layer to provide different combination of given layers.

        References : keras.layer.Merge

        into groups equally.

            # Input shape
                n 4D tensor with (nb_sample, x, y, z)
            # Output shape
                C(n,2) = n*(n-1)/2 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
            # Arguments
                   should make z/n an integer

        """

    def __init__(self, inputs, name=None):
        if K.backend() == 'theano' or K.image_dim_ordering() == 'th':
            raise RuntimeError("Only support tensorflow backend or image ordering")

        self.inputs = inputs

        # Layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None
        self.trainable = False

        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))

        self.name = name

        if inputs:
            # The inputs is a bunch of nodes shares the same input.
            self.built = True
            # self.add_inbound_node(inputs, node_indices, tensor_indices)
        else:
            self.built = False

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert (len(input_shape[0]) == 3)

        self.output_dim = input_shape[3] / self.n
        self.out_shape = input_shape[:2] + (self.output_dim,)
        self.built = True

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError("Regrouping must be taking more than one "
                            "tensor, Got: " + str(inputs))
        out = block_diagonal(inputs, K.floatx())
        return out

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        if input_mask is None or all([m is None for m in input_mask]):
            # return [None for _ in range(0, n_inputs - 1, 2)]
            return None
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))

    def compute_output_shape(self, input_shape):
        """ Return a list """
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == 3
        output_shape = list(input_shape[0])
        for i in range(1, len(input_shape)):
            output_shape[1] += input_shape[i][1]
            output_shape[2] += input_shape[i][2]
        return [tuple(output_shape), ]

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {
                  }
        base_config = super(MatrixConcat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MatrixReLU(Layer):
    """
    MatrixReLU layer supports the input of a 3D tensor, output a corresponding 3D tensor in
        Matrix diagnal ReLU case

    It implement the Matrix ReLU with a small shift (epsilon)

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
        self.out_dim = None
        # self.b = None
        super(MatrixReLU, self).__init__(**kwargs)

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
        base_config = super(MatrixReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        """
        2016.12.15 Implement with the theano.scan

        Returns
        -------
        3D tensor with same shape as input
        """
        if K.backend() == 'theano':
            # from theano import scan
            # components, update = scan(fn=lambda tx: self.logm(tx),
            #                           outputs_info=None,
            #                           sequences=[x],
            #                           non_sequences=None)
            #
            # return components
            raise ValueError("Matrix relu not supported for Theano")
        else:
            if self.built:
                import tensorflow as tf
                s, u = tf.self_adjoint_eig(x)
                comp = tf.zeros_like(s) + self.eps
                inner = tf.where(tf.less(s, comp), comp, s)
                # inner = tf.log(inner)
                # inner = tf.Print(inner, [inner], message='MatrixReLU_inner :', summarize=10)
                # inner = tf.where(tf.is_nan(inner), tf.zeros_like(inner), inner)
                inner = tf.matrix_diag(inner)
                tf_relu = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
                return tf_relu

            else:
                raise RuntimeError("Log transform layer should be built before using")


class ExpandDims(Layer):
    """ define expand dimension layer """
    def __init__(self, axis=-1):
        self.axis = axis
        super(ExpandDims, self).__init__()

    def compute_output_shape(self, input_shape):
        return input_shape[0:self.axis] + (input_shape[self.axis], 1,) + input_shape[self.axis: -1]

    def get_config(self):
        config = {'axis': self.axis
                  }
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        return tf.expand_dims(x, axis=self.axis)


class Squeeze(Layer):
    """ define expand dimension layer """
    def __init__(self, axis=-1):
        self.axis = axis
        super(Squeeze, self).__init__()

    def compute_output_shape(self, input_shape):
        assert input_shape[self.axis] == 1
        return input_shape[0:self.axis] + input_shape[self.axis: -1]

    def get_config(self):
        config = {'axis': self.axis
                  }
        base_config = super(Squeeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        return tf.squeeze(x, axis=self.axis)


class BatchNormalization_v2(BatchNormalization):
    """
    Support expand dimension batch-normalization
    """
    def __init__(self, expand_dim=True, **kwargs):
        self.expand_dim = expand_dim
        super(BatchNormalization_v2, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.expand_dim and x is not None:
            x = tf.expand_dims(x, axis=-1)

        if self.mode == 0 or self.mode == 2:
            assert self.built, 'Layer must be built before being called'
            input_shape = K.int_shape(x)

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            x_normed, mean, std = K.normalize_batch_in_training(
                x, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)

            if self.mode == 0:
                self.add_update([K.moving_average_update(self.running_mean, mean, self.momentum),
                                 K.moving_average_update(self.running_std, std, self.momentum)], x)

                if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
                    x_normed_running = K.batch_normalization(
                        x, self.running_mean, self.running_std,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        x, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                x_normed = K.in_train_phase(x_normed, x_normed_running)

        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(x, axis=-1, keepdims=True)
            std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (x - m) / (std + self.epsilon)
            x_normed = self.gamma * x_normed + self.beta

        if self.expand_dim and x is not None:
            x_normed = tf.squeeze(x_normed, squeeze_dims=-1)

        return x_normed