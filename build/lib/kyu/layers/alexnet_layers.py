from keras import backend as K
from keras.engine import Layer


class LocalResponsePooling(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(LocalResponsePooling, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        import tensorflow as tf
        return tf.nn.local_response_normalization(inputs,depth_radius=self.k,
                                                  alpha=self.alpha,
                                                  beta=self.beta,
                                                  bias=1.0)

    def get_config(self):
        config = {'alpha': self.alpha,
                  'k': self.k,
                  'beta': self.beta,
                  'n': self.n}
        base_config = super(LocalResponsePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CrossChannelNormalization(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(CrossChannelNormalization, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        # import tensorflow as tf
        # tf.nn.local_response_normalization()
        X = inputs
        b, r, c, ch = K.int_shape(X)
        half = self.n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
                                              , ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * extra_channels[:, :, :, i:i + ch]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {'alpha': self.alpha,
                  'k': self.k,
                  'beta': self.beta,
                  'n': self.n}
        base_config = super(CrossChannelNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SplitTensor(Layer):

    def __init__(self, axis=3, ratio_split=1, id_split=0, **kwargs):
        super(SplitTensor, self).__init__(**kwargs)
        self.axis = axis
        self.ratio_split = ratio_split
        self.id_split = id_split

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = output_shape[self.axis] // self.ratio_split
        return tuple(output_shape)

    def call(self, inputs, **kwargs):
        X = inputs
        axis = self.axis
        ratio_split = self.ratio_split
        id_split = self.id_split
        div = K.int_shape(X)[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def get_config(self):
        config ={'axis':self.axis,
                 'ratio_split':self.ratio_split,
                 'id_split':self.id_split}
        base_config = super(SplitTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))