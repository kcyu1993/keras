import warnings

from keras import backend as K, Input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import Layer, get_source_inputs, Model
from keras.layers import Lambda, Conv2D, MaxPool2D, ZeroPadding2D, Flatten, Dense, Dropout, Activation
from keras.utils import get_file, layer_utils
from kyu.utils.train_utils import toggle_trainable_layers

WEIGHTS_PATH = 'http://files.heuritech.com/weights/alexnet_weights.h5'


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


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        # For tensorflow tensor
        b, r, c, ch = K.int_shape(X)
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
                                              , ((0,0), (half,half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, :, :, i:i + ch]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


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


def splittensor(axis=3, ratio_split=1, id_split=0, **kwargs):
    def f(X):

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

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def AlexNet_v2(
        nb_class=1000,
        input_shape=None,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        last_pooling=True,
        # weight_decay=0,
        freeze_conv=False,
        nb_outputs=1):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and nb_class != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    model_outputs = []

    conv_1 = Conv2D(96, (11, 11), strides=(4, 4),
                    activation='relu', name='conv_1')(img_input)

    conv_2 = MaxPool2D((3, 3), strides=(2, 2))(conv_1)
    # conv_2 = CrossChannelNormalization(name='convpool_1')(conv_2)
    conv_2 = LocalResponsePooling(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    # conv_2 = concatenate([
    #                    Conv2D(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
    #                        SplitTensor(ratio_split=2, id_split=i)(conv_2)
    #                    ) for i in range(2)], axis=3, name='conv_2')

    conv_2 = Conv2D(256, (5,5), activation='relu', name='conv_2')(conv_2)

    conv_3 = MaxPool2D((3, 3), strides=(2, 2))(conv_2)
    # conv_3 = CrossChannelNormalization(name='convpool_2')(conv_3)
    conv_3 = LocalResponsePooling(name='convpool_2')(conv_3)
    # model_outputs.append(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    # conv_4 = concatenate([
    #                    Conv2D(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
    #                        SplitTensor(ratio_split=2, id_split=i)(conv_4)
    #                    ) for i in range(2)], axis=3, name='conv_4')
    conv_4 = Conv2D(384, (3,3), activation='relu', name='conv_4')(conv_4)
    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    # conv_5 = concatenate([
    #                    Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
    #                        SplitTensor(ratio_split=2, id_split=i)(conv_5)
    #                    ) for i in range(2)], axis=3, name='conv_5')
    conv_5 = Conv2D(256, (3,3), activation='relu', name='conv_5')(conv_5)

    if pooling == 'max':
        dense_1 = MaxPool2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    else:
        dense_1 = conv_5

    # process image input
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        # Create model.

    if include_top:

        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        if nb_class != 1000:
            pred_name = 'prediction'
        else:
            pred_name = 'dense_3'
        base_model = Model(inputs, dense_3, name='alexnet-base')
        toggle_trainable_layers(base_model, not freeze_conv)
        dense_3 = Dense(nb_class, name=pred_name)(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)
        model = Model(inputs, prediction, name='alexnet')
    else:
        nb_outputs = 1 if nb_outputs <= 1 else nb_outputs
        model_outputs.append(dense_1)
        model_outputs.reverse()
        model = Model(inputs, model_outputs[:nb_outputs],
                      name='alexnet-base-{}_out'.format(nb_outputs))
        toggle_trainable_layers(model, not freeze_conv)

    if weights == 'imagenet':
        weights_path = get_file('alexnet_weights.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')

        model.load_weights(weights_path, by_name=True)

        # default weight in Theano, so convert to tensorflow.
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                last_layer = model.get_layer(name='convpool_5') \
                    if last_pooling else model.get_layer(name='conv_5')
                shape = last_layer.output_shape[1:]
                dense = model.get_layer(name='dense_1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_last')
            if K.backend() == 'Theano':
                warnings.warn('You are using Theano backend, yte you '
                              'are using the TensorFlow '
                              'image data convention.')

    return model