# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function
from __future__ import absolute_import

import warnings

from ..layers import merge, Input
from ..layers import Dense, Activation, Flatten
from ..layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from ..layers import BatchNormalization, SecondaryStatistic, WeightedProbability, O2Transform
from ..models import Model
from .. import backend as K
from ..engine.topology import get_source_inputs
from ..utils.layer_utils import convert_all_kernels_in_model
from ..utils.data_utils import get_file
from .imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv_block_original(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, subsample=strides,
                      name=conv_name_base + '2a', border_mode="same")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Convolution2D(nb_filter2, kernel_size, kernel_size, subsample=strides,border_mode='same',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_original(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size, kernel_size, name=conv_name_base + '2a', border_mode='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def covariance_block_original(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu'):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
    x = WeightedProbability(nb_class, activation=activation, name=wp_name_base)(x)
    return x


def covariance_block_vector_space(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu'):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    dense_name_base = 'dense' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon)(input_tensor)
    x = Flatten()(x)
    for id, param in enumerate(parametric):
        x = Dense(param, activation=activation, name=dense_name_base + str(id))(x)
    x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `tf` dim ordering)
            or `(3, 224, 244)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
            else:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model



def ResCovNet50(weights='imagenet', nb_class=10, include_top=True):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    img_input = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    # Define the parametric layers
    parametrics = []

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    if include_top:
        # Make things concatenate
        cov_stat = covariance_block_vector_space(x, nb_class, stage=6, block='',
                                                 parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_stat], mode='concat', concat_axis=0)
        x = Dense(nb_class, activation='softmax', name='fcs')(x)
    else:
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
        return x, img_input, weights_path

    model = Model(img_input, x)

    # loads weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
            else:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


def ResNet50MINC(include_top=True, weights='imagenet',
                 nb_class=23):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape

    input_shape = (3, 224, 224)

    img_input = Input(shape=input_shape)
    if not K.is_keras_tensor(img_input):
        img_input = Input(tensor=img_input, shape=input_shape)

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc23')(x)
    else:
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
        return x, img_input, weights_path

    model = Model(img_input, x)

    # loads weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
            else:
                weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


def ResCovNet50MINC(parametrics=[], input_tensor=None, nb_class=23, mode=0):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.
        cov_mode:   {0,1,2,3 ..}
             cov_mode 0: concat in the final layer {FC256, WP 10} then FC 10
             cov_mode 1: concat in the second last {FC10, WP 10} then FC 10
             cov_mode 2: sum in the second last {relu(FC10) + relu(WP 10)} -> softmax
             cov_mode 3: sum in the second last {soft(FC10) + soft(WP 10)} -> softmax
             cov_mode 4 - 6 are Cov of other layers information, from resnet stage 2, 3, 4
                only differs during last combining phase
             cov_mode 4:  concat{FC4096, WP1, WP2, WP3} -> softmax
             cov_mode 5: concat{FC_23, WP1, WP2, WP3}
             cov_mode 6: concat{FC_23, sum(WP1, WP2, WP3)}

    # Returns
        A Keras model instance.
    '''

    basename = 'ResCov_MINC'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    block1_x = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    block2_x = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    block3_x = x

    cov_input = block3_x
    if mode == 0:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 1:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 2:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='sum', name='sum')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 3:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        cov_branch = Activation('softmax')(cov_branch)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc')(x)
        x = merge([x, cov_branch], mode='sum', name='sum')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 4:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 5:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 6:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='sum', name='sum')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(img_input, x, name=basename + "mode_" + str(mode))
    return model


def ResNet50CIFAR(include_top=True, second=False, input_tensor=None, nb_class=10):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 32, 32)
    else:
        input_shape = (32, 32, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # x = ZeroPadding2D((2, 2))(img_input)
    x = Convolution2D(16, 3, 3, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block_original(x, 3, [16, 16], stage=2, block='a', strides=(1,1))
    x = identity_block_original(x, 3, [16, 16], stage=2, block='b')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='c')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='d')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='e')

    x = conv_block_original(x, 3, [32, 32], stage=3, block='a', strides=(2,2))
    x = identity_block_original(x, 3, [32, 32], stage=3, block='b')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='c')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='d')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='e')

    x = conv_block_original(x, 3, [64, 64], stage=4, block='a', strides=(2,2))
    x = identity_block_original(x, 3, [64, 64], stage=4, block='b')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='c')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='d')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='e')

    if second:
        x = AveragePooling2D((3, 3), name='avg_pool')(x)

    if include_top:
        x = AveragePooling2D((3, 3), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc')(x)
    else:
        return x, img_input

    model = Model(img_input, x)

    return model


def ResCovNet50CIFAR(parametrics=[], input_tensor=None, nb_class=10, mode=0):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.
        cov_mode:   {0,1,2,3 ..}
             cov_mode 0: concat in the final layer {FC256, WP 10} then FC 10
             cov_mode 1: concat in the second last {FC10, WP 10} then FC 10
             cov_mode 2: sum in the second last {relu(FC10) + relu(WP 10)} -> softmax
             cov_mode 3: sum in the second last {soft(FC10) + soft(WP 10)} -> softmax

    # Returns
        A Keras model instance.
    '''

    basename = 'ResCov_CIFAR'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 32, 32)
    else:
        input_shape = (32, 32, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # x = ZeroPadding2D((2, 2))(img_input)
    x = Convolution2D(16, 3, 3, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block_original(x, 3, [16, 16], stage=2, block='a', strides=(1,1))
    x = identity_block_original(x, 3, [16, 16], stage=2, block='b')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='c')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='d')
    x = identity_block_original(x, 3, [16, 16], stage=2, block='e')
    block1_x = x

    x = conv_block_original(x, 3, [32, 32], stage=3, block='a', strides=(2,2))
    x = identity_block_original(x, 3, [32, 32], stage=3, block='b')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='c')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='d')
    x = identity_block_original(x, 3, [32, 32], stage=3, block='e')
    block2_x = x

    x = conv_block_original(x, 3, [64, 64], stage=4, block='a', strides=(2,2))
    x = identity_block_original(x, 3, [64, 64], stage=4, block='b')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='c')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='d')
    x = identity_block_original(x, 3, [64, 64], stage=4, block='e')
    block3_x = x

    x = AveragePooling2D((3, 3), name='avg_pool')(x)

    cov_input = block3_x
    if mode == 0:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 1:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 2:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='sum', name='sum')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 3:
        cov_branch = covariance_block_vector_space(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        cov_branch = Activation('softmax')(cov_branch)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc')(x)
        x = merge([x, cov_branch], mode='sum', name='sum')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 4:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 5:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 6:
        cov_branch1 = covariance_block_vector_space(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block_vector_space(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block_vector_space(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='sum', name='sum')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(img_input, x, name=basename + "mode_" + str(mode))
    return model
