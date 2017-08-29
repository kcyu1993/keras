"""
ilsvrc1000.py
ILSVRC 1000 Image Net Challenge

This file implements basic ImageNet VGG model training.

Base model:
    VGG 19

Second-layer structure:
    Cov-branch: O2Transform (Non-Parametric and Parametric layers)
    Following the same mode in CIFAR
        Mode 0: Base-line model
        Mode 1: Without top layer, but only the


"""
import warnings
from keras.engine import merge

from keras.utils.layer_utils import convert_all_kernels_in_model

from keras.applications.vgg16 import TH_WEIGHTS_PATH, TF_WEIGHTS_PATH

from keras.utils.data_utils import get_file

import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from kyu.models.so_cnn_helper import covariance_block_original, covariance_block_vector_space


def VGG16_with_second(parametrics=[], mode=0,
                      nb_classes=1000, init='glorot_normal',
                      cov_mode='o2transform', cov_branch_output=None,
                      dense_after_covariance=True,
                      weights='imagenet',
                      input_tensor=None, input_shape=None,
                      trainable=True):
    '''Instantiate the VGG16 architecture,
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
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    basename = 'VGG16_ILSVRC'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    model_name = basename + 'mode_{}'.format(str(mode))

    if cov_mode == 'o2transform':
        covariance_block = covariance_block_original
    elif cov_mode == 'dense':
        covariance_block = covariance_block_vector_space
    else:
        raise ValueError("Cov mode must be o2transform or dense")
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if cov_branch_output is None:
        cov_branch_output = nb_classes

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    block_1 = x

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    block_2 = x

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    block_3 = x

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    block_4 = x

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    block_5 = x

    if mode == 0:
        # VGG baseline model
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)
    elif mode == 1:
        x = covariance_block(x, cov_branch_output, stage=6, block='a', parametric=parametrics)
        x = Dense(1000, activation='softmax', name='n_predictions')(x)
    elif mode == 2:
        cov_branch_input = block_2
        cov_branch = covariance_block(cov_branch_input, cov_branch_output, stage=6, block='a', parametric=parametrics)

        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1', trainable=False)(x)
        x = Dense(4096, activation='relu', name='fc2', trainable=False)(x)
        x = Dense(1000, activation='relu', name='predictions_ori', trainable=True)(x)

        x = merge([cov_branch,x], mode='concat', name='concat')
        x = Dense(1000, activation='softmax', name='predictions_f')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name=model_name)

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':

            weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')

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

            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
            model.load_weights(weights_path, by_name=True)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


