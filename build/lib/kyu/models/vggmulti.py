"""
Re implement VGG model for general usage

"""
from keras.utils import get_file, layer_utils

from keras import Input

import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine import Model, get_source_inputs
from keras.layers import Flatten, Dense, warnings, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.applications.vgg16 import WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP
from kyu.models.so_cnn_helper import covariance_block_original
from kyu.legacy.so_wrapper import dcov_model_wrapper_v1, dcov_model_wrapper_v2, dcov_multi_out_model_wrapper


def VGG16_Multi(include_top=True, weights='imagenet',
                input_tensor=None, input_shape=None,
                pooling='max',
                last_avg=True):
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
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    if pooling == 'max':
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    elif pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block2_pool')(x)


    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    if pooling == 'max':
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    elif pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block3_pool')(x)
    block1_x = x

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    if pooling == 'max':
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    elif pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block4_pool')(x)
    block2_x = x

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)

    if last_avg:
        if pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        elif pooling == 'avg':
            x = AveragePooling2D((2,2), strides=(2,2), name='block5_pool')(x)
    block3_x = x

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, [block1_x, block2_x, block3_x], name='vgg16_multi')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def VGG16Multi_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
                  load_weights='imagenet',
                  cov_mode='channel',
                  cov_branch='o2transform',
                  cov_branch_output=None,
                  last_avg=False,
                  freeze_conv=False,
                  cov_regularizer=None,
                  nb_branch=1,
                  concat='concat',
                  last_conv_feature_maps=[],
                  pooling='max',
                  **kwargs
            ):
    """

    Parameters
    ----------
    parametrics
    mode
    nb_classes
    input_shape
    load_weights
    cov_mode
    cov_branch
    cov_branch_output
    cov_block_mode
    last_avg
    freeze_conv

    Returns
    -------

    """
    if load_weights in {"imagenet", "secondorder"}:
        base_model = VGG16_Multi(include_top=False, input_shape=input_shape, last_avg=last_avg, pooling=pooling)
    elif load_weights is None:
        base_model = VGG16_Multi(include_top=False, weights=None, input_shape=input_shape,last_avg=last_avg, pooling=pooling)
    else:
        base_model = VGG16_Multi(include_top=False, weights=None, input_shape=input_shape,last_avg=last_avg, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    basename = 'VGG16_o2_multi'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}_{}'.format(str(mode), cov_branch)

    if nb_branch == 1:
        # model = dcov_model_wrapper_v1(
        #     base_model, parametrics, mode, nb_classes, basename,
        #     cov_mode, cov_branch, cov_branch_output, freeze_conv,
        #     cov_regularizer, nb_branch, concat, last_conv_feature_maps,
        #     **kwargs
        # )
        raise NotImplementedError
    else:
        model = dcov_multi_out_model_wrapper(
            base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    return model


def VGG16_o2_with_config(param, mode, cov_output, config, nb_class, input_shape, **kwargs):
    """ API to ResNet50 o2, create by config """
    z = config.kwargs.copy()
    z.update(kwargs)
    return VGG16Multi_o2(parametrics=param, mode=mode, nb_classes=nb_class, cov_branch_output=cov_output,
                         input_shape=input_shape, last_avg=False, freeze_conv=False,
                         cov_regularizer=config.cov_regularizer, last_conv_feature_maps=config.last_conv_feature_maps,
                         nb_branch=config.nb_branch, cov_mode=config.cov_mode, epsilon=config.epsilon,
                         pooling=config.pooling,
                         **z
                         )


if __name__ == '__main__':
    # model = VGG16_o1([4096,4096,4096], input_shape=(224,224,3))
    model = VGG16Multi_o2([256, 256, 128], nb_branch=2, cov_mode='pmean', freeze_conv=True,
                          cov_branch_output=64,
                          robust=True, mode=1, nb_classes=47)
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
