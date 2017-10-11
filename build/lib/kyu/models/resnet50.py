from __future__ import absolute_import
from __future__ import print_function

import warnings

from kyu.utils.train_utils import toggle_trainable_layers

from keras.utils import get_file, layer_utils

from keras.engine import get_source_inputs, Model
from keras.layers import MaxPooling2D, Activation, BatchNormalization, Conv2D, AveragePooling2D, Flatten, Dense, \
    GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import add as merge_add
from keras.regularizers import l2
from keras.applications.resnet50 import WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP

from keras import Input
import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=1e-4):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge_add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), weight_decay=1e-4):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1',
                      kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge_add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50_v2(
        nb_class=1000, input_shape=None,
        include_top=True, weights='imagenet',
        input_tensor=None,
        pooling=None,
        last_avg=True,
        weight_decay=0,
        freeze_conv=False,
        nb_outputs=1,
        ):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', 'secondorder', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet) or '
                         '`secondorder` (pretrained on ImageNet for SO '
                         'structure.')

    if weights == 'imagenet' and include_top and nb_class != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
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
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    model_outputs = []

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1',
        kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay)
    model_outputs.append(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay)
    model_outputs.append(x)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay)
    model_outputs.append(x)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay)

    if last_avg:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        # Create model.

    if include_top:
        x = Flatten()(x)
        if nb_class != 1000:
            pred_name = 'new_pred'
        else:
            pred_name = 'fc1000'
        base_model = Model(inputs, x, name='resnet50-base')
        toggle_trainable_layers(base_model, not freeze_conv)
        x = Dense(nb_class, activation='softmax', name=pred_name)(x)
        model = Model(inputs, x, name='resnet50')
    else:
        # Handle multiple-outputs only here.
        nb_outputs = 1 if nb_outputs <= 1 else nb_outputs
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        model_outputs.append(x)
        model_outputs.reverse()

        model = Model(inputs, model_outputs[:nb_outputs],
                      name='resnet50-base-{}_out'.format(nb_outputs))
        toggle_trainable_layers(model, not freeze_conv)

    # load weights
    if weights is not None:
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
        elif weights == 'secondorder':
            weights_path = get_file('so_resnet50_weights_tf_dim_ordering_tf.h5',
                                    None,
                                    cache_subdir='models')
        else:
            return model
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model


def ResNet50_first_order(nb_class, denses=[], include_top=False, **kwargs):
    if include_top:
        return ResNet50_v2(nb_class=nb_class, include_top=include_top, **kwargs)
    base_model = ResNet50_v2(nb_class=nb_class, include_top=include_top, **kwargs)
    x = base_model.output
    x = Flatten(name='flatten')(x)
    for ind, para in enumerate(denses):
        x = Dense(para, activation='relu', name='new_fc{}'.format(str(ind + 1)),
                  kernel_initializer='glorot_uniform')(x)
    pred_name = 'new_pred'
    x = Dense(nb_class, activation='softmax', name=pred_name)(x)
    model = Model(base_model.input, x, name='resnet50-fo-{}'.format(denses))
    return model
