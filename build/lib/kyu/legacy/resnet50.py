import warnings

from keras import backend as K, Input
from keras.applications import ResNet50
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.resnet50 import conv_block, identity_block
from keras.engine import Model
from keras.layers import Convolution2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, ZeroPadding2D, \
    MaxPooling2D
from keras.legacy.layers import merge
from kyu.models.so_cnn_helper import dcov_multi_out_model_wrapper, dcov_model_wrapper_v1, dcov_model_wrapper_v2, \
    covariance_block_original, covariance_block_vector_space
from kyu.utils.train_utils import toggle_trainable_layers


def conv_block_original(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),batch_norm=True):
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
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Convolution2D(nb_filter2, kernel_size, kernel_size, subsample=strides,border_mode='same',
                             name=conv_name_base + '1')(input_tensor)
    if batch_norm:
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_original(input_tensor, kernel_size, filters, stage, block, batch_norm=True):
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
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50_o1(denses=[], nb_classes=1000, input_shape=None, load_weights=True, freeze_conv=False,
                last_conv_feature_maps=[]):
    """
    Create ResNet50 based on without_top.

    Parameters
    ----------
    denses : list[int]  dense layer parameters
    nb_classes : int    nb of classes
    input_shape : tuple input shape

    Returns
    -------
    Model
    """
    if last_conv_feature_maps == []:
        if load_weights:
            model = ResNet50(include_top=False, input_shape=input_shape)
        else:
            model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    else:
        if load_weights:
            res_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=False)
        else:
            res_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=False)
        x = res_model.output
        for ind, feature_dim in enumerate(last_conv_feature_maps):
            x = Convolution2D(feature_dim, 1, 1, activation='relu', name='1x1_conv_{}'.format(ind))(x)
        x = AveragePooling2D((7,7), name='avg_pool')(x)
        model = Model(res_model.input, x, name='resnet50_with_1x1')

    # Create Dense layers
    x = model.output
    x = Flatten()(x)
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc' + str(ind + 1))(x)
    # Prediction
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)
    if freeze_conv:
        toggle_trainable_layers(model, trainable=False)
    new_model = Model(model.input, x, name='resnet50_o1')
    return new_model


def ResNet50_o2_with_config(param, mode, cov_output, config, nb_class, input_shape, **kwargs):
    """ API to ResNet50 o2, create by config """
    z = config.kwargs.copy()
    z.update(kwargs)
    return ResNet50_o2(parametrics=param, mode=mode, nb_classes=nb_class, cov_branch_output=cov_output,
                       input_shape=input_shape, last_avg=False, freeze_conv=False,
                       cov_regularizer=config.cov_regularizer, last_conv_feature_maps=config.last_conv_feature_maps,
                       nb_branch=config.nb_branch, cov_mode=config.cov_mode, epsilon=config.epsilon, **z
                       )


def ResNet50_o2_multibranch(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
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
                            **kwargs
                            ):
    basename = 'ResNet_o2-multi-' + cov_branch
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if load_weights == 'imagenet':
        base_model = ResCovNet50(include_top=False, input_shape=input_shape, last_avg=last_avg)
    elif load_weights is None:
        base_model = ResCovNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
    else:
        base_model = ResCovNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
        base_model.load_weights(load_weights, by_name=True)

    model = dcov_multi_out_model_wrapper(
        base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
        cov_mode, cov_branch, cov_branch_output, freeze_conv,
        cov_regularizer, nb_branch, concat, last_conv_feature_maps,
        **kwargs
    )

    return model


def ResNet50_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
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
                **kwargs
                ):


    basename = 'ResNet_o2_' + cov_branch
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if load_weights == 'imagenet':
        base_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=last_avg)
    elif load_weights is None:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
    else:
        base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
        base_model.load_weights(load_weights, by_name=True)
    if nb_branch == 1:
        model = dcov_model_wrapper_v1(
            base_model, parametrics, mode, nb_classes, basename,
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    else:
        model = dcov_model_wrapper_v2(
            base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    return model


def ResNet50_cifar_o1(denses=[], nb_classes=10, input_shape=None, load_weights=True, freeze_conv=False,
                      last_conv_feature_maps=[], batch_norm=True):
    """
    Create ResNet50 based on without_top.

    Parameters
    ----------
    denses : list[int]  dense layer parameters
    nb_classes : int    nb of classes
    input_shape : tuple input shape

    Returns
    -------
    Model
    """
    if last_conv_feature_maps == []:
        if load_weights:
            model = ResNet50CIFAR(include_top=False, input_shape=input_shape)
        else:
            model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    else:
        if load_weights:
            res_model = ResNet50(include_top=False, input_shape=input_shape, last_avg=False)
        else:
            res_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, last_avg=False)
        x = res_model.output
        for ind, feature_dim in enumerate(last_conv_feature_maps):
            x = Convolution2D(feature_dim, 1, 1, activation='relu', name='1x1_conv_{}'.format(ind))(x)
        x = AveragePooling2D((7,7), name='avg_pool')(x)
        model = Model(res_model.input, x, name='resnet50_with_1x1')

    # Create Dense layers
    x = model.output
    x = Flatten()(x)
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc' + str(ind + 1))(x)
    # Prediction
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)
    if freeze_conv:
        toggle_trainable_layers(model, trainable=False)
    new_model = Model(model.input, x, name='resnet50_o1')
    return new_model


def ResNet50_cifar_o2(parametrics=[], mode=0, nb_classes=10, input_shape=(32,32,3),
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
                      **kwargs
                      ):


    basename = 'ResNetCIFAR_o2_' + cov_branch
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if load_weights == 'imagenet':
        base_model = ResNet50CIFAR(include_top=False, input_shape=input_shape, last_avg=last_avg)
    elif load_weights is None:
        base_model = ResNet50CIFAR(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
    else:
        base_model = ResNet50CIFAR(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg)
        base_model.load_weights(load_weights, by_name=True)
    if nb_branch == 1:
        model = dcov_model_wrapper_v1(
            base_model, parametrics, mode, nb_classes, basename,
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    else:
        model = dcov_model_wrapper_v2(
            base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    return model


def ResCovNet50(parametrics=[], epsilon=0., mode=0, nb_classes=23, input_shape=(3, 224, 224),
                init='glorot_normal', cov_branch='o2transform', cov_mode='channel',
                dropout=False, cov_branch_output=None,
                dense_after_covariance=True,
                cov_block_mode=3,
                last_softmax=True,
                independent_learning=False):
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

    # Function name
    if cov_branch == 'o2transform':
        covariance_block = covariance_block_original
    elif cov_branch == 'dense':
        covariance_block = covariance_block_vector_space
    else:
        raise ValueError('covariance cov_mode not supported')

    nb_class = nb_classes
    if cov_branch_output is None:
        cov_branch_output = nb_class

    basename = 'ResCovNet'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if epsilon > 0:
        basename += '-epsilon_{}'.format(str(epsilon))

    if input_shape[0] == 3:
        # Define the channel
        if K.image_dim_ordering() == 'tf':
            if input_shape[0] in {1, 3}:
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    input_tensor = Input(input_shape)

    x = ZeroPadding2D((3, 3))(input_tensor)
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

    block3_x = x
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if independent_learning:
        x = ZeroPadding2D((3, 3))(input_tensor)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='cov_conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='cov_bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='cov_a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='cov_b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='cov_c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='cov_a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='cov_b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='cov_c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='cov_d')
        block1_x = x

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='cov_a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='cov_b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='cov_c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='cov_d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='cov_e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='cov_f')
        block2_x = x

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='cov_a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='cov_b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='cov_c')

        block3_x = x
        x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if cov_block_mode == 3:
        cov_input = block3_x
    elif cov_block_mode == 2:
        cov_input = block2_x
    else:
        cov_input = block1_x
    basename += '_block_{}'.format(cov_block_mode)
    if mode == 0:
        warnings.warn("Mode 0 should be replaced with ResNet50")
        if nb_class == 1000:
            return ResNet50(input_tensor=input_tensor)
        else:
            raise ValueError("Only support 1000 class nb for ResNet50")

    if mode == 1:
        cov_branch = covariance_block(cov_input, nb_class, stage=5, block='a', parametric=parametrics,
                                      cov_mode=cov_mode)
        x = Dense(nb_class, activation='softmax', name='predictions')(cov_branch)

    elif mode == 2:
        cov_branch = covariance_block(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)

    elif mode == 3:
        cov_branch = covariance_block(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        cov_branch = Activation('softmax')(cov_branch)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc')(x)
        x = merge([x, cov_branch], mode='sum', name='sum')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 4:
        cov_branch1 = covariance_block(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 5:
        cov_branch1 = covariance_block(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch1, cov_branch2, cov_branch3], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 6:
        cov_branch1 = covariance_block(block1_x, nb_class, stage=2, block='a', parametric=parametrics)
        cov_branch2 = covariance_block(block2_x, nb_class, stage=3, block='b', parametric=parametrics)
        cov_branch3 = covariance_block(block3_x, nb_class, stage=4, block='c', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        cov_branch = merge([cov_branch1, cov_branch2, cov_branch3], mode='sum', name='sum')
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    elif mode == 7:
        cov_branch = covariance_block(cov_input, nb_class, stage=5, block='a', parametric=parametrics)
        x = Flatten()(x)
        x = Dense(nb_class, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_class, activation='softmax', name='predictions')(x)
    else:
        raise ValueError("Mode not supported {}".format(mode))

    model = Model(input_tensor, x, name=basename)
    return model


def ResNet50CIFAR(include_top=True,
                  input_tensor=None,
                  input_shape=None,
                  weights=None,
                  last_avg=False,
                  nb_class=10, batch_norm=True):
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
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=24,
                                      data_format=K.image_data_format(),
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

    # x = ZeroPadding2D((2, 2))(img_input)
    x = Convolution2D(16, 3, 3, name='conv1')(img_input)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block_original(x, 3, [16, 16], stage=2, block='a', strides=(1,1), batch_norm=batch_norm)
    x = identity_block_original(x, 3, [16, 16], stage=2, block='b', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [16, 16], stage=2, block='c', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [16, 16], stage=2, block='d', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [16, 16], stage=2, block='e', batch_norm=batch_norm)

    x = conv_block_original(x, 3, [32, 32], stage=3, block='a', strides=(2,2), batch_norm=batch_norm)
    x = identity_block_original(x, 3, [32, 32], stage=3, block='b', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [32, 32], stage=3, block='c', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [32, 32], stage=3, block='d', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [32, 32], stage=3, block='e', batch_norm=batch_norm)

    x = conv_block_original(x, 3, [64, 64], stage=4, block='a', strides=(2,2), batch_norm=batch_norm)
    x = identity_block_original(x, 3, [64, 64], stage=4, block='b', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [64, 64], stage=4, block='c', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [64, 64], stage=4, block='d', batch_norm=batch_norm)
    x = identity_block_original(x, 3, [64, 64], stage=4, block='e', batch_norm=batch_norm)

    if include_top:
        x = AveragePooling2D((3, 3), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(nb_class, activation='softmax', name='fc')(x)
    # else:
    #     return x, img_input

    model = Model(img_input, x)
    if weights:
        model.load_weights(weights, True)
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