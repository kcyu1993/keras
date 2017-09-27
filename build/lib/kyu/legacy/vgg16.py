from keras.applications import VGG16
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from kyu.legacy.so_wrapper import dcov_model_wrapper_v1, dcov_model_wrapper_v2
from kyu.utils.train_utils import toggle_trainable_layers


def VGG16_o1(denses=[], nb_classes=1000, input_shape=None, load_weights=True, freeze_conv=False, last_conv=False,
             last_pooling=False):
    """
    legacy support

    Create VGG 16 based on without_top.

    Parameters
    ----------
    denses : list[int]  dense layer parameters
    nb_classes : int    nb of classes
    input_shape : tuple input shape

    Returns
    -------
    Model
    """
    if load_weights:
        model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    else:
        model = VGG16(include_top=False, weights=None, input_shape=input_shape)

    # Create Dense layers
    x = model.output
    if last_conv:
        x = Conv2D(1024, (1, 1))(x)
    if last_pooling:
        x = MaxPooling2D((7,7))(x)
    x = Flatten()(x)
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc' + str(ind + 1))(x)
    # Prediction
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)
    if freeze_conv:
        toggle_trainable_layers(model, trainable=False)

    new_model = Model(model.input, x, name='VGG16_o1')
    return new_model


def VGG16_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
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
             pooling=None,
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
    if load_weights == 'imagenet':
        base_model = VGG16(include_top=False, input_shape=input_shape, pooling=pooling)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    basename = 'VGG16_o2'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}_{}'.format(str(mode), cov_branch)

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


def VGG16_o2_with_config(param, mode, cov_output, config, nb_class, input_shape, **kwargs):
    """ API to ResNet50 o2, create by config """
    z = config.kwargs.copy()
    z.update(kwargs)
    return VGG16_o2(parametrics=param, mode=mode, nb_classes=nb_class, cov_branch_output=cov_output,
                    input_shape=input_shape, last_avg=False, freeze_conv=False,
                    cov_regularizer=config.cov_regularizer, last_conv_feature_maps=config.last_conv_feature_maps,
                    nb_branch=config.nb_branch, cov_mode=config.cov_mode, epsilon=config.epsilon,
                    pooling=config.pooling,
                    **z
                    )