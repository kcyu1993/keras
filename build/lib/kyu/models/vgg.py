"""
Re implement VGG model for general usage

"""
import six

from kyu.engine.configs.model import ModelConfig

from kyu.engine.configs.generic import KCConfig

from keras.engine import Model
from keras.legacy.layers import merge

from keras.layers import Flatten, Dense, warnings, Conv2D, MaxPooling2D

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from kyu.models.bilinear import VGG16_bilinear
from kyu.models.single_second_order import DCovConfig, VGG16_second_order
from .generic_loader import deserialize_model_object, get_model_from_config

from kyu.models.secondstat import BiLinear
from kyu.models.so_cnn_helper import covariance_block_original, dcov_model_wrapper_v1, dcov_model_wrapper_v2
from kyu.models.so_cnn_helper import covariance_block_vector_space
from kyu.theano.general.train import toggle_trainable_layers
from kyu.utils.sys_utils import merge_dicts


def VGG16_o1(denses=[], nb_classes=1000, input_shape=None, load_weights=True, freeze_conv=False, last_conv=False,
             last_pooling=False):
    """
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


def second_order(config):
    """ Implement the original SO-VGG16 with single stream structure """
    if not isinstance(config, DCovConfig):
        raise ValueError("VGG: second-order only support DCovConfig")

    compulsory = ['input_shape', 'nb_class', 'cov_branch', 'cov_branch_kwargs']
    optional = ['concat', 'mode', 'cov_branch_output', 'name',
                'nb_branch', 'cov_output_vectorization',
                'upsample_method', 'last_conv_kernel', 'last_conv_feature_maps',
                'freeze_conv', 'load_weights']
    return get_model_from_config(VGG16_second_order, config, compulsory, optional)


def bilinear(config):
    """ Create the bilinear model and return """
    if not isinstance(config, ModelConfig):
        raise ValueError("VGG: get_model only support ModelConfig object")

    compulsory = ['nb_class', 'input_shape']
    optional = ['load_weights', 'input_shape', 'freeze_conv']

    # Return the model
    # return VGG16_bilinear(**args)
    return get_model_from_config(VGG16_bilinear, config, compulsory, optional)


def deserialize(name, custom_objects=None):
    return deserialize_model_object(name,
                                    module_objects=globals(),
                                    printable_module_name='vgg model'
                                    )


def get_model(config):
    if not isinstance(config, ModelConfig):
        raise ValueError("VGG: get_model only support ModelConfig object")

    model_id = config.model_id
    # Decompose the config with different files.
    # if model_id == ''
    if isinstance(model_id, six.string_types):
        model_id = str(model_id)
        model_function = deserialize(model_id)
    elif callable(model_id):
        model_function = model_id
    else:
        raise ValueError("Could not interpret the vgg model with {}".format(model_id))

    return model_function(config)


if __name__ == '__main__':
    # model = VGG16_o1([4096,4096,4096], input_shape=(224,224,3))
    model = VGG16_o2([256, 256, 128], nb_branch=2, cov_mode='pmean', freeze_conv=True,
                     cov_branch_output=64,
                     robust=True, mode=1, nb_classes=47)
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
