import six

from kyu.configs.engine_configs import ModelConfig
from kyu.configs.model_configs import ResNetFirstOrderConfig
from kyu.configs.model_configs.second_order import DCovConfig
from kyu.models.bilinear import ResNet50_bilinear
from kyu.models.resnet50 import ResNet50_v2, ResNet50_first_order
from kyu.models.generic_loader import get_model_from_config, deserialize_model_object
from kyu.models.single_second_order import ResNet50_second_order

RESNET_SUPPORTED_MODEL = ['resnet50']


def second_order(config):
    """ Implement the original ResNet with single stream structure """
    if not isinstance(config, DCovConfig):
        raise ValueError("ResNet: second-order only support DCovConfig")

    compulsory = DCovConfig.compulsoryArgs
    optional = DCovConfig.optionalArgs
    if config.class_id == RESNET_SUPPORTED_MODEL[0]:
        return get_model_from_config(ResNet50_second_order, config, compulsory, optional)
    else:
        raise NotImplementedError("ResNet model {} not supported yet".format(config.class_id))


def first_order(config):
    """
    Create for first order ResNet
    nb_dense_block=4,
    growth_rate=32,
    nb_filter=64,
    reduction=0.0,
    dropout_rate=0.0,
    weight_decay=1e-4,
    nb_class=1000,
    weights_path=None
    """
    if not isinstance(config, ResNetFirstOrderConfig):
        raise ValueError("ResNet first order: only support ResNetFirstOrderConfig")
    # TODO Change to ResENtFirstConfig
    compulsory = ['nb_class', 'input_shape']
    optional = ['include_top', 'weights', 'input_tensor', 'denses', 'pooling',
                'last_avg', 'weight_decay', 'freeze_conv', 'pred_activation']
    if config.class_id == RESNET_SUPPORTED_MODEL[0]:
        return get_model_from_config(ResNet50_first_order, config, compulsory, optional)
    else:
        raise NotImplementedError("DenseNet model {} not supported yet".format(config.class_id))


def bilinear(config):
    """ Create the bilinear model and return """
    if not isinstance(config, ModelConfig):
        raise ValueError("ResNet: get_model only support ModelConfig object")

    compulsory = ['nb_class', 'input_shape']
    optional = ['load_weights', 'input_shape', 'freeze_conv', 'last_conv_kernel']
    # Return the model
    if config.class_id == RESNET_SUPPORTED_MODEL[0]:
        return get_model_from_config(ResNet50_bilinear, config, compulsory, optional)


def deserialize(name, custom_objects=None):
    return deserialize_model_object(name,
                                    module_objects=globals(),
                                    printable_module_name='ResNet model'
                                    )


def get_model(config):

    if not isinstance(config, ModelConfig):
        raise ValueError("ResNet: get_model only support ModelConfig object")

    model_id = config.model_id
    # Decompose the config with different files.
    # if model_id == ''
    if isinstance(model_id, six.string_types):
        model_id = str(model_id)
        model_function = deserialize(model_id)
    elif callable(model_id):
        model_function = model_id
    else:
        raise ValueError("Could not interpret the densenet model with {}".format(model_id))

    return model_function(config)


if __name__ == '__main__':
    from kyu.configs.model_configs import ResNetFirstOrderConfig
    config = ResNetFirstOrderConfig(1000)

    model = get_model(config)
