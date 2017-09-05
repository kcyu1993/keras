import six

from kyu.configs.engine_configs import ModelConfig
from kyu.configs.model_configs import DenseNetFirstOrderConfig
from kyu.configs.model_configs.second_order import DCovConfig
from kyu.models.bilinear import DenseNet121_bilinear
from kyu.models.densenet121 import DenseNet121
from kyu.models.generic_loader import get_model_from_config, deserialize_model_object
from kyu.models.single_second_order import DenseNet121_second_order


def second_order(config):
    """ Implement the original SO-VGG16 with single stream structure """
    if not isinstance(config, DCovConfig):
        raise ValueError("DenseNet121: second-order only support DCovConfig")

    compulsory = ['input_shape', 'nb_class', 'cov_branch', 'cov_branch_kwargs']
    optional = ['concat', 'mode', 'cov_branch_output', 'name',
                'nb_branch', 'cov_output_vectorization',
                'upsample_method', 'last_conv_kernel', 'last_conv_feature_maps',
                'freeze_conv', 'load_weights']
    if config.class_id == 'densenet121':
        return get_model_from_config(DenseNet121_second_order, config, compulsory, optional)
    else:
        raise NotImplementedError("DenseNet model {} not supported yet".format(config.class_id))


def first_order(config):
    """
    Create for first order VGG
    nb_dense_block=4,
    growth_rate=32,
    nb_filter=64,
    reduction=0.0,
    dropout_rate=0.0,
    weight_decay=1e-4,
    nb_class=1000,
    weights_path=None
    """
    if not isinstance(config, DenseNetFirstOrderConfig):
        raise ValueError("DenseNet first order: only support DenseNetFirstOrderConfig")
    compulsory = ['nb_class', 'input_shape']
    optional = ['nb_dense_block', 'growth_rate', 'nb_filter', 'reduction', 'dropout_rate',
                'weight_decay', 'weights_path',
                'last_pooling', 'freeze_conv']
    if config.class_id == 'densenet121':
        return get_model_from_config(DenseNet121, config, compulsory, optional)
    else:
        raise NotImplementedError("DenseNet model {} not supported yet".format(config.class_id))


def bilinear(config):
    """ Create the bilinear model and return """
    if not isinstance(config, ModelConfig):
        raise ValueError("DenseNet121: get_model only support ModelConfig object")

    compulsory = ['nb_class', 'input_shape']
    optional = ['load_weights', 'input_shape', 'freeze_conv']

    # Return the model
    return get_model_from_config(DenseNet121_bilinear, config, compulsory, optional)


def deserialize(name, custom_objects=None):
    return deserialize_model_object(name,
                                    module_objects=globals(),
                                    printable_module_name='densenet121 model'
                                    )


def get_model(config):

    if not isinstance(config, ModelConfig):
        raise ValueError("DenseNet: get_model only support ModelConfig object")

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