"""
Design to serve as model getter like keras.loss.get(id), but switch by the configs it self.

"""
import six

from kyu.configs.engine_configs import ModelConfig
from kyu.utils.sys_utils import merge_dicts


def get_model_from_config(model_fn, config, compulsory, optional):
    """
    Get model from config with compulsory and optional arguments.

    Parameters
    ----------
    model_fn : function takes the compulsory and optional arguments
    config : ModelConfig
    compulsory : list[str ...] (args into model_fn)
    optional : list[str ...]  (optional args into model_fn

    Returns
    -------

    """
    if model_fn is None:
        return None

    if not isinstance(config, ModelConfig):
        raise ValueError("Generic Load Model: get_model only support ModelConfig object")

    if not all(hasattr(config, item) for item in compulsory):
        raise ValueError("{} config is not complete. \n {}".format(config.model_id, config))

    comp_dict = dict(zip(compulsory, [getattr(config, item) for item in compulsory]))
    opt_dict = {}
    for item in optional:
        # attr = getattr(config, item)
        if hasattr(config, item) and item not in comp_dict.keys():
            opt_dict[item] = getattr(config, item)

    args = merge_dicts(comp_dict, opt_dict)

    # Return the model
    return model_fn(**args)


def deserialize_model_object(identifier, module_objects=None,
                             printable_module_name='model object'):
    """
    deserialize the model object given a string as identifier.
    It will search the module objects (like create functions or class constructors) to
    construct the model based on selection.

    References:
        keras.utils.generic_utils deserialize_keras_object

    Parameters
    ----------
    identifier : one of six.string_types
    module_objects : the related globals()
    printable_module_name : module name

    Returns
    -------

    """
    if isinstance(identifier, six.string_types):
        function_name = identifier
        fn = module_objects.get(function_name)
        if fn is None:
            raise ValueError("generic_loader: Unknown " + printable_module_name + ":"
                             + function_name)
        return fn


# Generic loader for all models. (basically, replace the resnet.get_model )
def first_order():
    """
    Define the generic first order get model to replace individual first-order ones

    Returns
    -------

    """
    pass


def second_order():
    pass


def multiple_loss_second_order():
    pass


def bilinear():
    pass


def mpn():
    pass


def matrix_backprop():
    pass


def get_model(config):
    """
    Generic get-model

    Parameters
    ----------
    config: ModelConfig

    Returns
    -------

    """
    if not isinstance(config, ModelConfig):
        raise ValueError("GenericLoader: only takes ModelConfig")

