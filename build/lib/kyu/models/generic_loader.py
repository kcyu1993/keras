"""
Design to serve as model getter like keras.loss.get(id), but switch by the configs it self.

"""
from kyu.engine.configs.model import ModelConfig

from . import vgg
from . import resnet


def get_model(config):
    """
    Define the get model method from identifier

    Parameters
    ----------
    config

    Returns
    -------

    """
    if not isinstance(config, ModelConfig):
        raise ValueError("Get model must be a config file. ")

    identifier = str(config.class_id).lower()
    if identifier in ['vgg', 'vgg16', 'vgg19']:
        return vgg.get_model(config)
    elif identifier in ['resnet', 'resnet50',]:
        return resnet.get_model(config)

