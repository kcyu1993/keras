import kyu.models.densenet
from keras.engine import Model
from kyu.configs.engine_configs import ModelConfig
from . import resnet
from . import vgg
from . import densenet

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
        model = vgg.get_model(config)
    elif identifier in ['resnet', 'resnet50',]:
        model = resnet.get_model(config)
    elif str(identifier).find('densenet') >= 0:
        model = kyu.models.densenet.get_model(config)
    else:
        raise ValueError("Unkwown identifier {}".format(identifier))

    if not isinstance(model, Model):
        raise ValueError("Model is not keras model instance ! {}".format(model))

    model.name = config.name
    return model