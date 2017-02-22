"""
From config file to Keras model

Model supported:
    FitNet - DCov x
    General model design:
        config parser for Keras model
        Keras model to config (Caffe like model creation)

    Since Keras already support model serialization via config
    it is easy to write this class

"""

from keras.models import Model

def config2model(filename):
    """
    get model from config files

    Parameters
    ----------
    filename

    Returns
    -------

    """
    return Model.from_config(filename)

def model2config(model):
    """
    get config from model

    Parameters
    ----------
    model : Keras.models.Model

    Returns
    -------
    config_files
    """
    return model.get_config()
