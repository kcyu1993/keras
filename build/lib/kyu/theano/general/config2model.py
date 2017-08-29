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


def get_model_o2(o2model, config, parametrics=[], mode=0, cov_output=None, **kwargs):
    """
    Get a covariance output.
    Parameters
    ----------
    o2model
    parametrics
    mode
    cov_output
    config

    Returns
    -------
    Keras model
    """

    model = o2model(parametrics=parametrics, mode=mode, cov_branch=config.cov_branch, cov_mode=config.cov_mode,
                    nb_classes=config.nb_classes, cov_branch_output=cov_output, input_shape=config.input_shape,
                    cov_regularizer=config.cov_regularizer,
                    nb_branch=config.nb_branch,
                    last_conv_feature_maps=config.last_conv_feature_maps,
                    **kwargs
                    )
    return model

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
