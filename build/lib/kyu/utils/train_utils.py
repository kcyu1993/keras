"""
Training Utilities for Keras model
"""


def toggle_trainable_layers(model, trainable=True, keyword='', **kwargs):
    """
    Freeze the layers of a given model

    Parameters
    ----------
    model
    kwargs

    Returns
    -------
    model : keras.Model     need to be re-compiled once toggled.
    """
    for layer in model.layers:
        if keyword in layer.name:
            layer.trainable = trainable
    return model


def toggle_trainable_layers_with_target(model, trainable=True, key_instance=[], **kwargs):
    """

    Parameters
    ----------
    model : Keras Model
    trainable : True or False
    key_instance : pass the layers
    kwargs : Keywords

    Returns
    -------

    """

    for layer in model.layers:
        if any([isinstance(layer, i) for i in key_instance]):
            layer.trainable = trainable
    return model
