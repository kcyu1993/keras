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