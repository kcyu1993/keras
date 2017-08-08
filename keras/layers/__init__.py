from __future__ import absolute_import

from .advanced_activations import *
from .convolutional import *
from .convolutional_recurrent import *
from .convolutional_recurrent import *
from .core import *
from .embeddings import *
from .local import *
from .merge import *
from .noise import *
from .normalization import *
from .pooling import *
from .recurrent import *
from .wrappers import *
from ..legacy.layers import *


def serialize(layer):
    """Serialize a layer.

    # Arguments
        layer: a Layer object.

    # Returns
        dictionary with config.
    """
    return {'class_name': layer.__class__.__name__,
            'config': layer.get_config()}


def deserialize(config, custom_objects=None):
    """Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    """
    from .. import models
    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    return deserialize_keras_object(config,
                                    module_objects=globs,
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')

"""
Self Defined layers
"""
