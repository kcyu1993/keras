"""
Develop the AlexNet with the pre-trained weights from ImageNet,
following the same style like keras does.

"""

import six

from keras.engine import Model
from keras.layers import Dense
from kyu.configs.engine_configs import ModelConfig
from kyu.configs.model_configs.first_order import AlexNetFirstOrderConfig
from kyu.configs.model_configs.second_order import DCovConfig
from kyu.models.alexnet5 import AlexNet_v2
from kyu.models.generic_loader import deserialize_model_object, get_model_from_config
from kyu.models.single_second_order import AlexNet_second_order


def AlexNet_first_order(denses=[], include_top=False, **kwargs):
    if include_top:
        return AlexNet_v2(include_top=True, **kwargs)
    base_model = AlexNet_v2(include_top=False, **kwargs)
    x = base_model.output
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc{}'.format(ind+1))(x)
    nb_class = kwargs['nb_class']
    x = Dense(nb_class, activation='softmax', name='pred')(x)
    model = Model(base_model.input, x, name='alexnet_first_order')
    return model


def second_order(config):
    if not isinstance(config, DCovConfig):
        raise ValueError('Only support DCovConfig for second order')
    compulsory = DCovConfig.compulsoryArgs
    optional = DCovConfig.optionalArgs
    return get_model_from_config(AlexNet_second_order, config, compulsory, optional)


def first_order(config):
    if not isinstance(config, AlexNetFirstOrderConfig):
        raise ValueError('Only support AlexNetFirstOrderConfig for first-order')

    return get_model_from_config(AlexNet_first_order, config, config.compulsory, config.optional)


def deserialize(name, custom_objects=None):
    return deserialize_model_object(name,
                                    module_objects=globals(),
                                    printable_module_name='alexnet model'
                                    )


def get_model(config):
    if not isinstance(config, ModelConfig):
        raise ValueError("AlexNet: get_model only support ModelConfig object")

    model_id = config.model_id

    if isinstance(model_id, six.string_types):
        model_id = str(model_id)
        model_function = deserialize(model_id)
    elif callable(model_id):
        model_function = model_id
    else:
        raise ValueError("Could not interpret the alexnet with {}".format(model_id))

    return model_function(config)

if __name__ == '__main__':
    model = AlexNet_v2(1000, input_shape=(227, 227, 3))
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
