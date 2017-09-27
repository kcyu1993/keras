"""
Define the quick prototype of SO-CNN

"""
from keras.applications import VGG16
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.models import Model
from kyu.layers.secondstat import WeightedVectorization, SecondaryStatistic
from kyu.models.densenet121 import DenseNet121
from kyu.models.resnet50 import ResNet50_v2
from kyu.utils.train_utils import toggle_trainable_layers


def _compose_so_prototype_model(base_model, nb_class, freeze_conv=False, last_conv_kernel=[], name='Bilinear_default'):
    """
    Define a API for quick prototyping models.

    Parameters
    ----------
    base_model
    nb_class
    freeze_conv
    last_conv_kernel
    name

    Returns
    -------

    """
    x = base_model.output
    # Add a batch-normalization layer before stepping into the Covariance layer
    # Assume tensorflow backend
    x = BatchNormalization(axis=3, name='last_batchnorm')(x)

    # Add covariance branches
    x = SecondaryStatistic()(x)
    x = WeightedVectorization(1024, output_sqrt=True, use_bias=False)(x)

    # Add classifiers
    x = Dense(nb_class, activation='softmax', name='predictions')(x)
    if freeze_conv:
        toggle_trainable_layers(base_model, trainable=False)

    new_model = Model(base_model.input, x, name=name)
    return new_model


def VGG16_so_prototype(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, **kwargs

):
    if load_weights == 'imagenet':
        base_model = VGG16(include_top=False, input_shape=input_shape, pooling=pooling)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_so_prototype_model(base_model, nb_class, cov_branch, **kwargs)


def DenseNet121_so_prototype(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, **kwargs

):
    if load_weights == 'imagenet':
        base_model = DenseNet121(include_top=False, input_shape=input_shape, last_pooling=pooling,
                                 weights_path='imagenet')
    elif load_weights is None:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape, last_pooling=pooling)
    else:
        base_model = DenseNet121(include_top=False, weights_path=None, input_shape=input_shape, last_pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_so_prototype_model(base_model, nb_class, cov_branch, **kwargs)


def ResNet50_so_prototype(
        input_shape, nb_class, cov_branch, load_weights='imagenet', pooling=None, last_avg=False, **kwargs
):

    if load_weights == 'imagenet':
        base_model = ResNet50_v2(include_top=False, input_shape=input_shape, last_avg=last_avg, pooling=pooling)
    elif load_weights is None:
        base_model = ResNet50_v2(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                                 pooling=pooling)
    else:
        base_model = ResNet50_v2(include_top=False, weights=None, input_shape=input_shape, last_avg=last_avg,
                                 pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    return _compose_so_prototype_model(base_model, nb_class, cov_branch, **kwargs)

