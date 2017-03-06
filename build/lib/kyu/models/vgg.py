"""
Re implement VGG model for general usage

"""
from keras.engine import Model
from keras.engine import merge

from keras.layers import Flatten, Dense, warnings

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from kyu.models.keras_support import covariance_block_original, dcov_model_wrapper_v1, dcov_model_wrapper_v2
from kyu.models.keras_support import covariance_block_vector_space
from kyu.theano.general.train import toggle_trainable_layers


def VGG16_o1(denses=[], nb_classes=1000, input_shape=None, load_weights=True, freeze_conv=False):
    """
    Create VGG 16 based on without_top.

    Parameters
    ----------
    denses : list[int]  dense layer parameters
    nb_classes : int    nb of classes
    input_shape : tuple input shape

    Returns
    -------
    Model
    """
    if load_weights:
        model = VGG16(include_top=False, input_shape=input_shape)
    else:
        model = VGG16(include_top=False, weights=None, input_shape=input_shape)

    # Create Dense layers
    x = model.output
    x = Flatten()(x)
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc' + str(ind + 1))(x)
    # Prediction
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)
    if freeze_conv:
        toggle_trainable_layers(model, trainable=False)

    new_model = Model(model.input, x, name='VGG16_o1')
    return new_model


def VGG16_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
             load_weights='imagenet',
             cov_mode='channel',
             cov_branch='o2transform',
             cov_branch_output=None,
             last_avg=False,
             freeze_conv=False,
             cov_regularizer=None,
             nb_branch=1,
             concat='concat',
             last_conv_feature_maps=[],
             **kwargs
            ):
    """

    Parameters
    ----------
    parametrics
    mode
    nb_classes
    input_shape
    load_weights
    cov_mode
    cov_branch
    cov_branch_output
    cov_block_mode
    last_avg
    freeze_conv

    Returns
    -------

    """
    if load_weights == 'imagenet':
        base_model = VGG16(include_top=False, input_shape=input_shape)
    elif load_weights is None:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
    else:
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        base_model.load_weights(load_weights, by_name=True)

    basename = 'VGG16_o2'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if nb_branch == 1:
        model = dcov_model_wrapper_v1(
            base_model, parametrics, mode, nb_classes, basename,
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    else:
        model = dcov_model_wrapper_v2(
            base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    return model


if __name__ == '__main__':
    # model = VGG16_o1([4096,4096,4096], input_shape=(224,224,3))
    model = VGG16_o2([256, 256, 128], nb_branch=2, cov_mode='pmean', freeze_conv=True,
                     cov_branch_output=64,
                     robust=True, mode=1, nb_classes=47)
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
