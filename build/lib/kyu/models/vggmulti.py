"""
Re implement VGG model for general usage

"""
from keras.engine import Model
from keras.engine import merge

from keras.layers import Flatten, Dense, warnings, Convolution2D, MaxPooling2D, BiLinear

# from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16_Multi
from kyu.models.keras_support import covariance_block_original, dcov_model_wrapper_v1, \
    dcov_model_wrapper_v2, dcov_multi_out_model_wrapper
from kyu.models.keras_support import covariance_block_vector_space
from kyu.theano.general.train import toggle_trainable_layers


def VGG16Multi_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
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
                  pooling='max',
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
        base_model = VGG16_Multi(include_top=False, input_shape=input_shape, last_avg=last_avg, pooling=pooling)
    elif load_weights is None:
        base_model = VGG16_Multi(include_top=False, weights=None, input_shape=input_shape,last_avg=last_avg, pooling=pooling)
    else:
        base_model = VGG16_Multi(include_top=False, weights=None, input_shape=input_shape,last_avg=last_avg, pooling=pooling)
        base_model.load_weights(load_weights, by_name=True)

    basename = 'VGG16_o2_multi'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}_{}'.format(str(mode), cov_branch)

    if nb_branch == 1:
        # model = dcov_model_wrapper_v1(
        #     base_model, parametrics, mode, nb_classes, basename,
        #     cov_mode, cov_branch, cov_branch_output, freeze_conv,
        #     cov_regularizer, nb_branch, concat, last_conv_feature_maps,
        #     **kwargs
        # )
        raise NotImplementedError
    else:
        model = dcov_multi_out_model_wrapper(
            base_model, parametrics, mode, nb_classes, basename + 'nb_branch_' + str(nb_branch),
            cov_mode, cov_branch, cov_branch_output, freeze_conv,
            cov_regularizer, nb_branch, concat, last_conv_feature_maps,
            **kwargs
        )
    return model


def VGG16_o2_with_config(param, mode, cov_output, config, nb_class, input_shape, **kwargs):
    """ API to ResNet50 o2, create by config """
    z = config.kwargs.copy()
    z.update(kwargs)
    return VGG16Multi_o2(parametrics=param, mode=mode, nb_classes=nb_class, cov_branch_output=cov_output,
                         input_shape=input_shape, last_avg=False, freeze_conv=False,
                         cov_regularizer=config.cov_regularizer, last_conv_feature_maps=config.last_conv_feature_maps,
                         nb_branch=config.nb_branch, cov_mode=config.cov_mode, epsilon=config.epsilon,
                         pooling=config.pooling,
                         **z
                         )


if __name__ == '__main__':
    # model = VGG16_o1([4096,4096,4096], input_shape=(224,224,3))
    model = VGG16Multi_o2([256, 256, 128], nb_branch=2, cov_mode='pmean', freeze_conv=True,
                          cov_branch_output=64,
                          robust=True, mode=1, nb_classes=47)
    model.compile('sgd', 'categorical_crossentropy')
    model.summary()
