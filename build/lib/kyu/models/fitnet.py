"""
FitNet implementation

References
----------
    Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014, December 19).
    FitNets: Hints for Thin Deep Nets. arXiv.org.

"""
import keras.backend as K
from keras.engine import Input
from keras.engine import Model
from keras.engine import get_source_inputs
from keras.engine import merge
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from kyu.models.keras_support import covariance_block_original, dcov_model_wrapper_v1, dcov_model_wrapper_v2
from kyu.models.keras_support import covariance_block_vector_space


def convolution_block(input_tensor, feature_maps=[], kernel=[3,3], border_mode='same', init='glorot_normal',
                      activation='relu', last_activation=True,
                      stage=1, basename='conv'):
    """
    Generate convolutional block based on some input feature maps size and kernel.
    Parameters
    ----------
    input_tensor
    feature_maps
    kernel
    border_mode
    init
    activation
    last_activation
    stage
    basename

    Returns
    -------

    """
    if len(feature_maps) < 1:
        return input_tensor
    x = input_tensor
    for ind, f in enumerate(feature_maps[:-1]):
        x = Convolution2D(f, kernel[0], kernel[1], init=init, border_mode=border_mode,
                          name=basename + '_{}_{}'.format(stage, ind))(x)
        x = Activation(activation=activation)(x)
    x = Convolution2D(feature_maps[-1], kernel[0], kernel[1], init=init, border_mode=border_mode,
                      name=basename + "_{}_{}".format(stage, len(feature_maps)))(x)
    if last_activation:
        x = Activation(activation=activation)(x)
    return x


def cifar_fitnet_v5(parametrics=[], epsilon=0., mode=0, nb_classes=10, input_shape=(3,32,32),
                    init='glorot_normal', cov_mode='dense',
                    dropout=False, cov_branch_output=None,
                    dense_after_covariance=True,
                    last_softmax=True):
    """
        Implement the fit model has 205K param
        Without any Maxout design in this version
        Just follows the general architecture

        Update 12.09.2016
        Switch between Cov-O2Transform and Cov-Dense

        :return: model sequential
    """
    # Function name
    if cov_mode == 'o2transform':
        covariance_block = covariance_block_original
    elif cov_mode == 'dense':
        covariance_block = covariance_block_vector_space
    else:
        raise ValueError('covariance cov_mode not supported')

    nb_class = nb_classes
    if cov_branch_output is None:
        cov_branch_output = nb_class

    basename = 'fitnet_v5_indbranch'
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    if epsilon > 0:
        basename += '-epsilon_{}'.format(str(epsilon))

    if input_shape[0] == 3:
        # Define the channel
        if K.image_dim_ordering() == 'tf':
            if input_shape[0] in {1,3}:
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Convolutional branch
    input_tensor = Input(input_shape)
    conv_branch = convolution_block(input_tensor, [16, 16, 16], [3, 3], border_mode='same', init=init,
                                    stage=1, basename='conv')
    conv_branch = MaxPooling2D()(conv_branch)
    if dropout:
        conv_branch = Dropout(0.25)(conv_branch)
    conv_branch = convolution_block(conv_branch, [32, 32, 32], [3, 3], border_mode='same', init=init,
                                    stage=2, basename='conv')
    conv_branch = MaxPooling2D()(conv_branch)
    if dropout:
        conv_branch = Dropout(0.25)(conv_branch)

    conv_branch = convolution_block(conv_branch, [48, 48, 64], [3, 3], border_mode='same', init=init,
                                    stage=3, basename='conv')
    conv_branch = MaxPooling2D()(conv_branch)
    if dropout:
        conv_branch = Dropout(0.25)(conv_branch)

    fc = Flatten()(conv_branch)
    fc = Dense(500)(fc)
    fc = Dense(nb_classes, name='prediction_conv')(fc)

    # Covariance branch
    covariance_branch = convolution_block(input_tensor, [16, 16, 16], [3, 3], border_mode='same', init=init,
                                          stage=1, basename='cov_conv')
    covariance_branch = MaxPooling2D()(covariance_branch)
    if dropout:
        covariance_branch = Dropout(0.25)(covariance_branch)

    covariance_branch = convolution_block(covariance_branch, [32, 32, 32], [3, 3], border_mode='same', init=init,
                                          stage=2, basename='cov_conv')
    covariance_branch = MaxPooling2D()(covariance_branch)
    if dropout:
        covariance_branch = Dropout(0.25)(covariance_branch)

    covariance_branch = convolution_block(covariance_branch, [48, 48, 64], [3, 3], border_mode='same', init=init,
                                          stage=3, basename='cov_conv')
    covariance_branch = MaxPooling2D()(covariance_branch)
    if dropout:
        covariance_branch = Dropout(0.25)(covariance_branch)

    if mode == 1:
        x = covariance_block(covariance_branch, cov_branch_output, stage=4, block='a', epsilon=epsilon, parametric=parametrics)
        if dense_after_covariance and cov_branch_output != nb_class:
            x = Dense(nb_class, name='pre-prediction')(x)
        x = merge([fc,x], mode='concat', name='concat')
        x = Dense(nb_class, name='prediction', activation='softmax')(x)
    elif mode == 2: # Merge two ends.
        x = covariance_block(covariance_branch, cov_branch_output, stage=4, block='a', epsilon=epsilon,
                             parametric=parametrics)
        if dense_after_covariance and cov_branch_output != nb_class:
            x = Dense(nb_class, name='pre-prediction')(x)
            x = Activation('softmax')(x)
        fc = Activation('softmax')(fc)
        x = merge([fc, x], mode='ave', name='average')
    else:
        raise ValueError("Mode {} not supported".format(mode))

    model = Model(input_tensor, x, name=basename)
    return model


def fitnet_v1_top(input_shape=None, load_weights=None, dropout=False,
                  init='glorot_normal', last_avg=False):

    basename = 'fitnet_v1_top'

    if input_shape[0] == 3:
        # Define the channel
        if K.image_dim_ordering() == 'tf':
            if input_shape[0] in {1, 3}:
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Convolutional branch
    input_tensor = Input(input_shape)
    x = convolution_block(input_tensor, [16, 16, 16], [3, 3], border_mode='same', init=init,
                          stage=1, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)
    x = convolution_block(x, [32, 32, 32], [3, 3], border_mode='same', init=init,
                          stage=2, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)

    x = convolution_block(x, [48, 48, 64], [3, 3], border_mode='same', init=init,
                          stage=3, basename='conv')
    x = MaxPooling2D(pool_size=(2,2))(x)
    if last_avg:
        x = MaxPooling2D(pool_size=(8, 8))(x)
    if dropout:
        x = Dropout(0.25)(x)

    model = Model(input_tensor, x, name=basename)
    return model


def fitnet_v2_top(input_shape=None, load_weights=None, dropout=False,
                  init='glorot_normal', last_avg=False):

    basename = 'fitnet_v2_top'

    # Convolutional branch
    input_tensor = Input(input_shape)
    x = convolution_block(input_tensor, [16, 32, 32], [3, 3], border_mode='same', init=init,
                          stage=1, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)
    x = convolution_block(x, [48, 64, 80], [3, 3], border_mode='same', init=init,
                          stage=2, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)

    x = convolution_block(x, [96, 96, 128], [3, 3], border_mode='same', init=init,
                          stage=3, basename='conv')
    if last_avg:
        x = MaxPooling2D(pool_size=(8, 8))(x)
    if dropout:
        x = Dropout(0.25)(x)

    model = Model(input_tensor, x, name=basename)
    return model


def fitnet_v3_top(input_shape=None, load_weights=None, dropout=False,
                  init='glorot_normal', last_avg=False):

    basename = 'fitnet_v3_top'

    if input_shape[0] == 3:
        # Define the channel
        if K.image_dim_ordering() == 'tf':
            if input_shape[0] in {1, 3}:
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Convolutional branch
    input_tensor = Input(input_shape)
    x = convolution_block(input_tensor, [32, 48, 64, 64], [3, 3], border_mode='same', init=init,
                                    stage=1, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)
    x = convolution_block(x, [80, 80, 80, 80], [3, 3], border_mode='same', init=init,
                                    stage=2, basename='conv')
    x = MaxPooling2D()(x)
    if dropout:
        x = Dropout(0.25)(x)

    x = convolution_block(x, [128, 128, 128], [3, 3], border_mode='same', init=init,
                                    stage=3, basename='conv')
    if last_avg:
        x = MaxPooling2D(pool_size=(8, 8))(x)
    if dropout:
        x = Dropout(0.25)(x)

    model = Model(input_tensor, x, name=basename)
    return model


def fitnet_wrapper_o1(
        fn_model, basename='fitnet',
        denses=[], nb_classes=10, input_shape=None, load_weights=None,
        dropout=False, init='glorot_normal',
        freeze_conv=False, last_conv_feature_maps=[]):
    """
    FitNet Wrapper for o1

    Parameters
    ----------
    denses
    nb_classes
    input_shape
    load_weights
    dropout
    init

    Returns
    -------

    """

    if denses is not []:
        basename += '_dense-'
        for para in denses:
            basename += str(para) + '_'

    base_model = fn_model(input_shape=input_shape, load_weights=load_weights,
                          dropout=dropout,init=init)

    x = base_model.output

    x = Flatten()(x)
    for ind, dense in enumerate(denses):
        x = Dense(dense, activation='relu', name='fc' + str(ind + 1))(x)
    # Prediction
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)

    # Create model.
    model = Model(base_model.input, x, name=basename)

    if load_weights is not None:
        model.load_weights(load_weights, True)

    return model


def get_fitnet_top_by_version(version):
    if version == 1:
        model_fn = fitnet_v1_top
    elif version == 2:
        model_fn = fitnet_v2_top
    elif version == 3:
        model_fn = fitnet_v3_top
    else:
        raise RuntimeError("Not supported version")
    return model_fn


def fitnet_o1(
        version=1,
        denses=[], nb_classes=10, input_shape=None,
        load_weights=None, dropout=False, init='glorot_normal'):

    model_fn = get_fitnet_top_by_version(version)
    model = fitnet_wrapper_o1(model_fn, 'fitnet_v1', denses=denses, nb_classes=nb_classes,
                              input_shape=input_shape, dropout=dropout, init=init,
                              load_weights=load_weights)
    return model


def fitnet_o2(parametrics=[], mode=0, nb_classes=1000, input_shape=(224,224,3),
                load_weights=None,
                cov_mode='channel',
                cov_branch='o2transform',
                cov_branch_output=None,
                last_avg=False,
                freeze_conv=False,
                cov_regularizer=None,
                nb_branch=1,
                concat='concat',
                last_conv_feature_maps=[],
                version=1,
               dropout=False,
                **kwargs
                ):


    if cov_branch_output is None:
        cov_branch_output = nb_classes
    basename = 'fitnet_v{}_o2'.format(version)
    if parametrics is not []:
        basename += '_para-'
        for para in parametrics:
            basename += str(para) + '_'
    basename += 'mode_{}'.format(str(mode))

    base_model_fn = get_fitnet_top_by_version(version)

    base_model = base_model_fn(input_shape=input_shape, dropout=dropout, last_avg=last_avg)

    if input_shape[0] == 3:
        # Define the channel
        if K.image_dim_ordering() == 'tf':
            if input_shape[0] in {1, 3}:
                input_shape = (input_shape[1], input_shape[2], input_shape[0])

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
    if load_weights:
        model.load_weights(load_weights, by_name=True)
    return model