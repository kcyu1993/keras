import warnings

from keras.engine import merge
from keras.layers import SecondaryStatistic, O2Transform, WeightedVectorization, Flatten, Dense, LogTransform, \
    Convolution2D, Deconvolution2D, SeparateConvolutionFeatures, MatrixReLU

from kyu.theano.general.train import toggle_trainable_layers, Model


def covariance_block_original(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                              cov_mode='channel', cov_regularizer=None, vectorization='wv',
                              **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
    x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base)(x)
    return x


def covariance_block_log(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                         cov_mode='channel', cov_regularizer=None, vectorization='wv',
                         **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    log_name_base = 'log' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
    # add log layer here.
    x = LogTransform(epsilon, name=log_name_base)(x)
    if vectorization == 'wv':
        x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base)(x)
    elif vectorization == 'dense':
        x = Flatten()(x)
        x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    elif vectorization == 'flatten':
        x = Flatten()(x)
    else:
        ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def covariance_block_vector_space(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                                  **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    dense_name_base = 'dense' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon, **kwargs)(input_tensor)
    x = Flatten()(x)
    for id, param in enumerate(parametric):
        x = Dense(param, activation=activation, name=dense_name_base + str(id))(x)
    x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    return x


def covariance_block_mix(input_tensor, nb_class, stage, block, epsilon=0,
                         parametric=[], denses=[], wv=True, wv_param=None, activation='relu',
                         **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + ''
    o2t_name_base = 'o2t' + str(stage) + block
    dense_name_base = 'dense' + str(stage) + block + ''

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon, **kwargs)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
    if wv:
        if wv_param is None:
            wv_param = nb_class
        x = WeightedVectorization(wv_param, activation=activation, name='wv' + str(stage) + block)(x)
    else:
        x = Flatten()(x)
    for id, param in enumerate(denses):
        x = Dense(param, activation=activation, name=dense_name_base + str(id))(x)
    x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    return x


def covariance_block_residual(input_tensor, nb_class, stage, block, epsilon=0,
                              parametric=[], denses=[], wv=True, wv_param=None, activation='relu',
                              **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + ''
    o2t_name_base = 'o2t' + str(stage) + block
    dense_name_base = 'dense' + str(stage) + block + ''

    second_layer = SecondaryStatistic(name=cov_name_base, eps=epsilon, **kwargs)
    x = second_layer(input_tensor)

    cov_dim = second_layer.out_dim
    input_cov = x

    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
        if not param == cov_dim:
            x = O2Transform(cov_dim, activation='relu', name=o2t_name_base + str(id) + 'r')(x)
        x = merge([x, input_cov], mode='sum', name='residualsum_{}_{}'.format(str(id), str(block)))

    if wv:
        if wv_param is None:
            wv_param = nb_class
        x = WeightedVectorization(wv_param, activation=activation, name='wv' + str(stage) + block)(x)
    else:
        x = Flatten()(x)
    for id, param in enumerate(denses):
        x = Dense(param, activation=activation, name=dense_name_base + str(id))(x)

    x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    return x


def covariance_block_aaai(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                          cov_mode='channel', cov_regularizer=None, vectorization='wv',
                          **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    relu_name_base = 'matrelu' + str(stage) + block + '_branch'
    log_name_base = 'log' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
        x = MatrixReLU(epsilon=epsilon, name=relu_name_base + str(id))(x)
    # add log layer here.
    x = LogTransform(epsilon, name=log_name_base)(x)
    if vectorization == 'wv':
        x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base)(x)
    elif vectorization == 'dense':
        x = Flatten()(x)
        x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    elif vectorization == 'flatten':
        x = Flatten()(x)
    else:
        ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def upsample_wrapper_v1(x, last_conv_feature_maps=[],method='conv',kernel=[1,1], **kwargs):
    """
    Wrapper to decrease the dimension feed into SecondStat layers.

    Parameters
    ----------
    last_conv_feature_maps
    method
    kernel

    Returns
    -------

    """
    if method == 'conv':
        for feature_dim in last_conv_feature_maps:
            x = Convolution2D(feature_dim, kernel[0], kernel[1], **kwargs)(x)
    elif method == 'deconv':
        for feature_dim in last_conv_feature_maps:
            x = Deconvolution2D(feature_dim, kernel[0], kernel[1] **kwargs)(x)
    else:
        raise ValueError("Upsample wrapper v1 : Error in method {}".format(method))
    return x



def dcov_model_wrapper_v1(
        base_model, parametrics=[], mode=0, nb_classes=1000,
        basename='',
        cov_mode='channel',
        cov_branch='o2transform',
        cov_branch_output=None,
        freeze_conv=False,
        cov_regularizer=None,
        nb_branch=1,
        concat='concat',
        last_conv_feature_maps=[],
        upsample_method='conv',
        **kwargs
    ):
    """
    Wrapper for any base model, attach right after the last layer of given model

    Parameters
    ----------
    base_model
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

    # Function name
    if cov_branch == 'o2transform':
        covariance_block = covariance_block_original
    elif cov_branch == 'dense':
        covariance_block = covariance_block_vector_space
    elif cov_branch == 'mix':
        covariance_block = covariance_block_mix
    elif cov_branch == 'residual':
        covariance_block = covariance_block_residual
    elif cov_branch == 'aaai':
        covariance_block = covariance_block_aaai
    elif cov_branch == 'log':
        covariance_block = covariance_block_log
    else:
        raise ValueError('covariance cov_mode not supported')

    if cov_branch_output is None:
        cov_branch_output = nb_classes

    x = base_model.output

    x = upsample_wrapper_v1(x, last_conv_feature_maps, upsample_method, kernel=[1,1])

    cov_input = x
    if mode == 0:
        x = Flatten()(x)
        for ind, param in enumerate(parametrics):
            x = Dense(param, activation='relu', name='fc{}'.format(ind))(x)
        x = Dense(nb_classes, activation='softmax')(x)

    if mode == 1:
        if nb_branch == 1:
            cov_branch = covariance_block(cov_input, cov_branch_output, stage=5, block='a', parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)
            x = Dense(nb_classes, activation='softmax', name='predictions')(cov_branch)
        elif nb_branch > 1:
            pass

    elif mode == 2:
        cov_branch = covariance_block(cov_input, cov_branch_output, stage=5, block='a', parametric=parametrics,
                                      cov_regularizer=cov_regularizer, **kwargs)
        x = Flatten()(x)
        x = Dense(nb_classes, activation='relu', name='fc')(x)
        x = merge([x, cov_branch], mode='concat', name='concat')
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    if freeze_conv:
        toggle_trainable_layers(base_model, not freeze_conv)

    model = Model(base_model.input, x, name=basename)
    return model


def dcov_model_wrapper_v2(
        base_model, parametrics=[], mode=0, nb_classes=1000,
        basename='',
        cov_mode='channel',
        cov_branch='o2transform',
        cov_branch_output=None,
        freeze_conv=False,
        cov_regularizer=None,
        nb_branch=1,
        concat='concat',
        last_conv_feature_maps=[],
        upsample_method='conv',
        **kwargs
    ):
    """
    Wrapper for any base model, attach right after the last layer of given model

    Parameters
    ----------
    base_model
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

    # Function name
    if cov_branch == 'o2transform':
        covariance_block = covariance_block_original
    elif cov_branch == 'dense':
        covariance_block = covariance_block_vector_space
    elif cov_branch == 'mix':
        covariance_block = covariance_block_mix
    elif cov_branch == 'residual':
        covariance_block = covariance_block_residual
    elif cov_branch == 'aaai':
        covariance_block = covariance_block_aaai
    else:
        raise ValueError('covariance cov_mode not supported')

    if cov_branch_output is None:
        cov_branch_output = nb_classes

    x = base_model.output

    x = upsample_wrapper_v1(x, last_conv_feature_maps, upsample_method, kernel=[1,1])

    def split_keras_tensor_according_axis(x, nb_split, axis, axis_dim):
        outputs = []
        split_dim = axis_dim / nb_split
        split_loc = [split_dim * i for i in range(nb_split)]
        split_loc.append(-1)
        for i in range(nb_split):
            outputs.append(x[:,:,:, split_loc[i]:split_loc[i+1]])
        return outputs

    cov_input = SeparateConvolutionFeatures(nb_branch)(x)
    cov_outputs = []
    for ind, x in enumerate(cov_input):
        if mode == 0:
            x = Flatten()(x)
            for ind, param in enumerate(parametrics):
                x = Dense(param, activation='relu', name='fc{}'.format(ind))(x)
            # x = Dense(nb_classes, activation='softmax')(x)

        if mode == 1:
            cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind), parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)
            x = cov_branch
            # x = Dense(nb_classes, activation='softmax', name='predictions')(cov_branch)

        elif mode == 2:
            cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind), parametric=parametrics,
                                          cov_regularizer=cov_regularizer, **kwargs)
            x = Flatten()(x)
            x = Dense(nb_classes, activation='relu', name='fc')(x)
            x = merge([x, cov_branch], mode='concat', name='concat')
            # x = Dense(nb_classes, activation='softmax', name='predictions')(x)
        cov_outputs.append(x)

    if concat == 'concat':
        x = merge(cov_outputs, mode='concat', name='merge')

    if freeze_conv:
        toggle_trainable_layers(base_model, not freeze_conv)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(base_model.input, x, name=basename)
    return model

