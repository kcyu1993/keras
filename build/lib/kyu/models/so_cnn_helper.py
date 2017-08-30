import warnings

from keras.legacy.layers import merge
from keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, MaxPooling2D

from kyu.models.secondstat import SecondaryStatistic, O2Transform, WeightedVectorization,  LogTransform, \
    SeparateConvolutionFeatures, MatrixReLU, Regrouping, MatrixConcat, \
    PowTransform, BatchNormalization_v2, ExpandDims, Squeeze, FlattenSymmetric, BiLinear, \
    Correlation
from kyu.tensorflow.ops.normalization import SecondOrderBatchNormalization

from keras.models import Model
from kyu.utils.train_utils import toggle_trainable_layers

import tensorflow as tf


def get_cov_name_base(stage, block, **kwargs):
    return 'cov-{}-br_{}'.format(str(stage), block)


def fn_regroup(tensors):
    """
    Python function which takes a list of tensors, returning their combinations.

    Parameters
    ----------
    tensors

    Returns
    -------
    Combinations of them
    """
    outputs = []
    n_inputs = len(tensors)
    for i in range(n_inputs - 1):
        for j in range(i + 1, n_inputs):
            outputs.append(tf.concat([tf.identity(tensors[i]), tf.identity(tensors[j])]))
    return outputs


def covariance_block_original(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                              cov_mode='channel', cov_regularizer=None,
                              o2t_constraints=None, o2t_regularizer=None, o2t_activation='relu',
                              use_bias=False, robust=False, cov_alpha=0.1, cov_beta=0.3,
                              **kwargs):
    cov_name_base = get_cov_name_base(stage, block)
    o2t_name_base = 'o2t' + str(stage) + '_branch' + block
    wp_name_base = 'wp' + str(stage) + '_branch' + block
    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                               robust=robust, cov_alpha=cov_alpha,
                               cov_beta=cov_beta)(input_tensor)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            x = O2Transform(param, activation=o2t_activation, name=o2t_name_base + str(id),
                            kernel_constraint=o2t_constraints, kernel_regularizer=o2t_regularizer,
                            )(x)
    with tf.name_scope(wp_name_base):
        x = WeightedVectorization(nb_class, use_bias=use_bias, activation=activation, name=wp_name_base)(x)
    return x


def covariance_block_log(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                         cov_mode='channel', cov_regularizer=None, vectorization='wv',
                         o2tconstraints=None,
                         **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block
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
                              o2tconstraints=None, vectorization='wv',
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


def covariance_block_no_wv(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                           cov_mode='channel', cov_regularizer=None, vectorization='wv',
                           o2t_constraints=None, o2t_regularizer=None,
                           normalization=False, so_mode=1, robust=False, cov_alpha=0.3,
                           cov_beta=0.1,
                           **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                               cov_alpha=cov_alpha, cov_beta=cov_beta,
                               robust=robust)(input_tensor)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            if normalization:
                x = SecondOrderBatchNormalization(so_mode=so_mode, momentum=0.8, axis=-1)(x)
            x = O2Transform(param, activation=activation, name=o2t_name_base + str(id),
                            kernel_constraint=o2t_constraints,
                            kernel_regularizer=o2t_regularizer)(x)
    return x


def covariance_block_matbp(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                           cov_mode='channel', cov_regularizer=None, vectorization='dense',
                           o2tconstraints=None,
                           **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    log_name_base = 'log' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon, normalization='mean',
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
    # x = BiLinear()(input_tensor)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)

    # add log layer here.
    x = LogTransform(epsilon, name=log_name_base)(x)
    x = Flatten()(x)
    # if vectorization == 'dense':
    #     x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    # else:
    #     ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def covariance_block_pow(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                         cov_mode='channel', cov_regularizer=None, vectorization='mat_flatten',
                         o2tconstraints=None, cov_beta=0.3,
                         **kwargs):
    # if epsilon > 0:
    #     cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    # else:
    #     cov_name_base = 'cov' + str(stage) + block + '_branch'
    cov_name_base = get_cov_name_base(stage, block)
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    pow_name_base = 'pow' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon, cov_beta=cov_beta,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer)(input_tensor)

    # Try the power transform before and after.
    x = PowTransform(alpha=0.5, name=pow_name_base, normalization=None)(x)
    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)

    if vectorization == 'wv':
        x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base)(x)
    elif vectorization == 'dense':
        x = Flatten()(x)
        x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
    elif vectorization == 'flatten':
        x = Flatten()(x)
    elif vectorization == 'mat_flatten':
        x = FlattenSymmetric()(x)
    else:
        ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def covariance_block_multi_o2t(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                               cov_mode='channel', cov_regularizer=None, vectorization=None,
                               o2t_constraints=None, nb_o2t=4, o2t_concat='concat',
                               **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)

    # Try to implement multiple o2t layers out of the same x.
    cov_input = x
    cov_br = []
    for i in range(nb_o2t):
        x = cov_input
        for id, param in enumerate(parametric):
            x = O2Transform(param, activation='relu', name=o2t_name_base + str(id) + '_'+str(i))(x)
        if vectorization == 'wv':
            x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base + str(id) + '_' + str(i))(x)
        elif vectorization == 'dense':
            x = Flatten()(x)
            x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
        elif vectorization == 'flatten':
            x = Flatten()(x)
        elif vectorization == 'mat_flatten':
            x = FlattenSymmetric()(x)
        elif vectorization is None:
            pass
        else:
            ValueError("vectorization parameter not recognized : {}".format(vectorization))

        cov_br.append(x)

    if o2t_concat == 'concat' and vectorization is None:
        # use matrix concat
        x = MatrixConcat(cov_br)(cov_br)
        x = WeightedVectorization(nb_class * nb_o2t / 2)(x)
    elif o2t_concat == 'concat':
        # use vector concat
        x = merge(cov_br, mode='concat')
    else:
        raise NotImplementedError

    return x


def covariance_block_sobn_multi_o2t(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                                    cov_mode='channel', cov_regularizer=None, vectorization=None,
                                    o2t_constraints=None, nb_o2t=1, o2t_concat='concat', so_mode=2,
                                    **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)

    # Try to implement multiple o2t layers out of the same x.
    cov_input = x
    cov_br = []
    for i in range(nb_o2t):
        x = cov_input
        for id, param in enumerate(parametric):
            x = SecondOrderBatchNormalization(so_mode=so_mode, momentum=0.8, axis=-1)(x)
            x = O2Transform(param, activation='relu', name=o2t_name_base + str(id) + '_'+str(i))(x)
        if vectorization == 'wv':
            x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base + str(id) + '_' + str(i))(x)
        elif vectorization == 'dense':
            x = Flatten()(x)
            x = Dense(nb_class, activation=activation, name=dense_name_base)(x)
        elif vectorization == 'flatten':
            x = Flatten()(x)
        elif vectorization == 'mat_flatten':
            x = FlattenSymmetric()(x)
        elif vectorization is None:
            pass
        else:
            ValueError("vectorization parameter not recognized : {}".format(vectorization))

        cov_br.append(x)

    if o2t_concat == 'concat' and vectorization is None:
        # use matrix concat
        x = MatrixConcat(cov_br)(cov_br)
        x = WeightedVectorization(nb_class * nb_o2t / 2)(x)
    elif o2t_concat == 'concat':
        # use vector concat
        x = merge(cov_br, mode='concat')
    else:
        raise NotImplementedError

    return x


def covariance_block_batch(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                           cov_mode='pmean', cov_regularizer=None, vectorization='wv',
                           o2tconstraints=None,
                           **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    pow_name_base = 'pow' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)

    # Try the power transform before and after.

    for id, param in enumerate(parametric):
        x = O2Transform(param, activation='relu', name=o2t_name_base + str(id))(x)
        x = ExpandDims()(x)
        x = BatchNormalization_v2(axis=-1)(x)
        x = Squeeze()(x)

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


def covariance_block_corr(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                          cov_mode='channel', cov_regularizer=None, vectorization='wv',
                          o2t_constraints=None, normalization=False, so_mode=1
                          ,
                          **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'
    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
        x = Correlation()(x)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            if normalization:
                x = SecondOrderBatchNormalization(so_mode=so_mode, momentum=0.8, axis=-1)(x)
            x = O2Transform(param, activation='relu', name=o2t_name_base + str(id), kernel_constraint=o2t_constraints)(x)
    with tf.name_scope(wp_name_base):
        x = WeightedVectorization(nb_class, activation=activation, name=wp_name_base)(x)
    return x


def covariance_block_corr_no_wv(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                                cov_mode='channel', cov_regularizer=None, vectorization='wv',
                                o2t_constraints=None, normalization=False, so_mode=1
                                ,
                                **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'
    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
        x = Correlation()(x)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            if normalization:
                x = SecondOrderBatchNormalization(so_mode=so_mode, momentum=0.8, axis=-1)(x)
            x = O2Transform(param, activation='relu', name=o2t_name_base + str(id), kernel_constraint=o2t_constraints)(x)
    return x


def covariance_block_new_wv(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                            cov_mode='channel', cov_regularizer=None, vectorization='wv',
                            o2t_constraints=None, normalization=False, so_mode=1
                            ,
                            **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'
    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)(input_tensor)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            x = O2Transform(param, activation='relu', name=o2t_name_base + str(id), kernel_constraint=o2t_constraints)(x)

    x = WeightedVectorization(nb_class, output_sqrt=False, activation_regularizer=None, name=wp_name_base)(x)
    return x


def upsample_wrapper_v1(x, last_conv_feature_maps=[],method='conv',kernel=(1,1), stage='', **kwargs):
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
        for ind, feature_dim in enumerate(last_conv_feature_maps):
            x = Conv2D(feature_dim, kernel, name='1x1_conv_' + str(ind) + stage, **kwargs)(x)
    elif method == 'deconv':
        for feature_dim in last_conv_feature_maps:
            x = Conv2DTranspose(feature_dim, kernel, name='dconv_' + stage, **kwargs)(x)
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
        regroup=False,
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
    covariance_block = get_cov_block(cov_branch)

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
    elif mode == 3:
        if nb_branch == 1:

            cov_branch = covariance_block(cov_input, cov_branch_output, stage=5, block='a', parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                                          o2t_constraints='UnitNorm',
                                          **kwargs)
            x = Dense(nb_classes, activation='softmax', name='predictions')(cov_branch)
        elif nb_branch > 1:
            pass

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
        regroup=False,
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
    cov_branch_mode = cov_branch
    # Function name
    covariance_block = get_cov_block(cov_branch)

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
    if regroup:
        with tf.device('/gpu:0'):
            cov_input = Regrouping(None)(cov_input)
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
        elif mode == 3:
            cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind), parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                                          o2t_constraints='UnitNorm',
                                          **kwargs)
            x = cov_branch
        cov_outputs.append(x)

    if concat == 'concat':
        if cov_branch_mode == 'o2t_no_wv' or cov_branch_mode == 'corr_no_wv':
            x = MatrixConcat(cov_outputs, name='Matrix_diag_concat')(cov_outputs)
            x = WeightedVectorization(cov_branch_output*nb_branch, name='WV_big')(x)
        else:
            x = merge(cov_outputs, mode='concat', name='merge')
    elif concat == 'sum':
        x = merge(cov_outputs, mode='sum', name='sum')
        if cov_branch_mode == 'o2t_no_wv':
            x = WeightedVectorization(cov_branch_output, name='wv_sum')(x)
    elif concat == 'ave':
        x = merge(cov_outputs, mode='ave', name='ave')
        if cov_branch_mode == 'o2t_no_wv':
            x = WeightedVectorization(cov_branch_output, name='wv_sum')(x)
    else:
        raise RuntimeError("concat mode not support : " + concat)

    if freeze_conv:
        toggle_trainable_layers(base_model, not freeze_conv)

    # x = Dense(cov_branch_output * nb_branch, activation='relu', name='Dense_b')(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(base_model.input, x, name=basename)
    return model


def dcov_multi_out_model_wrapper(
        base_model, parametrics=[], mode=0, nb_classes=1000,
        basename='',
        cov_mode='channel',
        cov_branch='o2t_no_wv',
        cov_branch_output=None,
        freeze_conv=False,
        cov_regularizer=None,
        nb_branch=1,
        concat='concat',
        last_conv_feature_maps=[],
        upsample_method='conv',
        regroup=False,

        **kwargs
    ):
    """
    Wrapper for any multi output base model, attach right after the last layer of given model

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

    mode 1: 1x1 reduce dim

    Returns
    -------

    """
    cov_branch_mode = cov_branch
    # Function name
    covariance_block = get_cov_block(cov_branch)

    if cov_branch_output is None:
        cov_branch_output = nb_classes
    # 256, 512, 512
    block1, block2, block3 = outputs = base_model.outputs
    print("===================")
    cov_outputs = []
    if mode == 1:
        print("Model design : ResNet_o2_multi_branch 1x1 conv to reduce dim ")
        """ 1x1 conv to reduce dim """
        # Starting from block3
        block3 = upsample_wrapper_v1(block3, [1024, 512])
        block2 = upsample_wrapper_v1(block2, [512])
        block2 = MaxPooling2D()(block2)
        block1 = MaxPooling2D(pool_size=(4,4))(block1)
        outputs = [block1, block2, block3]
        for ind, x in enumerate(outputs):
            cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind), parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)
            x = cov_branch
            cov_outputs.append(x)
    elif mode == 2 or mode == 3:
        """ Use branchs to reduce dim """
        block3 = SeparateConvolutionFeatures(4)(block3)
        block2 = SeparateConvolutionFeatures(2)(block2)
        block1 = MaxPooling2D()(block1)
        block1 = [block1]
        outputs = [block1, block2, block3]
        for ind, outs in enumerate(outputs):
            block_outs = []
            for ind2, x in enumerate(outs):
                cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind) + '_' + str(ind2),
                                              parametric=parametrics,
                                              cov_mode=cov_mode, cov_regularizer=cov_regularizer, **kwargs)
                x = cov_branch
                block_outs.append(x)
            if mode == 3:
                """ Sum block covariance output """
                if len(block_outs) > 1:
                    o = merge(block_outs, mode='sum', name='multibranch_sum_{}'.format(ind))
                    o = WeightedVectorization(cov_branch_output)(o)
                    cov_outputs.append(o)
                else:
                    a = block_outs[0]
                    if 'o2t' in a.name:
                        a = WeightedVectorization(cov_branch_output)(a)
                    cov_outputs.append(a)
            else:
                cov_outputs.extend(block_outs)
    elif mode == 4:
        """ Use the similar structure to Feature Pyramid Network """
        # supplimentary stream
        block1 = upsample_wrapper_v1(block1, [256], stage='block1')
        block2 = upsample_wrapper_v1(block2, [256], stage='block2')
        # main stream
        block3 = upsample_wrapper_v1(block3, [512], stage='block3')

        cov_input = SeparateConvolutionFeatures(nb_branch)(block3)
        cov_outputs = []
        for ind, x in enumerate(cov_input):

            cov_branch = covariance_block(x, cov_branch_output, stage=5, block=str(ind), parametric=parametrics,
                                          cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                                          normalization=False,
                                          **kwargs)
            x = cov_branch
            cov_outputs.append(x)

        x = MatrixConcat(cov_outputs, name='Matrix_diag_concat')(cov_outputs)
        x = O2Transform(64, activation='relu', name='o2t_mainst_1')(x)

        block2 = SecondaryStatistic(name='cov_block2', cov_mode='pmean', robust=False, eps=1e-5)(block2)
        block2 = O2Transform(64, activation='relu', name='o2t_block2')(block2)

        # fuse = merge([block2, x], mode='sum')
        # fuse = O2Transform(64, activation='relu', name='o2t_mainst_2')(fuse)

        block1 = SecondaryStatistic(name='cov_block1', cov_mode='pmean', robust=False, eps=1e-5)(block1)
        block1 = O2Transform(64, activation='relu', name='o2t_block1')(block1)

        # fuse = merge([fuse, block1], mode='sum')

        x = MatrixConcat([x, block1, block2], name='Matrix_diag_concat_all')([x, block1, block2])
        x = WeightedVectorization(128, activation='relu', name='wv_fuse')(x)

        # Merge the last matrix for matrix concat

    if freeze_conv:
        toggle_trainable_layers(base_model, not freeze_conv)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(base_model.input, x, name=basename)
    return model


def get_cov_block(cov_branch):
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
    elif cov_branch == 'o2t_no_wv':
        covariance_block = covariance_block_no_wv
    elif cov_branch == 'matbp':
        covariance_block = covariance_block_matbp
    elif cov_branch == 'pow_o2t':
        covariance_block = covariance_block_pow
    elif cov_branch == 'o2t_batch_norm':
        covariance_block = covariance_block_batch
    elif cov_branch == 'multiple_o2t':
        covariance_block = covariance_block_multi_o2t
    elif cov_branch == 'sobn_multiple_o2t':
        covariance_block = covariance_block_sobn_multi_o2t
    elif cov_branch == 'corr':
        covariance_block = covariance_block_corr
    elif cov_branch == 'new_wv':
        covariance_block = covariance_block_new_wv
    else:
        raise ValueError('covariance cov_mode not supported')

    return covariance_block
