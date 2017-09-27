import tensorflow as tf

from keras.layers import Flatten, Dense
from keras.legacy.layers import merge
from kyu.layers.secondstat import WeightedVectorization, \
    O2Transform, SecondaryStatistic, BatchNormalization_v2, Correlation, \
    LogTransform
from kyu.layers.assistants import FlattenSymmetric, MatrixConcat, ExpandDims, Squeeze
from kyu.tensorflow.ops.normalization import SecondOrderBatchNormalization


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