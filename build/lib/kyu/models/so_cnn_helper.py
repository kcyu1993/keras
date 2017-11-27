import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, Reshape, BatchNormalization
from keras.layers import concatenate, add, average
from kyu.layers.secondstat import SecondaryStatistic, O2Transform, WeightedVectorization, LogTransform, \
    PowTransform, BiLinear, GlobalSquarePooling
from kyu.layers.assistants import FlattenSymmetric, MatrixConcat, MatrixReLU, SignedSqrt
from kyu.legacy.so_cnn_helper import covariance_block_multi_o2t, covariance_block_sobn_multi_o2t, \
    covariance_block_batch, covariance_block_corr, covariance_block_vector_space, covariance_block_mix, \
    covariance_block_residual
from kyu.tensorflow.ops.normalization import SecondOrderBatchNormalization


def get_tensorboard_layer_name_keys():
    return ['cov', 'o2t', 'pv', '1x1', 'bn-', 'last_bn', 'pow', ]


def get_cov_name_base(stage, block, epsilon):
    if epsilon > 0:
        cov_name_base = 'cov-{}-br_{}-eps_{}'.format(str(stage), block, epsilon)
    else:
        cov_name_base = 'cov-{}-br_{}'.format(str(stage), block)
    return cov_name_base


def get_o2t_name_base(stage, block):
    return 'o2t-{}-br_{}'.format(str(stage), block)


def get_pv_name_base(stage, block):
    return 'pv-{}-br_{}'.format(str(stage), block)


def get_name_base(name, stage, block):
    return '{}-{}-br_{}'.format(name, str(stage), block)


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
                              pv_constraints=None, pv_regularizer=None, pv_activation='relu',
                              use_bias=False, robust=False, cov_alpha=0.1, cov_beta=0.3,
                              **kwargs):
    cov_name_base = get_cov_name_base(stage, block)
    o2t_name_base = get_o2t_name_base(stage, block)
    wp_name_base = get_pv_name_base(stage, block)
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
        x = WeightedVectorization(nb_class,
                                  kernel_regularizer=pv_regularizer,
                                  kernel_constraint=pv_constraints,
                                  use_bias=use_bias,
                                  activation=pv_activation,
                                  name=wp_name_base)(x)
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


def covariance_block_matbp(input_tensor, nb_class, stage, block,
                           epsilon=0,
                           parametric=[],
                           vectorization='wv',
                           batch_norm=True,
                           log_norm=False,
                           cov_kwargs=None,
                           o2t_kwargs=None,
                           pv_kwargs=None,
                           **kwargs):
    """ Matrix Backprop """
    cov_name_base = get_cov_name_base(stage, block, epsilon)
    o2t_name_base = get_o2t_name_base(stage, block)
    wp_name_base = get_pv_name_base(stage, block)

    # Add a normalization before goinging into secondary statistics
    x = input_tensor
    if batch_norm:
        x = BatchNormalization(axis=3, name='last_BN_{}_{}'.format(stage, block))(x)

    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base,
                               **cov_kwargs)(x)
    if log_norm:
        x = LogTransform()(x)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            if o2t_kwargs.has_key('output_dim'):
                del o2t_kwargs['output_dim']
            x = O2Transform(output_dim=param,
                            name=o2t_name_base + str(id),
                            **o2t_kwargs
                            )(x)
    if vectorization == 'pv' or vectorization == 'wv':
        x = WeightedVectorization(nb_class,
                                  name=wp_name_base,
                                  **pv_kwargs)(x)
    elif vectorization == 'mat_flatten' or vectorization == 'flatten':
        x = FlattenSymmetric()(x)
    return x


def covariance_block_pow(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                         cov_mode='channel', cov_regularizer=None, vectorization='mat_flatten',
                         o2tconstraints=None, cov_beta=0.3,
                         **kwargs):
    # if epsilon > 0:
    #     cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    # else:
    #     cov_name_base = 'cov' + str(stage) + block + '_branch'
    cov_name_base = get_cov_name_base(stage, block, epsilon)
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    pow_name_base = 'pow' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    x = input_tensor

    # add baseline test.
    # x = BatchNormalization(axis=3, name='last_BN')(input_tensor)

    x = SecondaryStatistic(name=cov_name_base, eps=epsilon, cov_beta=cov_beta,
                           cov_mode=cov_mode, cov_regularizer=cov_regularizer)(x)

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
    elif vectorization == 'no':
        pass
    else:
        ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def covariance_block_bilinear(input_tensor, nb_class, stage, block, epsilon=0, parametric=[], activation='relu',
                              cov_mode='channel', cov_regularizer=None, vectorization='mat_flatten',
                              o2tconstraints=None, cov_beta=0.3,
                              **kwargs):
    """ Updated bilinear block, reference to pow branch """
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    bilinear_name_base = 'bilinear' + str(stage) + block + '_branch'
    dense_name_base = 'fc' + str(stage) + block + '_branch'
    wp_name_base = 'wp' + str(stage) + block + '_branch'

    # Try the power transform before and after.
    x = BiLinear(eps=0., activation='linear', name=bilinear_name_base)(input_tensor)
    dim = K.int_shape(x)[1]
    x = Reshape(target_shape=(int(np.sqrt(dim)), int(np.sqrt(dim))))(x)

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
    elif vectorization == 'no':
        pass
    else:
        ValueError("vectorization parameter not recognized : {}".format(vectorization))
    return x


def covariance_block_norm_wv(input_tensor, nb_class, stage, block, epsilon=0, parametric=[],
                             activation='relu',
                             vectorization='wv',
                             cov_mode='channel', cov_regularizer=None,
                             o2t_constraints=None, o2t_regularizer=None, o2t_activation='relu',
                             use_bias=False, robust=False, cov_alpha=0.1, cov_beta=0.3,
                             pv_constraints=None, pv_regularizer=None, pv_activation='relu',
                             pv_normalization=False,
                             pv_output_sqrt=True,
                             pv_use_bias=False,
                             pv_use_gamma=False,
                             # pv_gamma_constraints=None,
                             # pv_gamma_initializer='ones',
                             # pv_gamma_regularizer=None,
                             **kwargs):
    if epsilon > 0:
        cov_name_base = 'cov' + str(stage) + block + '_branch_epsilon' + str(epsilon)
    else:
        cov_name_base = 'cov' + str(stage) + block + '_branch'
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'

    # Add a normalization before goinging into secondary statistics
    # x = input_tensor
    x = BatchNormalization(axis=3, name='last_BN')(input_tensor)

    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base, eps=epsilon,
                               cov_mode=cov_mode, cov_regularizer=cov_regularizer,
                               cov_alpha=cov_alpha, cov_beta=cov_beta, robust=robust,
                               **kwargs)(x)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            x = O2Transform(param, activation=o2t_activation,
                            kernel_constraint=o2t_constraints,
                            kernel_regularizer=o2t_regularizer,
                            use_bias=use_bias,
                            name=o2t_name_base + str(id),
                            )(x)
    if vectorization == 'pv' or vectorization == 'wv':
        x = WeightedVectorization(nb_class,
                                  output_sqrt=pv_output_sqrt,
                                  use_bias=pv_use_bias,
                                  normalization=pv_normalization,
                                  kernel_regularizer=pv_regularizer,
                                  kernel_constraint=pv_constraints,
                                  use_gamma=pv_use_gamma,
                                  # gamma_constraint=pv_gamma_constraints,
                                  # gamma_initializer=pv_gamma_initializer,
                                  # gamma_regularizer=pv_gamma_regularizer,
                                  activation=pv_activation,
                                  name=wp_name_base)(x)
    return x


def covariance_block_newn_wv(input_tensor, nb_class, stage, block,
                             epsilon=0,
                             parametric=[],
                             vectorization='wv',
                             batch_norm=True,
                             batch_norm_end=False,
                             batch_norm_kwargs={},
                             pow_norm=False,
                             cov_kwargs={},
                             o2t_kwargs={},
                             pv_kwargs={},
                             **kwargs):
    cov_name_base = get_cov_name_base(stage, block, epsilon)
    o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'

    # Add a normalization before goinging into secondary statistics
    x = input_tensor
    if batch_norm:
        print(batch_norm_kwargs)
        batch_norm_layer = BatchNormalization(axis=3, name='last_BN_{}_{}'.format(stage, block),
                                              **batch_norm_kwargs)
        x = batch_norm_layer(x)

    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base,
                               **cov_kwargs)(x)
    if pow_norm:
        x = PowTransform()(x)
    for id, param in enumerate(parametric):
        with tf.name_scope(o2t_name_base + str(id)):
            if o2t_kwargs.has_key('output_dim'):
                del o2t_kwargs['output_dim']
            x = O2Transform(output_dim=param,
                            name=o2t_name_base + str(id),
                            **o2t_kwargs
                            )(x)
    if vectorization == 'pv' or vectorization == 'wv':
        pv_kwargs['batch_norm_moving_variance'] = batch_norm_layer.moving_variance if batch_norm else None
        x = WeightedVectorization(nb_class,
                                  name=wp_name_base,
                                  **pv_kwargs)(x)
        use_sqrt = pv_kwargs['output_sqrt']
        if use_sqrt:
            x = SignedSqrt(2)(x)
        if batch_norm_end:
            x = Reshape((1,1, nb_class))(x)
            x = BatchNormalization(axis=3, name='BN-end{}_{}'.format(stage, block),
                                   **batch_norm_kwargs)(x)
            x = Flatten()(x)
    return x


def covariance_block_pv_equivelent(input_tensor, nb_class, stage, block,
                                   batch_norm=True,
                                   batch_norm_kwargs={},
                                   batch_norm_end=False,
                                   conv_kwargs={},
                                   gsp_kwargs={},
                                   **kwargs):
    conv_name_base = get_name_base('1x1_conv', stage, block)
    gsp_name_base = get_name_base('globalsquarepool', stage, block)

    # Add a normalization before goinging into secondary statistics
    x = input_tensor
    if batch_norm:
        print(batch_norm_kwargs)
        x = BatchNormalization(axis=3, name='last_BN_{}_{}'.format(stage, block),
                               **batch_norm_kwargs)(x)

    x = Conv2D(filters=nb_class, kernel_size=(1,1), name=conv_name_base, **conv_kwargs)(x)
    x = GlobalSquarePooling(nb_class, name=gsp_name_base, **gsp_kwargs)(x)
    use_sqrt = gsp_kwargs['output_sqrt']
    if use_sqrt:
        x = SignedSqrt(2)(x)
    if batch_norm_end:
        print(batch_norm_kwargs)
        x = Reshape((1, 1, nb_class))(x)
        x = BatchNormalization(axis=3, name='BN_{}_{}-end'.format(stage, block),
                               **batch_norm_kwargs)(x)
        x = Flatten()(x)
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
    elif cov_branch == 'bilinear':
        covariance_block = covariance_block_bilinear
    elif cov_branch == 'o2t_batch_norm':
        covariance_block = covariance_block_batch
    elif cov_branch == 'multiple_o2t':
        covariance_block = covariance_block_multi_o2t
    elif cov_branch == 'sobn_multiple_o2t':
        covariance_block = covariance_block_sobn_multi_o2t
    elif cov_branch == 'corr':
        covariance_block = covariance_block_corr
    elif cov_branch == 'norm_wv':
        covariance_block = covariance_block_norm_wv
    elif cov_branch == 'new_norm_wv':
        covariance_block = covariance_block_newn_wv
    elif cov_branch == "1x1_gsp":
        covariance_block = covariance_block_pv_equivelent
    else:
        raise ValueError('covariance cov_mode not supported')

    return covariance_block


def merge_branches_with_method(concat, cov_outputs,
                               cov_output_vectorization=None,
                               cov_output_dim=1024,
                               pv_constraints=None, pv_regularizer=None, pv_activation='relu',
                               pv_normalization=False,
                               pv_output_sqrt=True, pv_use_bias=False,
                               **kwargs
                               ):
    """
    Helper to merge all branches with given methods.
    Parameters
    ----------
    concat
    cov_outputs
    cov_output_vectorization
    cov_output_dim
    **kwargs (Passed into vectorization layer)
    Returns
    -------

    """
    if len(cov_outputs) < 2:
        return cov_outputs[0]
    else:
        if concat == 'concat':
            if all(len(cov_output.shape) == 3 for cov_output in cov_outputs):
                # use matrix concat
                x = MatrixConcat(cov_outputs)(cov_outputs)
            else:
                x = concatenate(cov_outputs)
        elif concat == 'sum':
            x = add(cov_outputs)
        elif concat == 'ave' or concat == 'avg':
            x = average(cov_outputs)
        else:
            raise ValueError("Concat mode not supported {}".format(concat))
    if cov_output_vectorization == 'wv' or cov_output_vectorization == 'pv':
        x = WeightedVectorization(output_dim=cov_output_dim,
                                  output_sqrt=pv_output_sqrt,
                                  use_bias=pv_use_bias,
                                  normalization=pv_normalization,
                                  kernel_regularizer=pv_regularizer,
                                  kernel_constraint=pv_constraints,
                                  activation=pv_activation,
                                  )(x)
    else:
        raise ValueError("Merge branches with method only support pv layer now")

    return x
