import tensorflow as tf

from keras.engine import Model
from keras.layers import Flatten, Dense, MaxPooling2D
from kyu.layers.secondstat import WeightedVectorization, O2Transform, SecondaryStatistic
from kyu.layers.assistants import SeparateConvolutionFeatures, Regrouping, MatrixConcat
from kyu.models.so_cnn_helper import get_cov_block, upsample_wrapper_v1
from kyu.utils.train_utils import toggle_trainable_layers
from keras.legacy.layers import merge


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