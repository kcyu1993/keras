from kyu.configs.model_configs.second_order import NewNormWVBranchConfig, PVEquivalentConfig
from kyu.layers.secondstat import get_default_secondstat_args
# from kyu.tensorflow.ops.cov_reg import L2InnerNorm
from kyu.utils.dict_utils import update_source_dict_by_given_kwargs, create_dict_by_given_kwargs
from kyu.utils.inspect_util import get_default_args
from ..model_configs import O2TBranchConfig, NoWVBranchConfig, NormWVBranchConfig


def get_single_o2transform(exp):
    """ Same as before, updated with the new framework """
    if exp == 1:
        return O2TBranchConfig(
            parametric=[],
            activation='relu',
            cov_mode='channel',
            # cov_mode='pmean',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            input_shape=(224,224,3),
            nb_class=67,
            cov_branch_output=128,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='TestingO2T-bias',
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    elif exp == 2:
        return O2TBranchConfig(
            parametric=[64, 32,],
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=32,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-testing',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    elif exp == 3:
        # o2t_regularizer = 'l1'
        # o2t_regularizer = 'l2'
        o2t_regularizer = 'l1_l2'
        # o2t_regularizer = None
        return O2TBranchConfig(
            parametric=[256],
            # parametric=[256, 128, 64, ],
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            o2t_activation='relu',
            o2t_constraints=None,
            o2t_regularizer=o2t_regularizer,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=256,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-comparing_regularizer-{}'.format(o2t_regularizer),
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',

        )
    elif exp == 4:
        o2t_constraints = 'UnitNorm'
        # o2t_constraints = None
        return O2TBranchConfig(
            parametric=[256, 128, ],
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.1,
            o2t_activation='relu',
            o2t_constraints=o2t_constraints,
            o2t_regularizer='l1_l2',
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=64,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-comparing_constraints-{}'.format(o2t_constraints),
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[512],
            last_conv_kernel=[1, 1],
            upsample_method='conv',

        )
    elif exp == 5:
        # o2t_regularizer = 'l1'
        # o2t_regularizer = 'l2'
        o2t_regularizer = 'l1_l2'
        # o2t_regularizer = None
        return O2TBranchConfig(
            parametric=[256],
            # parametric=[256, 128, 64, ],
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            o2t_activation='relu',
            o2t_constraints=None,
            o2t_regularizer=o2t_regularizer,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=128,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-comparing_regularizer_{}-branch2'.format(o2t_regularizer),
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[512],
            last_conv_kernel=[1, 1],
            upsample_method='conv',

        )
    elif exp == 6:
        o2t_regularizer = 'l1_l2'
        # o2t_regularizer = None
        return O2TBranchConfig(
            parametric=[128],
            # parametric=[256, 128, 64, ],
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            o2t_activation='relu',
            o2t_constraints=None,
            o2t_regularizer=o2t_regularizer,
            robust=False,
            vectorization='wv',
            use_bias=False,
            epsilon=1e-5,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=1024,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='Original-O2T-reg_{}-PV_1024'.format(o2t_regularizer),
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',

        )
    else:

        raise ValueError("N")


def get_no_wv_config(exp=1):

    if exp == 1:
        o2t_regularizer = 'l1_l2'
        model_config = NoWVBranchConfig(
            parametric=[128],
            epsilon=1e-7,
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            cov_beta=0.1,
            o2t_regularizer=o2t_regularizer,
            robust=False,
            normalization=False,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=128,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='No-PV-2branch-128',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    elif exp == 2:
        # o2t_regularizer = None
        o2t_regularizer = 'l2'
        # o2t_regularizer = 'l1'
        # o2t_regularizer = 'l1_l2'
        model_config = NoWVBranchConfig(
            parametric=[128, 64, 32],
            epsilon=1e-7,
            activation='relu',
            cov_mode='pmean',
            cov_alpha=0.3,
            cov_beta=0.1,
            o2t_regularizer=o2t_regularizer,
            robust=False,
            normalization=False,
            # input_shape=(256, 256, 3),
            input_shape=(224, 224, 3),
            nb_class=67,
            cov_branch_output=64,
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1,
            freeze_conv=False, name='No-PV-2branch-128',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[512],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    else:
        raise ValueError("N")

    return model_config


def get_wv_norm_config(exp):
    o2t_regularizer = 'l2'
    parametric = []
    name = None
    nb_branch = 1
    pv_use_bias = False
    pv_use_gamma = False
    pv_output_sqrt = True
    mode = 1
    load_weights = 'imagenet'
    if exp == 1:
        parametric = []
        cov_branch_output = 1024
        name = 'BN-Cov-PV{}'.format(cov_branch_output)
        pv_use_bias = False
        pv_normalization = False
        pv_output_sqrt = False
        mode = 1

    elif exp == 2:
        parametric = [256]
        cov_branch_output = 1024
        name = 'BN-Cov-O2T{}-PV{}'.format(parametric, cov_branch_output)
        nb_branch = 1
        pv_normalization = False

    elif exp == 3:
        parametric = []
        cov_branch_output = 512
        name = 'FOSO-BN-Cov-PV{}'.format(cov_branch_output)
        pv_use_bias = True
        pv_normalization = True
        mode = 2
    elif exp == 4:
        cov_branch_output = 1024
        name = 'BN-Cov-PV{}_gamma'.format(cov_branch_output)
        pv_use_gamma = True
        pv_normalization = True
        pv_output_sqrt = True
        pv_use_bias = True

    model_config = NormWVBranchConfig(
        parametric=parametric,
        activation='relu',
        cov_mode='channel',
        vectorization='wv',
        epsilon=1e-5,
        use_bias=False,
        pv_use_bias=pv_use_bias,
        pv_use_gamma=pv_use_gamma,
        pv_output_sqrt=pv_output_sqrt,
        robust=False,
        cov_alpha=0.1,
        cov_beta=0.3,
        o2t_constraints=None,
        o2t_regularizer=o2t_regularizer,
        o2t_activation='relu',
        pv_constraints=None,
        pv_regularizer=o2t_regularizer,
        pv_activation='relu',
        pv_normalization=pv_normalization,
        # Other
        input_shape=(224, 224, 3),
        nb_class=67,
        cov_branch_output=cov_branch_output,
        class_id=None,
        load_weights='imagenet',
        # configs for _compose_second_order_things
        mode=mode,
        freeze_conv=False, name=name + '-{}_branch'.format(nb_branch),
        nb_branch=nb_branch,
        concat='concat',
        cov_output_vectorization='pv',
        last_conv_feature_maps=[256],
        last_conv_kernel=[1, 1],
        upsample_method='conv',
    )
    return model_config


def get_new_wv_norm_general(exp=1):
    cov_branch_output = 1024
    batch_size = 32
    dense_branch_output = None
    last_conv_feature_maps = [256]
    parametric = []
    nb_branch = 1
    vectorization = 'wv'
    load_weights = 'imagenet'
    mode = 1
    concat = None
    separate_conv_features = True
    cov_kwargs = get_default_secondstat_args('Cov')
    cov_use_kernel = False

    # O2t
    o2t_kwargs = get_default_secondstat_args('O2T')

    # PV layers
    pv_kwargs = get_default_secondstat_args('PV')
    pow_norm = False
    use_gamma = True
    batch_norm_end = False
    pv_reg = None
    normalization = True  # normalize to further fit Chi-square distribution
    use_bias = True  # use bias for normalization additional
    input_shape = (224, 224, 3)
    weight_decay = 0

    pred_activation = 'softmax'

    # batch norm
    batch_norm_kwargs = {'scale': True}
    if exp == 1:
        cov_branch_output = 2048
        use_gamma = True
        normalization = True
        name = 'BN-Cov-PV{}-mode1_complete-gamma{}-norm{}'.format(cov_branch_output, use_gamma, normalization)
        mode = 1
    elif exp == 2:
        use_gamma = False
        cov_branch_output = 2048
        name = "BN-Cov-PV{}-mode2_gamma{}".format(cov_branch_output, use_gamma)
        mode = 2
    elif exp == 3:
        nb_branch = 1
        last_conv_feature_maps = [256]
        pow_norm = True
        parametric = [256]
        vectorization = 'wv'
        name = "BN-Cov-Pow-PV{}-mode1_complete".format(cov_branch_output)
    elif exp == 4:
        last_conv_feature_maps = [512]
        use_gamma = True
        use_bias = True
        batch_norm_end = False
        normalization = True
        cov_branch_output = 2048
        name = "448-Conv-512-BN-Cov-PV{}".format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16

    elif exp == 5:
        last_conv_feature_maps = [512]
        use_gamma = False
        use_bias = False
        batch_norm_end = True
        normalization = True
        cov_branch_output = 2048
        name = "448-Conv-512-BN-Cov-PV{}-BN".format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16

    elif exp == 6:
        cov_branch_output = 2048
        use_gamma = True
        load_weights = None
        name = 'Scratch-BN-Cov-PV{}-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
        mode = 1
    elif exp == 7:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-Cov-PV{}-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
        batch_norm_kwargs['scale'] = False
    elif exp == 8:
        cov_branch_output = 2048
        use_gamma = False
        parametric = [32, 256]
        nb_branch = 8
        separate_conv_features = False
        name = 'BN-Cov-O2T{}-PV{}-mode1-gamma{}-Sep{}'.\
            format(cov_branch_output, parametric, use_gamma, separate_conv_features)
        concat = 'sum'
        vectorization = None
        batch_norm_kwargs['scale'] = True
    elif exp == 9:
        cov_branch_output = 2048
        use_gamma = True
        batch_norm_end = True
        normalization = True
        use_bias = False
        name = 'BN-Cov-PV{}-FromFO-gamma{}'.format(cov_branch_output, use_gamma)
        mode = 1
        load_weights = 'secondorder'
    elif exp == 10:
        cov_branch_output = 2048
        use_gamma = False
        use_bias = False
        normalization = False
        batch_norm_end = True
        mode = 1
        nb_branch = 4
        parametric = [32, 256]
        concat = 'sum'
        vectorization = None
        batch_norm_kwargs['scale'] = True
        name = 'BN-Cov-PV{}-BN-mode1-gamma_{}-br_{}'.format(cov_branch_output, use_gamma, nb_branch)
    elif exp == 11:
        # Test with the new parameterized covariance layer with the simplest structure
        cov_branch_output = 2048
        cov_use_kernel = True
        use_gamma = True
        normalization = True
        use_bias = True

        name = 'BN-Para_Cov-PV{}-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
        mode = 1

    elif exp == 12:
        cov_branch_output = 2048
        use_gamma = False
        name = 'BN-Cov-PV_gamma{}-BN'. \
            format(cov_branch_output, use_gamma)
        normalization = True
        batch_norm_end = True
        batch_norm_kwargs['scale'] = True

    elif exp == 13:
        cov_branch_output = 2048
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN'.format(cov_branch_output)
        # input_shape = (448, 448, 3)
        # batch_size = 16
    elif exp == 14:
        cov_branch_output = 512
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN-448'.format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16
        # weight_decay = 1e-5
        # load_weights = None
    elif exp == 15:
        # from pretrained weights
        cov_branch_output = 512
        batch_norm_end = True
        use_gamma = False
        normalization = True
        use_bias = False
        name = 'BN-Cov-PY{}_final-BN-448-finetune'.format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16
        load_weights = 'secondorder'
    elif exp == 16:
        cov_branch_output = [128, 256, 512, ]
        # cov_branch_output = [4096, 8192]
        batch_norm_end = True
        use_gamma = False
        normalization = True
        use_bias = False
        name = 'BN-Cov-PV_final-BN-compare pv'
    elif exp == 17:
        cov_branch_output = [4096]
        # cov_branch_output = [512, 1024, 4096, 8192]
        batch_norm_end = True
        use_gamma = False
        normalization = True
        use_bias = False
        input_shape = (448, 448, 3)
        batch_size = 16
        name = 'BN-Cov-PV_final-BN-448-compare pv'
    elif exp == 18:
        cov_branch_output = 512
        pv_reg = 'l2'
        # pv_reg = L2InnerNorm(0.01)
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN-448'.format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16
    elif exp == 19:
        cov_branch_output = 2048
        # pv_reg = 'l2'
        # pv_reg = L2InnerNorm(0.01)
        last_conv_feature_maps = [512]
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN-448'.format(cov_branch_output)
        input_shape = (448, 448, 3)
        batch_size = 16
    elif exp == 20:
        cov_branch_output = 2048
        batch_norm_end = False
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final'.format(cov_branch_output)
        # input_shape = (448, 448, 3)
        # batch_size = 16
    elif exp == 21:
        cov_branch_output = 2048
        # pv_reg = 'l2'
        # pv_reg = L2InnerNorm(0.01)
        last_conv_feature_maps = [512]
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN'.format(cov_branch_output)
    elif exp == 22:
        cov_branch_output = 2048
        batch_norm_end = True
        use_gamma = False
        normalization = True  # normalize to further fit Chi-square distribution
        use_bias = False  # use bias for normalization additional
        name = 'BN-Cov-PV{}_final-BN'.format(cov_branch_output)
        pred_activation = 'sigmoid'
    else:
        raise ValueError("exp not reg {}".format(exp))

    cov_kwargs = update_source_dict_by_given_kwargs(
        cov_kwargs,
        cov_mode='channel',
        normalization='mean',
        use_kernel=cov_use_kernel,
    )
    o2t_kwargs = update_source_dict_by_given_kwargs(
        o2t_kwargs,
        activation='relu',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        kernel_constraint=None,
    )
    pv_kwargs = update_source_dict_by_given_kwargs(
        pv_kwargs,
        activation='linear',
        eps=1e-8,
        output_sqrt=True,  # Normalization
        normalization=normalization,  # normalize to further fit Chi-square distribution
        kernel_initializer='glorot_uniform',
        kernel_regularizer=pv_reg,
        use_bias=use_bias,  # use bias for normalization additional
        bias_initializer='zeros',
        bias_regularizer=None,
        use_gamma=use_gamma,  # use gamma for general gaussian distribution
        gamma_initializer='ones',
        gamma_regularizer='l2',
    )

    model_config = NewNormWVBranchConfig(
        # For cov-branch_kwargs
        epsilon=0,
        parametric=parametric,
        vectorization=vectorization,
        batch_norm=True,
        batch_norm_end=batch_norm_end,
        batch_norm_kwargs=batch_norm_kwargs,
        pow_norm=pow_norm,
        cov_kwargs=cov_kwargs,
        o2t_kwargs=o2t_kwargs,
        pv_kwargs=pv_kwargs,
        # Other
        input_shape=input_shape,
        nb_class=67,
        class_id=None,  # Not set here.
        # configs for _compose_second_order_things
        cov_branch_output=cov_branch_output,
        dense_branch_output=dense_branch_output,
        load_weights=load_weights,
        weight_decay=weight_decay,
        mode=mode,
        freeze_conv=False, name=name + '-{}_branch'.format(nb_branch),
        nb_branch=nb_branch,
        concat='concat' if concat is None else concat,
        cov_output_vectorization='pv',
        last_conv_feature_maps=last_conv_feature_maps,
        last_conv_kernel=[1, 1],
        separate_conv_features=separate_conv_features,
        upsample_method='conv',
        batch_size=batch_size,
        pred_activation=pred_activation,
    )
    return model_config


def get_pv_equivalent(exp=1):
    cov_branch_output = 1024
    batch_size = 32
    last_conv_feature_maps = [256]
    load_weights = 'imagenet'
    from keras.layers import Conv2D
    conv_kwargs = create_dict_by_given_kwargs(
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation='linear',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )
    use_bias = True

    gsp_kwargs = get_default_secondstat_args('GlobalSquarePooling')
    output_sqrt = False
    use_gamma = True
    gsp_use_bias = True
    batch_norm_kwargs = {'scale': True}
    nb_branch = 1
    batch_norm_end = False
    if exp == 1:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-1x1_{}-GSP-useGamme_{}'.format(cov_branch_output, use_gamma)
        mode = 1
    elif exp == 2:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-1x1_{}-GSP-useGamme_{}'.format(cov_branch_output, use_gamma)
        mode = 1
        load_weights = 'secondorder'
    elif exp == 3:
        cov_branch_output = 2048
        use_gamma = True
        name = 'SC-BN-1x1_{}-GSP-useGamme_{}'.format(cov_branch_output, use_gamma)
        mode = 1
        load_weights = None
    elif exp == 4:
        cov_branch_output = 512
        name = 'BN-1x1_{}-GSP-FC_mode1-gsplayer'.format(cov_branch_output, use_gamma)
        mode = 1

        # GSP layer
        use_gamma = True
        gsp_use_bias = True
        batch_norm_end = False
        output_sqrt = True

        # Conv layer
        use_bias = False
    elif exp == 5:
        cov_branch_output = 2048
        name = 'BN-1x1_{}-GSP-FC_mode1-convlayer'.format(cov_branch_output, use_gamma)
        mode = 1

        # GSP layer
        use_gamma = True
        gsp_use_bias = True
        batch_norm_end = False
        output_sqrt = True

        # Conv layer
        use_bias = False
    elif exp == 6:
        cov_branch_output = 2048
        mode = 1
        name = '1x1_{}-BN-GSP-BN-FC'.format(cov_branch_output)

        # GSP layer
        gsp_use_bias = False
        use_gamma = False
        output_sqrt = True
        batch_norm_end = True

        # Conv layer
        use_bias = False

        # Training from scratch
        load_weights = None
    else:
        raise ValueError("exp not reg {}".format(exp))

    conv_kwargs = update_source_dict_by_given_kwargs(
        conv_kwargs,
        use_bias=use_bias
    )

    gsp_kwargs = update_source_dict_by_given_kwargs(
        gsp_kwargs,
        activation='linear',
        output_sqrt=output_sqrt,  # Normalization
        normalization=False,  # normalize to further fit Chi-square distribution
        # kernel_initializer='glorot_uniform',
        # kernel_constraint=None,
        # kernel_regularizer=None,
        use_bias=gsp_use_bias,  # use bias for normalization additional
        bias_initializer='zeros',
        bias_regularizer=None,
        bias_constraint=None,
        use_gamma=use_gamma,  # use gamma for general gaussian distribution
        gamma_initializer='ones',
        gamma_regularizer='l2',
        gamma_constraint=None,
        activation_regularizer=None,
    )
    model_config = PVEquivalentConfig(
        # For cov-branch_kwargs
        batch_norm=True,
        batch_norm_end=batch_norm_end,
        conv_kwargs=conv_kwargs,
        gsp_kwargs=gsp_kwargs,
        batch_norm_kwargs=batch_norm_kwargs,
        # Other
        input_shape=(224, 224, 3),
        nb_class=67,
        class_id=None,  # Not set here.
        # configs for _compose_second_order_things
        cov_branch_output=cov_branch_output,
        load_weights=load_weights,
        mode=mode,
        freeze_conv=False, name=name + '-{}_branch'.format(nb_branch),
        cov_output_vectorization='pv',
        last_conv_feature_maps=last_conv_feature_maps,
        last_conv_kernel=[1, 1],
        upsample_method='conv',
    )
    return model_config
