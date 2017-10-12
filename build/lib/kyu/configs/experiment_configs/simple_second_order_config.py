from kyu.configs.model_configs.second_order import NewNormWVBranchConfig, PVEquivalentConfig
from kyu.layers.secondstat import get_default_secondstat_args
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
    dense_branch_output = None
    last_conv_feature_maps = [256]
    parametric = []
    nb_branch = 1
    vectorization = 'wv'
    load_weights = 'imagenet'
    mode = 1
    concat = None
    cov_kwargs = get_default_secondstat_args('Cov')
    o2t_kwargs = get_default_secondstat_args('O2T')
    pv_kwargs = get_default_secondstat_args('PV')
    batch_norm_kwargs = {'scale': True}

    # PV layers
    pow_norm = False
    use_gamma = True
    batch_norm_end = False
    normalization = True,  # normalize to further fit Chi-square distribution
    use_bias = True,  # use bias for normalization additional

    if exp == 1:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-Cov-PV{}-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
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
        nb_branch = 2
        last_conv_feature_maps = [512]
        pow_norm = True
        vectorization = None
        name = "BN-Cov-Pow-PV{}-mode1_complete".format(cov_branch_output)
    elif exp == 5:
        nb_branch = 2
        last_conv_feature_maps = [512]
        pow_norm = False
        vectorization = None
        parametric = [256]
        concat = 'concat'
        use_gamma = True
        cov_branch_output = 512
        name = "BN-Cov-O2T{}-PV{}-mode1_complete".format(parametric, cov_branch_output)
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
        use_gamma = True
        parametric = [128,]
        name = 'BN-Cov-O2T-PV{}-mode1_complete-gamma{}'.\
            format(cov_branch_output, parametric, use_gamma)
        batch_norm_kwargs['scale'] = False
    elif exp == 9:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-Cov-PV{}-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
        mode = 1
        load_weights = 'secondorder'
    elif exp == 10:
        cov_branch_output = 2048
        use_gamma = False
        use_bias = False
        normalization = False
        batch_norm_end = True
        mode = 1
        name = 'BN-Cov-PV{}-BN-mode1_complete-gamma{}'.format(cov_branch_output, use_gamma)
    else:
        raise ValueError("exp not reg {}".format(exp))

    cov_kwargs = update_source_dict_by_given_kwargs(
        cov_kwargs,
        cov_mode='channel',
        normalization='mean',
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
        kernel_regularizer=None,
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
        input_shape=(224, 224, 3),
        nb_class=67,
        class_id=None,  # Not set here.
        # configs for _compose_second_order_things
        cov_branch_output=cov_branch_output,
        dense_branch_output=dense_branch_output,
        load_weights=load_weights,
        mode=mode,
        freeze_conv=False, name=name + '-{}_branch'.format(nb_branch),
        nb_branch=nb_branch,
        concat='concat' if concat is None else concat,
        cov_output_vectorization='pv',
        last_conv_feature_maps=last_conv_feature_maps,
        last_conv_kernel=[1, 1],
        upsample_method='conv',
    )
    return model_config


def get_pv_equivalent(exp=1):
    cov_branch_output = 1024
    last_conv_feature_maps = [256]
    load_weights = 'imagenet'
    from keras.layers import Conv2D
    conv_kwargs = create_dict_by_given_kwargs(
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )
    gsp_kwargs = get_default_secondstat_args('GlobalSquarePooling')
    batch_norm_kwargs = {'scale': True}
    nb_branch = 1
    batch_norm_end = False
    use_gamma = True
    if exp == 1:
        cov_branch_output = 2048
        use_gamma = True
        name = 'BN-1x1_{}-GSP-useGamme_{}'.format(cov_branch_output, use_gamma)
        mode = 1
    else:
        raise ValueError("exp not reg {}".format(exp))

    conv_kwargs = update_source_dict_by_given_kwargs(
        conv_kwargs,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )

    gsp_kwargs = update_source_dict_by_given_kwargs(
        gsp_kwargs,
        activation='linear',
        output_sqrt=False,  # Normalization
        normalization=False,  # normalize to further fit Chi-square distribution
        # kernel_initializer='glorot_uniform',
        # kernel_constraint=None,
        # kernel_regularizer=None,
        use_bias=False,  # use bias for normalization additional
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
