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