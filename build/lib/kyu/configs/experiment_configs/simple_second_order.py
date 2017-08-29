from kyu.configs.model_configs import NoWVBranchConfig
from ..model_configs import O2TBranchConfig


def get_single_o2transform(exp):
    """ Same as before, updated with the new framework """
    if exp == 1:
        return O2TBranchConfig(
            parametric=[128],
            activation='relu',
            cov_mode='pmean',
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

    else:

        raise ValueError("N")


def get_no_wv_config(exp=1):

    if exp == 1:
        model_config = NoWVBranchConfig(
            parametric=[128],
            epsilon=1e-7,
            activation='relu',
            cov_mode='channel',
            cov_alpha=0.3,
            cov_beta=0.1,
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
            freeze_conv=False, name='No-PV-2branch-128-128',
            nb_branch=2,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[256],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    else:
        raise ValueError("N")

    return model_config