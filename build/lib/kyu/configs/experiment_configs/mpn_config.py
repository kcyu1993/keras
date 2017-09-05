"""
Define the MPN-Cov structure

"""
from ..model_configs import MPNConfig


def get_basic_mpn_model_and_run(exp):

    if exp == 1:
        mpn_config = MPNConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[],
            activation='relu',
            cov_mode='channel',
            vectorization='mat_flatten',
            epsilon=1e-5,
            use_bias=False,
            cov_alpha=0.1,
            cov_beta=0.3,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[256],
            name='MPN-Cov-baseline no normalization'
        )
    elif exp == 2:
        mpn_config = MPNConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[256],
            activation='relu',
            cov_mode='channel',
            vectorization='wv',
            epsilon=1e-5,
            use_bias=True,
            cov_alpha=0.1,
            cov_beta=0.3,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[256],
            cov_branch_output=1024,
            name='MPN-Cov-with-O2T-and-pv'

        )
    elif exp == 3:
        mpn_config = MPNConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[],
            activation='relu',
            cov_mode='channel',
            vectorization='wv',
            epsilon=1e-5,
            use_bias=True,
            cov_alpha=0.1,
            cov_beta=0.3,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[256],
            cov_branch_output=1024,
            name='MPN-Cov-PV-only'
        )
    else:
        raise ValueError
    return mpn_config


def get_multiple_branch_mpn_model(exp=1):
    if exp == 1:
        mpn_config = MPNConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[],
            activation='relu',
            cov_mode='channel',
            vectorization='no',
            use_bias=False,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[512],
            nb_branch=2,
            cov_branch_output=1024,
            concat='concat',
            name='MPN-Cov with 2 branch by mat concat'
        )
    else:
        raise ValueError("Not supported")
    return mpn_config
