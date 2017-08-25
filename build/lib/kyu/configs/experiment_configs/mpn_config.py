"""
Define the MPN-Cov structure

"""
from .running_configs import get_running_config_no_debug_withSGD
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
            last_conv_feature_maps=[256]

        )
        running_config = get_running_config_no_debug_withSGD('MPN-Cov-baseline no normalization', mpn_config)
    elif exp == 2:
        mpn_config = MPNConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[128],
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
            cov_branch_output=128

        )
        running_config = get_running_config_no_debug_withSGD('MPN-Cov-with-O2T-and-pv', mpn_config)
        running_config.comments = 'test with WV and bias enable'
    else:
        raise ValueError

    return mpn_config, running_config