from kyu.configs.model_configs.iccv_pow_transform import MPNConfig
from kyu.theano.dtd.new_train import get_running_config, finetune_with_model


def mpn_cov_baseline(exp=1):

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
        running_config = get_running_config('MPN-Cov-baseline no normalization', mpn_config)
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
            last_conv_feature_maps=[256]

        )
        running_config = get_running_config('MPN-Cov-with-O2T-and-pv', mpn_config)
        running_config.comments = 'test with WV and bias enable'
    else:
        raise ValueError
    finetune_with_model(
        mpn_config,
        4,
        running_config
    )


if __name__ == '__main__':
    mpn_cov_baseline(2)