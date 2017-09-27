from kyu.configs.model_configs import BilinearConfig, BilinearSOConfig


def get_bilinear_baseline_exp(exp=1):

    if exp == 1:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224,224,3),
            load_weights='imagenet',
            name='BCNN-Baseline',
        )
    elif exp == 2:
        return BilinearConfig(
            nb_class=10,
            input_shape=(224, 224, 3),
            last_conv_kernel=[256],
            load_weights='imagenet',
            name='BCNN-Baseline',
        )
    else:
        raise NotImplementedError


def get_bilinear_so_structure_exp(exp=1):
    if exp == 2:
        branch = 2
        return BilinearSOConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            parametric=[256],
            activation='relu',
            cov_mode='channel',
            vectorization='wv',
            use_bias=False,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[512],
            nb_branch=branch,
            cov_branch_output=1024,
            concat='concat',
            name='BCNN-Cov {} branch by mat concat'.format(branch)
        )
    elif exp == 1:
        return BilinearSOConfig(
            input_shape=(224, 224, 3),
            nb_class=67,
            # parametric=[],
            parametric=[256],
            activation='relu',
            cov_mode='channel',
            vectorization='wv',
            use_bias=False,
            normalization=None,
            mode=1,
            last_conv_feature_maps=[512],
            nb_branch=1,
            cov_branch_output=1024,
            concat='concat',
            name='BCNN-Cov with PV only'
        )
