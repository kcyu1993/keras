"""
Get Model Config
"""

from kyu.models.single_second_order import O2TBranchConfig


def get_o2t_testing(exp):
    """ Same as before, updated with the new framework """
    if exp == 1:
        return O2TBranchConfig(
            parametric=[],
            activation='relu',
            cov_mode='channel',
            vectorization='wv',
            use_bias=True,
            epsilon=1e-5,
            input_shape=(256,256,3),
            nb_class=47,
            cov_branch='o2transform',
            class_id='vgg',
            load_weights='imagenet',
            # configs for _compose_second_order_things
            mode=1, cov_branch_output=None,
            freeze_conv=False, name='TestingO2T-bias',
            nb_branch=1,
            concat='concat',
            cov_output_vectorization='pv',
            last_conv_feature_maps=[],
            last_conv_kernel=[1, 1],
            upsample_method='conv',
        )
    else:
        raise ValueError("N")
