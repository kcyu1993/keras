from kyu.configs.model_configs.second_order import MatrixBackPropConfig
from kyu.layers.secondstat import get_default_secondstat_args
from kyu.utils.dict_utils import update_source_dict_by_given_kwargs


def get_baseline_matbp_exp(exp=1):
    cov_branch_output = 1024
    dense_branch_output = None
    last_conv_feature_maps = [256]
    parametric = []
    nb_branch = 1
    vectorization = 'mat_flatten'
    mode = 1
    concat = None
    cov_kwargs = get_default_secondstat_args('Cov')
    o2t_kwargs = get_default_secondstat_args('O2T')
    pv_kwargs = get_default_secondstat_args('PV')
    log_norm = True
    use_gamma = True
    batch_norm = True
    if exp == 1:
        batch_norm = False
        name = 'MatBP-baseline{}'.format('-BN' if batch_norm else '')
        mode = 1
    # elif exp == 2:

    else:
        raise ValueError("wrong")
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
        normalization=True,  # normalize to further fit Chi-square distribution
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None,
        use_bias=True,  # use bias for normalization additional
        bias_initializer='zeros',
        bias_regularizer=None,
        use_gamma=use_gamma,  # use gamma for general gaussian distribution
        gamma_initializer='ones',
        gamma_regularizer='l2',
    )
    model_config = MatrixBackPropConfig(
        # For cov-branch_kwargs
        epsilon=0,
        parametric=parametric,
        vectorization=vectorization,
        batch_norm=batch_norm,
        log_norm=log_norm,
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
        load_weights='imagenet',
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