"""
Define the API for model with config
"""


def get_so_model_from_config(fn_model, param, mode, cov_output, nb_classes, input_shape, config, **kwargs):
    kwargs = config.kwargs
    model = fn_model(parametrics=param, mode=mode, cov_branch=config.cov_branch,
                     cov_mode=config.cov_mode,
                     nb_classes=nb_classes, cov_branch_output=cov_output,
                     input_shape=input_shape,
                     last_avg=False,
                     freeze_conv=True,
                     cov_regularizer=config.cov_regularizer,
                     nb_branch=config.nb_branch,
                     last_conv_feature_maps=config.last_conv_feature_maps,
                     epsilon=config.epsilon,
                     vectorization=config.vectorization,
                     **kwargs
                     )

    return model
