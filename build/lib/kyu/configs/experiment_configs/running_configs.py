from kyu.utils.dict_utils import create_dict_by_given_kwargs
from ..engine_configs import RunningConfig
from tensorflow.python import debug as tfdbg


def get_running_config_no_debug_withSGD(title='general-testing', model_config=None):

    config = RunningConfig(
        _title=title,
        nb_epoch=150,
        batch_size=getattr(model_config,'batch_size') if hasattr(model_config, 'batch_size') else 32,
        verbose=2,
        lr_decay=False,
        sequence=8,
        patience=8,
        early_stop=False,
        save_weights=True,
        load_weights=False,
        init_weights_location=None,
        save_per_epoch=True,
        tensorboard=None,
        lr=0.01,
        optimizer=None,
        model_config=model_config,
        # Image Generator Config
        train_image_gen_configs=None,
        valid_image_gen_configs=None,
        tf_debug=False,
        tf_debug_filters_func=[tfdbg.has_inf_or_nan,],
        tf_debug_filters_name=['has_inf_or_nan',],
    )

    config.train_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=[449, 500], random_crop=True, horizontal_flip=True)
    config.valid_image_gen_configs = create_dict_by_given_kwargs(
            rescaleshortedgeto=449, random_crop=False, horizontal_flip=True)

    return config


