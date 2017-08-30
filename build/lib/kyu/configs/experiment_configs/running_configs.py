from ..engine_configs import RunningConfig
from tensorflow.python import debug as tfdbg

from keras.optimizers import SGD

def get_running_config_no_debug_withSGD(title='general-testing', model_config=None):
    return RunningConfig(
        _title=title,
        nb_epoch=200,
        batch_size=32,
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
        optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-5),
        model_config=model_config,
        # Image Generator Config
        rescale_small=296,
        random_crop=True,
        horizontal_flip=True,
        tf_debug=False,
        tf_debug_filters_func=[tfdbg.has_inf_or_nan,],
        tf_debug_filters_name=['has_inf_or_nan',],
    )


