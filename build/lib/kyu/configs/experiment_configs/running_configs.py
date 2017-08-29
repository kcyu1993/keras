from ..engine_configs import RunningConfig


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
        optimizer='SGD',
        lr=0.01,
        model_config=model_config,
        # Image Generator Config
        rescale_small=296,
        random_crop=True,
        horizontal_flip=True,
    )


