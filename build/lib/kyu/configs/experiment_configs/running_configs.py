from kyu.datasets.chestxray14 import preprocess_image_for_chestxray, chestxray_labels
from kyu.datasets.cub import preprocess_image_for_cub
from kyu.tensorflow.metrics import IndexBinaryAccuracy
from kyu.utils.dict_utils import create_dict_by_given_kwargs
from ..engine_configs import RunningConfig
from tensorflow.python import debug as tfdbg


def get_running_config_no_debug_withSGD(title='general-testing', model_config=None):

    config = RunningConfig(
        _title=title,
        nb_epoch=120,
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
        metrics=['acc'],
        optimizer=None,
        model_config=model_config,
        # Image Generator Config
        train_image_gen_configs=None,
        valid_image_gen_configs=None,
        tf_debug=False,
        tf_debug_filters_func=[tfdbg.has_inf_or_nan,],
        tf_debug_filters_name=['has_inf_or_nan',],
    )
    # config.train_image_gen_configs = create_dict_by_given_kwargs(
    #         rescaleshortedgeto=[449, 500], random_crop=True, horizontal_flip=True)
    # config.valid_image_gen_configs = create_dict_by_given_kwargs(
    #         rescaleshortedgeto=449, random_crop=False, horizontal_flip=True)
    return config


def get_running_config_for_cub(title='general-cub', model_config=None):
    config = RunningConfig(
        _title=title,
        nb_epoch=100,
        batch_size=getattr(model_config, 'batch_size') if hasattr(model_config, 'batch_size') else 32,
        verbose=2,
        lr_decay=True,
        sequence=30,
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
        tf_debug_filters_func=[tfdbg.has_inf_or_nan, ],
        tf_debug_filters_name=['has_inf_or_nan', ],
    )

    config.train_image_gen_configs = create_dict_by_given_kwargs(
        # rescaleshortedgeto=[449, 500], random_crop=True, horizontal_flip=True,
        rescaleshortedgeto=[450, 592], random_crop=True, horizontal_flip=True,
        preprocessing_function=preprocess_image_for_cub,
    )
    config.valid_image_gen_configs = create_dict_by_given_kwargs(
        rescaleshortedgeto=450, random_crop=False, horizontal_flip=True,
        preprocessing_function=preprocess_image_for_cub,
    )

    return config


def get_running_config_for_chest(title='general-chestxray', model_config=None):

    metrics = ['binary_accuracy']
    categories = chestxray_labels()
    for ind, cate in enumerate(categories):
        metrics.append(IndexBinaryAccuracy(ind, '{}_{}'.format(ind, cate[:4])))

    config = RunningConfig(
        _title=title,
        nb_epoch=200,
        batch_size=getattr(model_config, 'batch_size') if hasattr(model_config, 'batch_size') else 32,
        verbose=2,
        lr_decay=True,
        sequence=30,
        patience=8,
        early_stop=False,
        save_weights=True,
        load_weights=False,
        init_weights_location=None,
        save_per_epoch=True,
        tensorboard=None,
        lr=0.01,
        optimizer=None,
        loss='categorical_crossentropy',
        metrics=['binary_accuracy', IndexBinaryAccuracy(0, '0_Atele')],
        model_config=model_config,
        # Image Generator Config
        train_image_gen_configs=None,
        valid_image_gen_configs=None,
        tf_debug=False,
        tf_debug_filters_func=[tfdbg.has_inf_or_nan, ],
        tf_debug_filters_name=['has_inf_or_nan', ],
    )

    config.train_image_gen_configs = create_dict_by_given_kwargs(
        # rescaleshortedgeto=[449, 500], random_crop=True, horizontal_flip=True,
        rescaleshortedgeto=[225, 256], random_crop=True, horizontal_flip=True,
        preprocessing_function=preprocess_image_for_chestxray,
    )
    config.valid_image_gen_configs = create_dict_by_given_kwargs(
        rescaleshortedgeto=256, random_crop=False, horizontal_flip=True,
        preprocessing_function=preprocess_image_for_chestxray,
    )

    return config


def get_custom_metrics_objects_for_chest():
    """
    Define the custom objects
    Returns
    -------

    """

    return {'0_Atele_acc': IndexBinaryAccuracy(0, '0_Atele')}
