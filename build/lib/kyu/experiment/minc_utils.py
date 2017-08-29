"""
Define the minc training pipeline
"""
from kyu.datasets.minc import Minc2500_v2
from kyu.experiment.general_train import get_dirhelper, finetune_with_model_data


def minc_finetune_with_model(model_config, nb_epoch_finetune, running_config):
    """
    General training with finetuning for MINC dataset.

    Parameters
    ----------
    model_config
    nb_epoch_finetune
    running_config

    Returns
    -------

    """
    data = Minc2500_v2('/home/kyu/.keras/datasets/minc-2500', image_dir=None)
    dirhelper = get_dirhelper(dataset_name=data.name, model_category=model_config.class_id)
    finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config)


def mpn_baseline(exp=1):

    from kyu.configs.model_configs import MPNConfig
    if exp == 1:
        config = MPNConfig(

        )
