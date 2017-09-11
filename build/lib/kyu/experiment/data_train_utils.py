"""
Define the DTD training pipeline

"""
from kyu.datasets.imagenet import ImageNetData
from kyu.datasets.mit import MitIndoor

from kyu.datasets.sun import SUN397, SUN397_v2

from kyu.datasets.dtd import DTD
from kyu.datasets.minc import Minc2500_v2
from kyu.experiment.general_train import finetune_with_model_data, get_dirhelper, get_debug_dirhelper
from kyu.configs.experiment_configs import *
from kyu.utils.io_utils import ProjectFile

from argparse import ArgumentParser


def dtd_finetune_with_model(model_config, nb_epoch_finetune, running_config):
    """
    General training pipeline for DTD dataset. Where passing a model config is enough.
    
    :param model_config: 
    :param nb_epoch_finetune: 
    :param running_config: 
    :return: 
    """
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    data_finetune_with_model(data, model_config, nb_epoch_finetune, running_config)


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
    data_finetune_with_model(data, model_config, nb_epoch_finetune, running_config)


def sun_finetune_with_model(**kwargs):
    """
    SUN397 dataset training

    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    data = SUN397_v2('/home/kyu/.keras/datasets/sun')
    data_finetune_with_model(data, **kwargs)


def mit_finetune_with_model(**kwargs):
    """
    MitIndoor training
    Parameters
    ----------
    kwargs

    Returns
    -------

    """
    data = MitIndoor('/home/kyu/.keras/datasets/mit_indoor')
    data_finetune_with_model(data, **kwargs)


def imagenet_finetune_with_model(**kwargs):
    data = ImageNetData('/home/kyu/.keras/datasets/ILSVRC2015')
    data_finetune_with_model(data, **kwargs)


def data_finetune_with_model(data, model_config, nb_epoch_finetune, running_config):
    """
    Finetune with data passing

    Parameters
    ----------
    data
    model_config
    nb_epoch_finetune
    running_config

    Returns
    -------

    """
    if running_config.debug:
        dirhelper = get_debug_dirhelper(dataset_name=data.name, model_category=model_config.class_id)
    else:
        dirhelper = get_dirhelper(dataset_name=data.name, model_category=model_config.class_id)

    finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config)
