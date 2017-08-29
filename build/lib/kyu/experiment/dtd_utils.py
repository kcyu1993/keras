"""
Define the DTD training pipeline

"""
from kyu.datasets.dtd import DTD
from kyu.experiment.general_train import finetune_with_model_data, get_dirhelper
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
    dirhelper = get_dirhelper(dataset_name=data.name, model_category=model_config.class_id)
    finetune_with_model_data(data, model_config, dirhelper, nb_epoch_finetune, running_config)


if __name__ == '__main__':
    # Test the new argument parser with supported logic
    parser = ArgumentParser()
    parser.add_argument()