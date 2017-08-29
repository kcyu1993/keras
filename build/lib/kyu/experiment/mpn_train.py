"""
Train with MPN network

"""

import argparse

from kyu.configs.experiment_configs.mpn_config import get_basic_mpn_model_and_run
from kyu.experiment.dtd_utils import dtd_finetune_with_model
from kyu.experiment.minc_utils import minc_finetune_with_model

def baseline_mpn_train(dataset, model_exp=1, nb_epoch_finetune=0):
    """
    Baseline mpn train pipeline

    Parameters
    ----------
    dataset
    model_exp

    Returns
    -------

    """
    dataset = str(dataset).lower()
    mpn_config, running_config = get_basic_mpn_model_and_run(model_exp)

    if dataset == 'dtd':
        dtd_finetune_with_model(mpn_config, nb_epoch_finetune, running_config)

    elif dataset == 'minc2500' or dataset == 'minc-2500':
        minc_finetune_with_model(mpn_config, nb_epoch_finetune, running_config)

    else:
        raise NotImplementedError("Dataset not support {}".format(dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPN Train with different dataset and model settings ')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name: support dtd, minc2500')
    parser.add_argument('-me', '--model_exp', help='model experiment index', type=int, default=1)
    parser.add_argument('-ef', '--nb_epoch_finetune', help='number of epoch to be finetuned', default=0, type=int)
    args = parser.parse_args()
    baseline_mpn_train(**vars(args))
