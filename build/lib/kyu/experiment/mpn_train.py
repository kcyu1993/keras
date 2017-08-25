"""
Train with MPN network

"""


from kyu.configs.experiment_configs.mpn_config import get_basic_mpn_model_and_run
from kyu.experiment.dtd_utils import dtd_finetune_with_model
from kyu.experiment.general_train import get_argparser
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
    parser = get_argparser(description='MPN Train with different dataset and model settings ')
    baseline_mpn_train(**vars(parser.parse_args()))
