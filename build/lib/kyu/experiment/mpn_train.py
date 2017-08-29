"""
Train with MPN network

"""


from kyu.configs.experiment_configs.mpn_config import get_basic_mpn_model_and_run
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train


def baseline_mpn_train(model_class, **kwargs):
    """
    Baseline mpn train pipeline

    Parameters
    ----------
    dataset
    model_exp

    Returns
    -------

    """

    return so_cnn_train(model_class=model_class, model_exp_fn=get_basic_mpn_model_and_run,
                        title='MPN-{}-baseline'.format(str(model_class).upper()),
                        **kwargs)

if __name__ == '__main__':
    parser = get_argparser(description='MPN Train with different dataset and model settings ')
    baseline_mpn_train(**vars(parser.parse_args()))
