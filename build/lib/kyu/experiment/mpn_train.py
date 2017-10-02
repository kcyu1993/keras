"""
Train with MPN network

"""


from kyu.configs.experiment_configs import mpn_config, matrixbp_config
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train

BRANCH_CHOICES = ['mpn_baseline', 'multi_mpn', 'matbp_baseline']


def mpn_train(model_class, model_exp_fn, **kwargs):
    """
    Baseline mpn train pipeline

    Parameters
    ----------
    dataset
    model_exp

    Returns
    -------

    """

    return so_cnn_train(model_class=model_class, model_exp_fn=model_exp_fn,
                        title='MPN-{}'.format(str(model_class).upper()),
                        **kwargs)


def matbp_train(model_class, model_exp_fn, **kwargs):
    return so_cnn_train(model_class=model_class, model_exp_fn=model_exp_fn,
                        title='MatBP-{}'.format(str(model_class).upper()),
                        **kwargs)


if __name__ == '__main__':
    parser = get_argparser(description='MPN or MatBP Train with different dataset and model settings ')
    parser.add_argument('-b', '--branch', default='mpn_baseline', choices=BRANCH_CHOICES)

    args = parser.parse_args()
    branch = args.branch
    del args.branch

    if branch == BRANCH_CHOICES[0]:
        mpn_train(model_exp_fn=mpn_config.get_basic_mpn_model_and_run, **vars(args))
    elif branch == BRANCH_CHOICES[1]:
        mpn_train(model_exp_fn=mpn_config.get_multiple_branch_mpn_model, **vars(args))
    elif branch == BRANCH_CHOICES[2]:
        matbp_train(model_exp_fn=matrixbp_config.get_baseline_matbp_exp, **vars(args))
    else:
        raise ValueError


