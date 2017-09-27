from kyu.configs.experiment_configs import bilinear_config
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train

BRANCH_CHOICES = ['bilinear_baseline', 'multi_bilinear']


def bilinear_train(model_class, model_exp_fn,  **kwargs):
    """
    B-CNN baseline testing

    Parameters
    ----------
    model_class
    kwargs

    Returns
    -------

    """
    return so_cnn_train(model_class=model_class, model_exp_fn=model_exp_fn,
                        title='B-CNN-{}'.format(str(model_class).upper()),
                        **kwargs)


if __name__ == '__main__':
    parser = get_argparser(description='Bilinear Train with different dataset and model settings ')
    parser.add_argument('-b', '--branch', default='bilinear_baseline', choices=BRANCH_CHOICES)

    args = parser.parse_args()
    branch = args.branch
    del args.branch

    if branch == BRANCH_CHOICES[0]:
        bilinear_train(model_exp_fn=bilinear_config.get_bilinear_baseline_exp, **vars(args))
    elif branch == BRANCH_CHOICES[1]:
        bilinear_train(model_exp_fn=bilinear_config.get_bilinear_so_structure_exp, **vars(args))
    else:
        raise ValueError
