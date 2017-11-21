"""
Baseline for first order tests

"""
from kyu.configs.experiment_configs.first_order import get_fo_vgg_exp, get_fo_dense_exp, get_fo_resnet_exp, \
    get_fo_alexnet_exp
from kyu.experiment.general_train import get_argparser
from kyu.experiment.so_train import so_cnn_train


def baseline_first_order_train_multilabel(model_class, **kwargs):
    """
    Multiple label first order train environment
    Parameters
    ----------
    model_class
    kwargs

    Returns
    -------

    """




def baseline_first_order_train(model_class, **kwargs):
    if 'chest' in kwargs['dataset']:
        baseline_first_order_train_multilabel(model_class, **kwargs)
        return

    if str(model_class).lower() == 'vgg16':
        model_exp_fn = get_fo_vgg_exp
    elif str(model_class).lower() == 'densenet121':
        model_exp_fn = get_fo_dense_exp
    elif str(model_class).lower() == 'resnet50':
        model_exp_fn = get_fo_resnet_exp
    elif str(model_class).lower() == 'alexnet':
        model_exp_fn = get_fo_alexnet_exp
    else:
        raise NotImplementedError

    return so_cnn_train(model_class=model_class,
                        model_exp_fn=model_exp_fn,
                        title="FO-{}-baseline".format(str(model_class).upper()),
                        **kwargs)


if __name__ == '__main__':
    parser = get_argparser(description='FO Train with different settings ')
    args = parser.parse_args()

    # Model class
    baseline_first_order_train(**vars(args))
